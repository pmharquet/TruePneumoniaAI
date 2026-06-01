"""
TruePneumoniaAI — Serveur d'inférence
Usage : cd ai && python inference_server.py
Accessible sur http://localhost:8001/
"""

import os
import sys
import base64
import zipfile
import tempfile
import numpy as np
import cv2

from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

sys.path.insert(0, os.path.dirname(__file__))

from xp import xp, GPU
from ConvolutionLayer import ConvolutionLayer as CONV
from RectifiedLinearUnitLayer import RectifiedLinearUnitLayer as RELU
from PoolingLayer import PoolingLayer as POOL
from GlobalAveragePoolingLayer import GlobalAveragePoolingLayer as GAP
from ClassActivationMapLayer import ClassActivationMapLayer as CAM_LAYER
from FullyConnected import FullyConnectedLayer as FC
from SoftmaxLayer import SoftmaxLayer as SOFTMAX

NB_BLOCS     = 5
NB_CONV_BLOC = 3
NB_FILTRES   = 48
KERNEL_SIZE  = 3
STRIDE_CONV  = 1
POOL_SIZE    = 2
STRIDE_POOL  = 2
FC_HIDDEN    = 128
TARGET_W     = 660   # largeur cible (même logique que imageResize.py : 1320/2)
TARGET_H     = 484   # hauteur cible (968/2)

CLASS_NAMES = ["Normal", "Bactérien", "Viral"]

_ROOT     = Path(__file__).parent.parent
_HTML     = Path(__file__).parent / "inference.html"
_RESULTS  = _ROOT / "training_results"

app = FastAPI(title="TruePneumoniaAI — Inference")

_model_cache: dict = {}


def preprocess_image(img):
    """Même logique que data/2_image_resize/imageResize.py :
    - ratio conservé
    - bandes noires (letterbox/pillarbox) pour atteindre TARGET_W × TARGET_H
    - interpolation AREA si réduction, CUBIC si agrandissement
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orig_h, orig_w = img.shape
    ratio = orig_w / orig_h

    if TARGET_H * ratio > TARGET_W:
        nex_w, nex_h = TARGET_W, int(TARGET_W / ratio)
    else:
        nex_w, nex_h = int(TARGET_H * ratio), TARGET_H

    interp = cv2.INTER_AREA if (orig_w > TARGET_W or orig_h > TARGET_H) else cv2.INTER_CUBIC
    resized = cv2.resize(img, (nex_w, nex_h), interpolation=interp)

    canvas = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
    y_off = (TARGET_H - nex_h) // 2
    x_off = (TARGET_W - nex_w) // 2
    canvas[y_off:y_off + nex_h, x_off:x_off + nex_w] = resized
    return canvas


def _to_cpu(x):
    return x.get() if hasattr(x, "get") else x


def build_network():
    blocs = []
    in_ch = 1
    for _ in range(NB_BLOCS):
        convs, relus = [], []
        for _ in range(NB_CONV_BLOC):
            convs.append(CONV.create(NB_FILTRES, KERNEL_SIZE, KERNEL_SIZE, in_ch, STRIDE_CONV))
            relus.append(RELU())
            in_ch = NB_FILTRES
        blocs.append({"convs": convs, "relus": relus, "pool": POOL(POOL_SIZE, STRIDE_POOL)})
    fc1 = FC(NB_FILTRES, FC_HIDDEN)
    fc2 = FC(FC_HIDDEN, 3)
    return blocs, GAP(), RELU(), fc1, fc2, SOFTMAX()


def _open_checkpoint(path: str):
    """Ouvre un checkpoint .npz ou .zip contenant un .npz."""
    p = Path(path)
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist() if n.endswith(".npz")]
            if not names:
                raise ValueError(f"Aucun .npz trouvé dans {p.name}")
            with z.open(names[0]) as f:
                return np.load(f, allow_pickle=False)
    return np.load(path, allow_pickle=False)


def load_checkpoint(path, blocs, fc1, fc2):
    data = _open_checkpoint(path)
    for bi, bloc in enumerate(blocs):
        for ci, conv in enumerate(bloc["convs"]):
            conv.kernel = xp.asarray(data[f"conv_{bi}_{ci}"])
            conv.d_kernel = xp.zeros_like(conv.kernel)
    for ni, neuron in enumerate(fc1.neurons):
        neuron.weights = data[f"fc1_w_{ni}"]
        neuron.bias    = data[f"fc1_b_{ni}"]
    for ni, neuron in enumerate(fc2.neurons):
        neuron.weights = data[f"fc2_w_{ni}"]
        neuron.bias    = data[f"fc2_b_{ni}"]
    return int(data["epoch"][0]), float(data["val_acc"][0])


def get_model(model_path: str):
    if model_path not in _model_cache:
        blocs, gap, relu_fc, fc1, fc2, softmax = build_network()
        epoch, val_acc = load_checkpoint(model_path, blocs, fc1, fc2)
        _model_cache[model_path] = (blocs, gap, relu_fc, fc1, fc2, softmax, epoch, val_acc)
    return _model_cache[model_path]


def run_inference(image_np, blocs, gap, relu_fc, fc1, fc2, softmax):
    data = xp.asarray(image_np.astype(np.float32))
    if data.ndim == 2:
        data = data[:, :, xp.newaxis]

    for bloc in blocs:
        for conv, relu in zip(bloc["convs"], bloc["relus"]):
            data = relu.forward(conv.forward(data))
        data = bloc["pool"].forward(data)

    last_feature_maps = _to_cpu(data)  # [H', W', 48]

    gap_out  = _to_cpu(gap.forward(data))        # [48]
    fc1_out  = relu_fc.forward(fc1.forward(gap_out))  # [128]
    fc2_out  = fc2.forward(fc1_out)              # [3]
    probs    = softmax.forward(fc2_out)          # [3]
    pred_cls = int(np.argmax(probs))

    # CAM — poids effectifs via la composition linéaire FC2 ∘ FC1
    # fc1_W[k, d] = fc1.neurons[k].weights[d]  →  shape [128, 48]
    # fc2_w[k]    = fc2.neurons[pred_cls].weights[k]  →  shape [128]
    # cam_w[d]    = sum_k(fc2_w[k] * fc1_W[k, d])  →  shape [48]
    fc1_W  = np.array([n.weights for n in fc1.neurons])        # [128, 48]
    fc2_w  = np.array(fc2.neurons[pred_cls].weights)           # [128]
    cam_w  = fc2_w @ fc1_W                                      # [48]

    cam_map = CAM_LAYER().forward(last_feature_maps, cam_w)     # [H', W'] uint8
    return probs, pred_cls, cam_map


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return HTMLResponse(_HTML.read_text(encoding="utf-8"))


@app.get("/api/models")
def list_models():
    models = []
    if not _RESULTS.exists():
        return models
    for version_dir in sorted(_RESULTS.iterdir()):
        if not version_dir.is_dir():
            continue
        ckpt = next(
            (version_dir / name for name in ("checkpoint_best.npz", "checkpoint_best.zip")
             if (version_dir / name).exists()),
            None,
        )
        if ckpt is None:
            continue
        eval_path = version_dir / "eval.txt"
        summary = ""
        if eval_path.exists():
            lines = eval_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            summary = " | ".join(l.strip() for l in lines[:4] if l.strip())
        models.append({
            "name": version_dir.name,
            "path": str(ckpt),
            "summary": summary,
        })
    return models


@app.post("/api/predict")
async def predict(model_path: str = Form(...), image: UploadFile = File(...)):
    raw = await image.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return JSONResponse({"error": "Image invalide ou format non supporté."}, status_code=400)

    img_resized = preprocess_image(img)

    blocs, gap, relu_fc, fc1, fc2, softmax, epoch, val_acc = get_model(model_path)
    probs, pred_cls, cam_map = run_inference(img_resized, blocs, gap, relu_fc, fc1, fc2, softmax)

    # Superposition CAM sur l'image originale
    img_color  = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    cam_up     = cv2.resize(cam_map, (TARGET_W, TARGET_H))
    heatmap    = cv2.applyColorMap(cam_up, cv2.COLORMAP_JET)
    overlay    = cv2.addWeighted(img_color, 0.55, heatmap, 0.45, 0)

    _, enc = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cam_b64 = "data:image/jpeg;base64," + base64.b64encode(enc).decode()

    _, enc_orig = cv2.imencode(".jpg", img_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    orig_b64 = "data:image/jpeg;base64," + base64.b64encode(enc_orig).decode()

    return {
        "probabilities": {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(3)},
        "prediction": CLASS_NAMES[pred_cls],
        "cam_image": cam_b64,
        "orig_image": orig_b64,
        "model_epoch": epoch,
        "model_val_acc": round(val_acc * 100, 1),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
