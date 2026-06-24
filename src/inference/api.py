"""
FastAPI inference server.

Accepts a chest X-ray image, returns the prediction (NORMAL/PNEUMONIA),
the probability, and the Grad-CAM heatmap as a base64-encoded PNG.

Usage:
    uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
"""

import base64
import io
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src.data.transforms import get_val_transforms_albumentations
from src.explainability.gradcam import GradCAM, get_target_layer
from src.models.classifier import PneumoniaClassifier


CONFIG_PATH = "configs/default.yaml"
with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

_THRESHOLD = _cfg["threshold"]["default"]
_IMAGE_SIZE = _cfg["data"]["image_size"]
_ONNX_PATH = _cfg["paths"]["onnx_export"]
_CKPT_PATH = Path(_cfg["paths"]["checkpoints"])

app = FastAPI(title="TruePneumoniaAI v0.2", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ONNX session for fast inference
_ort_session: ort.InferenceSession | None = None
# Full PyTorch model kept for Grad-CAM only
_pt_model: PneumoniaClassifier | None = None
_transform = get_val_transforms_albumentations(_IMAGE_SIZE)


def _get_ort_session() -> ort.InferenceSession:
    global _ort_session
    if _ort_session is None:
        _ort_session = ort.InferenceSession(_ONNX_PATH, providers=["CPUExecutionProvider"])
    return _ort_session


def _get_pt_model() -> PneumoniaClassifier:
    global _pt_model
    if _pt_model is None:
        # rglob: checkpoints now live in per-run subdirs (checkpoints/<task>/<ts>/).
        ckpts = sorted(_CKPT_PATH.rglob("best-*.ckpt"), key=lambda p: p.stat().st_mtime)
        if not ckpts:
            raise RuntimeError(f"No checkpoint found in {_CKPT_PATH}")
        _pt_model = PneumoniaClassifier.load_from_checkpoint(str(ckpts[-1]))
        _pt_model.eval()
    return _pt_model


class PredictionResponse(BaseModel):
    label: str
    probability: float
    threshold: float
    gradcam_png_b64: str | None = None


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), gradcam: bool = True):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    augmented = _transform(image=img_np)
    tensor = augmented["image"].unsqueeze(0).float() # (1, 3, H, W)

    ort_input = {_get_ort_session().get_inputs()[0].name: tensor.numpy()}
    logit = float(_get_ort_session().run(None, ort_input)[0][0])
    prob = float(1 / (1 + np.exp(-logit)))
    label = "PNEUMONIA" if prob >= _THRESHOLD else "NORMAL"

    gradcam_b64 = None
    if gradcam:
        model = _get_pt_model()
        backbone_name = _cfg["model"]["backbone"]
        target_layer = get_target_layer(model, backbone_name)
        cam_gen = GradCAM(model, target_layer)
        cam = cam_gen.generate(tensor)
        overlay = cam_gen.overlay(img_np, cam)
        pil_overlay = Image.fromarray(overlay)
        buf = io.BytesIO()
        pil_overlay.save(buf, format="PNG")
        gradcam_b64 = base64.b64encode(buf.getvalue()).decode()

    return PredictionResponse(
        label=label,
        probability=round(prob, 4),
        threshold=_THRESHOLD,
        gradcam_png_b64=gradcam_b64,
    )


@app.get("/health")
def health():
    return {"status": "ok", "threshold": _THRESHOLD}
