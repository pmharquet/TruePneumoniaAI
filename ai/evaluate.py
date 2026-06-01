"""
TruePneumoniaAI — Évaluation du meilleur modèle
Usage : cd ai && python evaluate.py
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from xp import xp, GPU

from ConvolutionLayer import ConvolutionLayer as CONV
from RectifiedLinearUnitLayer import RectifiedLinearUnitLayer as RELU
from PoolingLayer import PoolingLayer as POOL
from GlobalAveragePoolingLayer import GlobalAveragePoolingLayer as GAP
from FullyConnected import FullyConnectedLayer as FC
from SoftmaxLayer import SoftmaxLayer as SOFTMAX

CLASS_NAMES = ["Normal", "Bactérien", "Viral"]
SEP = "=" * 55

NB_BLOCS     = 5
NB_CONV_BLOC = 3
NB_FILTRES   = 48
KERNEL_SIZE  = 3
STRIDE_CONV  = 1
POOL_SIZE    = 2
STRIDE_POOL  = 2
FC_HIDDEN    = 128

_ROOT     = os.path.join(os.path.dirname(__file__), "..")
TRAIN_DIR = os.path.join(_ROOT, "data", "dataset", "train")
VAL_DIR   = os.path.join(_ROOT, "data", "dataset", "val")
CKPT      = os.path.join(_ROOT, "checkpoints", "checkpoint_best.npz")


# ─── Chargement du checkpoint ────────────────────────────────────────────────

def load_checkpoint(path, blocs, fc1, fc2):
    data = np.load(path, allow_pickle=False)
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
    epoch   = int(data["epoch"][0])
    val_acc = float(data["val_acc"][0])
    return epoch, val_acc


# ─── Construction du réseau ──────────────────────────────────────────────────

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
    return blocs, GAP(), RELU(), SOFTMAX()


# ─── Inférence image seule ───────────────────────────────────────────────────

def _to_cpu(x):
    return x.get() if hasattr(x, "get") else x


def predict(image_np, blocs, gap, relu_fc, fc1, fc2, softmax):
    data = xp.asarray(image_np.astype(np.float32))
    if data.ndim == 2:
        data = data[:, :, xp.newaxis]
    for bloc in blocs:
        for conv, relu in zip(bloc["convs"], bloc["relus"]):
            data = relu.forward(conv.forward(data))
        data = bloc["pool"].forward(data)
    data = _to_cpu(gap.forward(data))
    data = relu_fc.forward(fc1.forward(data))
    return int(np.argmax(softmax.forward(fc2.forward(data))))


# ─── Chargement dataset (nom de fichier → label) ─────────────────────────────

def load_items_by_name(directory, label_map, ext=".jpg", resize=None):
    """Charge toutes les images dont le nom contient une clé de label_map."""
    items = []
    for fname in sorted(os.listdir(directory)):
        if not fname.lower().endswith(ext):
            continue
        fl = fname.lower()
        label = None
        for key, idx in label_map.items():
            if key in fl:
                label = idx
                break
        if label is None:
            continue
        img = cv2.imread(os.path.join(directory, fname), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            items.append((img, label))
    return items


def load_items_flat(directory, label, ext=(".jpg", ".jpeg")):
    """Charge toutes les images d'un répertoire avec un label fixe."""
    items = []
    for fname in sorted(os.listdir(directory)):
        if not any(fname.lower().endswith(e) for e in ext):
            continue
        img = cv2.imread(os.path.join(directory, fname), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            items.append((img, label))
    return items


# ─── Évaluation ─────────────────────────────────────────────────────────────

def evaluate_set(items, blocs, gap, relu_fc, fc1, fc2, softmax, name):
    n = len(items)
    if n == 0:
        print(f"  [{name}] Aucune image trouvée.")
        return

    confusion = np.zeros((3, 3), dtype=int)
    for i, (img, true_lbl) in enumerate(items):
        pred = predict(img, blocs, gap, relu_fc, fc1, fc2, softmax)
        confusion[true_lbl, pred] += 1
        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"  {i + 1}/{n}", end='\r')

    print()
    correct = np.diag(confusion).sum()
    print(f"\n  {name}  —  {n} images  |  Acc globale : {correct / n:.3f}")
    print(f"  {'Classe':<14} {'Vrais':<8} {'Total':<8} {'Acc':<8}")
    print(f"  {'-'*40}")
    for c, cname in enumerate(CLASS_NAMES):
        tot = confusion[c].sum()
        acc = confusion[c, c] / tot if tot > 0 else 0.0
        print(f"  {cname:<14} {confusion[c, c]:<8} {tot:<8} {acc:.3f}")

    print(f"\n  Matrice de confusion (lignes=réel, cols=prédit) :")
    header = "  " + " ".join(f"{n:>10}" for n in CLASS_NAMES)
    print(header)
    for c, row in enumerate(confusion):
        print(f"  {CLASS_NAMES[c]:<12}" + " ".join(f"{v:>10}" for v in row))

    # Erreurs typiques
    print(f"\n  Erreurs fréquentes :")
    for true_c in range(3):
        for pred_c in range(3):
            if true_c != pred_c and confusion[true_c, pred_c] > 0:
                pct = confusion[true_c, pred_c] / confusion[true_c].sum() * 100
                print(f"    {CLASS_NAMES[true_c]} -> predit {CLASS_NAMES[pred_c]} : "
                      f"{confusion[true_c, pred_c]} fois ({pct:.1f}%)")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  TruePneumoniaAI — Évaluation")
    print(SEP)

    print("\n[1] Construction et chargement du modèle…")
    blocs, gap, relu_fc, softmax = build_network()
    fc1 = FC(NB_FILTRES, FC_HIDDEN)
    fc2 = FC(FC_HIDDEN, 3)

    epoch, saved_acc = load_checkpoint(CKPT, blocs, fc1, fc2)
    print(f"     Checkpoint : epoch {epoch}, val_acc={saved_acc:.3f}")

    # ── Validation set (nommage : bacteria-/normal-/virus-) ─────────────────
    print("\n[2] Évaluation — Val set…")
    val_items = load_items_by_name(VAL_DIR, {"bacteria": 1, "normal": 0, "virus": 2})
    evaluate_set(val_items, blocs, gap, relu_fc, fc1, fc2, softmax, "Val")

    # ── Train set (mesure l'overfitting par classe) ─────────────────────────
    print("\n[3] Evaluation — Train set (detection overfitting par classe)…")
    train_items = load_items_by_name(TRAIN_DIR, {"bacteria": 1, "normal": 0, "virus": 2})
    evaluate_set(train_items, blocs, gap, relu_fc, fc1, fc2, softmax, "Train")

    print(f"\n{SEP}")
    print("  Évaluation terminée")
    print(SEP)


if __name__ == "__main__":
    main()
