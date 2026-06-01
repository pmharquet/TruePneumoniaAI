"""
TruePneumoniaAI — Boucle d'entraînement
Architecture : [[CONV→RELU]*3 → POOL]*5 → GAP → FC(D,128) → RELU → FC(128,3) → SOFTMAX

Lancement :
    cd ai && python main_train.py

Le dashboard est accessible sur http://localhost:8000/ pendant l'entraînement.
"""

import os
import sys
import time
import base64

import cv2
import numpy as np

# Assure que les imports locaux fonctionnent
sys.path.insert(0, os.path.dirname(__file__))

from xp import xp, GPU  # noqa: E402


def _to_cpu(x):
    """Transfère un tableau CuPy vers NumPy (no-op si déjà NumPy)."""
    return x.get() if hasattr(x, "get") else x


def save_checkpoint(path, blocs, fc1, fc2, epoch, val_acc):
    """Sauvegarde les poids du modèle (CONV + FC) dans un fichier .npz."""
    data = {}
    for bi, bloc in enumerate(blocs):
        for ci, conv in enumerate(bloc["convs"]):
            data[f"conv_{bi}_{ci}"] = _to_cpu(conv.kernel)
    for ni, neuron in enumerate(fc1.neurons):
        data[f"fc1_w_{ni}"] = neuron.weights
        data[f"fc1_b_{ni}"] = neuron.bias
    for ni, neuron in enumerate(fc2.neurons):
        data[f"fc2_w_{ni}"] = neuron.weights
        data[f"fc2_b_{ni}"] = neuron.bias
    data["epoch"]   = np.array([epoch])
    data["val_acc"] = np.array([val_acc])
    np.savez(path, **data)


from ConvolutionLayer import ConvolutionLayer as CONV
from RectifiedLinearUnitLayer import RectifiedLinearUnitLayer as RELU
from PoolingLayer import PoolingLayer as POOL
from GlobalAveragePoolingLayer import GlobalAveragePoolingLayer as GAP
from ClassActivationMapLayer import ClassActivationMapLayer as CAM
from FullyConnected import FullyConnectedLayer as FC
from SoftmaxLayer import SoftmaxLayer as SOFTMAX
from DropoutLayer import DropoutLayer as DROPOUT
from CrossEntropyLoss import CrossEntropyLoss
from SGDOptimizer import SGDOptimizer
from DatasetLoader import DatasetLoader
import dashboard_server as dashboard

# ─────────────────────────────────────────────
#  Hyperparamètres
# ─────────────────────────────────────────────
NB_EPOCHS     = 50
LEARNING_RATE = 0.001
MOMENTUM      = 0.9
NB_FILTRES    = 48    # ≥32 requis pour matrices cuBLAS efficaces
NB_BLOCS      = 5
NB_CONV_BLOC  = 3
FC_HIDDEN     = 128
KERNEL_SIZE   = 3
STRIDE_CONV   = 1
POOL_SIZE     = 2
STRIDE_POOL   = 2
BATCH_SIZE    = 8     # optimal RTX 5080 484×660 float32 (batch=16+ → OOM activations >16GB VRAM)
GRAD_CLIP     = 1.0   # clip élément par élément des gradients (anti-explosion)
LOG_INTERVAL  = 1     # log + broadcast dashboard toutes les N batches
DROPOUT_RATE  = 0.5   # taux de dropout entre relu_fc et fc2
EARLY_STOP_PATIENCE = 8   # arrêt si pas d'amélioration val_acc pendant N epochs
LR_MIN        = 1e-5  # LR minimum pour le cosine annealing

# Dataset régénéré à 484×660 px (÷2, ratio identique à l'original 968×1320)
IMAGE_TARGET_SIZE = None  # images déjà à la bonne taille

# Mode debug : limite le nombre d'images chargées (None = tout le dataset)
DEBUG_MAX_IMAGES = None  # ex: 32 pour vérifier le pipeline rapidement

# Chemins depuis la racine du projet (D:/Docker/TruePneumoniaAI/)
_ROOT          = os.path.join(os.path.dirname(__file__), "..")
TRAIN_DIR      = os.path.join(_ROOT, "data", "dataset", "train")
VAL_DIR        = os.path.join(_ROOT, "data", "dataset", "val")
CHECKPOINT_DIR = os.path.join(_ROOT, "checkpoints")

CLASS_NAMES = ["Normal", "Bactérien", "Viral"]
SEP = "=" * 55


# ─────────────────────────────────────────────
#  Augmentation de données (CPU, uint8)
# ─────────────────────────────────────────────
def augment_batch(images):
    """
    images : numpy uint8 [batch, H, W]
    Applique aléatoirement sur chaque image :
      - Flip horizontal (p=0.5)
      - Jitter brightness/contrast (facteur ∈ [0.85, 1.15])
    Retourne un tableau uint8 de même forme.
    """
    result = images.copy()
    for i in range(result.shape[0]):
        if np.random.rand() < 0.5:
            result[i] = np.fliplr(result[i])
        factor = np.random.uniform(0.85, 1.15)
        result[i] = np.clip(result[i].astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return result


# ─────────────────────────────────────────────
#  Cosine annealing LR
# ─────────────────────────────────────────────
def cosine_lr(epoch, total_epochs, lr_max, lr_min=LR_MIN):
    """Décroissance cosinus de lr_max à lr_min sur total_epochs."""
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * (epoch - 1) / total_epochs))


# ─────────────────────────────────────────────
#  Construction du réseau
# ─────────────────────────────────────────────
def build_network():
    """Crée tous les blocs CONV+RELU+POOL avec initialisation He."""
    blocs = []
    in_ch = 1  # image grayscale = 1 canal d'entrée

    for _ in range(NB_BLOCS):
        convs = []
        relus = []
        for _ in range(NB_CONV_BLOC):
            convs.append(CONV.create(NB_FILTRES, KERNEL_SIZE, KERNEL_SIZE, in_ch, STRIDE_CONV))
            relus.append(RELU())
            in_ch = NB_FILTRES
        blocs.append({
            "convs": convs,
            "relus": relus,
            "pool": POOL(POOL_SIZE, STRIDE_POOL),
        })

    gap     = GAP()
    relu_fc = RELU()
    dropout = DROPOUT(DROPOUT_RATE)
    softmax = SOFTMAX()
    return blocs, gap, relu_fc, dropout, softmax


# ─────────────────────────────────────────────
#  Forward GPU — couches conv/pool/gap (batché)
# ─────────────────────────────────────────────
def forward_gpu(batch_np, blocs, gap):
    """
    batch_np : numpy uint8 [batch, H, W] — images brutes
    Retourne :
      gap_outs_cpu : numpy float32 [batch, NB_FILTRES]
      last_fmaps   : numpy float32 [batch, H', W', NB_FILTRES]
    """
    # CPU → GPU + ajout dimension canal (float32 : FP32 = 56 TFLOPS vs FP64 = 875 GFLOPS)
    data = xp.asarray(batch_np.astype(np.float32))[:, :, :, xp.newaxis]  # [B,H,W,1]

    for bloc in blocs:
        for conv, relu in zip(bloc["convs"], bloc["relus"]):
            data = relu.forward(conv.forward(data))
        data = bloc["pool"].forward_batch(data)

    last_fmaps = data                          # GPU [batch, H', W', NB_FILTRES]
    gap_outs   = gap.forward(data)             # GPU [batch, NB_FILTRES]

    return _to_cpu(gap_outs), _to_cpu(last_fmaps)


# ─────────────────────────────────────────────
#  Backward GPU — couches gap/pool/conv (batché)
# ─────────────────────────────────────────────
def backward_gpu(grad_gap_np, blocs, gap):
    """
    grad_gap_np : numpy [batch, NB_FILTRES] — gradients depuis les FC (CPU)
    Met à jour d_kernel de toutes les CONV (somme sur le batch).
    """
    # Cast float32 pour GPU (FP64 consumer = 875 GFLOPS vs FP32 = 56 TFLOPS)
    dtype = np.float32 if GPU else np.float64
    grad = xp.asarray(grad_gap_np.astype(dtype))   # CPU → GPU [batch, NB_FILTRES]
    grad = gap.backward(grad)        # [batch, H', W', NB_FILTRES]

    for bloc in reversed(blocs):
        grad = bloc["pool"].backward(grad)
        for conv, relu in zip(reversed(bloc["convs"]), reversed(bloc["relus"])):
            grad = conv.backward(relu.backward(grad))


# ─────────────────────────────────────────────
#  Forward image seule (validation)
# ─────────────────────────────────────────────
def forward_single(image, blocs, gap, relu_fc, dropout, fc1, fc2, softmax):
    """Utilisé uniquement pour la validation (image seule, pas de gradient).
    Le dropout est désactivé (training=False) pendant l'inférence.
    """
    data = xp.asarray(image.astype(np.float32))
    if data.ndim == 2:
        data = data[:, :, xp.newaxis]

    for bloc in blocs:
        for conv, relu in zip(bloc["convs"], bloc["relus"]):
            data = relu.forward(conv.forward(data))
        data = bloc["pool"].forward(data)

    last_fmaps = data
    data = gap.forward(data)
    data_cpu = _to_cpu(data)
    data_cpu = relu_fc.forward(fc1.forward(data_cpu))
    # dropout désactivé en validation
    output = softmax.forward(fc2.forward(data_cpu))
    return output, _to_cpu(last_fmaps)


# ─────────────────────────────────────────────
#  Utilitaires
# ─────────────────────────────────────────────
def compute_cam_b64(last_fmaps, fc1, fc2, predicted_class):
    """Génère une CAM colorisée encodée en base64 (JPEG). last_fmaps : numpy CPU."""
    W1 = np.array([n.weights for n in fc1.neurons])   # [FC_HIDDEN, D]
    w2 = fc2.neurons[predicted_class].weights          # [FC_HIDDEN]
    cam_weights = W1.T @ w2                            # [D]

    cam_map = np.dot(last_fmaps, cam_weights)
    cam_map = np.maximum(cam_map, 0)
    if cam_map.max() > 0:
        cam_map = (cam_map / cam_map.max() * 255).astype(np.uint8)
    else:
        cam_map = cam_map.astype(np.uint8)

    cam_resized = cv2.resize(cam_map, (224, 224))
    cam_colored = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    _, buf = cv2.imencode(".jpg", cam_colored, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


def evaluate(val_data, blocs, gap, relu_fc, dropout, fc1, fc2, softmax):
    """Calcule loss, accuracy globale et accuracy par classe sur la validation."""
    loss_fn = CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    correct_per_class = [0, 0, 0]
    total_per_class   = [0, 0, 0]

    for image, label in val_data:
        output, _ = forward_single(image, blocs, gap, relu_fc, dropout, fc1, fc2, softmax)
        total_loss += loss_fn.forward(output, label)
        predicted = int(np.argmax(output))
        total_per_class[label] += 1
        if predicted == label:
            correct += 1
            correct_per_class[label] += 1

    n = len(val_data)
    class_acc_val = [
        round(correct_per_class[c] / total_per_class[c], 4) if total_per_class[c] > 0 else 0.0
        for c in range(3)
    ]
    return total_loss / n, correct / n, class_acc_val


def zero_all_grads(learnable_layers):
    for layer in learnable_layers:
        layer.zero_grads()


# ─────────────────────────────────────────────
#  Boucle principale
# ─────────────────────────────────────────────
def main():
    print(SEP)
    print("  TruePneumoniaAI — Entraînement")
    print(SEP)
    print(f"  Backend : {'GPU (CuPy — RTX 5080)' if GPU else 'CPU (NumPy)'}")
    print(f"  Batch size : {BATCH_SIZE} images")

    # Démarrage du dashboard
    dashboard.start_background()
    time.sleep(1)

    print("\n[1] Indexation du dataset…")
    loader = DatasetLoader(
        TRAIN_DIR, VAL_DIR,
        target_size=IMAGE_TARGET_SIZE,
        max_images=DEBUG_MAX_IMAGES,
    )
    loader.load()
    train_data = loader.get_train()
    val_data   = loader.get_val()

    if len(train_data) == 0:
        print("[ERREUR] Aucune image trouvée dans le répertoire d'entraînement.")
        return

    print("\n[2] Construction du réseau…")
    blocs, gap, relu_fc, dropout, softmax = build_network()

    D = NB_FILTRES
    fc1 = FC(D, FC_HIDDEN)
    fc2 = FC(FC_HIDDEN, 3)
    print(f"     GAP → {D} canaux  |  FC({D}→{FC_HIDDEN})  |  Dropout({DROPOUT_RATE})  |  FC({FC_HIDDEN}→3)")

    learnable_layers = []
    for bloc in blocs:
        learnable_layers.extend(bloc["convs"])
    learnable_layers.extend([fc1, fc2])
    optimizer = SGDOptimizer(learnable_layers, learning_rate=LEARNING_RATE, momentum=MOMENTUM)

    loss_fn = CrossEntropyLoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_acc     = 0.0
    early_stop_count = 0

    n_train    = len(train_data)
    n_batches  = (n_train + BATCH_SIZE - 1) // BATCH_SIZE   # arrondi haut
    history    = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n[3] Entraînement : {NB_EPOCHS} epochs — {n_train} images ({n_batches} batches/epoch)")
    print(f"     LR={LEARNING_RATE}→{LR_MIN} (cosine)  momentum={MOMENTUM}  filtres={NB_FILTRES}  batch={BATCH_SIZE}")
    print(f"     Dropout={DROPOUT_RATE}  EarlyStopping patience={EARLY_STOP_PATIENCE}")
    print(SEP)

    for epoch in range(1, NB_EPOCHS + 1):
        # Cosine annealing : met à jour le LR au début de chaque epoch
        lr_epoch = cosine_lr(epoch, NB_EPOCHS, LEARNING_RATE)
        optimizer.set_lr(lr_epoch)

        dropout.training = True   # activer le dropout en entraînement

        t_start    = time.time()
        epoch_loss = 0.0
        correct    = 0
        correct_per_class = [0, 0, 0]
        total_per_class   = [0, 0, 0]
        pred_counts       = [0, 0, 0]
        batch_loss_history = []   # rolling 500 derniers batches
        batch_acc_history  = []   # rolling 500 derniers batches

        train_data.shuffle()

        # _items : liste de (uint8_image, label) — PAS de conversion float64 ici
        # (évite de charger 15000×2.56MB=38GB en RAM)
        items_raw = train_data._items
        last_fmaps_cam = None
        last_pred_cam  = 0

        for batch_idx in range(n_batches):
            b_start = batch_idx * BATCH_SIZE
            b_end   = min(b_start + BATCH_SIZE, n_train)
            batch_raw  = items_raw[b_start:b_end]
            actual_bs  = len(batch_raw)

            # uint8 → stacked numpy + augmentation
            batch_images = np.stack([img for img, _ in batch_raw])  # [bs, H, W] uint8
            batch_images = augment_batch(batch_images)
            batch_labels = [lbl for _, lbl in batch_raw]

            # 1. Mise à zéro des gradients (une fois par batch)
            zero_all_grads(learnable_layers)

            # 2. Forward GPU : conv/pool/gap en parallèle sur tout le batch
            gap_outs, fmaps_batch = forward_gpu(batch_images, blocs, gap)
            # gap_outs  : [bs, NB_FILTRES]  CPU numpy
            # fmaps_batch : [bs, H', W', NB_FILTRES] CPU numpy

            # 3. FC / Softmax / Loss par image (CPU) + backward FC
            batch_loss = 0.0
            grad_gaps  = []

            for b in range(actual_bs):
                # Forward FC (CPU)
                fc1_out     = fc1.forward(gap_outs[b])
                relu_out    = relu_fc.forward(fc1_out)
                dropout_out = dropout.forward(relu_out)
                fc2_out     = fc2.forward(dropout_out)
                output      = softmax.forward(fc2_out)

                lbl = batch_labels[b]
                loss_val = loss_fn.forward(output, lbl)
                batch_loss += float(loss_val)

                predicted = int(np.argmax(output))
                total_per_class[lbl] += 1
                if predicted == lbl:
                    correct += 1
                    correct_per_class[lbl] += 1
                pred_counts[predicted] += 1

                # Backward FC (accumule les gradients dans les neurones)
                grad = loss_fn.backward()
                grad = fc2.backward(grad)
                grad = dropout.backward(grad)
                grad = relu_fc.backward(grad)
                grad = fc1.backward(grad)   # CPU [NB_FILTRES]
                grad_gaps.append(grad)

            epoch_loss += batch_loss
            avg_batch_loss = batch_loss / actual_bs
            batch_loss_history.append(avg_batch_loss)
            batch_acc_history.append(round(correct / b_end, 4))
            if len(batch_loss_history) > 100:
                batch_loss_history.pop(0)
            if len(batch_acc_history) > 100:
                batch_acc_history.pop(0)

            # 4. Backward GPU : gap/pool/conv avec gradients empilés
            grad_gap_batch = np.stack(grad_gaps)            # [bs, NB_FILTRES]
            backward_gpu(grad_gap_batch, blocs, gap)

            # 5. Normaliser les gradients par batch_size + clip
            for layer in learnable_layers:
                for _, g in layer.get_params_and_grads():
                    g /= actual_bs
                    g[...] = g.clip(-GRAD_CLIP, GRAD_CLIP)

            # 6. Mise à jour des paramètres
            optimizer.step()

            # CAM sur la dernière image du batch
            last_fmaps_cam = fmaps_batch[-1]
            last_pred_cam  = pred_counts.index(max(pred_counts))

            # Log + dashboard toutes les LOG_INTERVAL batches
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                images_done = b_end
                avg_loss    = epoch_loss / images_done
                acc         = correct / images_done
                class_acc_train = [
                    round(correct_per_class[c] / total_per_class[c], 4) if total_per_class[c] > 0 else 0.0
                    for c in range(3)
                ]
                global_step = (epoch - 1) * n_train + images_done

                elapsed   = time.time() - t_start
                speed     = images_done / elapsed                       # img/s
                remaining = (n_train - images_done) / speed
                eta_total = remaining + (NB_EPOCHS - epoch) * n_train / speed

                print(f"  Epoch {epoch:02d} | {images_done:5d}/{n_train} | "
                      f"Loss={avg_loss:.4f} | Acc={acc:.3f} | "
                      f"{speed:.1f}img/s | ETA epoch {remaining/60:.0f}min", end='\r')

                cam_b64 = None
                if last_fmaps_cam is not None:
                    try:
                        cam_b64 = compute_cam_b64(last_fmaps_cam, fc1, fc2, last_pred_cam)
                    except Exception:
                        pass

                dashboard.broadcast({
                    "type": "batch_update",
                    "epoch": epoch,
                    "total_epochs": NB_EPOCHS,
                    "batch": images_done,
                    "total_batches": n_train,
                    "global_step": global_step,
                    "current_loss": float(avg_loss),
                    "current_accuracy": float(acc),
                    "prediction_distribution": list(pred_counts),
                    "batch_loss_history": batch_loss_history,
                    "batch_acc_history": batch_acc_history,
                    "images_per_sec": round(float(speed), 1),
                    "elapsed_epoch": round(elapsed),
                    "eta_epoch": round(remaining),
                    "eta_total": round(eta_total),
                    "cam_image": cam_b64,
                    "class_accuracy_train": class_acc_train,
                })

        epoch_time = time.time() - t_start
        train_loss = epoch_loss / n_train
        train_acc  = correct / n_train

        # Validation (dropout désactivé)
        dropout.training = False
        print(f"\n  Epoch {epoch:02d} — validation…", end='\r')
        val_loss, val_acc, class_acc_val = evaluate(val_data, blocs, gap, relu_fc, dropout, fc1, fc2, softmax)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

        print(f"  Epoch {epoch:02d}/{NB_EPOCHS} "
              f"| Loss train={train_loss:.4f} val={val_loss:.4f} "
              f"| Acc train={train_acc:.3f} val={val_acc:.3f} "
              f"| LR={lr_epoch:.2e} | {epoch_time:.1f}s")

        cam_b64 = None
        if last_fmaps_cam is not None:
            try:
                cam_b64 = compute_cam_b64(last_fmaps_cam, fc1, fc2, last_pred_cam)
            except Exception as e:
                print(f"  [AVERT] CAM non générée : {e}")

        dashboard.broadcast({
            "type": "epoch_update",
            "epoch": epoch,
            "total_epochs": NB_EPOCHS,
            "global_step": epoch * n_train,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "epoch_time": round(epoch_time, 1),
            "prediction_distribution": pred_counts,
            "cam_image": cam_b64,
            "history": history,
            "class_accuracy_val": class_acc_val,
            "class_accuracy_train_final": [
                round(correct_per_class[c] / total_per_class[c], 4) if total_per_class[c] > 0 else 0.0
                for c in range(3)
            ],
        })

        # Sauvegarde checkpoint
        ckpt_latest = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.npz")
        save_checkpoint(ckpt_latest, blocs, fc1, fc2, epoch, val_acc)
        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            early_stop_count = 0
            ckpt_best = os.path.join(CHECKPOINT_DIR, "checkpoint_best.npz")
            save_checkpoint(ckpt_best, blocs, fc1, fc2, epoch, val_acc)
            print(f"  -> Meilleur modele sauvegarde (epoch {epoch}, val_acc={val_acc:.3f})")
        else:
            early_stop_count += 1
            if early_stop_count >= EARLY_STOP_PATIENCE:
                print(f"\n  [Early Stopping] Pas d'amelioration depuis {EARLY_STOP_PATIENCE} epochs.")
                break

    print(f"\n{SEP}")
    print("  Entraînement terminé")
    print(SEP)


if __name__ == "__main__":
    main()
