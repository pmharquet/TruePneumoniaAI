# TruePneumoniaAI

CNN from-scratch en NumPy pur (sans PyTorch/TensorFlow) pour la classification de radiographies thoraciques en trois classes :
- **Normal**
- **Bactérien** (pneumonie bactérienne)
- **Viral** (pneumonie virale)

---

## Architecture

```
[[CONV → RELU] × 3 → POOL] × 5 → GAP → FC(8, 128) → RELU → FC(128, 3) → SOFTMAX
```

- 5 blocs de 3 convolutions + ReLU + max-pooling
- Global Average Pooling (GAP) → vecteur de 8 valeurs
- 2 couches fully-connected (FC)
- Softmax en sortie

Toutes les couches sont implémentées en NumPy avec rétropropagation complète.

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Docker (stack v0.2, GPU)

Lance toute la stack (dashboard d'entraînement GPU + MLflow + API d'inférence) en une commande :

```bash
docker compose -f docker/docker-compose.yml up --build
```

| Service   | URL                     | Rôle                                            |
|-----------|-------------------------|-------------------------------------------------|
| dashboard | http://localhost:8501   | Piloter l'entraînement (utilise le GPU)         |
| mlflow    | http://localhost:5000   | Suivi des expériences                            |
| api       | http://localhost:8000   | Inférence (FastAPI)                              |

**Pré-requis :** Docker Desktop avec support GPU (WSL2 + NVIDIA Container Toolkit). L'image est en **CUDA 12.8 / PyTorch cu128**, requis pour la RTX 5080 (Blackwell). Les datasets, checkpoints, exports, `outputs/` et `mlruns/` sont montés en volumes (rien n'est figé dans l'image).

Vérifier que le GPU est bien vu dans le conteneur :

```bash
docker compose -f docker/docker-compose.yml run --rm dashboard \
  python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Interface de test

```bash
python ai\inference_server.py
```

---

## Entraînement

```bash
cd ai
python main_train.py
```

Le dashboard de monitoring est accessible sur **http://localhost:8000/** pendant l'entraînement.

### Dashboard

Le dashboard affiche en temps réel :
- **Barres de progression** — epoch courante + images dans l'epoch
- **Loss et Accuracy** — courbes denses par image (train) + points par epoch (val)
- **Loss live** — sparkline des 100 derniers batches
- **Distribution des prédictions** — histogramme Normal / Bactérien / Viral
- **CAM** (Class Activation Map) — carte de chaleur sur la dernière image traitée
- **Métriques de temps** — images/s, temps écoulé, ETA epoch, ETA total

---

## Dataset

Le dataset est situé dans `data/dataset/` :

```
data/dataset/
├── train/   # 15 000 images (484 × 660 px, niveaux de gris)
└── val/     #    640 images
```

Les labels sont extraits du nom de fichier :
- `normal-*.jpg` → classe 0 (Normal)
- `bacteria-*.jpg` → classe 1 (Bactérien)
- `virus-*.jpg` → classe 2 (Viral)

Les images sont stockées à 484 × 660 px (÷ 2 par rapport à la taille originale 968 × 1320, ratio identique).

Le pipeline de génération du dataset est dans `data/` :
1. `1_image_size/` — analyse des tailles originales
2. `2_image_resize/` — redimensionnement et normalisation
3. `3_image_generates/` — génération du split train/val

---

## Structure du projet

```
TruePneumoniaAI/
├── ai/
│   ├── main_train.py               # Boucle d'entraînement principale
│   ├── main.py                     # Pipeline d'inférence (ne pas modifier)
│   ├── ConvolutionLayer.py         # CONV vectorisée (as_strided + matmul)
│   ├── RectifiedLinearUnitLayer.py # ReLU
│   ├── PoolingLayer.py             # Max-pooling (3D et 4D)
│   ├── GlobalAveragePoolingLayer.py# GAP
│   ├── ClassActivationMapLayer.py  # CAM
│   ├── FullyConnected.py           # Couche fully-connected
│   ├── Neuron.py                   # Neurone individuel
│   ├── SoftmaxLayer.py             # Softmax
│   ├── CrossEntropyLoss.py         # Cross-entropy loss (gradient combiné)
│   ├── SGDOptimizer.py             # SGD avec momentum
│   ├── DatasetLoader.py            # Chargement dataset en RAM (uint8)
│   ├── dashboard_server.py         # Serveur FastAPI + WebSocket
│   └── dashboard.html              # Interface de monitoring
├── data/
│   └── dataset/
│       ├── train/                  # Images d'entraînement
│       └── val/                    # Images de validation
├── requirements.txt
└── README.md
```

---

## Hyperparamètres par défaut

| Paramètre       | Valeur  | Description                              |
|-----------------|---------|------------------------------------------|
| `NB_EPOCHS`     | 20      | Nombre d'epochs                          |
| `LEARNING_RATE` | 0.001   | Taux d'apprentissage                     |
| `MOMENTUM`      | 0.9     | Momentum SGD                             |
| `NB_FILTRES`    | 8       | Filtres par couche de convolution        |
| `NB_BLOCS`      | 5       | Nombre de blocs CONV×3+POOL              |
| `NB_CONV_BLOC`  | 3       | Convolutions par bloc                    |
| `FC_HIDDEN`     | 128     | Neurones dans la couche FC cachée        |
| `GRAD_CLIP`     | 1.0     | Clip élémentaire des gradients           |
| `LOG_INTERVAL`  | 10      | Fréquence de mise à jour du dashboard    |
