# TruePneumoniaAI

Classification binaire **NORMAL vs PNEUMONIA** sur radiographies thoraciques, avec explainabilité clinique via Grad-CAM.

---

## Stack technique

| Composant           | Technologie           | Rôle                                                 |
|---------------------|-----------------------|------------------------------------------------------|
| Données             | `dataset/chest_Xray*/` | Images JPEG, splits train/val/test                  |
| Transforms médicaux | MONAI                 | ScaleIntensity, EnsureChannelFirst, support DICOM    |
| Augmentation        | Albumentations        | Flip, rotation, CLAHE, bruit gaussien                |
| Backbone            | DenseNet121 (CheXNet) | Référence historique classification radio thoracique |
| Entraînement        | PyTorch Lightning     | Boucle structurée, callbacks, AMP                    |
| Tracking            | MLflow                | Hyperparams, métriques, artefacts par run            |
| Explainabilité      | Grad-CAM              | Heatmap — où le modèle regarde sur la radio          |
| Export              | ONNX                  | Inférence optimisée                                  |
| API                 | FastAPI               | Endpoint `/predict` + Grad-CAM en réponse            |
| Infra               | Docker + CUDA         | Reproductibilité GPU                                 |

---

## Structure du projet

```
dataset/                   # tous les datasets
├── chest_Xray/            # brut (train/val/test × NORMAL|PNEUMONIA)
├── chest_Xray_augmented/  # binaire augmenté/letterboxé
├── chest_Xray_patient/    # re-split patient-level sans fuite
└── chest_Xray_subtype/    # BACTERIA|VIRUS (étage 2)

checkpoints/               # runs groupés par tâche
├── normal-pneumonia/<timestamp>/   # *.ckpt + config + events + state
└── bacteria-virus/<timestamp>/

src/
├── data/
│   ├── dataset.py         # ChestXrayDataset — label 0=NORMAL, 1=PNEUMONIA
│   ├── transforms.py      # pipelines MONAI + Albumentations
│   └── datamodule.py      # LightningDataModule + pos_weight automatique
├── models/
│   ├── backbone.py        # factory DenseNet121 / ResNet50
│   └── classifier.py      # LightningModule, métriques, optimiseur
├── training/
│   ├── train.py           # script d'entraînement principal
│   └── threshold_tuning.py# calibration du seuil de décision
├── explainability/
│   └── gradcam.py         # GradCAM + overlay heatmap
└── inference/
    ├── api.py             # FastAPI server
    └── export_onnx.py     # export ONNX opset 17

configs/default.yaml       # tous les hyperparamètres
docker/
├── Dockerfile
└── docker-compose.yml     # API + serveur MLflow
notebooks/
└── 01_data_analysis.ipynb # EDA — distribution, qualité, augmentation
```

---

## Métriques et seuil clinique

Le modèle est évalué sur :

- **AUC-ROC** — performance globale indépendante du seuil
- **Sensibilité** (recall PNEUMONIA) — ne pas manquer un cas
- **Spécificité** — ne pas sur-alarmer
- **F1-score**

Le seuil de décision (défaut 0.5) est **calibré après l'entraînement** pour atteindre une sensibilité cible de **95%** sur le set de validation. Dans un contexte clinique, manquer une pneumonie est plus grave qu'un faux positif.

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```
