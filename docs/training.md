# Entraînement

---

## 1. Configuration

Tous les hyperparamètres sont dans `configs/default.yaml` :

```yaml
data:
  data_dir: "dataset/chest_Xray_NP_augmented"
  image_size: 128
  batch_size: 128
  num_workers: 8

model:
  backbone: "densenet121"   # densenet121 | resnet50
  pretrained: true
  dropout: 0.5

training:
  max_epochs: 30
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  precision: "16-mixed"     # AMP — nécessite GPU CUDA

threshold:
  default: 0.5
  target_sensitivity: 0.95
```

Modifier ce fichier pour changer de backbone, ajuster le learning rate, etc.

---

## 2. Lancer via le dashboard

```bash
uvicorn src.dashboard.app:app --host 127.0.0.1 --port 8501
```

Puis ouvrir [http://127.0.0.1:8501](http://127.0.0.1:8501).

Le dashboard permet de choisir le dataset (`dataset/chest_Xray_NP_augmented/`, `dataset/chest_Xray_VB_augmented/`, …), les hyperparamètres, puis de suivre l'entraînement en temps réel : progression epoch/batch, loss, AUC, sensibilité, spécificité, logs et checkpoints. Le dataset sélectionné détermine la tâche (binaire NORMAL/PNEUMONIA ou sous-type BACTERIA/VIRUS) et donc la config, les classes et le dossier de checkpoints.

---

## 3. Lancer en CLI

```bash
python -m src.training.train --config configs/default.yaml
```

Ce script :

1. Instancie le `ChestXrayDataModule` et calcule le `pos_weight`
2. Charge le backbone DenseNet121 pretrained ImageNet
3. Remplace la tête de classification par `Dropout → Linear(1)`
4. Lance l'entraînement avec :
    - **Optimiseur** : AdamW (lr=1e-4, weight_decay=1e-5)
    - **Scheduler** : CosineAnnealingLR sur `max_epochs`
    - **Loss** : BCEWithLogitsLoss avec `pos_weight`
    - **Précision** : 16-mixed (AMP pour GPU)
5. Sauvegarde chaque run dans `checkpoints/<tâche>/<timestamp>/` (`.ckpt` + `config.yaml` + `events.jsonl` + `state.json`)
6. Log chaque run dans MLflow

> Pour entraîner l'étage 2 (bactérien vs viral) : `python -m src.training.train --config configs/subtype.yaml`.

---

## 4. Suivre l'entraînement avec MLflow

### Lancer l'interface MLflow

```bash
mlflow ui --backend-store-uri mlruns
```

Puis ouvrir [http://localhost:5000](http://localhost:5000).

Avec les versions récentes de MLflow, si le backend fichier est refusé, définir :

```bash
$env:MLFLOW_ALLOW_FILE_STORE="true"
```

### Ce qui est loggé par run

| Paramètre | Valeur |
|---|---|
| `backbone` | densenet121 |
| `pretrained` | true |
| `dropout` | 0.5 |
| `lr` | 1e-4 |
| `batch_size` | 32 |
| `pos_weight` | calculé depuis le dataset |

| Métrique (par epoch) | Description |
|---|---|
| `train/loss` | BCE loss entraînement |
| `train/auroc` | AUC-ROC entraînement |
| `val/loss` | BCE loss validation |
| `val/auroc` | AUC-ROC validation — **métrique principale** |
| `val/sensitivity` | Recall sur PNEUMONIA |
| `val/specificity` | Recall sur NORMAL |
| `val/f1` | F1-score |
| `val/accuracy` | Accuracy |

### Callbacks actifs

| Callback | Monitore | Action |
|---|---|---|
| `ModelCheckpoint` | `val/auroc` | Sauvegarde le meilleur checkpoint |
| `ModelCheckpoint` | `val/sensitivity` | Sauvegarde le checkpoint max sensibilité |
| `EarlyStopping` | `val/auroc` | Arrête si pas d'amélioration sur 7 epochs |
| `LearningRateMonitor` | — | Log le LR à chaque epoch |

---

## 5. Calibrer le seuil de décision

Après l'entraînement, le seuil par défaut (0.5) n'est pas optimisé cliniquement. Lancer :

```bash
python -m src.training.threshold_tuning \
    --ckpt checkpoints/normal-pneumonia/<timestamp>/best-loss-epochXX.ckpt \
    --config configs/default.yaml
```

Le script :

1. Charge le checkpoint et calcule les probabilités sur le set de validation
2. Trace la courbe ROC complète
3. Trouve le seuil qui atteint `target_sensitivity = 0.95` avec la meilleure spécificité possible
4. Affiche le résultat :

```
--- Threshold Tuning Results ---
  threshold:   0.3412
  sensitivity: 0.9503
  specificity: 0.8721
  auc_roc:     0.9847
```

5. Met à jour `configs/default.yaml` avec le nouveau seuil — il sera utilisé automatiquement par l'API.

---

## 6. Via Docker (API + MLflow)

```bash
docker compose -f docker/docker-compose.yml up
```

Lance deux services :

- **api** sur le port `8000` — inférence FastAPI
- **mlflow** sur le port `5000` — interface de tracking
