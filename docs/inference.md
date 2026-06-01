# Utilisation du modèle

---

## 1. Exporter en ONNX

Avant de lancer l'API, exporter le checkpoint entraîné en ONNX pour des inférences rapides sur CPU :

```bash
python -m src.inference.export_onnx \
    --ckpt checkpoints/best-epoch=XX-val_auroc=X.XXXX.ckpt \
    --output exports/model.onnx
```

Le fichier `exports/model.onnx` est généré avec des axes dynamiques (batch variable). Il tourne sur CPU sans dépendance PyTorch.

---

## 2. Lancer l'API

```bash
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

L'API lit automatiquement le seuil depuis `configs/default.yaml`. Si le threshold tuning a été fait, il est déjà calibré cliniquement.

Vérifier que l'API répond :

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "threshold": 0.3412}
```

---

## 3. Faire une prédiction

### Via curl

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@ma_radio.jpg" \
  -F "gradcam=true"
```

### Via Python

```python
import requests, base64
from PIL import Image
import io

with open("ma_radio.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f},
        data={"gradcam": "true"},
    )

result = response.json()
print(f"Label      : {result['label']}")
print(f"Probabilité: {result['probability']:.1%}")
print(f"Seuil      : {result['threshold']}")

# Sauvegarder la heatmap Grad-CAM
if result["gradcam_png_b64"]:
    img_bytes = base64.b64decode(result["gradcam_png_b64"])
    Image.open(io.BytesIO(img_bytes)).save("gradcam_overlay.png")
```

### Réponse JSON

```json
{
  "label": "PNEUMONIA",
  "probability": 0.8731,
  "threshold": 0.3412,
  "gradcam_png_b64": "<base64 PNG>"
}
```

---

## 4. Interpréter le Grad-CAM

Le champ `gradcam_png_b64` contient la radio originale avec une heatmap superposée :

- **Zone rouge/chaude** → région ayant fortement influencé la décision
- **Zone bleue/froide** → région ignorée par le modèle

En clinique, vérifier que les zones chaudes correspondent aux lobes pulmonaires et non à des artefacts (étiquettes, matériel médical). Si le modèle regarde autre chose que les poumons, la prédiction n'est pas cliniquement valide.

---

## 5. Via Docker

```bash
docker compose -f docker/docker-compose.yml up
```

L'API est disponible sur `http://localhost:8000`, identique à l'utilisation locale. Le volume `exports/` doit contenir `model.onnx` avant de démarrer le conteneur.

---

## 6. Comprendre le seuil de décision

Le modèle retourne une **probabilité** (0 à 1). La décision binaire dépend du seuil :

```
probabilité >= seuil  →  PNEUMONIA
probabilité <  seuil  →  NORMAL
```

Un seuil bas (ex. 0.34) augmente la sensibilité — le modèle sera plus prudent et signalera plus de cas positifs, au prix de plus de faux positifs. C'est le compromis choisi pour une application clinique d'aide au diagnostic.

Le seuil est ajustable dans `configs/default.yaml` sans réentraîner le modèle.
