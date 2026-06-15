# Analyse des données

Le notebook `notebooks/01_data_analysis.ipynb` couvre l'intégralité de l'EDA. Cette page en documente les résultats et décisions.

---

## Lancer le notebook

```bash
jupyter notebook notebooks/01_data_analysis.ipynb
```

Ou pour générer un rapport PDF :

```bash
jupyter nbconvert --to pdf notebooks/01_data_analysis.ipynb
```

---

## Structure du dataset

Le dataset `chest_Xray/` est organisé en trois splits avec deux classes :

```
chest_Xray/
├── train/
│   ├── NORMAL/      ~1341 images
│   └── PNEUMONIA/   ~3875 images
├── val/
│   ├── NORMAL/      8 images
│   └── PNEUMONIA/   8 images
└── test/
    ├── NORMAL/      234 images
    └── PNEUMONIA/   390 images
```

### Déséquilibre de classes

Le dataset d'entraînement est **déséquilibré** : environ 3x plus de cas PNEUMONIA que NORMAL.

Le `ChestXrayDataModule` calcule automatiquement le `pos_weight` :

```
pos_weight = n_NORMAL / n_PNEUMONIA ≈ 0.35
```

Ce poids est passé à la `BCEWithLogitsLoss` pour compenser le déséquilibre sans sur-échantillonner.

---

## Pipeline de transforms

### Entraînement (Albumentations)

| Transform | Paramètres | Justification |
|---|---|---|
| `LongestMaxSize` + `PadIfNeeded` | 224×224, ratio conservé | Taille d'entrée ImageNet sans déformation anatomique |
| `HorizontalFlip` | p=0.5 | Radio thoracique symétrique |
| `Rotate` | ±15°, p=0.5 | Variabilité de positionnement patient |
| `RandomBrightnessContrast` | ±0.2, p=0.5 | Variabilité d'exposition |
| `CLAHE` | clip=2.0, p=0.3 | Rehausse les détails dans les zones pulmonaires denses |
| `GaussNoise` | std 0.01–0.03, p=0.2 | Simule le bruit des capteurs |
| `Normalize` | mean/std ImageNet | Cohérence avec le backbone pretrained |

### Validation / Test

Uniquement `LongestMaxSize` + `PadIfNeeded` + `Normalize` — pas d'augmentation pour l'évaluation reproductible.

### Génération offline du dataset augmenté

Le script `src.data.generate_augmentations` permet de créer un dataset séparé, sans modifier `chest_Xray/` :

```bash
python -m src.data.generate_augmentations
```

Par défaut, il écrit dans `chest_Xray_augmented/`, convertit toutes les images en 224×224 par letterbox avec padding noir, garde `val/` et `test/` sans augmentation, et équilibre `train/` en générant des variantes uniquement pour la classe minoritaire `NORMAL`.

---

## Contenu du notebook

Le notebook `01_data_analysis.ipynb` exécute les analyses suivantes :

1. **Distribution des classes** — histogramme train/val/test par classe
2. **Galerie d'images** — exemples NORMAL vs PNEUMONIA côte à côte
3. **Analyse des dimensions** — distribution hauteur/largeur, aspect ratio
4. **Qualité image** — histogrammes d'intensité, contraste moyen par classe
5. **Aperçu des augmentations** — même image avant/après chaque transform
6. **pos_weight calculé** — confirmation du facteur de rééquilibrage
