# Dashboard

Le dashboard local permet de piloter l'entraînement sans modifier `configs/default.yaml`.

---

## Lancer

```bash
uvicorn src.dashboard.app:app --host 127.0.0.1 --port 8501
```

Ouvrir ensuite [http://127.0.0.1:8501](http://127.0.0.1:8501).

---

## Fonctionnalités

| Zone | Rôle |
|---|---|
| Lancement | Choix dataset, backbone, epochs, batch size, learning rate, precision |
| Progression | Epoch, batch, statut du run, dossier du run |
| Métriques live | Loss, AUC, sensibilité, spécificité, F1, accuracy, learning rate |
| Courbes | Loss entraînement/validation et métriques validation |
| Logs | Sortie du processus d'entraînement |
| Artefacts | Checkpoints, export ONNX, état MLflow |
| Dataset | Comptages par dataset et aperçu d'images |
| Système | CUDA, GPU, espace disque |

Chaque run lancé depuis le dashboard écrit une config runtime dans `outputs/dashboard/runs/<timestamp>/config.yaml`.

---

## Données temps réel

Le callback `DashboardEventLogger` écrit :

- `events.jsonl` — événements batch/epoch/validation
- `state.json` — dernier état synthétique
- `train.log` — logs du processus

Ces fichiers sont sous `outputs/dashboard/runs/<timestamp>/`.
