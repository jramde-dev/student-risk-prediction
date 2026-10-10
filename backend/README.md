# Student Risk API

Backend FastAPI pour **entraîner** et **déployer** les modèles du projet capstone
« Impact de l'IA et des réseaux sociaux sur la santé des étudiants » :

- **Régression** : prédit `Mental_Health_Score` (Random Forest par défaut)
- **Classification** : probabilité qu'un étudiant soit **à risque** (`Mental_Health_Score < 60`) — LightGBM par défaut, Optuna en option

Le feature engineering du notebook est **intégré au pipeline** : l'API reçoit les colonnes brutes.

## Démarrage (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Linux/macOS : `source .venv/bin/activate` à la place de l'activation Windows.
Documentation interactive : <http://127.0.0.1:8000/docs>

> Lancez **un seul worker** (pas de `--workers N`) : l'état des jobs et les modèles sont en mémoire.

## Utilisation

```powershell
# 1. Entraîner (asynchrone) -> renvoie un job_id
curl.exe -X POST http://127.0.0.1:8000/train -H "Content-Type: application/json" -d "{}"

# 2. Suivre le job
curl.exe http://127.0.0.1:8000/train/<job_id>

# 3. Métriques et algorithmes des modèles servis
curl.exe http://127.0.0.1:8000/models
```

Pour `/predict`, le plus simple est la page `/docs` (« Try it out »), qui contient un exemple prérempli.

### Endpoints

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/health` | État + modèles chargés |
| POST | `/train` | Lance un entraînement (202 + `job_id`) ; 409 si déjà en cours |
| GET | `/train/{job_id}` | Statut du job (`running` / `succeeded` / `failed`) |
| GET | `/models` | Algorithmes, métriques de test, versions des libs |
| POST | `/predict` | Un étudiant → score prédit + probabilité de risque |
| POST | `/predict/batch` | Jusqu'à 1000 étudiants |

`POST /train` accepte (tout est optionnel) :

```json
{
  "tasks": ["regression", "classification"],
  "regression_algorithm": "random_forest",
  "classification_algorithm": "lightgbm",
  "optuna_trials": 0
}
```

Algorithmes : régression `random_forest | ridge | lightgbm` ; classification `lightgbm | random_forest | logistic_regression`.
`optuna_trials > 0` optimise LightGBM (PR-AUC, CV 3 plis, comme dans le notebook).

`/predict?threshold=0.3` change le seuil de décision (défaut `0.5`) : l'abaisser privilégie le
**rappel** (détecter plus d'étudiants à risque, au prix de fausses alertes).

## Configuration

Variables `APP_*` ou fichier `.env` (voir `.env.example`) : chemin du dataset, dossier des modèles,
seed, seuil de risque (60), seuil de décision (0.5).

## Structure

```
app/
  config.py     paramètres (pydantic-settings)
  features.py   feature engineering du notebook
  training.py   split, pipelines, Optuna, métriques, sauvegarde atomique
  registry.py   chargement à chaud des modèles + prédiction
  schemas.py    validation des entrées/sorties
  main.py       routes FastAPI
tests/          pytest (données synthétiques)
models/         artefacts générés (*.joblib, metadata.json), ignorés par git
```

Tests : `pytest -q`

## Limites connues (hors périmètre du minimum)

- Modèle évalué sur le jeu de test puis sauvegardé **sans ré-entraînement sur 100 % des données**.
- Un seul modèle « courant » (pas d'historique de versions ni de rollback).
- Jobs en mémoire (perdus au redémarrage), pas d'authentification.
- Recharger les modèles avec les **mêmes versions** de scikit-learn/LightGBM (voir `/models` → `versions`).
- Outil d'aide au repérage, pas un diagnostic : PR-AUC ≈ 0,34, beaucoup de faux positifs.


## Run the projects
- Create env : $ & "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\python.exe" -m venv .venv
- Activate env : $ .venv\Scripts\Activate.ps1
- backend : $ uvicorn app.main:app --reload
- Frontend : $ npm run dev        # http://localhost:5173