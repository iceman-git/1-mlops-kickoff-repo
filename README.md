# Titanic Survival Prediction — MLOps Pipeline

**Authors:** Dalton Kern, Max Tiefenbacher, Issam El Hoss, Jaime Moncayo, Maria Cruz
**Course:** Machine Learning Ops: Master in Business Analytics and Data Science

---

## 1. Business Objective

This project uses passenger data from the RMS Titanic disaster to build a binary survival classifier. The primary purpose is for the student team to practice end-to-end MLOps by transitioning a working Jupyter Notebook into a production-grade, modular pipeline with experiment tracking, containerized serving, and automated CI/CD.

- **The Goal:** Predict whether a passenger survived the Titanic sinking based on their demographic profile, ticket class, and travel characteristics.
- **The User:** Data science students and practitioners learning to operationalize ML models beyond the notebook.
- **In Scope:** A repeatable, auditable MLOps pipeline with binary survival classifications, probability scores, experiment tracking via W&B, a REST API, and automated deployment.
- **Out of Scope:** Causal analysis of survival factors or policy recommendations.

---

## 2. Success Metrics

- **Business KPI:** Maximize the identification of true survivors to support accurate historical analysis and safety planning benchmarks.
- **Technical Metric:** Weighted F1-Score on the held-out test set — balancing Precision (avoiding false alarms) and Recall (catching true survivors) across the imbalanced class distribution.
- **Acceptance Criteria:** The pipeline must run end-to-end without errors, produce all required artifacts, pass 100% of the CI test suite, and be accessible via the deployed API before any branch is merged to `main`.

---

## 3. The Data

### Source and Unit of Analysis
- Classic Kaggle dataset: *Titanic: Machine Learning from Disaster*
- Unit of analysis is an individual passenger record from the RMS Titanic manifest
- Fallback: a small scaffold CSV is auto-generated when the raw data file is not present (for CI/test environments)

### Dataset Snapshot
- Rows: 891 (training manifest)
- Columns: 12 raw features (including `PassengerId` and target)
- Positive class prevalence (`Survived=1`): ~38.4% (342 of 891)

### Target Definition
- `Survived`: whether the passenger survived the sinking (1 = survived, 0 = did not survive)

### Data Sensitivity
- Fully public historical dataset containing no living individuals
- No special data handling restrictions apply in an academic context

### Data Dictionary

| Feature | Description |
|---|---|
| Survived | Survival indicator — 1 survived, 0 did not survive (target) |
| Pclass | Passenger ticket class — 1 = First, 2 = Second, 3 = Third |
| Name | Full passenger name (used to engineer Title; dropped after cleaning) |
| Sex | Passenger sex — male or female |
| Age | Passenger age in years; median-imputed where missing |
| SibSp | Number of siblings or spouses aboard |
| Parch | Number of parents or children aboard |
| Ticket | Ticket number (dropped after cleaning) |
| Fare | Passenger fare paid in pounds; median-imputed where missing |
| Cabin | Cabin number (dropped after cleaning) |
| Embarked | Port of embarkation — C = Cherbourg, Q = Queenstown, S = Southampton; mode-imputed where missing |

### Engineered Features

| Feature | Description |
|---|---|
| FamilySize | Total family members aboard, computed as `SibSp + Parch + 1` |
| Title | Title extracted from passenger Name (Mr, Mrs, Miss, etc.); rare titles grouped as `Other` |

---

## 4. ML Approach

This repository transitions a fragile Jupyter Notebook into a testable, modular MLOps architecture.

- **Separation of Concerns:** Every step (Loading, Cleaning, Validating, Feature Engineering, Training, Evaluating, Inferring) has a dedicated single-purpose Python module.
- **Fail-Fast Gates:** `validate.py` blocks empty DataFrames and missing required columns before any compute begins.
- **Leakage Prevention:** The train/test split is performed in `main.py` *before* the feature recipe is built or fitted, ensuring the `ColumnTransformer` never sees test data.
- **Deployable Artifacts:** Preprocessing and the algorithm are bundled into a single `.joblib` Pipeline, preventing training-serving skew at inference time.
- **Enriched Inference Output:** `infer.py` returns the binary prediction, survival probability, a human-readable outcome label, and a high-confidence flag.

### Model Card

| Property | Value |
|---|---|
| **Algorithm** | Logistic Regression (`max_iter=1000`, `random_state=42`) |
| **Preprocessing** | `StandardScaler` on numeric features (`Age`, `Fare`, `FamilySize`); `OneHotEncoder` on categorical features (`Pclass`, `Sex`, `Embarked`, `Title`) |
| **Problem Type** | Binary classification |
| **Primary Metric** | Weighted F1-Score |
| **Training Data** | 80% of 891 Titanic passenger records |
| **Test Data** | 20% held-out split (stratified) |
| **Artifact** | `models/model.joblib` — sklearn Pipeline (ColumnTransformer + LogisticRegression) |
| **Experiment Tracking** | Weights & Biases (W&B) — metrics, parameters, and model artifact logged per run |
| **Model Registry** | W&B model registry — production model promoted with `prod` alias |
| **Known Limitations** | Trained on a small historical dataset; not suitable for real-world safety-critical decisions |

---

## 5. Repository Structure

```text
.
├── README.md
├── environment.yml          # Conda environment (mlops)
├── config.yaml              # Central configuration hub
├── Dockerfile               # Container definition for API serving
├── .env.example             # Template for secrets (copy to .env)
├── .github/
│   └── workflows/
│       ├── ci.yml           # CI — runs pytest on PRs to dev and main
│       └── deploy.yml       # CD — triggers Render deploy on GitHub Release
├── data/
│   ├── raw/
│   │   └── titanic.csv
│   └── processed/
│       └── clean.csv
├── models/
│   └── model.joblib
├── reports/
│   └── predictions.csv
├── src/
│   ├── __init__.py
│   ├── main.py              # Pipeline orchestrator
│   ├── load_data.py
│   ├── clean_data.py
│   ├── validate.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── api.py               # FastAPI application
│   ├── utils.py
│   └── logger.py            # Centralised logging setup
└── tests/
    ├── __init__.py
    └── main_pipline_test.py
```

---

## 6. How to Run & Test

### Step 1: Environment Setup
```bash
conda env create -f environment.yml
conda activate mlops
```

### Step 2: Configure Secrets
Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```
```
WANDB_API_KEY=your_wandb_api_key
WANDB_ENTITY=your_wandb_username
MODEL_SOURCE=local          # or "wandb" to load from registry
WANDB_MODEL_ALIAS=prod
```

### Step 3: Add Your Data
Place the Titanic CSV at:
```
data/raw/titanic.csv
```
> If no file is found, the pipeline automatically falls back to a small scaffold CSV so it can run end-to-end in test/CI environments.

### Step 4: Run the Pipeline
```bash
python -m src.main
```

### Step 5: Run the Test Suite
```bash
pytest tests/ -v
```

### Step 6: Serve the API Locally
```bash
uvicorn src.api:app --reload
```
- Health check: `GET /health`
- Predict: `POST /predict` with JSON body:
```json
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 29.0,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 211.3,
  "Embarked": "S"
}
```

### Step 7: Run with Docker
```bash
docker build -t mlops-api .
docker run -p 8000:8000 --env-file .env -v $(pwd)/models:/app/models mlops-api
```

---

## 7. CI/CD Pipeline

| Trigger | Workflow | Action |
|---|---|---|
| PR to `dev` or `main` | `ci.yml` | Runs full pytest suite — must pass before merge |
| GitHub Release published | `deploy.yml` | Triggers Render deploy hook to redeploy the API |

Branch protection on `main` requires CI to pass before any merge.

---

## 8. Outputs Generated

| Artifact | Path | Description |
|---|---|---|
| Cleaned data | `data/processed/clean.csv` | Deterministically cleaned and feature-engineered input data |
| Trained model | `models/model.joblib` | Deployable sklearn Pipeline (preprocessor + LogisticRegression) |
| Predictions | `reports/predictions.csv` | Inference log with prediction, probability, outcome label, and confidence flag |

### Sample Predictions Output

| prediction | survival_probability | outcome | high_confidence |
|---|---|---|---|
| 0 | 0.21 | Did not survive | True |
| 1 | 0.91 | Survived | True |
| 0 | 0.48 | Did not survive | False |

---

## 9. Changelog

### v1.0.0 — Production Release
- Deployed FastAPI application to Render with `/health` and `/predict` endpoints
- Added GitHub Actions CI (`ci.yml`) running pytest on all PRs to `dev` and `main`
- Added GitHub Actions CD (`deploy.yml`) triggering Render deploy on GitHub Release
- Added Dockerfile for containerized API serving
- Integrated Weights & Biases for experiment tracking, metric logging, and model registry with `prod` alias
- Added `load_model_for_serving()` supporting both local and W&B registry model loading via `MODEL_SOURCE` env var
- Centralised all configuration into `config.yaml` (removed `SETTINGS` dict from `main.py`)
- Replaced all `print()` statements with structured logging via `logger.py`
- Added `.env` / `.env.example` for secrets management via `python-dotenv`
- Added `api.py` (FastAPI) with Pydantic request/response models and lifespan model loading
- Expanded `environment.yml` with `wandb`, `fastapi`, `uvicorn`, `pydantic`

### v0.1.0 — Initial Scaffold
- Modular pipeline: `load_data` → `clean_data` → `validate` → `features` → `train` → `evaluate` → `infer`
- sklearn Pipeline bundling `ColumnTransformer` + `LogisticRegression`
- Leakage-safe train/test split before feature fitting
- Fail-fast validation gate blocking bad data before training
- Enriched inference output with probability, outcome label, and confidence flag
