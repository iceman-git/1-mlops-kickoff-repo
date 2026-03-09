# Titanic Survival Prediction

**Authors:** Dalton Kern, Max Tiefenbacher, Issam El Hoss, Jaime Moncayo, Maria Cruz
**Course:** Machine Learning Ops: Master in Business Analytics and Data Science  

---

## 1. Business Objective

This project uses passenger data from the RMS Titanic disaster to build a binary survival classifier. The primary purpose of the project is for the student team to practice classification models and build a modularized ML pipeline.

* **The Goal:** Predict whether a passenger survived the Titanic sinking based on their demographic profile, ticket class, and travel characteristics.
* **The User:** Data science students learning to transition a working Jupyter Notebook into a production-oriented, modular MLOps pipeline.
* **In Scope:** A repeatable, auditable MLOps pipeline generating binary survival classifications and operational probability scores.
* **Out of Scope:** Real-time inference serving, causal analysis of survival factors, or policy recommendations.

---

## 2. Success Metrics

* **Business KPI:** Maximize the identification of true survivors to support accurate historical analysis and safety planning benchmarks.
* **Technical Metric:** Weighted F1-Score on the held-out test set. We balance Precision (avoiding false alarms) and Recall (catching true survivors) across the imbalanced class distribution.
* **Acceptance Criteria:** The pipeline must run end-to-end without errors, produce all three required artifacts, and pass 100% of the test suite before any branch is merged to `dev`.

---

## 3. The Data

### Source and unit of analysis
- Classic Kaggle dataset: *Titanic: Machine Learning from Disaster*
- Unit of analysis is an individual passenger record from the RMS Titanic manifest
- Fallback: the `seaborn` built-in Titanic dataset is used when Kaggle credentials are unavailable

### Dataset snapshot
- Rows: 891 (training manifest)
- Columns: 12 raw features (including `PassengerId` and target)
- Positive class prevalence (`Survived=1`): ~38.4% (342 of 891)

### Target definition
- `Survived`: whether the passenger survived the sinking (1 = survived, 0 = did not survive)

### Data sensitivity
- This is a fully public historical dataset containing no living individuals
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

## 4. Academic Purpose & ML Approach

This repository is a teaching scaffold for Machine Learning Operations. We transition from a fragile Jupyter Notebook into a testable, modular software engineering architecture.

* **Separation of Concerns:** Every step (Loading, Cleaning, Validating, Feature Engineering, Training, Evaluating, Inferring) has a dedicated, single-purpose Python module.
* **Fail-Fast Security Gates:** `validate.py` blocks empty DataFrames and missing required columns before any compute begins.
* **Leakage Prevention:** The train/test split is performed in `main.py` *before* the feature recipe is built or fitted, ensuring the `ColumnTransformer` never sees test data.
* **Deployable Artifacts:** The orchestrator bundles preprocessing and the algorithm into a single `.joblib` Pipeline, preventing training-serving skew at inference time.
* **Enriched Inference Output:** `infer.py` returns not just the binary prediction but also the survival probability, a human-readable outcome label, and a high-confidence flag — making predictions immediately actionable without a data dictionary.

### ML Model
- **Algorithm:** Logistic Regression (`LogisticRegression(max_iter=1000, random_state=42)`)
- **Preprocessing:** `StandardScaler` on numeric features (`Age`, `Fare`, `FamilySize`); `OneHotEncoder` on categorical features (`Pclass`, `Sex`, `Embarked`, `Title`)
- **Problem Type:** Binary classification

### Future Roadmap (Upcoming Sessions)
* Move `SETTINGS` into `config.yaml`.
* Replace `print()` statements with structured logging.
* Add MLflow for experiment tracking and model registry.
* Containerize and serve predictions via a FastAPI application.

---

## 5. Repository Structure

```text
.
├── README.md
├── environment.yml
├── data
│   ├── raw
│   │   └── titanic.csv
│   └── processed
│       └── clean.csv
├── models
│   └── model.joblib
├── reports
│   └── predictions.csv
├── src
│   ├── __init__.py
│   ├── clean_data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── infer.py
│   ├── load_data.py
│   ├── main.py
│   ├── train.py
│   ├── utils.py
│   └── validate.py
└── tests
    ├── __init__.py
    └── test_infer.py
    └── test_clean_data.py
    └── test_features.py
    └── test_load_data.py
    └── test_validate.py

---

### 6. How to Run & Test

### Step 1: Environment Setup
Build and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate mlops-student-env
```

### Step 2: Add Your Data
Place the Titanic CSV at the following path before running the pipeline:
```
data/raw/titanic.csv
```
> If no file is found, `load_data.py` will automatically generate a small dummy CSV at that path so the pipeline runs end-to-end. The dummy dataset is for scaffolding only — replace it with real data before meaningful training.

### Step 3: Configure the Pipeline
Open `src/main.py` and update the `SETTINGS` dictionary to match your dataset. At minimum, set `is_example_config` to `False` once real data is in place:
```python
SETTINGS = {
    "is_example_config": False,
    "target_column": "Survived",
    "problem_type": "classification",
    ...
}
```

### Step 4: Run the Test Suite
Ensure all pipeline contracts are unbroken before training:
```bash
python -m pytest tests/ -v
```
> You should see 100% passing tests.

### Step 5: Execute the Orchestrator
Run the end-to-end pipeline to clean data, train the model, and generate artifacts:
```bash
python -m src.main
```

---

## 7. Outputs Generated

| Artifact | Path | Description |
|---|---|---|
| Cleaned data | `data/processed/clean.csv` | Deterministically cleaned and feature-engineered input data |
| Trained model | `models/model.joblib` | Deployable sklearn Pipeline (preprocessor + LogisticRegression) |
| Predictions | `reports/predictions.csv` | Inference log with prediction, survival probability, outcome label, and confidence flag |

### Sample predictions output

| prediction | survival_probability | outcome | high_confidence |
|---|---|---|---|
| 0 | 0.21 | Did not survive | True |
| 1 | 0.91 | Survived | True |
| 0 | 0.48 | Did not survive | False |

---

## 8. Academic Purpose

This repository is a teaching scaffold for Machine Learning Operations (MLOps).

**Learning outcomes**

- Translate a notebook workflow into a `src/` layout with explicit module contracts
- Implement quality gates before training to prevent silent failure modes
- Enforce leakage prevention by design through split-before-fit boundaries
- Produce model and data artifacts for auditability and reproducibility
- Add tests that validate behaviour and output contracts, not just that code runs
- Practice the full Git workflow: branching, committing, pushing, and opening pull requests to `dev`

**Roadmap for later sessions**
- Move `SETTINGS` into `config.yaml` and add environment-based secrets via `.env`
- Replace `print()` statements with standard library logging and structured logs
- Add experiment tracking, model registry, and continuous integration
- Containerize and serve predictions via a FastAPI application
