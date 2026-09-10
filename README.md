# Loan Default Prediction — End-to-End MLflow Pipeline

An end-to-end ML project demonstrating a real MLOps workflow: tracked experiments, model comparison, a registry-backed promotion gate, and registry-driven serving — using MLflow as the backbone.

## Problem Statement
A lender wants to predict whether a loan applicant is likely to default, so risk teams can price loans appropriately or flag high-risk applications for manual review.

## Pipeline Overview
```
Data generation → Preprocessing → Multiple tracked experiment runs (MLflow)
→ Best-run selection (by AUC) → Model Registry registration + promotion gate
→ Serving from the registry (not a static file)
```

This is the core pattern that separates a "trained a model" project from an "MLOps" project: **the served model is always pulled from the registry's current Production version**, not a hardcoded file — so promoting a new model in the registry is enough to update what's served, with zero code changes to the serving layer.

## What Makes This "MLflow-native"
- **Every run is logged**, not just the winner — 4 experiment configs (Logistic Regression, 2 Random Forest depths, Gradient Boosting), each as its own MLflow run with params, metrics, and the model artifact
- **Model comparison happens via the MLflow UI**, not manual print statements — `mlflow ui` gives a sortable table across all runs
- **Promotion is gated**, not automatic — a run only gets registered to the Model Registry if it clears a minimum AUC bar (0.75), and only the best run is promoted to the `Production` stage
- **Serving reads from the registry by name + stage** (`models:/loan-default-classifier/Production`), so retraining and promoting a better model doesn't require redeploying the API

## Results (this run)
| Run | Model | AUC | Precision | Recall |
|---|---|---|---|---|
| logreg_default | Logistic Regression | 0.831 | 0.738 | 0.640 |
| rf_shallow | Random Forest (depth=5) | 0.849 | 0.786 | 0.593 |
| **rf_deep** | **Random Forest (depth=12)** | **0.858** | **0.783** | **0.647** |
| gbm_default | Gradient Boosting | 0.854 | 0.759 | 0.645 |

`rf_deep` won on AUC and was auto-registered and promoted to Production.

## How to Run
```bash
pip install -r requirements.txt

# 1. Generate data
python src/generate_data.py

# 2. Run all 4 tracked experiments; best one gets registered + promoted
python src/train.py

# 3. Inspect all runs in the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Visit http://localhost:5000 to compare runs, view metrics, and see registered model versions

# 4. Serve the current Production model
uvicorn src.app:app --reload --port 8000
# Visit http://localhost:8000/docs to test interactively
```

## Example API Call
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "age": 29, "annual_income": 400000, "loan_amount": 900000, "credit_score": 580,
  "employment_type": "Self-Employed", "existing_loans": 3, "loan_term_months": 60,
  "missed_payments_last_year": 2
}'
```
Response:
```json
{"default_probability": 0.8458, "default_prediction": 1, "model_name": "loan-default-classifier", "model_stage": "Production"}
```

## What I'd Do With More Time
- Add a drift-detection job (e.g. Evidently) that compares live prediction inputs to training data and triggers retraining automatically
- Add a CI/CD gate (GitHub Actions) so a new model only gets promoted if it beats the *current* Production model, not just a fixed threshold
- Swap the SQLite-backed tracking store for a proper MLflow tracking server + Postgres backend for multi-user use
- Containerize the serving app and deploy it, with the MLflow tracking URI pointed at a remote server

## Tech Stack
Python, scikit-learn, MLflow (tracking + Model Registry), FastAPI, pandas
