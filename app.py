"""
FastAPI service that loads the current 'Production' stage model
from the MLflow Model Registry and serves predictions.

Run: uvicorn src.app:app --port 8000
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "loan-default-classifier"
MODEL_STAGE = "Production"

app = FastAPI(title="Loan Default Prediction API", version="1.0")

# Loaded via the sklearn flavor (not pyfunc) so we can call predict_proba
# for a calibrated probability rather than just a hard 0/1 label.
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")


class LoanRequest(BaseModel):
    age: int = Field(..., ge=18, le=80, example=32)
    annual_income: float = Field(..., ge=0, example=600000)
    loan_amount: float = Field(..., ge=0, example=500000)
    credit_score: int = Field(..., ge=300, le=900, example=650)
    employment_type: str = Field(..., example="Self-Employed")
    existing_loans: int = Field(..., ge=0, example=2)
    loan_term_months: int = Field(..., example=36)
    missed_payments_last_year: int = Field(..., ge=0, example=1)


class LoanResponse(BaseModel):
    default_probability: float
    default_prediction: int
    model_name: str
    model_stage: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=LoanResponse)
def predict(request: LoanRequest):
    try:
        input_df = pd.DataFrame([request.dict()])
        prob = float(model.predict_proba(input_df)[0][1])
        pred = int(prob >= 0.5)

        return LoanResponse(
            default_probability=round(prob, 4),
            default_prediction=pred,
            model_name=MODEL_NAME,
            model_stage=MODEL_STAGE,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
