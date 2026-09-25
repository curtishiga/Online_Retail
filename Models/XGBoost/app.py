import pandas as pd
from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

import column_transformers

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgb_best_estim.joblib"

xgb_model = joblib.load(MODEL_PATH)

app = FastAPI(title = "Online Retail Churn Prediction API",
                description = "API for predicting customer churn using an XGBoost model",
                version = "1.0.0")

class CustOrder(BaseModel):
    RollingTotalPrice: float
    DaysSinceLastPurchase: float
    RollingNumInvoices: int
    InvoiceDate: str
    FirstInvoiceDate: str
    Country: str

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model": "xgb_best_estim.joblib"
    }

@app.post("/predict")
def predict_churn(order: CustOrder):
    row = pd.DataFrame([order.model_dump()])
    prediction = xgb_model.predict(row)
    predict_prob = xgb_model.predict_proba(row)
    return {'churn_prediction': int(prediction[0]), 'churn_probability': f"{predict_prob[0][1]:.2%}"}