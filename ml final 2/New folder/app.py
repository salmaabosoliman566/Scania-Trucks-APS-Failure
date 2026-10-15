import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Scania APS Failure Prediction API")
model = joblib.load("scania_aps_pipeline.joblib")

class TruckData(BaseModel):
    features: dict

@app.get("/")
def home():
    return {"message": "Scania APS Failure Prediction API is running"}

@app.post("/predict")
def predict(data: TruckData):
    df_input = pd.DataFrame([data.features])
    pred = int(model.predict(df_input)[0])
    prob = float(model.predict_proba(df_input)[0][1])
    return {
        "prediction": pred,
        "status": "APS Failure (pos)" if pred == 1 else "Healthy (neg)",
        "failure_probability": round(prob, 4)
    }