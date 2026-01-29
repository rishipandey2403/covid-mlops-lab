from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model ONCE when app starts
MODEL_PATH = "artifacts/models/strong_xgboost.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="COVID Mortality Prediction API")

class Patient(BaseModel):
    features: list

@app.post("/predict")
def predict(patient: Patient):
    X = np.array(patient.features).reshape(1, -1)
    prediction = model.predict(X)[0]
    return {
        "prediction": int(prediction),
        "meaning": "Death" if prediction == 1 else "Survived"
    }
