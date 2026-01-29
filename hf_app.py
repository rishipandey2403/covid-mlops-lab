import gradio as gr
import joblib
import numpy as np
from pathlib import Path

# Load model
MODEL_PATH = Path("artifacts/models/strong_xgboost.joblib")
model = joblib.load(MODEL_PATH)

def predict(*features):
    x = np.array(features).reshape(1, -1)
    pred = model.predict(x)[0]
    return int(pred)

inputs = [
    gr.Number(label=f"Feature {i+1}") for i in range(model.n_features_in_)
]

app = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Number(label="Prediction"),
    title="COVID Risk Prediction",
    description="MLOps COVID Model Inference"
)

if __name__ == "__main__":
    app.launch()
