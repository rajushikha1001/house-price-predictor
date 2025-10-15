from fastapi import FastAPI
from app.schema import HouseFeatures
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("app/model.pkl")
scaler = joblib.load("app/scaler.pkl")


@app.post("/predict")
def predict(data: HouseFeatures):
    scaled = scaler.transform([data.features])
    prediction = model.predict(scaled)
    return {"predicted_price": prediction[0]}
