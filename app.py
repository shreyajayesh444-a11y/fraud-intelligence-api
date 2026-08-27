from fastapi import FastAPI
import joblib
import os

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "fraud_model.pkl")
model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {
        "status": "Fraud Intelligence API Running"
    }
