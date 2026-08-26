from fastapi import FastAPI
import pandas as pd
import joblib
import google.generativeai as genai

app = FastAPI()

model = joblib.load("fraud_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

@app.get("/")
def home():
    return {"status": "Fraud Intelligence API Running"}

@app.post("/predict")
def predict():

    sample = pd.DataFrame([{
        "amt": 500,
        "city_pop": 100000,
        "hour": 2,
        "day_of_week": 5,
        "location_distance": 50,
        "category": "shopping_pos"
    }])

    X = preprocessor.transform(sample)

    prediction = model.predict(X)[0]

    return {
        "prediction": int(prediction)
    }
