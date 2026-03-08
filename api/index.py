import pandas as pd
import numpy as np
import pickle
import os
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- Load Pre-trained Artifacts ---
# In Vercel, paths are relative to the root of the project.
MODEL_PATH = "models/heart_model.pkl"
SCALER_PATH = "models/scaler.pkl"
EXPLAINER_PATH = "models/explainer.pkl"

app = FastAPI(title="Heart Disease Risk Predictor API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model/scaler/explainer
model = None
scaler = None
explainer = None

def load_artifacts():
    global model, scaler, explainer
    if model is None:
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            with open(EXPLAINER_PATH, 'rb') as f:
                explainer = pickle.load(f)
        except Exception as e:
            print(f"Error loading models: {e}")

class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/api/predict")
async def predict(data: PatientData):
    load_artifacts()
    if model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        input_df = pd.DataFrame([data.dict()])
        input_scaled = scaler.transform(input_df)
        
        prob = model.predict_proba(input_scaled)[0][1]
        risk_score = round(prob * 100, 2)
        
        shap_values = explainer.shap_values(input_scaled)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]
            
        feature_importance = dict(zip(input_df.columns, sv))
        sorted_factors = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        main_factors = []
        for feat, imp in sorted_factors:
            direction = "High" if imp > 0 else "Low"
            main_factors.append(f"{direction} {feat.replace('_', ' ').title()}")

        return {
            "risk_probability": risk_score,
            "main_factors": main_factors,
            "status": "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    return {"status": "healthy"}
