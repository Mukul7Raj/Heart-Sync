import pandas as pd
import numpy as np
import pickle
import os
import shap
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- Load Pre-trained Artifacts ---
MODEL_PATH = "models/heart_model.pkl"
SCALER_PATH = "models/scaler.pkl"
EXPLAINER_PATH = "models/explainer.pkl"

app = FastAPI(title="HeartSync AI API")

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
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
                with open(SCALER_PATH, 'rb') as f:
                    scaler = pickle.load(f)
                with open(EXPLAINER_PATH, 'rb') as f:
                    explainer = pickle.load(f)
                print("Models loaded successfully")
            else:
                print(f"Model file not found at {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading models: {e}")
            traceback.print_exc()

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

@app.get("/api")
async def root():
    return {"message": "HeartSync AI API is running", "endpoints": ["/api/predict", "/api/health"]}

@app.get("/api/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/api/predict")
async def predict(data: PatientData):
    load_artifacts()
    if model is None:
        raise HTTPException(status_code=500, detail="Machine learning models are not initialized on the server.")
    
    try:
        data_dict = data.model_dump()
        feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        input_df = pd.DataFrame([data_dict])[feature_order]
        input_scaled = scaler.transform(input_df)
        
        prob = model.predict_proba(input_scaled)[0][1]
        risk_score = round(prob * 100, 2)
        
        # --- Robust SHAP Calculation ---
        sv = np.zeros(len(feature_order))
        try:
            s_vals = explainer.shap_values(input_scaled)
            if isinstance(s_vals, list):
                sv = s_vals[1][0] if len(s_vals) > 1 else s_vals[0][0]
            elif hasattr(s_vals, 'values'):
                sv = s_vals.values[0]
                if len(sv.shape) > 1:
                    sv = sv[:, 1] if sv.shape[1] > 1 else sv[:, 0]
            elif hasattr(s_vals, 'shape'):
                if len(s_vals.shape) == 3:
                    sv = s_vals[0, :, 1] if s_vals.shape[2] > 1 else s_vals[0, :, 0]
                elif len(s_vals.shape) == 2:
                    sv = s_vals[0]
            sv = np.array(sv).flatten()
        except Exception as shap_err:
            print(f"SHAP Error: {shap_err}")
            if hasattr(model, 'feature_importances_'):
                sv = model.feature_importances_

        feature_importance = {}
        for i, val in enumerate(sv):
            feature_importance[feature_order[i]] = float(val)
            
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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
