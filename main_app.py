import pandas as pd
import numpy as np
import pickle
import os
import traceback
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

print("\n" + "="*40)
print("🚀 HEART DISEASE PREDICTOR BACKEND V2.0")
print("="*40 + "\n")

# --- ML Training & Setup ---
DATA_PATH = "data/heart.csv"
MODEL_PATH = "models/heart_model.pkl"
SCALER_PATH = "models/scaler.pkl"
EXPLAINER_PATH = "models/explainer.pkl"

def train_model():
    print("Training Heart Disease Prediction Model...")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    explainer = shap.TreeExplainer(model)
    
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(EXPLAINER_PATH, 'wb') as f:
        pickle.dump(explainer, f)
        
    print(f"Model trained. Accuracy: {model.score(X_test_scaled, y_test):.2f}")
    return model, scaler, explainer

# Load or train
if not os.path.exists(MODEL_PATH):
    model, scaler, explainer = train_model()
else:
    print("Loading existing model artifacts...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(EXPLAINER_PATH, 'rb') as f:
        explainer = pickle.load(f)

# --- FastAPI Backend ---
app = FastAPI(title="HeartSync AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    print("\n--- NEW PREDICTION REQUEST ---")
    try:
        # Use model_dump() for V2 compatibility
        data_dict = data.model_dump()
        print(f"Input Data: {data_dict}")
        
        feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        input_df = pd.DataFrame([data_dict])[feature_order]
        
        input_scaled = scaler.transform(input_df)
        
        # Prediction
        prob = model.predict_proba(input_scaled)[0][1]
        risk_score = round(prob * 100, 2)
        print(f"Risk Score: {risk_score}%")
        
        # --- Super-Robust SHAP Calculation ---
        print("Calculating SHAP values...")
        sv = np.zeros(len(feature_order)) # Default
        try:
            # SHAP calculation
            s_vals = explainer.shap_values(input_scaled)
            
            # Extract the correct array. Binary classification for TreeExplainer 
            # often returns a list [class0_arr, class1_arr].
            if isinstance(s_vals, list):
                if len(s_vals) > 1:
                    sv = s_vals[1][0] # Class 1 for heart disease
                else:
                    sv = s_vals[0][0]
            elif hasattr(s_vals, 'values'): # Newer Explanation object
                sv = s_vals.values[0]
                if len(sv.shape) > 1: # Multiclass Explanation
                    sv = sv[:, 1] if sv.shape[1] > 1 else sv[:, 0]
            elif hasattr(s_vals, 'shape'): # Numpy array
                if len(s_vals.shape) == 3: # (samples, features, classes)
                    sv = s_vals[0, :, 1] if s_vals.shape[2] > 1 else s_vals[0, :, 0]
                elif len(s_vals.shape) == 2: # (samples, features)
                    sv = s_vals[0]
            
            # Ensure sv is a 1D numpy array of floats
            sv = np.array(sv).flatten()
            if len(sv) != len(feature_order):
                print(f"SHAP shape mismatch: {len(sv)} vs {len(feature_order)}. Using zeros.")
                sv = np.zeros(len(feature_order))

        except Exception as shap_err:
            print(f"SHAP Error: {shap_err}. Using model's feature importance fallback.")
            if hasattr(model, 'feature_importances_'):
                sv = model.feature_importances_
            else:
                sv = np.zeros(len(feature_order))

        # Map back to feature names
        feature_importance = {}
        for i, val in enumerate(sv):
            feature_importance[feature_order[i]] = float(val)
            
        sorted_factors = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        sorted_factors = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        main_factors = []
        for feat, imp in sorted_factors:
            direction = "High" if imp > 0 else "Low"
            main_factors.append(f"{direction} {feat.replace('_', ' ').title()}")

        print(f"Main Factors: {main_factors}")
        return {
            "risk_probability": risk_score,
            "main_factors": main_factors,
            "status": "Success"
        }
    except Exception as e:
        print("\n!!! ERROR DETECTED !!!")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
