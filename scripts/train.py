import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap

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
        
    print(f"✅ Model trained. Accuracy: {model.score(X_test_scaled, y_test):.2f}")

if __name__ == "__main__":
    train_model()
