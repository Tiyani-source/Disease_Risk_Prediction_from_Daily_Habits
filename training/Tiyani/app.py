from typing import Dict, Any
import streamlit as st

SELECTED_FEATURES = [
    "sugar_intake","bmi","cholesterol","sleep_hours",
    "physical_activity","work_hours","blood_pressure",
    "calorie_intake","water_intake"
]

R = {
    "bmi": (10, 50, 22),
    "blood_pressure": (80, 200, 120),
    "cholesterol": (100, 400, 180),
    "sleep_hours": (0, 12, 7),
    "physical_activity": (0, 30, 5),
    "work_hours": (0, 16, 8),
    "calorie_intake": (1000, 5000, 2000),
    "sugar_intake": (0, 300, 50),
    "water_intake": (0, 10, 2),
}

st.caption("Only the inputs shown are used by the current reduced model.")

step_labels = ["🩺 Vitals", "🧭 Daily Habits"]
cols = st.columns(2)

# ---- Step state (0=Vitals, 1=Daily Habits)
if "step" not in st.session_state:
    st.session_state.step = 0

for i, label in enumerate(step_labels):
    if i == st.session_state.step:
        cols[i].button(label, key=f"step_{i}", disabled=True)
    else:
        if cols[i].button(label, key=f"step_{i}"):
            st.session_state.step = i
            st.experimental_rerun()

# ---------- Step 0: Vitals ----------
if st.session_state.step == 0:
    v1, v2 = st.columns(2)
    with v1:
        st.session_state.bmi = st.slider("BMI", *R["bmi"])                          # used
        st.session_state.blood_pressure = st.slider("Blood Pressure (systolic mmHg)", *R["blood_pressure"])  # used
        st.session_state.cholesterol = st.slider("Cholesterol (mg/dL)", *R["cholesterol"])                  # used
    with v2:
        st.empty(); st.empty(); st.empty(); st.empty()

# ---------- Step 1: Daily Habits ----------
elif st.session_state.step == 1:
    h1, h2 = st.columns(2)
    with h1:
        st.session_state.physical_activity = st.slider("Physical Activity (hours/week)", *R["physical_activity"])   # used
        st.session_state.work_hours = st.slider("Work Hours (per day)", *R["work_hours"])                         # used
        st.session_state.calorie_intake = st.slider("Daily Calorie Intake", *R["calorie_intake"])                 # used
    with h2:
        st.session_state.sugar_intake = st.slider("Daily Sugar Intake (grams)", *R["sugar_intake"])               # used
        st.session_state.water_intake = st.slider("Water Intake (liters/day)", *R["water_intake"])                 # used
        st.session_state.sleep_hours = st.slider("Sleep Hours", *R["sleep_hours"])                                # used

if st.button("Submit"):
    payload = {
        "sugar_intake": st.session_state.sugar_intake,
        "bmi": st.session_state.bmi,
        "cholesterol": st.session_state.cholesterol,
        "sleep_hours": st.session_state.sleep_hours,
        "physical_activity": st.session_state.physical_activity,
        "work_hours": st.session_state.work_hours,
        "blood_pressure": st.session_state.blood_pressure,
        "calorie_intake": st.session_state.calorie_intake,
        "water_intake": st.session_state.water_intake,
    }
    # Assume a function `get_prediction` exists to call the backend API
    result = get_prediction(payload)
    st.write(result)
# FastAPI backend for Lifestyle Disease Risk Predictor
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

# ------------------------------
# Load model artifacts
# ------------------------------
# Try to load trained model and selected features from disk
MODEL_PATH = os.getenv("MODEL_PATH", "xgb_model.pkl")
FEATS_PATH = os.getenv("FEATS_PATH", "selected_features.pkl")
THRESH_PATH = os.getenv("THRESH_PATH", "best_threshold.pkl")

# Fallback selected features (your reduced 9-feature set)
FALLBACK_FEATURES = [
    "sugar_intake","bmi","cholesterol","sleep_hours",
    "physical_activity","work_hours","blood_pressure",
    "calorie_intake","water_intake",
]

# Load model
model = joblib.load(MODEL_PATH)

# Load features list, fallback to hardcoded
try:
    sel_feats = joblib.load(FEATS_PATH)
    if not isinstance(sel_feats, (list, tuple)):
        sel_feats = FALLBACK_FEATURES
except Exception:
    sel_feats = FALLBACK_FEATURES

# Load tuned threshold, fallback to 0.5
try:
    THRESH = float(joblib.load(THRESH_PATH))
except Exception:
    THRESH = 0.5

MODEL_VERSION = os.getenv("MODEL_VERSION", "xgb_red_v1")

# ------------------------------
# App + CORS
# ------------------------------
app = FastAPI(title="Lifestyle Risk API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Defaults for missing fields (kept reasonable)
# ------------------------------
DEFAULTS: Dict[str, float] = {
    "sugar_intake": 50.0,
    "bmi": 22.0,
    "cholesterol": 180.0,
    "sleep_hours": 7.0,
    "physical_activity": 5.0,
    "work_hours": 8.0,
    "blood_pressure": 120.0,
    "calorie_intake": 2000.0,
    "water_intake": 2.0,
}

# ------------------------------
# Routes
# ------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "features": sel_feats,
        "threshold": float(THRESH),
        "model_version": MODEL_VERSION,
    }

@app.post("/predict")
def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Accept flexible JSON, fill defaults for missing, and return prediction."""
    data = dict(payload or {})

    # Ensure all required features exist
    for f in sel_feats:
        if data.get(f) is None:
            data[f] = DEFAULTS.get(f, 0.0)

    # Build dataframe in correct order
    df = pd.DataFrame([data]).reindex(columns=sel_feats, fill_value=0)

    proba = model.predict_proba(df)[:, 1]
    p = float(proba[0])
    pred = "At Risk" if p > THRESH else "Healthy"

    return {
        "prediction": pred,
        "probability": p,
        "threshold": float(THRESH),
        "features_used": sel_feats,
        "model_version": MODEL_VERSION,
    }