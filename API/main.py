from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load trained model (replace with your own path/model)
#model = joblib.load("model.pkl")

# Initialize FastAPI
app = FastAPI(title="ML Prediction API", version="1.0")

# Input schema
class InputData(BaseModel):
    features: list[float]

# Health check
@app.get("/")
def home():
    return {"message": "Model Prediction API is running!"}

# Prediction endpoint
@app.post("/predict/")
def predict(data: InputData):
    # Convert to numpy array (reshape for single sample)
    X = np.array(data.features).reshape(1, -1)
    
    # Get prediction
    prediction = model.predict(X)
    
    # For probability (if classifier)
    # proba = model.predict_proba(X).tolist()
    
    return {
        "input": data.features,
        "prediction": prediction.tolist()
        # "probabilities": proba
    }

