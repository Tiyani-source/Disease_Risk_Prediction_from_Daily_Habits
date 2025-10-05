import joblib
import xgboost as xgb
import os


#MODEL_PATH = os.getenv("MODEL_PATH", "xgb_model.pkl")
#print(MODEL_PATH)
model = joblib.load("xgb_model.pkl")
model.save_model("xgb_model.json")