import joblib
import numpy as np
from typing import Tuple

MODEL_PATH = 'models/lgb_rolling_graph_struct_full.joblib'
SCALER_PATH = 'models/scaler_rolling_graph_struct_full.joblib'

def load_model():
    """Return (model, scaler, model_features)"""
    model = None
    scaler = None
    model_features = None
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception:
        scaler = None
    try:
        model_features = model.booster_.feature_name()
    except Exception:
        try:
            model_features = model.feature_name_
        except Exception:
            model_features = None
    return model, scaler, model_features


def predict_proba(model, X):
    return model.predict_proba(X)[:, 1]
