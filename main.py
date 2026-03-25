from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib

# ----------------------------
# CONFIG
# ----------------------------
FEATURES = [
    "cam_count", "cam_duration", "cam_bg_ratio",
    "loc_freq", "loc_bg_ratio",
    "mic_duration", "mic_bg_flag",
    "data_upload", "data_download",
    "bg_data", "reserved"
]

MODEL_PATH = "model\lstm_permission_model.keras"
SCALER_PATH = "model\scaler.pkl"

# ----------------------------
# LOAD MODEL
# ----------------------------
from tensorflow.keras.models import load_model
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ----------------------------
# API
# ----------------------------
app = FastAPI(title="Permission Abuse Detection API")

# ----------------------------
# SCHEMAS
# ----------------------------
class Event(BaseModel):
    cam_count: float
    cam_duration: float
    cam_bg_ratio: float
    loc_freq: float
    loc_bg_ratio: float
    mic_duration: float
    mic_bg_flag: int
    data_upload: float
    data_download: float
    bg_data: float
    reserved: float


class SessionRequest(BaseModel):
    session_id: str
    events: List[Event]


# ----------------------------
# ENDPOINT
# ----------------------------
@app.post("/predict")
def predict_session(data: SessionRequest):

    # 1. convert to numpy
    X = np.array([[getattr(e, f) for f in FEATURES] for e in data.events])

    # 2. scale
    X_scaled = scaler.transform(X)

    # 3. add batch dimension (1, timesteps, features)
    X_scaled = np.expand_dims(X_scaled, axis=0)

    # 4. predict
    probs = model.predict(X_scaled)[0]
    pred_class = int(np.argmax(probs))

    label_map = {
        0: "Safe",
        1: "Suspicious",
        2: "Dangerous"
    }

    return {
        "session_id": data.session_id,
        "prediction": pred_class,
        "label": label_map[pred_class],
        "confidence": float(probs[pred_class]),
        "probabilities": {
            "safe": float(probs[0]),
            "suspicious": float(probs[1]),
            "dangerous": float(probs[2])
        }
    }
