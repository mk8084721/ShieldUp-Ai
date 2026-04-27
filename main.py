from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator
from typing import List
import numpy as np
import joblib
import logging
import hashlib
import json
from datetime import datetime
from collections import defaultdict

# ----------------------------
# LOGGING SETUP
# ----------------------------
logging.basicConfig(
    filename="security_log.txt",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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

# حدود منطقية لكل feature — أي قيمة بره الحدود دي مشبوهة
FEATURE_BOUNDS = {
    "cam_count":      (0, 1000),
    "cam_duration":   (0, 86400),   # ثانية (يوم كامل max)
    "cam_bg_ratio":   (0.0, 1.0),   # نسبة من 0 لـ 1
    "loc_freq":       (0, 10000),
    "loc_bg_ratio":   (0.0, 1.0),
    "mic_duration":   (0, 86400),
    "mic_bg_flag":    (0, 1),        # binary فقط
    "data_upload":    (0, 1e9),      # bytes
    "data_download":  (0, 1e9),
    "bg_data":        (0, 1e9),
    "reserved":       (0, 1e9),
}

# لو الموديل مش واثق بأكتر من كده، نرفض النتيجة
CONFIDENCE_THRESHOLD = 0.60

# أقصى عدد events في session واحدة
MAX_EVENTS_PER_SESSION = 200
MIN_EVENTS_PER_SESSION = 1

# Rate limiting — أقصى requests في الدقيقة لكل IP
RATE_LIMIT_PER_MINUTE = 30

MODEL_PATH = "model/lstm_permission_model.keras"
SCALER_PATH = "model/scaler.pkl"

# ----------------------------
# LOAD MODEL
# ----------------------------
from tensorflow.keras.models import load_model
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# تحميل الـ scaler bounds للكشف عن الشذوذ
scaler_center = scaler.center_
scaler_scale = scaler.scale_

# ----------------------------
# RATE LIMITER (بسيط في الميموري)
# ----------------------------
request_counts = defaultdict(list)

def check_rate_limit(ip: str) -> bool:
    now = datetime.now().timestamp()
    minute_ago = now - 60
    request_counts[ip] = [t for t in request_counts[ip] if t > minute_ago]
    if len(request_counts[ip]) >= RATE_LIMIT_PER_MINUTE:
        return False
    request_counts[ip].append(now)
    return True

# ----------------------------
# API
# ----------------------------
app = FastAPI(title="Permission Abuse Detection API — Protected")

# ----------------------------
# SCHEMAS مع Validation
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

    # ✅ الحماية الأولى: التحقق من حدود كل قيمة
    @field_validator('*', mode='before')
    def check_not_nan_inf(cls, v, info: ValidationInfo):
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                raise ValueError(f"Field '{info.field_name}' contains NaN or Inf")
        return v

    @field_validator('cam_bg_ratio', 'loc_bg_ratio')
    def ratio_must_be_0_to_1(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Ratio value {v} out of range [0,1]")
        return v

    @field_validator('mic_bg_flag')
    def flag_must_be_binary(cls, v):
        if v not in (0, 1):
            raise ValueError(f"mic_bg_flag must be 0 or 1, got {v}")
        return v

    @field_validator(
        'cam_count', 'cam_duration', 'loc_freq', 'mic_duration',
        'data_upload', 'data_download', 'bg_data', 'reserved'
    )
    def must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError(f"Negative value {v} not allowed")
        return v


class SessionRequest(BaseModel):
    session_id: str
    events: List[Event]

    @field_validator('session_id')
    def session_id_safe(cls, v):
        if len(v) > 64:
            raise ValueError("session_id too long")
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("session_id contains invalid characters")
        return v

    @field_validator('events')
    def events_count_valid(cls, v):
        if len(v) < MIN_EVENTS_PER_SESSION:
            raise ValueError(f"Too few events: minimum is {MIN_EVENTS_PER_SESSION}")
        if len(v) > MAX_EVENTS_PER_SESSION:
            raise ValueError(f"Too many events: maximum is {MAX_EVENTS_PER_SESSION}")
        return v


# ----------------------------
# HELPER: كشف الشذوذ الإحصائي
# ----------------------------
def detect_statistical_anomaly(X: np.ndarray) -> dict:
    """
    ✅ الحماية الثانية: مقارنة البيانات الجاية بالـ scaler
    اللي اتدرب عليه الموديل.
    لو في قيم بعيدة جداً عن المتوسط → مشبوهة.
    """
    X_scaled = (X - scaler_center) / scaler_scale
    z_scores = np.abs(X_scaled)

    # Z-score أكبر من 5 → شاذ جداً
    extreme_mask = z_scores > 5
    extreme_count = int(extreme_mask.sum())
    extreme_features = []

    if extreme_count > 0:
        feature_indices = np.where(extreme_mask.any(axis=0))[0]
        extreme_features = [FEATURES[i] for i in feature_indices]

    return {
        "has_anomaly": extreme_count > 0,
        "extreme_count": extreme_count,
        "extreme_features": extreme_features,
        "max_z_score": float(z_scores.max())
    }


def check_feature_bounds(X: np.ndarray) -> dict:
    """
    ✅ الحماية الثالثة: التحقق من الحدود المنطقية لكل feature
    """
    violations = []
    for i, feature in enumerate(FEATURES):
        low, high = FEATURE_BOUNDS[feature]
        col = X[:, i]
        if col.min() < low or col.max() > high:
            violations.append({
                "feature": feature,
                "value_min": float(col.min()),
                "value_max": float(col.max()),
                "allowed_range": [low, high]
            })
    return {
        "has_violation": len(violations) > 0,
        "violations": violations
    }


def compute_data_hash(X: np.ndarray) -> str:
    """
    ✅ الحماية الرابعة: hash للبيانات للـ audit trail
    """
    return hashlib.sha256(X.tobytes()).hexdigest()[:16]


# ----------------------------
# ENDPOINT
# ----------------------------
@app.post("/predict")
async def predict_session(data: SessionRequest, request: Request):

    client_ip = request.client.host

    # ✅ الحماية الخامسة: Rate Limiting
    if not check_rate_limit(client_ip):
        logger.warning(f"RATE_LIMIT_EXCEEDED | IP: {client_ip} | session: {data.session_id}")
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down."
        )

    # 1. تحويل البيانات لـ numpy
    X = np.array([[getattr(e, f) for f in FEATURES] for e in data.events])

    data_hash = compute_data_hash(X)

    # ✅ الحماية السادسة: فحص الحدود المنطقية
    bounds_check = check_feature_bounds(X)
    if bounds_check["has_violation"]:
        logger.warning(
            f"BOUNDS_VIOLATION | IP: {client_ip} | session: {data.session_id} "
            f"| hash: {data_hash} | violations: {bounds_check['violations']}"
        )
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Input data contains out-of-range values",
                "violations": bounds_check["violations"]
            }
        )

    # ✅ الحماية السابعة: كشف الشذوذ الإحصائي
    anomaly = detect_statistical_anomaly(X)
    if anomaly["has_anomaly"]:
        logger.warning(
            f"STATISTICAL_ANOMALY | IP: {client_ip} | session: {data.session_id} "
            f"| hash: {data_hash} | max_z: {anomaly['max_z_score']:.2f} "
            f"| features: {anomaly['extreme_features']}"
        )
        # مش بنرفض هنا، بس بنسجل ونضيف warning في الرد
        anomaly_warning = (
            f"Statistical anomaly detected in features: {anomaly['extreme_features']}"
        )
    else:
        anomaly_warning = None

    # 2. Scale البيانات
    X_scaled = scaler.transform(X)

    # 3. Batch dimension
    X_scaled = np.expand_dims(X_scaled, axis=0)

    # 4. Predict
    probs = model.predict(X_scaled)[0]
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    label_map = {0: "Safe", 1: "Suspicious", 2: "Dangerous"}

    # ✅ الحماية الثامنة: Confidence Threshold
    if confidence < CONFIDENCE_THRESHOLD:
        logger.warning(
            f"LOW_CONFIDENCE | IP: {client_ip} | session: {data.session_id} "
            f"| hash: {data_hash} | confidence: {confidence:.3f}"
        )
        return {
            "session_id": data.session_id,
            "prediction": None,
            "label": "Uncertain",
            "confidence": confidence,
            "warning": (
                f"Model confidence ({confidence:.1%}) is below threshold "
                f"({CONFIDENCE_THRESHOLD:.1%}). Manual review recommended."
            ),
            "probabilities": {
                "safe": float(probs[0]),
                "suspicious": float(probs[1]),
                "dangerous": float(probs[2])
            },
            "data_hash": data_hash
        }

    # رد سليم
    response = {
        "session_id": data.session_id,
        "prediction": pred_class,
        "label": label_map[pred_class],
        "confidence": confidence,
        "probabilities": {
            "safe": float(probs[0]),
            "suspicious": float(probs[1]),
            "dangerous": float(probs[2])
        },
        "data_hash": data_hash
    }

    if anomaly_warning:
        response["anomaly_warning"] = anomaly_warning

    return response


# ----------------------------
# ENDPOINT: فحص صحة الـ API
# ----------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "max_events": MAX_EVENTS_PER_SESSION
    }