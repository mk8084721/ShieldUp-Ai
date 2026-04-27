"""
====================================================
  ShieldUp — Poisoning Attack Protection Tests
  شغّل الملف ده مباشرة: python test_protections.py
====================================================
مش محتاج model أو server — بيحاكي كل الحمايات
"""

import numpy as np
import json
import hashlib
import time
from datetime import datetime
from collections import defaultdict

# =============================================
# CONFIG (نفس اللي في main_protected.py)
# =============================================
FEATURES = [
    "cam_count", "cam_duration", "cam_bg_ratio",
    "loc_freq", "loc_bg_ratio",
    "mic_duration", "mic_bg_flag",
    "data_upload", "data_download",
    "bg_data", "reserved"
]

FEATURE_BOUNDS = {
    "cam_count":     (0, 1000),
    "cam_duration":  (0, 86400),
    "cam_bg_ratio":  (0.0, 1.0),
    "loc_freq":      (0, 10000),
    "loc_bg_ratio":  (0.0, 1.0),
    "mic_duration":  (0, 86400),
    "mic_bg_flag":   (0, 1),
    "data_upload":   (0, 1e9),
    "data_download": (0, 1e9),
    "bg_data":       (0, 1e9),
    "reserved":      (0, 1e9),
}

CONFIDENCE_THRESHOLD = 0.60
RATE_LIMIT_PER_MINUTE = 5  # صغير للتجربة

# بيانات التدريب التقريبية (بتمثل scaler)
SCALER_CENTER = np.array([2.0, 30.0, 0.1, 5.0, 0.05,
                           10.0, 0.0, 1e6, 5e6, 5e5, 0.0])
SCALER_SCALE  = np.array([5.0, 60.0, 0.15, 10.0, 0.1,
                           20.0, 0.5, 2e6, 8e6, 1e6, 1.0])

# =============================================
# COLORS للـ output
# =============================================
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✅ PASS:{RESET} {msg}")
def fail(msg): print(f"  {RED}❌ BLOCKED:{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠️  WARN:{RESET} {msg}")
def info(msg): print(f"  {BLUE}ℹ️  INFO:{RESET} {msg}")

def header(title):
    print(f"\n{BOLD}{'='*55}{RESET}")
    print(f"{BOLD}  🧪 {title}{RESET}")
    print(f"{BOLD}{'='*55}{RESET}")

# =============================================
# الحمايات (مستخرجة من main_protected.py)
# =============================================

request_log = defaultdict(list)

def check_rate_limit(ip: str, limit: int = RATE_LIMIT_PER_MINUTE) -> dict:
    now = time.time()
    minute_ago = now - 60
    request_log[ip] = [t for t in request_log[ip] if t > minute_ago]
    count = len(request_log[ip])
    if count >= limit:
        return {"allowed": False, "count": count}
    request_log[ip].append(now)
    return {"allowed": True, "count": count + 1}

def validate_event(event: dict) -> dict:
    errors = []
    for feat, val in event.items():
        if feat not in FEATURE_BOUNDS:
            continue
        # NaN / Inf
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            errors.append(f"'{feat}' = {val} → NaN أو Inf مش مسموح")
        # Negative
        if val < 0:
            errors.append(f"'{feat}' = {val} → قيمة سالبة مش مسموح")
        # Bounds
        low, high = FEATURE_BOUNDS[feat]
        if not (low <= val <= high):
            errors.append(f"'{feat}' = {val} → خارج النطاق [{low}, {high}]")
    # Binary check
    if "mic_bg_flag" in event and event["mic_bg_flag"] not in (0, 1):
        errors.append(f"'mic_bg_flag' = {event['mic_bg_flag']} → لازم 0 أو 1 بس")
    return {"valid": len(errors) == 0, "errors": errors}

def detect_anomaly(X: np.ndarray) -> dict:
    z = np.abs((X - SCALER_CENTER) / SCALER_SCALE)
    extreme = z > 5
    extreme_feats = [FEATURES[i] for i in np.where(extreme.any(axis=0))[0]]
    return {
        "has_anomaly": bool(extreme.any()),
        "max_z": float(z.max()),
        "extreme_features": extreme_feats
    }

def check_bounds_batch(X: np.ndarray) -> dict:
    violations = []
    for i, feat in enumerate(FEATURES):
        low, high = FEATURE_BOUNDS[feat]
        col = X[:, i]
        if col.min() < low or col.max() > high:
            violations.append({"feature": feat,
                                "got": [float(col.min()), float(col.max())],
                                "allowed": [low, high]})
    return {"ok": len(violations) == 0, "violations": violations}

def fake_predict(X: np.ndarray, force_low_confidence=False):
    """موديل وهمي للتجربة"""
    if force_low_confidence:
        return np.array([0.38, 0.32, 0.30])
    # بناءً على متوسط cam_bg_ratio
    avg_cam = X[:, 2].mean()
    avg_mic = X[:, 6].mean()
    if avg_cam > 0.7 or avg_mic > 0.8:
        return np.array([0.05, 0.15, 0.80])   # Dangerous
    elif avg_cam > 0.3:
        return np.array([0.20, 0.65, 0.15])   # Suspicious
    else:
        return np.array([0.85, 0.10, 0.05])   # Safe

def compute_hash(X: np.ndarray) -> str:
    return hashlib.sha256(X.tobytes()).hexdigest()[:16]

def full_pipeline(session_id: str, events: list,
                  ip="1.2.3.4", force_low_conf=False) -> dict:
    """Pipeline كامل بكل الحمايات"""
    print(f"\n  📦 Session: {session_id}  |  IP: {ip}  |  Events: {len(events)}")

    # 1. Rate Limit
    rl = check_rate_limit(ip)
    if not rl["allowed"]:
        fail(f"Rate limit! ({rl['count']} requests/min)")
        return {"blocked": "rate_limit"}

    # 2. عدد الـ events
    if len(events) < 1 or len(events) > 200:
        fail(f"Invalid events count: {len(events)}")
        return {"blocked": "events_count"}

    # 3. Validate كل event
    for i, ev in enumerate(events):
        v = validate_event(ev)
        if not v["valid"]:
            fail(f"Event #{i} فشل التحقق:")
            for e in v["errors"]:
                print(f"       → {e}")
            return {"blocked": "validation", "errors": v["errors"]}

    # 4. تحويل لـ numpy
    X = np.array([[ev[f] for f in FEATURES] for ev in events])
    data_hash = compute_hash(X)
    info(f"Data hash: {data_hash}")

    # 5. Bounds check
    bc = check_bounds_batch(X)
    if not bc["ok"]:
        fail("Bounds violation في البيانات:")
        for v in bc["violations"]:
            print(f"       → {v['feature']}: got {v['got']}, allowed {v['allowed']}")
        return {"blocked": "bounds", "violations": bc["violations"]}

    # 6. Anomaly detection
    anomaly = detect_anomaly(X)
    if anomaly["has_anomaly"]:
        warn(f"Anomaly! Z-score={anomaly['max_z']:.1f} في {anomaly['extreme_features']}")
    else:
        info(f"Z-score max = {anomaly['max_z']:.2f} → طبيعي")

    # 7. Predict
    probs = fake_predict(X, force_low_confidence=force_low_conf)
    pred  = int(np.argmax(probs))
    conf  = float(probs[pred])
    label = ["Safe", "Suspicious", "Dangerous"][pred]

    # 8. Confidence threshold
    if conf < CONFIDENCE_THRESHOLD:
        warn(f"Confidence منخفضة جداً ({conf:.1%}) → Uncertain")
        return {
            "session_id": session_id,
            "label": "Uncertain",
            "confidence": conf,
            "note": "Manual review recommended",
            "hash": data_hash
        }

    ok(f"نتيجة: {label} ({conf:.1%} confidence)")
    return {
        "session_id": session_id,
        "label": label,
        "confidence": conf,
        "probabilities": {
            "safe": float(probs[0]),
            "suspicious": float(probs[1]),
            "dangerous": float(probs[2])
        },
        "anomaly_warning": anomaly["extreme_features"] if anomaly["has_anomaly"] else None,
        "hash": data_hash
    }

# =============================================
# بيانات مساعدة
# =============================================
def safe_event():
    return {
        "cam_count": 2.0, "cam_duration": 30.0, "cam_bg_ratio": 0.05,
        "loc_freq": 3.0, "loc_bg_ratio": 0.02,
        "mic_duration": 5.0, "mic_bg_flag": 0,
        "data_upload": 500000.0, "data_download": 2000000.0,
        "bg_data": 100000.0, "reserved": 0.0
    }

def dangerous_event():
    return {
        "cam_count": 50.0, "cam_duration": 3600.0, "cam_bg_ratio": 0.85,
        "loc_freq": 500.0, "loc_bg_ratio": 0.90,
        "mic_duration": 7200.0, "mic_bg_flag": 1,
        "data_upload": 50000000.0, "data_download": 10000000.0,
        "bg_data": 20000000.0, "reserved": 0.0
    }

# =============================================
# ============  الاختبارات  =================
# =============================================

print(f"\n{BOLD}{GREEN}{'='*55}")
print("  ShieldUp — Poisoning Attack Protection Tests")
print(f"{'='*55}{RESET}")

# -----------------------------------------------
header("TEST 1: طلب سليم — لعبة آمنة")
# -----------------------------------------------
result = full_pipeline(
    session_id="safe-game-001",
    events=[safe_event() for _ in range(5)]
)
print(f"  النتيجة: {json.dumps(result, ensure_ascii=False, indent=4)}")

# -----------------------------------------------
header("TEST 2: لعبة خطيرة — cam_bg_ratio عالي")
# -----------------------------------------------
result = full_pipeline(
    session_id="dangerous-game-002",
    events=[dangerous_event() for _ in range(5)]
)
print(f"  النتيجة: {json.dumps(result, ensure_ascii=False, indent=4)}")

# -----------------------------------------------
header("TEST 3: Poisoning — قيمة NaN في البيانات")
# -----------------------------------------------
poisoned = safe_event()
poisoned["cam_bg_ratio"] = float("nan")   # ← هجوم
full_pipeline(
    session_id="poisoned-nan-003",
    events=[poisoned]
)

# -----------------------------------------------
header("TEST 4: Poisoning — قيمة Infinity")
# -----------------------------------------------
poisoned = safe_event()
poisoned["data_upload"] = float("inf")    # ← هجوم
full_pipeline(
    session_id="poisoned-inf-004",
    events=[poisoned]
)

# -----------------------------------------------
header("TEST 5: Poisoning — قيمة سالبة")
# -----------------------------------------------
poisoned = safe_event()
poisoned["cam_duration"] = -999.0         # ← هجوم
full_pipeline(
    session_id="poisoned-neg-005",
    events=[poisoned]
)

# -----------------------------------------------
header("TEST 6: Poisoning — cam_bg_ratio > 1")
# -----------------------------------------------
poisoned = safe_event()
poisoned["cam_bg_ratio"] = 5.7            # ← هجوم (النسبة ما تعدش 1)
full_pipeline(
    session_id="poisoned-ratio-006", ip="2.2.2.2",
    events=[poisoned]
)

# -----------------------------------------------
header("TEST 7: Poisoning — mic_bg_flag = 99 (مش binary)")
# -----------------------------------------------
poisoned = safe_event()
poisoned["mic_bg_flag"] = 99              # ← هجوم
full_pipeline(
    session_id="poisoned-flag-007", ip="3.3.3.3",
    events=[poisoned]
)

# -----------------------------------------------
header("TEST 8: Poisoning — قيمة كبيرة جداً (Z-score عالي)")
# -----------------------------------------------
poisoned = safe_event()
poisoned["cam_count"] = 999999.0          # ← شاذ إحصائياً
full_pipeline(
    session_id="poisoned-zcore-008", ip="4.4.4.4",
    events=[poisoned]
)

# -----------------------------------------------
header("TEST 9: Rate Limiting — نفس IP بيبعت كتير")
# -----------------------------------------------
print("  بنبعت 7 طلبات من نفس الـ IP (الـ limit = 5)...")
for i in range(7):
    rl = check_rate_limit("attacker-ip", limit=5)
    if rl["allowed"]:
        ok(f"طلب #{i+1} → مسموح ({rl['count']}/5)")
    else:
        fail(f"طلب #{i+1} → محجوب! ({rl['count']} requests/min)")

# -----------------------------------------------
header("TEST 10: Confidence منخفضة — Uncertain")
# -----------------------------------------------
result = full_pipeline(
    session_id="low-conf-010", ip="5.5.5.5",
    events=[safe_event() for _ in range(3)],
    force_low_conf=True
)
print(f"  النتيجة: {json.dumps(result, ensure_ascii=False, indent=4)}")

# -----------------------------------------------
header("TEST 11: Events كتير جداً — فوق الـ limit")
# -----------------------------------------------
full_pipeline(
    session_id="too-many-events-011", ip="6.6.6.6",
    events=[safe_event() for _ in range(250)]   # max = 200
)

# -----------------------------------------------
header("TEST 12: session_id فيه SQL Injection محاولة")
# -----------------------------------------------
bad_id = "'; DROP TABLE games; --"
print(f"  session_id المشبوه: {bad_id}")
# محاكاة الـ validator
allowed_chars = bad_id.replace("-", "").replace("_", "")
if not allowed_chars.isalnum():
    fail(f"session_id رُفض — يحتوي على أحرف غير مسموح بها")
else:
    ok("session_id سليم")

# -----------------------------------------------
header("TEST 13: Data Hash — التحقق من سلامة البيانات")
# -----------------------------------------------
events = [safe_event() for _ in range(3)]
X = np.array([[ev[f] for f in FEATURES] for ev in events])
original_hash = compute_hash(X)
info(f"Hash قبل التعديل:  {original_hash}")

# محاكاة تعديل البيانات في الطريق
X_tampered = X.copy()
X_tampered[0, 0] = 9999.0
tampered_hash = compute_hash(X_tampered)
info(f"Hash بعد التعديل:  {tampered_hash}")

if original_hash != tampered_hash:
    ok("الـ Hash اكتشف التلاعب في البيانات!")
else:
    fail("الـ Hash ما اكتشفش التعديل!")

# -----------------------------------------------
print(f"\n{BOLD}{GREEN}{'='*55}")
print("  ✅ كل الاختبارات خلصت!")
print(f"{'='*55}{RESET}\n")
