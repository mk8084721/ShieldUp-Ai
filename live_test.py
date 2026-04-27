"""
====================================================
  ShieldUp — Live Server Tests
  شغّله على جهازك: python live_test.py
  السيرفر لازم يكون شغال على localhost:8000
====================================================
"""

import urllib.request
import urllib.error
import json
import time

BASE_URL = "http://localhost:8000"

# =============================================
# COLORS
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
# HTTP HELPER
# =============================================
def post(endpoint, payload):
    url = BASE_URL + endpoint
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
            return resp.status, body
    except urllib.error.HTTPError as e:
        body = {}
        try:
            body = json.loads(e.read().decode())
        except:
            pass
        return e.code, body
    except Exception as ex:
        return None, {"error": str(ex)}

def get(endpoint):
    url = BASE_URL + endpoint
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except Exception as ex:
        return None, {"error": str(ex)}

def print_result(status, body):
    print(f"  {BLUE}Status:{RESET} {status}")
    print(f"  {BLUE}Response:{RESET}")
    print("  " + json.dumps(body, ensure_ascii=False, indent=4).replace("\n", "\n  "))

# =============================================
# بيانات مساعدة
# =============================================
def safe_event():
    return {
        "cam_count": 2.0,
        "cam_duration": 30.0,
        "cam_bg_ratio": 0.05,
        "loc_freq": 3.0,
        "loc_bg_ratio": 0.02,
        "mic_duration": 5.0,
        "mic_bg_flag": 0,
        "data_upload": 500000.0,
        "data_download": 2000000.0,
        "bg_data": 100000.0,
        "reserved": 0.0
    }

def dangerous_event():
    return {
        "cam_count": 50.0,
        "cam_duration": 3600.0,
        "cam_bg_ratio": 0.90,
        "loc_freq": 500.0,
        "loc_bg_ratio": 0.95,
        "mic_duration": 7200.0,
        "mic_bg_flag": 1,
        "data_upload": 80000000.0,
        "data_download": 20000000.0,
        "bg_data": 30000000.0,
        "reserved": 0.0
    }

# =============================================
# التأكد إن السيرفر شغال
# =============================================
print(f"\n{BOLD}{GREEN}{'='*55}")
print("  ShieldUp — Live Server Tests")
print(f"  Target: {BASE_URL}")
print(f"{'='*55}{RESET}")

print("\n🔌 بنتحقق إن السيرفر شغال...")
status, body = get("/health")
if status is None:
    print(f"\n{RED}{BOLD}❌ السيرفر مش شغال على {BASE_URL}{RESET}")
    print(f"{YELLOW}تأكد إنك شغّلت: uvicorn main:app --reload{RESET}\n")
    exit(1)
ok(f"السيرفر شغال! — {json.dumps(body, ensure_ascii=False)}")

# =============================================
# TEST 1: لعبة آمنة
# =============================================
header("TEST 1: لعبة آمنة — Minecraft مثلاً")
payload = {
    "session_id": "minecraft-safe-001",
    "events": [safe_event() for _ in range(5)]
}
status, body = post("/predict", payload)
print_result(status, body)
if status == 200 and body.get("label") in ("Safe", "Uncertain"):
    ok(f"النتيجة المتوقعة ✓")
else:
    warn(f"نتيجة غير متوقعة: {body.get('label')}")

# =============================================
# TEST 2: لعبة خطيرة
# =============================================
header("TEST 2: لعبة خطيرة — Talking Angela مثلاً")
payload = {
    "session_id": "talking-angela-dangerous-002",
    "events": [dangerous_event() for _ in range(5)]
}
status, body = post("/predict", payload)
print_result(status, body)
if status == 200 and body.get("label") in ("Dangerous", "Suspicious"):
    ok("اتكشفت كلعبة خطيرة ✓")
elif status == 200:
    warn(f"اتصنفت كـ: {body.get('label')}")

# =============================================
# TEST 3: NaN Poisoning
# =============================================
header("TEST 3: Poisoning — NaN في cam_bg_ratio")
ev = safe_event()
ev["cam_bg_ratio"] = None   # JSON null → سيتم رفضه
payload = {
    "session_id": "poisoned-nan-003",
    "events": [ev]
}
status, body = post("/predict", payload)
print_result(status, body)
if status in (422, 400):
    ok(f"السيرفر رفض الـ NaN (status {status}) ✓")
else:
    warn(f"السيرفر قبل البيانات الفاسدة! status={status}")

# =============================================
# TEST 4: Infinity Poisoning
# =============================================
header("TEST 4: Poisoning — قيمة ضخمة جداً في data_upload")
ev = safe_event()
ev["data_upload"] = 9.9e99   # أكبر بكتير من الـ bound
payload = {
    "session_id": "poisoned-huge-004",
    "events": [ev]
}
status, body = post("/predict", payload)
print_result(status, body)
if status in (422, 400):
    ok(f"السيرفر رفض القيمة الضخمة (status {status}) ✓")
else:
    warn(f"القيمة الضخمة اتقبلت! status={status}")

# =============================================
# TEST 5: قيمة سالبة
# =============================================
header("TEST 5: Poisoning — cam_duration سالبة")
ev = safe_event()
ev["cam_duration"] = -500.0
payload = {
    "session_id": "poisoned-negative-005",
    "events": [ev]
}
status, body = post("/predict", payload)
print_result(status, body)
if status in (422, 400):
    ok(f"السيرفر رفض القيمة السالبة (status {status}) ✓")
else:
    warn(f"القيمة السالبة اتقبلت! status={status}")

# =============================================
# TEST 6: cam_bg_ratio > 1
# =============================================
header("TEST 6: Poisoning — cam_bg_ratio = 9.5 (نسبة مستحيلة)")
ev = safe_event()
ev["cam_bg_ratio"] = 9.5
payload = {
    "session_id": "poisoned-ratio-006",
    "events": [ev]
}
status, body = post("/predict", payload)
print_result(status, body)
if status in (422, 400):
    ok(f"رُفضت النسبة المستحيلة (status {status}) ✓")
else:
    warn(f"نسبة 9.5 اتقبلت! status={status}")

# =============================================
# TEST 7: mic_bg_flag مش binary
# =============================================
header("TEST 7: Poisoning — mic_bg_flag = 99")
ev = safe_event()
ev["mic_bg_flag"] = 99
payload = {
    "session_id": "poisoned-flag-007",
    "events": [ev]
}
status, body = post("/predict", payload)
print_result(status, body)
if status in (422, 400):
    ok(f"رُفض الـ flag الخاطئ (status {status}) ✓")
else:
    warn(f"flag=99 اتقبل! status={status}")

# =============================================
# TEST 8: events فاضية
# =============================================
header("TEST 8: Poisoning — events list فاضية")
payload = {
    "session_id": "empty-events-008",
    "events": []
}
status, body = post("/predict", payload)
print_result(status, body)
if status in (422, 400):
    ok(f"رُفض الطلب الفاضي (status {status}) ✓")
else:
    warn(f"Events فاضية اتقبلت! status={status}")

# =============================================
# TEST 9: Rate Limiting
# =============================================
header("TEST 9: Rate Limiting — 35 طلب متتالي")
print(f"  {YELLOW}ده ممكن ياخد ثواني...{RESET}")
blocked = 0
passed  = 0
for i in range(35):
    p = {"session_id": f"rate-test-{i:03d}", "events": [safe_event()]}
    s, b = post("/predict", p)
    if s == 429:
        blocked += 1
        if blocked == 1:
            fail(f"أول حجب عند الطلب #{i+1}")
    elif s == 200:
        passed += 1
    time.sleep(0.05)

info(f"مسموح: {passed} | محجوب: {blocked} من أصل 35 طلب")
if blocked > 0:
    ok("الـ Rate Limiter شغال ✓")
else:
    warn("ما فيش Rate Limiting! راجع الـ RATE_LIMIT_PER_MINUTE في main_protected.py")

# =============================================
# TEST 10: session_id فيه SQL Injection
# =============================================
header("TEST 10: SQL Injection في session_id")
payload = {
    "session_id": "'; DROP TABLE games; --",
    "events": [safe_event()]
}
status, body = post("/predict", payload)
print_result(status, body)
if status in (422, 400):
    ok(f"رُفض الـ SQL Injection (status {status}) ✓")
else:
    warn(f"session_id المشبوه اتقبل! status={status}")

# =============================================
# TEST 11: events كتير جداً (250)
# =============================================
header("TEST 11: Flood — 250 event في طلب واحد")
payload = {
    "session_id": "flood-test-011",
    "events": [safe_event() for _ in range(250)]
}
status, body = post("/predict", payload)
print_result(status, body)
if status in (422, 400):
    ok(f"رُفض الـ flood (status {status}) ✓")
else:
    warn(f"250 events اتقبلوا! status={status}")

# =============================================
# TEST 12: mixed events (بعضها سليم وبعضها مشبوه)
# =============================================
header("TEST 12: Mixed Session — Roblox بيطلب كاميرا في الخلفية")
events = []
for i in range(10):
    ev = safe_event()
    if i % 3 == 0:   # كل 3 events، في spike مشبوه
        ev["cam_bg_ratio"] = 0.75
        ev["mic_bg_flag"] = 1
        ev["data_upload"] = 25000000.0
    events.append(ev)

payload = {
    "session_id": "roblox-mixed-012",
    "events": events
}
status, body = post("/predict", payload)
print_result(status, body)
if status == 200:
    label = body.get("label", "?")
    conf  = body.get("confidence", 0)
    if label in ("Suspicious", "Dangerous"):
        ok(f"اكتشف السلوك المشبوه → {label} ({conf:.1%}) ✓")
    else:
        warn(f"صنّفها كـ {label} — ممكن تراجع الـ threshold")

# =============================================
# SUMMARY
# =============================================
print(f"\n{BOLD}{GREEN}{'='*55}")
print("  ✅ كل الاختبارات خلصت!")
print(f"  السيرفر: {BASE_URL}")
print(f"{'='*55}{RESET}\n")
