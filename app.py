"""
LeafScan AI — Full Frontend
Supports both .keras (web/desktop) and .tflite (mobile/lite) inference.
Place beside: plant_disease_model.keras  OR  plant_disease_model.tflite
              class_names.json

Auth system uses SQLite (leafscan_users.db) — created automatically on first run.

For persistent login across browser sessions (remember-me), install:
    pip install extra-streamlit-components
Without it, users will need to log in on every new browser session.
"""

import streamlit as st
import numpy as np
import json
from PIL import Image
import os
import time
import datetime
import io
import re as _re
import sqlite3
import hashlib
import uuid

# ── Optional imports (graceful fallback) ──────────────────────────────────────
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_RUNTIME = True
except ImportError:
    TFLITE_RUNTIME = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.platypus import Image as RLImage
    from reportlab.lib import colors
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    import extra_streamlit_components as stx
    COOKIES_AVAILABLE = True
except ImportError:
    COOKIES_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LeafScan AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# PATHS & BASE DIR (defined here so DB can use it)
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
KERAS_PATH       = os.path.join(BASE_DIR, "plant_disease_model.keras")
TFLITE_PATH      = os.path.join(BASE_DIR, "plant_disease_model.tflite")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
DB_PATH          = os.path.join(BASE_DIR, "leafscan_users.db")

IMG_SIZE  = (224, 224)
TTA_STEPS = 8

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE — SQLite user & session storage
# ══════════════════════════════════════════════════════════════════════════════
def _db():
    """Open a SQLite connection."""
    return sqlite3.connect(DB_PATH)

def init_db():
    """Create tables if they don't exist yet."""
    with _db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL,
                phone         TEXT    UNIQUE NOT NULL,
                password_hash TEXT    NOT NULL,
                salt          TEXT    NOT NULL,
                created_at    TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                token      TEXT    UNIQUE NOT NULL,
                created_at TEXT    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS plant_profiles (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                name       TEXT    NOT NULL,
                plant_type TEXT,
                notes      TEXT,
                created_at TEXT    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS scan_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL,
                plant_id      INTEGER,
                raw_label     TEXT    NOT NULL,
                display_label TEXT    NOT NULL,
                confidence    REAL    NOT NULL,
                is_healthy    INTEGER NOT NULL,
                severity      TEXT    NOT NULL,
                spread        TEXT,
                season        TEXT,
                treatment     TEXT,
                backend       TEXT,
                scan_time     TEXT    NOT NULL,
                image_thumb   BLOB,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)

init_db()

# ── Auth helpers ───────────────────────────────────────────────────────────────
def _hash_pw(pw: str, salt: str = None):
    if salt is None:
        salt = uuid.uuid4().hex
    digest = hashlib.sha256((pw + salt).encode()).hexdigest()
    return digest, salt

def phone_exists(phone: str) -> bool:
    with _db() as conn:
        return conn.execute(
            "SELECT 1 FROM users WHERE phone=?", (phone,)
        ).fetchone() is not None

def register_user(name: str, phone: str, pw: str):
    """Returns (True, user_id) or (False, reason)."""
    if phone_exists(phone):
        return False, "phone_exists"
    h, s = _hash_pw(pw)
    try:
        with _db() as conn:
            cur = conn.execute(
                "INSERT INTO users (name,phone,password_hash,salt,created_at) "
                "VALUES (?,?,?,?,?)",
                (name, phone, h, s, datetime.datetime.now().isoformat())
            )
            return True, cur.lastrowid
    except sqlite3.IntegrityError:
        return False, "phone_exists"

def login_user(phone: str, pw: str):
    """Returns user dict or None."""
    with _db() as conn:
        row = conn.execute(
            "SELECT id,name,phone,password_hash,salt FROM users WHERE phone=?",
            (phone,)
        ).fetchone()
    if row is None:
        return None
    uid, name, ph, stored_h, salt = row
    h, _ = _hash_pw(pw, salt)
    if h == stored_h:
        return {"id": uid, "name": name, "phone": ph}
    return None

def make_session(user_id: int) -> str:
    token = uuid.uuid4().hex + uuid.uuid4().hex
    with _db() as conn:
        conn.execute(
            "INSERT INTO sessions (user_id,token,created_at) VALUES (?,?,?)",
            (user_id, token, datetime.datetime.now().isoformat())
        )
    return token

def check_session(token: str):
    """Returns user dict if token is valid, else None."""
    if not token:
        return None
    with _db() as conn:
        row = conn.execute(
            "SELECT u.id,u.name,u.phone FROM sessions s "
            "JOIN users u ON s.user_id=u.id WHERE s.token=?",
            (token,)
        ).fetchone()
    return {"id": row[0], "name": row[1], "phone": row[2]} if row else None

def end_session(token: str):
    if token:
        with _db() as conn:
            conn.execute("DELETE FROM sessions WHERE token=?", (token,))

def delete_account(user_id: int):
    with _db() as conn:
        conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))

def reset_password(phone: str, new_pw: str) -> bool:
    with _db() as conn:
        row = conn.execute("SELECT id FROM users WHERE phone=?", (phone,)).fetchone()
        if not row:
            return False
        h, s = _hash_pw(new_pw)
        conn.execute(
            "UPDATE users SET password_hash=?,salt=? WHERE phone=?", (h, s, phone)
        )
    return True

# ── Scan history helpers ────────────────────────────────────────────────────
def save_scan(user_id, raw_label, display_label, confidence, is_healthy,
              severity, spread, season, treatment, backend, pil_image, plant_id=None):
    thumb = None
    try:
        t = pil_image.convert("RGB").resize((128, 128), Image.LANCZOS)
        buf = io.BytesIO(); t.save(buf, format="JPEG", quality=70)
        thumb = buf.getvalue()
    except Exception:
        pass
    with _db() as conn:
        conn.execute(
            "INSERT INTO scan_history (user_id,plant_id,raw_label,display_label,confidence,"
            "is_healthy,severity,spread,season,treatment,backend,scan_time,image_thumb) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (user_id, plant_id, raw_label, display_label, confidence, int(is_healthy),
             severity, spread, season, treatment, backend,
             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), thumb)
        )

def get_user_scans(user_id):
    with _db() as conn:
        rows = conn.execute(
            "SELECT id,plant_id,raw_label,display_label,confidence,is_healthy,severity,"
            "spread,season,treatment,backend,scan_time,image_thumb "
            "FROM scan_history WHERE user_id=? ORDER BY id DESC LIMIT 50", (user_id,)
        ).fetchall()
    return [{"id":r[0],"plant_id":r[1],"raw_label":r[2],"display_label":r[3],
             "confidence":r[4],"is_healthy":bool(r[5]),"severity":r[6],
             "spread":r[7],"season":r[8],"treatment":r[9],"backend":r[10],
             "scan_time":r[11],"image_thumb":r[12]} for r in rows]

def delete_scan(scan_id, user_id):
    with _db() as conn:
        conn.execute("DELETE FROM scan_history WHERE id=? AND user_id=?", (scan_id, user_id))

# ── Plant profile helpers ───────────────────────────────────────────────────
def get_plants(user_id):
    with _db() as conn:
        rows = conn.execute(
            "SELECT id,name,plant_type,notes,created_at FROM plant_profiles "
            "WHERE user_id=? ORDER BY id ASC", (user_id,)
        ).fetchall()
    return [{"id":r[0],"name":r[1],"plant_type":r[2],"notes":r[3],"created_at":r[4]} for r in rows]

def add_plant(user_id, name, plant_type, notes):
    with _db() as conn:
        conn.execute(
            "INSERT INTO plant_profiles (user_id,name,plant_type,notes,created_at) VALUES (?,?,?,?,?)",
            (user_id, name, plant_type, notes, datetime.datetime.now().isoformat())
        )

def delete_plant(plant_id, user_id):
    with _db() as conn:
        conn.execute("DELETE FROM plant_profiles WHERE id=? AND user_id=?", (plant_id, user_id))
        conn.execute("UPDATE scan_history SET plant_id=NULL WHERE plant_id=? AND user_id=?",
                     (plant_id, user_id))

def get_plant_scans(plant_id, user_id):
    with _db() as conn:
        rows = conn.execute(
            "SELECT id,raw_label,display_label,confidence,is_healthy,severity,scan_time,image_thumb "
            "FROM scan_history WHERE plant_id=? AND user_id=? ORDER BY id DESC",
            (plant_id, user_id)
        ).fetchall()
    return [{"id":r[0],"raw_label":r[1],"display_label":r[2],"confidence":r[3],
             "is_healthy":bool(r[4]),"severity":r[5],"scan_time":r[6],"image_thumb":r[7]}
            for r in rows]

# ══════════════════════════════════════════════════════════════════════════════
# COOKIE MANAGER  (persistent login across browser sessions)
# ══════════════════════════════════════════════════════════════════════════════
SESSION_COOKIE = "ls_sess"   # stores session token
VISITED_COOKIE = "ls_vis"    # marks that user has seen onboarding

if COOKIES_AVAILABLE:
    _cm = stx.CookieManager(key="leafscan_cm")
else:
    _cm = None

def _get_cookie(key: str):
    if _cm is None:
        return st.session_state.get(f"__ck_{key}")
    try:
        return _cm.get(key)
    except Exception:
        return None

def _set_cookie(key: str, val: str, days: int = 365):
    if _cm is None:
        st.session_state[f"__ck_{key}"] = val
        return
    try:
        exp = datetime.datetime.now() + datetime.timedelta(days=days)
        _cm.set(key, val, expires_at=exp)
    except Exception:
        st.session_state[f"__ck_{key}"] = val

def _del_cookie(key: str):
    if _cm is None:
        st.session_state.pop(f"__ck_{key}", None)
        return
    try:
        _cm.delete(key)
    except Exception:
        st.session_state.pop(f"__ck_{key}", None)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — initialise all keys
# ══════════════════════════════════════════════════════════════════════════════
_defaults = {
    "stage":         None,
    "user":          None,
    "token":         None,
    "auth_mode":     "login",
    "history":       [],
    "total_scans":   0,
    "healthy_count": 0,
    "lang":          "en",
    "scan_submitted": False,
    "last_uploaded_name": None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Language strings ──────────────────────────────────────────────────────────
LANG = {
    "en": {
        "scan_tab": "🔬 Scan Leaf", "plants_tab": "🌱 My Plants", "guide_tab": "📖 Disease Guide",
        "upload_hdr": "📷 Upload or Capture Leaf",
        "camera_lbl": "📸 Take a Photo", "file_lbl": "📁 Upload from Device",
        "analyse_btn": "🔬  Analyse This Leaf",
        "result_hdr": "🤖 AI Diagnosis", "image_hdr": "🍃 Uploaded Leaf",
        "top5_hdr": "📊 Top-5 Predictions", "report_hdr": "📄 Download Report",
        "download_btn": "⬇️  Download Full Diagnosis Report (PDF)",
        "scan_again": "🔄  Scan Another Leaf",
        "treatment_title": "💊 Recommended Treatment",
        "healthy_status": "✅ Plant Status",
        "spread_lbl": "🌬 How It Spreads", "season_lbl": "📅 Peak Season",
        "tip_title": "🌿 Plant Care Tip",
        "low_conf": "⚠️ Low confidence ({c:.1f}%) — try a clearer photo.",
        "history_hdr": "📋 Scan History", "no_history": "No scans yet.",
        "plants_hdr": "🌱 My Plants", "add_plant": "➕ Add New Plant",
        "plant_name": "Plant Name", "plant_type": "Plant Type", "plant_notes": "Notes (optional)",
        "save_plant": "Save Plant", "no_plants": "No plants added yet.",
        "profile_hdr": "👤 My Profile",
        "logout_btn": "🚪 Logout", "delete_btn": "🗑 Delete Account",
        "backend_hdr": "⚙️ Inference Backend",
        "stats_hdr": "📊 Session Stats",
        "lang_label": "🌐 Language",
    },
    "hi": {
        "scan_tab": "🔬 पत्ती स्कैन", "plants_tab": "🌱 मेरे पौधे", "guide_tab": "📖 रोग गाइड",
        "upload_hdr": "📷 पत्ती अपलोड या कैप्चर करें",
        "camera_lbl": "📸 फोटो लें", "file_lbl": "📁 डिवाइस से अपलोड",
        "analyse_btn": "🔬  इस पत्ती का विश्लेषण करें",
        "result_hdr": "🤖 AI निदान", "image_hdr": "🍃 अपलोड की पत्ती",
        "top5_hdr": "📊 टॉप-5 भविष्यवाणियाँ", "report_hdr": "📄 रिपोर्ट डाउनलोड",
        "download_btn": "⬇️  पूरी निदान रिपोर्ट डाउनलोड करें (PDF)",
        "scan_again": "🔄  दूसरी पत्ती स्कैन करें",
        "treatment_title": "💊 अनुशंसित उपचार",
        "healthy_status": "✅ पौधे की स्थिति",
        "spread_lbl": "🌬 कैसे फैलता है", "season_lbl": "📅 मुख्य मौसम",
        "tip_title": "🌿 पौधे की देखभाल टिप",
        "low_conf": "⚠️ कम विश्वास ({c:.1f}%) — साफ फोटो लें।",
        "history_hdr": "📋 स्कैन इतिहास", "no_history": "अभी कोई स्कैन नहीं।",
        "plants_hdr": "🌱 मेरे पौधे", "add_plant": "➕ नया पौधा जोड़ें",
        "plant_name": "पौधे का नाम", "plant_type": "पौधे का प्रकार", "plant_notes": "नोट्स (वैकल्पिक)",
        "save_plant": "पौधा सहेजें", "no_plants": "अभी कोई पौधा नहीं।",
        "profile_hdr": "👤 मेरा प्रोफ़ाइल",
        "logout_btn": "🚪 लॉगआउट", "delete_btn": "🗑 अकाउंट हटाएं",
        "backend_hdr": "⚙️ मॉडल बैकएंड",
        "stats_hdr": "📊 सत्र आँकड़े",
        "lang_label": "🌐 भाषा",
    },
}

def T(key): return LANG.get(st.session_state.lang, LANG["en"]).get(key, key)
if st.session_state.stage is None:
    tok = _get_cookie(SESSION_COOKIE)
    if tok:
        u = check_session(tok)
        if u:
            # Valid persistent session → go straight to app
            st.session_state.user  = u
            st.session_state.token = tok
            st.session_state.stage = "app"
        else:
            # Token expired/invalid → clean up and show onboarding
            _del_cookie(SESSION_COOKIE)
            st.session_state.stage = "splash"
    else:
        st.session_state.stage = "splash"

# ══════════════════════════════════════════════════════════════════════════════
# STYLES — original app styles + new auth / onboarding styles
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,400;1,600&family=Outfit:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: #050d06 !important;
    color: #ddeedd !important;
    font-family: 'Outfit', sans-serif !important;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 70% 50% at 15% 25%, rgba(0, 90, 25, 0.16) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 85% 75%, rgba(0, 70, 18, 0.12) 0%, transparent 55%),
        radial-gradient(ellipse 80% 60% at 50% 50%, rgba(5, 20, 8, 0.3) 0%, transparent 80%);
    pointer-events: none;
    z-index: 0;
    animation: bgPulse 10s ease-in-out infinite alternate;
}
@keyframes bgPulse {
    0%   { opacity: 0.6; }
    100% { opacity: 1; }
}

#MainMenu, footer { visibility: hidden; }
header { visibility: hidden; }

[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    color: rgba(0, 200, 80, 0.75) !important;
    background: rgba(3, 8, 4, 0.92) !important;
    border: 1px solid rgba(0, 180, 70, 0.22) !important;
    border-radius: 0 10px 10px 0 !important;
    transition: background 0.2s, color 0.2s !important;
}
[data-testid="collapsedControl"]:hover,
[data-testid="stSidebarCollapsedControl"]:hover {
    background: rgba(0, 40, 12, 0.98) !important;
    color: #3dff80 !important;
    border-color: rgba(0, 220, 90, 0.4) !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg { 
    fill: currentColor !important; 
    color: inherit !important; 
}
.block-container {
    padding: 1.5rem 2rem !important;
    max-width: 1400px !important;
    position: relative; z-index: 1;
}

/* ══ SIDEBAR ══ */
section[data-testid="stSidebar"] {
    background: rgba(3, 8, 4, 0.98) !important;
    border-right: 1px solid rgba(0, 180, 70, 0.1) !important;
}
section[data-testid="stSidebar"] * { color: #b0ccb0 !important; }

/* ══ MODEL BADGE ══ */
.model-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.35rem 0.9rem; border-radius: 999px;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; margin-bottom: 0.5rem;
}
.badge-keras  { background: rgba(0,80,200,0.12); color: #7ad4f5; border: 1px solid rgba(30,120,200,0.28); }
.badge-tflite { background: rgba(80,200,0,0.1);  color: #a0f070; border: 1px solid rgba(80,180,0,0.28); }
.badge-dot { width: 6px; height: 6px; border-radius: 50%; animation: pulse-dot 1.8s ease-in-out infinite; display: inline-block; }
.badge-keras .badge-dot  { background: #7ad4f5; }
.badge-tflite .badge-dot { background: #a0f070; }
@keyframes pulse-dot { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(0.7); } }

/* ══ STAT TILES ══ */
.stat-tile {
    background: rgba(0, 22, 7, 0.75); border: 1px solid rgba(0, 160, 55, 0.15);
    border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: 0.55rem;
    position: relative; overflow: hidden; transition: border-color 0.3s ease;
}
.stat-tile::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 220, 90, 0.18), transparent);
}
.stat-tile:hover { border-color: rgba(0, 200, 80, 0.28); }
.stat-num { font-family: 'Cormorant Garamond', serif; font-size: 2.1rem; color: #3dff80; line-height: 1; font-weight: 600; }
.stat-label { font-size: 0.62rem; color: rgba(100, 150, 100, 0.55); text-transform: uppercase; letter-spacing: 0.11em; margin-top: 0.18rem; }

/* ══ UPLOAD ══ */
[data-testid="stFileUploader"] {
    background: rgba(0, 38, 12, 0.45) !important; border: 2px dashed rgba(0, 200, 80, 0.38) !important;
    border-radius: 18px !important; padding: 0.5rem !important; width: 100% !important;
    animation: uploadGlow 3.5s ease-in-out infinite;
}
@keyframes uploadGlow {
    0%, 100% { border-color: rgba(0, 200, 80, 0.38); box-shadow: none; }
    50%       { border-color: rgba(0, 220, 90, 0.6);  box-shadow: 0 0 28px rgba(0,200,80,0.07); }
}
[data-testid="stFileUploader"] > div { width: 100% !important; }
[data-testid="stFileUploader"] section {
    background: transparent !important; border: none !important;
    padding: 2.5rem 1.5rem !important; text-align: center !important; width: 100% !important;
}
[data-testid="stFileUploader"] label { color: rgba(0, 230, 110, 0.9) !important; font-size: 1.05rem !important; font-weight: 600 !important; }
[data-testid="stFileUploadDropzone"] {
    background: rgba(0, 45, 14, 0.3) !important; border: 1px dashed rgba(0, 180, 70, 0.28) !important;
    border-radius: 12px !important; min-height: 148px !important; width: 100% !important;
}
[data-testid="stFileUploadDropzone"] p     { color: rgba(0, 220, 100, 0.82) !important; font-size: 1rem !important; font-weight: 600 !important; }
[data-testid="stFileUploadDropzone"] small { color: rgba(0, 160, 60, 0.55) !important; font-size: 0.8rem !important; }
[data-testid="stFileUploader"] button,
[data-testid="stFileUploadDropzone"] button {
    background: linear-gradient(135deg, #00b84d, #00d966) !important;
    color: #010e04 !important; border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; font-size: 0.9rem !important; padding: 0.75rem 0 !important;
    margin-top: 0.75rem !important; width: 100% !important; cursor: pointer !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(135deg, #00d966, #3dff80) !important;
    transform: translateY(-1px) !important; box-shadow: 0 6px 22px rgba(0, 200, 80, 0.28) !important;
}

/* ══ HERO BANNER ══ */
.hero-banner {
    position: relative; background: linear-gradient(135deg, #060f07 0%, #0b1e0d 45%, #060f07 100%);
    border: 1px solid rgba(0, 200, 75, 0.18); border-radius: 22px;
    padding: 3rem 3rem 2.75rem; margin-bottom: 2rem; overflow: hidden;
}
.hero-banner::before {
    content: ''; position: absolute; top: 0; left: -80%; width: 55%; height: 1.5px;
    background: linear-gradient(90deg, transparent, rgba(0, 255, 110, 0.75), transparent);
    animation: scanH 5s ease-in-out infinite; pointer-events: none;
}
@keyframes scanH {
    0%   { left: -60%; top: 20%; } 45%  { left: 160%; top: 20%; }
    46%  { left: 160%; top: 78%; } 90%  { left: -60%; top: 78%; } 100% { left: -60%; top: 20%; }
}
.hero-banner::after {
    content: ''; position: absolute; top: -100px; right: -100px;
    width: 340px; height: 340px;
    background: radial-gradient(circle, rgba(0, 200, 75, 0.07) 0%, transparent 65%);
    border-radius: 50%; pointer-events: none;
}
.hero-eyebrow { font-size: 0.66rem; font-weight: 600; letter-spacing: 0.22em; text-transform: uppercase; color: rgba(0, 210, 90, 0.6); margin-bottom: 0.8rem; }
.hero-title { font-family: 'Cormorant Garamond', serif; font-size: 3.6rem; font-weight: 600; color: #e4f4e6; line-height: 1.04; letter-spacing: -0.02em; }
.hero-title em { color: #3dff80; font-style: italic; }
.hero-sub { font-size: 0.93rem; color: rgba(120, 170, 120, 0.7); margin-top: 0.8rem; line-height: 1.75; max-width: 540px; }
.hero-pills { display: flex; gap: 0.55rem; flex-wrap: wrap; margin-top: 1.6rem; }
.hero-pill {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: rgba(0, 180, 65, 0.07); border: 1px solid rgba(0, 200, 75, 0.2);
    color: rgba(0, 220, 95, 0.72); font-size: 0.65rem; font-weight: 600; letter-spacing: 0.07em;
    text-transform: uppercase; padding: 0.28rem 0.85rem; border-radius: 999px;
    animation: pillGlow 3.5s ease-in-out infinite;
}
.hero-pill:nth-child(2) { animation-delay: 0.6s; }
.hero-pill:nth-child(3) { animation-delay: 1.2s; }
.hero-pill:nth-child(4) { animation-delay: 1.8s; }
.hero-pill:nth-child(5) { animation-delay: 2.4s; }
@keyframes pillGlow {
    0%, 100% { border-color: rgba(0, 200, 75, 0.2); }
    50%       { border-color: rgba(0, 220, 95, 0.45); box-shadow: 0 0 10px rgba(0,200,75,0.06); }
}

/* ══ SECTION HEADERS ══ */
.section-hdr {
    font-size: 0.64rem; font-weight: 600; color: rgba(0, 200, 80, 0.55);
    letter-spacing: 0.18em; text-transform: uppercase;
    margin: 1.5rem 0 0.85rem; display: flex; align-items: center; gap: 0.6rem;
    font-family: 'Outfit', sans-serif;
}
.section-hdr::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(0,200,80,0.12), transparent);
    margin-left: 0.5rem;
}

/* ══ RESULT CARDS ══ */
.result-card { border-radius: 20px; padding: 1.8rem; margin-bottom: 1rem; border: 1px solid transparent; backdrop-filter: blur(14px); animation: cardReveal 0.55s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
@keyframes cardReveal { from { opacity: 0; transform: translateY(14px) scale(0.99); } to { opacity: 1; transform: translateY(0) scale(1); } }
.result-healthy  { background: linear-gradient(145deg, rgba(0,38,10,0.88), rgba(0,48,14,0.78)); border-color: rgba(0, 200, 75, 0.28); box-shadow: 0 10px 40px rgba(0, 200, 75, 0.08), inset 0 1px 0 rgba(0,255,100,0.04); }
.result-diseased { background: linear-gradient(145deg, rgba(38,18,0,0.88), rgba(32,14,0,0.78)); border-color: rgba(230, 155, 20, 0.28); box-shadow: 0 10px 40px rgba(230, 155, 20, 0.08), inset 0 1px 0 rgba(255,200,0,0.04); }
.result-tag      { font-size: 0.6rem; color: rgba(120,160,120,0.55); text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 0.45rem; }
.result-title    { font-family: 'Cormorant Garamond', serif; font-size: 1.75rem; line-height: 1.18; margin-bottom: 0.25rem; font-weight: 600; }
.result-healthy  .result-title { color: #3dff80; }
.result-diseased .result-title { color: #ffc235; }
.result-subtitle { font-size: 0.76rem; color: rgba(120,160,120,0.52); }

/* Confidence meter */
.meter-label { font-size: 0.7rem; color: rgba(120,160,120,0.65); margin: 1.25rem 0 0.45rem; display: flex; justify-content: space-between; align-items: center; }
.meter-pct   { font-size: 0.98rem; font-weight: 700; }
.result-healthy  .meter-pct { color: #3dff80; }
.result-diseased .meter-pct { color: #ffc235; }
.meter-wrap { background: rgba(0,28,6,0.7); border-radius: 999px; height: 7px; overflow: hidden; position: relative; }
.meter-fill-h { height: 100%; background: linear-gradient(90deg, #00b84d, #3dff80); border-radius: 999px; position: relative; overflow: hidden; animation: growBar 0.9s cubic-bezier(0.4, 0, 0.2, 1) both; }
.meter-fill-d { height: 100%; background: linear-gradient(90deg, #b86a00, #ffc235); border-radius: 999px; position: relative; overflow: hidden; animation: growBar 0.9s cubic-bezier(0.4, 0, 0.2, 1) both; }
@keyframes growBar { from { width: 0% !important; } }
.meter-fill-h::after, .meter-fill-d::after {
    content: ''; position: absolute; top: 0; left: -100%; width: 55%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.28), transparent);
    animation: shimmer 2.2s ease-in-out 0.9s infinite;
}
@keyframes shimmer { 0% { left: -55%; } 100% { left: 160%; } }

/* Severity badges */
.sev-badge { display: inline-flex; align-items: center; gap: 0.3rem; padding: .28rem 1rem; border-radius: 999px; font-size: .65rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; margin-top: 1rem; font-family: 'Outfit', sans-serif; }
.sev-healthy  { background: rgba(0,120,38,0.18);  color: #3dff80; border: 1px solid rgba(0,200,75,0.28); }
.sev-mild     { background: rgba(0,80,140,0.18);  color: #7ad4f5; border: 1px solid rgba(30,120,180,0.28); }
.sev-moderate { background: rgba(145,88,0,0.18);  color: #ffc235; border: 1px solid rgba(200,130,0,0.28); }
.sev-severe   { background: rgba(145,18,0,0.18);  color: #ff7a5a; border: 1px solid rgba(200,48,18,0.28); }

/* ══ INFO BOXES ══ */
.treatment-box { background: rgba(0, 28, 8, 0.65); border-left: 3px solid #00b84d; border-radius: 12px; padding: 1.1rem 1.35rem; font-size: .85rem; color: #b5d2b5; line-height: 1.88; margin-top: .95rem; backdrop-filter: blur(8px); animation: cardReveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.12s both; }
.box-title { color: rgba(0, 210, 85, 0.82); font-size: .6rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; display: block; margin-bottom: .5rem; }
.meta-grid { display: flex; gap: .7rem; margin-top: .85rem; }
.meta-tile { flex: 1; background: rgba(0, 18, 5, 0.65); border: 1px solid rgba(0, 100, 28, 0.2); border-radius: 12px; padding: .82rem 1rem; animation: cardReveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.22s both; }
.meta-tile .mt-label { font-size: .6rem; color: rgba(100,148,100,0.55); text-transform: uppercase; letter-spacing: .08em; }
.meta-tile .mt-value { font-size: .84rem; color: #c5d8c5; margin-top: .2rem; font-weight: 500; }
.tip-box { background: rgba(0, 18, 5, 0.55); border: 1px solid rgba(0, 100, 28, 0.18); border-radius: 12px; padding: .9rem 1.1rem; font-size: .82rem; color: #88b888; margin-top: .82rem; line-height: 1.78; animation: cardReveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.32s both; }

/* ══ HEALTHY PRECAUTIONS ══ */
.precaution-box { background: rgba(0, 28, 8, 0.65); border: 1px solid rgba(0, 180, 70, 0.2); border-left: 3px solid #00b84d; border-radius: 12px; padding: 1.1rem 1.35rem; margin-top: .95rem; animation: cardReveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.14s both; }
.precaution-item { display: flex; align-items: flex-start; gap: .55rem; font-size: .83rem; color: #b5d2b5; line-height: 1.7; padding: .32rem 0; border-bottom: 1px solid rgba(0,100,28,0.1); }
.precaution-item:last-child { border-bottom: none; }
.farmer-tip-box { background: linear-gradient(135deg, rgba(0,40,12,0.75), rgba(0,28,8,0.65)); border: 1px solid rgba(61,255,128,0.14); border-radius: 12px; padding: 1rem 1.2rem; margin-top: .85rem; animation: cardReveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.28s both; }
.farmer-tip-icon { font-size: 1.4rem; margin-bottom: .4rem; display: block; }
.farmer-tip-text { font-size: .82rem; color: #a0c8a0; line-height: 1.78; }

/* ══ TOP-5 BARS ══ */
.prob-row { margin-bottom: .78rem; animation: cardReveal 0.45s ease-out both; }
.prob-row:nth-child(1) { animation-delay: 0.08s; } .prob-row:nth-child(2) { animation-delay: 0.16s; }
.prob-row:nth-child(3) { animation-delay: 0.24s; } .prob-row:nth-child(4) { animation-delay: 0.32s; }
.prob-row:nth-child(5) { animation-delay: 0.40s; }
.prob-lbl { font-size: .76rem; color: rgba(138,178,138,0.78); margin-bottom: .28rem; display: flex; justify-content: space-between; }
.prob-bg  { background: rgba(0,22,5,0.75); border-radius: 999px; height: 5px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 999px; animation: growBar 0.8s cubic-bezier(0.4,0,0.2,1) both; }

/* ══ TABS ══ */
.stTabs [data-baseweb="tab-list"] { background: rgba(0,12,4,0.85) !important; border-radius: 14px; padding: 5px; gap: 4px; border: 1px solid rgba(0, 100, 28, 0.14) !important; }
.stTabs [data-baseweb="tab"]      { background: transparent !important; color: rgba(100,148,100,0.65) !important; border-radius: 10px !important; font-family: 'Outfit' !important; font-size: .84rem !important; transition: all 0.2s ease !important; }
.stTabs [aria-selected="true"]    { background: rgba(0, 155, 52, 0.14) !important; color: #3dff80 !important; box-shadow: 0 0 22px rgba(0,200,75,0.07) !important; }

/* ══ SCAN HISTORY ══ */
.hist-card { background: rgba(0, 18, 5, 0.65); border: 1px solid rgba(0, 100, 28, 0.17); border-radius: 13px; padding: .9rem 1.1rem; margin-bottom: .5rem; font-size: .82rem; display: flex; justify-content: space-between; align-items: center; transition: border-color 0.25s ease, background 0.25s ease; animation: cardReveal 0.4s ease-out both; }
.hist-card:hover { border-color: rgba(0, 180, 60, 0.28); background: rgba(0, 28, 8, 0.72); }
.hist-name   { color: #c5d8c5; font-weight: 500; }
.hist-meta   { color: rgba(100,138,100,0.55); font-size: .7rem; margin-top: 0.14rem; }
.conf-h      { color: #3dff80; font-weight: 700; font-size: 0.98rem; }
.conf-d      { color: #ffc235; font-weight: 700; font-size: 0.98rem; }
.hist-backend { font-size: 0.6rem; padding: 0.1rem 0.5rem; border-radius: 999px; margin-left: 0.5rem; }
.hist-keras  { color: #7ad4f5; background: rgba(30,120,200,0.12); border: 1px solid rgba(30,120,200,0.2); }
.hist-tflite { color: #a0f070; background: rgba(80,180,0,0.1);   border: 1px solid rgba(80,180,0,0.2); }

/* ══ EXPANDERS ══ */
.streamlit-expanderHeader   { background: rgba(0, 18, 5, 0.65) !important; color: rgba(118,158,118,0.8) !important; border-radius: 11px !important; border: 1px solid rgba(0, 100, 28, 0.17) !important; }
.streamlit-expanderContent  { background: rgba(0, 10, 3, 0.82) !important; border: 1px solid rgba(0, 100, 28, 0.14) !important; border-top: none !important; border-radius: 0 0 11px 11px !important; }

hr { border-color: rgba(0, 100, 28, 0.12) !important; }

/* ══ EMPTY STATE ══ */
.empty-state { text-align: center; padding: 3rem 2rem; animation: cardReveal 0.5s ease-out both; }
.empty-state .icon { font-size: 2.8rem; margin-bottom: 1rem; display: block; animation: leafFloat 3.5s ease-in-out infinite; }
@keyframes leafFloat { 0%, 100% { transform: translateY(0) rotate(-6deg); } 50% { transform: translateY(-9px) rotate(6deg); } }
.empty-state .etitle { font-family: 'Cormorant Garamond', serif; font-size: 1.55rem; font-weight: 600; color: #c5d8c5; margin-bottom: .5rem; }
.empty-state .edesc  { font-size: .84rem; color: rgba(100,140,100,0.55); max-width: 380px; margin: auto; line-height: 1.82; }

/* ══ BUTTONS ══ */
.stButton > button { background: rgba(0, 100, 28, 0.14) !important; border: 1px solid rgba(0, 150, 50, 0.22) !important; color: rgba(0, 220, 90, 0.78) !important; border-radius: 11px !important; font-family: 'Outfit', sans-serif !important; font-weight: 500 !important; transition: all 0.22s ease !important; }
.stButton > button:hover { background: rgba(0, 150, 50, 0.2) !important; border-color: rgba(0, 200, 80, 0.38) !important; color: #3dff80 !important; box-shadow: 0 4px 18px rgba(0, 200, 80, 0.1) !important; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, rgba(0,140,45,0.35), rgba(0,180,60,0.3)) !important; border-color: rgba(0, 200, 80, 0.5) !important; color: #3dff80 !important; font-weight: 600 !important; }

/* ══ SELECT BOX / RADIO ══ */
div[data-testid="stSelectbox"] > div > div { background: rgba(0, 22, 7, 0.8) !important; border: 1px solid rgba(0, 160, 55, 0.2) !important; border-radius: 10px !important; color: #b0ccb0 !important; }
.stRadio > div { gap: 0.5rem !important; }
.stRadio > div > label { background: rgba(0, 22, 7, 0.6) !important; border: 1px solid rgba(0, 140, 48, 0.2) !important; border-radius: 10px !important; padding: 0.5rem 1rem !important; font-size: 0.85rem !important; color: #a0c4a0 !important; transition: all 0.2s ease !important; }
.stRadio > div > label:hover { border-color: rgba(0, 200, 80, 0.35) !important; background: rgba(0, 40, 14, 0.7) !important; }

/* ══ INPUTS ══ */
.stTextInput > div > div > input {
    background: rgba(0, 18, 6, 0.85) !important;
    border: 1px solid rgba(0, 150, 55, 0.22) !important;
    border-radius: 10px !important;
    color: #ddeedd !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(0, 220, 90, 0.48) !important;
    box-shadow: 0 0 0 3px rgba(0, 200, 80, 0.09) !important;
}
.stTextInput label { color: rgba(100, 160, 100, 0.7) !important; font-size: 0.82rem !important; }

/* ══ SPINNER ══ */
.stSpinner > div > div { border-top-color: #00e668 !important; }

/* ══ DOWNLOAD BUTTON ══ */
div[data-testid="stDownloadButton"] > button { background: linear-gradient(135deg, #003d14, #004d1a) !important; border: 1px solid rgba(0, 200, 75, 0.4) !important; color: #3dff80 !important; border-radius: 12px !important; font-family: 'Outfit', sans-serif !important; font-weight: 600 !important; font-size: 0.92rem !important; padding: 0.7rem 1.5rem !important; width: 100% !important; transition: all 0.22s ease !important; }
div[data-testid="stDownloadButton"] > button:hover { background: linear-gradient(135deg, #005020, #006628) !important; border-color: rgba(0, 220, 90, 0.6) !important; box-shadow: 0 6px 24px rgba(0, 200, 75, 0.18) !important; transform: translateY(-1px) !important; }

/* ══ FOOTER ══ */
.leaf-footer { text-align: center; padding: 1.5rem 0 0.5rem; color: rgba(50, 78, 50, 0.45); font-size: 0.7rem; border-top: 1px solid rgba(0, 100, 28, 0.1); margin-top: 1.5rem; letter-spacing: 0.07em; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #050d06; }
::-webkit-scrollbar-thumb { background: rgba(0, 160, 55, 0.22); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0, 210, 80, 0.38); }

/* ══════════════════════════════════════════════════════════
   AUTH & ONBOARDING STYLES
══════════════════════════════════════════════════════════ */

/* Splash screen */
.splash-screen {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: #050d06; z-index: 9999;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    animation: splashFadeIn 0.6s ease-out both;
}
@keyframes splashFadeIn { from { opacity: 0; } to { opacity: 1; } }
.splash-icon {
    font-size: 7rem; line-height: 1;
    filter: drop-shadow(0 0 40px rgba(61,255,128,0.55));
    animation: splashPulse 2s ease-in-out infinite alternate;
}
@keyframes splashPulse {
    from { transform: scale(1);    filter: drop-shadow(0 0 20px rgba(61,255,128,0.35)); }
    to   { transform: scale(1.1); filter: drop-shadow(0 0 55px rgba(61,255,128,0.72)); }
}
.splash-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4rem; font-weight: 600; color: #e4f4e6;
    margin-top: 1.2rem; letter-spacing: -0.02em;
    animation: fadeSlideUp 0.7s ease-out 0.35s both;
}
.splash-name em { color: #3dff80; font-style: italic; }
.splash-tag {
    font-size: 0.74rem; color: rgba(0,200,80,0.55);
    letter-spacing: 0.25em; text-transform: uppercase;
    margin-top: 0.5rem;
    animation: fadeSlideUp 0.7s ease-out 0.55s both;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.splash-dots {
    display: flex; gap: 0.5rem; margin-top: 3rem;
    animation: fadeSlideUp 0.7s ease-out 0.75s both;
}
.splash-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: rgba(61,255,128,0.45);
    animation: dotBounce 1.4s ease-in-out infinite;
}
.splash-dot:nth-child(2) { animation-delay: 0.18s; }
.splash-dot:nth-child(3) { animation-delay: 0.36s; }
@keyframes dotBounce {
    0%, 100% { transform: translateY(0);    opacity: 0.45; }
    50%       { transform: translateY(-12px); opacity: 1; }
}

/* Welcome screen */
.welcome-wrap {
    max-width: 700px; margin: 0 auto; padding: 3.5rem 2rem 2rem;
    animation: fadeSlideUp 0.6s ease-out both;
}
.welcome-eyebrow {
    font-size: 0.64rem; color: rgba(0,210,90,0.6);
    letter-spacing: 0.25em; text-transform: uppercase;
    text-align: center; margin-bottom: 1rem;
}
.welcome-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.4rem; font-weight: 600; color: #e4f4e6;
    line-height: 1.06; text-align: center; letter-spacing: -0.02em;
}
.welcome-title em { color: #3dff80; font-style: italic; }
.welcome-sub {
    font-size: 0.98rem; color: rgba(120,170,120,0.72);
    text-align: center; line-height: 1.82; margin-top: 1.1rem;
    max-width: 520px; margin-left: auto; margin-right: auto;
}
.feature-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 1rem; margin-top: 2.5rem; margin-bottom: 2rem;
}
@media (max-width: 600px) { .feature-grid { grid-template-columns: 1fr; } }
.feature-card {
    background: rgba(0, 20, 7, 0.75); border: 1px solid rgba(0, 155, 55, 0.15);
    border-radius: 18px; padding: 1.5rem 1.1rem; text-align: center;
    transition: border-color 0.3s, transform 0.3s;
}
.feature-card:hover { border-color: rgba(0, 200, 80, 0.3); transform: translateY(-4px); }
.feature-icon  { font-size: 2.2rem; display: block; margin-bottom: 0.7rem; }
.feature-title { font-size: 0.9rem; font-weight: 600; color: #c5d8c5; margin-bottom: 0.38rem; }
.feature-desc  { font-size: 0.76rem; color: rgba(100,140,100,0.55); line-height: 1.64; }

/* Auth card */
.auth-card {
    background: rgba(4, 14, 6, 0.92);
    border: 1px solid rgba(0, 170, 60, 0.18);
    border-radius: 24px; padding: 2.5rem 2rem;
    box-shadow: 0 20px 60px rgba(0,0,0,0.45), 0 0 0 1px rgba(0,200,75,0.04);
    animation: cardReveal 0.5s ease-out both;
}
.auth-logo {
    font-size: 2.8rem; text-align: center; display: block;
    filter: drop-shadow(0 0 18px rgba(61,255,128,0.4));
    animation: logoFloat 3.5s ease-in-out infinite;
    margin-bottom: 0.6rem;
}
@keyframes logoFloat { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-7px); } }
.auth-title { font-family: 'Cormorant Garamond', serif; font-size: 2rem; font-weight: 600; color: #e4f4e6; text-align: center; }
.auth-title em { color: #3dff80; font-style: italic; }
.auth-sub { font-size: 0.78rem; color: rgba(100,148,100,0.58); text-align: center; margin: 0.2rem 0 1.8rem; letter-spacing: 0.04em; }
.auth-divider {
    display: flex; align-items: center; gap: 0.75rem; margin: 0.5rem 0 1rem;
    color: rgba(80,120,80,0.35); font-size: 0.72rem; letter-spacing: 0.08em;
}
.auth-divider::before, .auth-divider::after { content: ''; flex: 1; height: 1px; background: rgba(0,150,50,0.12); }

/* User badge in sidebar */
.user-badge {
    display: flex; align-items: center; gap: 0.6rem;
    background: rgba(0, 180, 65, 0.07); border: 1px solid rgba(0, 170, 58, 0.18);
    border-radius: 12px; padding: 0.65rem 0.9rem; margin-bottom: 0.5rem;
}
.user-avatar { font-size: 1.4rem; }
.user-info-name { font-size: 0.82rem; font-weight: 600; color: rgba(200, 230, 200, 0.88) !important; }
.user-info-phone { font-size: 0.65rem; color: rgba(100, 148, 100, 0.5) !important; letter-spacing: 0.04em; }

/* ══ NAVBAR LOGOUT BUTTON ══ */
div[data-testid="column"]:has(button[kind="secondary"][data-testid="baseButton-secondary"]) button {
    background: rgba(255, 60, 60, 0.06) !important;
    border: 1px solid rgba(255, 80, 80, 0.2) !important;
    color: rgba(255, 130, 110, 0.75) !important;
    border-radius: 999px !important;
    font-size: 0.7rem !important;
    padding: 0.3rem 0.8rem !important;
    font-weight: 500 !important;
    margin-top: -3.6rem !important;
}
div[data-testid="column"]:has(button[kind="secondary"]) button:hover {
    background: rgba(255, 80, 80, 0.14) !important;
    border-color: rgba(255, 80, 80, 0.38) !important;
    color: #ff7a5a !important;
    box-shadow: 0 0 14px rgba(255,80,80,0.1) !important;
}

/* ══ TOP NAVBAR ══ */
.ls-navbar {
    position: relative;
    display: flex; align-items: center; justify-content: space-between;
    background: rgba(3, 10, 4, 0.92);
    border: 1px solid rgba(0, 180, 70, 0.14);
    border-radius: 16px;
    padding: 0.65rem 1.4rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(18px);
    overflow: hidden;
    animation: navbarReveal 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
}
@keyframes navbarReveal {
    from { opacity: 0; transform: translateY(-10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.ls-navbar::before {
    content: '';
    position: absolute; top: 0; left: -100%; width: 60%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(61,255,128,0.5), transparent);
    animation: navScan 4s ease-in-out infinite;
}
@keyframes navScan {
    0%   { left: -60%; }
    100% { left: 160%; }
}
.ls-navbar-brand {
    display: flex; align-items: center; gap: 0.5rem;
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.25rem; font-weight: 600; color: #3dff80;
    white-space: nowrap;
}
.ls-navbar-brand span { font-size: 1.4rem; }
.ls-navbar-center {
    display: flex; align-items: center; gap: 1.2rem; flex-wrap: wrap;
}
.ls-nav-welcome {
    display: flex; align-items: center; gap: 0.45rem;
}
.ls-nav-avatar {
    width: 30px; height: 30px; border-radius: 50%;
    background: linear-gradient(135deg, rgba(0,160,55,0.25), rgba(0,220,90,0.15));
    border: 1px solid rgba(0, 200, 75, 0.3);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem;
    animation: avatarPulse 3s ease-in-out infinite;
}
@keyframes avatarPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(61,255,128,0); }
    50%       { box-shadow: 0 0 0 4px rgba(61,255,128,0.08); }
}
.ls-nav-user-text { display: flex; flex-direction: column; }
.ls-nav-greeting  { font-size: 0.58rem; color: rgba(100,148,100,0.5); letter-spacing: 0.08em; text-transform: uppercase; }
.ls-nav-name      { font-size: 0.82rem; font-weight: 600; color: rgba(210,235,210,0.9); line-height: 1.1; }
.ls-nav-phone     {
    display: flex; align-items: center; gap: 0.3rem;
    font-size: 0.7rem; color: rgba(100,148,100,0.6);
    background: rgba(0,100,28,0.1); border: 1px solid rgba(0,150,50,0.15);
    border-radius: 999px; padding: 0.18rem 0.65rem;
}
.ls-nav-scan-badge {
    display: flex; align-items: center; gap: 0.35rem;
    background: rgba(0, 180, 65, 0.08); border: 1px solid rgba(0, 200, 75, 0.18);
    border-radius: 999px; padding: 0.28rem 0.75rem;
    font-size: 0.7rem; color: rgba(0, 220, 90, 0.75);
    cursor: default;
    transition: background 0.2s, border-color 0.2s;
}
.ls-nav-scan-badge:hover {
    background: rgba(0, 200, 75, 0.14); border-color: rgba(0, 220, 90, 0.3);
}
.ls-nav-scan-count {
    font-weight: 700; color: #3dff80;
    font-size: 0.82rem;
    animation: countPop 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}
@keyframes countPop {
    from { transform: scale(0.6); opacity: 0; }
    to   { transform: scale(1);   opacity: 1; }
}
.ls-nav-logout-btn {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: rgba(255, 90, 90, 0.06); border: 1px solid rgba(255, 80, 80, 0.18);
    color: rgba(255, 130, 110, 0.72); border-radius: 999px;
    padding: 0.28rem 0.85rem; font-size: 0.7rem; font-weight: 500;
    cursor: pointer; transition: all 0.2s ease; text-decoration: none;
    font-family: 'Outfit', sans-serif;
}
.ls-nav-logout-btn:hover {
    background: rgba(255, 80, 80, 0.12); border-color: rgba(255, 80, 80, 0.35);
    color: #ff7a5a; box-shadow: 0 0 12px rgba(255,80,80,0.08);
}
.ls-nav-guest {
    font-size: 0.72rem; color: rgba(100,148,100,0.5); font-style: italic;
}

/* ══ ANIMATED FOOTER ══ */
.ls-footer-wrap {
    position: relative;
    margin-top: 3.5rem;
    overflow: hidden;
    border-radius: 18px 18px 0 0;
}
.ls-footer-glow {
    position: absolute; bottom: 0; left: 50%; transform: translateX(-50%);
    width: 70%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(61,255,128,0.4), transparent);
    animation: footerGlowPulse 3s ease-in-out infinite;
}
@keyframes footerGlowPulse {
    0%, 100% { opacity: 0.4; width: 50%; }
    50%       { opacity: 1;   width: 85%; }
}
.ls-footer {
    background: linear-gradient(180deg, rgba(3, 10, 4, 0.0) 0%, rgba(3, 10, 4, 0.94) 30%);
    border-top: 1px solid rgba(0, 180, 70, 0.1);
    padding: 2.5rem 2rem 1.8rem;
    text-align: center;
}
.ls-footer-logo {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.9rem; font-weight: 600; color: #3dff80;
    letter-spacing: -0.01em;
    animation: logoFloat 3.5s ease-in-out infinite;
    display: inline-block; margin-bottom: 0.5rem;
}
.ls-footer-logo span { font-style: italic; }
.ls-footer-tagline {
    font-size: 0.7rem; color: rgba(100,148,100,0.45);
    letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 1.5rem;
}
.ls-footer-pills {
    display: flex; flex-wrap: wrap; gap: 0.5rem;
    justify-content: center; margin-bottom: 1.5rem;
}
.ls-footer-pill {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: rgba(0,100,28,0.08); border: 1px solid rgba(0,160,55,0.14);
    color: rgba(0, 200, 80, 0.55); font-size: 0.63rem; font-weight: 500;
    letter-spacing: 0.06em; padding: 0.22rem 0.75rem; border-radius: 999px;
    transition: border-color 0.3s, color 0.3s;
}
.ls-footer-pill:hover { border-color: rgba(0,220,90,0.3); color: rgba(0,230,95,0.8); }
.ls-footer-pills .ls-footer-pill:nth-child(1) { animation: pillGlow 3.5s 0.0s ease-in-out infinite; }
.ls-footer-pills .ls-footer-pill:nth-child(2) { animation: pillGlow 3.5s 0.5s ease-in-out infinite; }
.ls-footer-pills .ls-footer-pill:nth-child(3) { animation: pillGlow 3.5s 1.0s ease-in-out infinite; }
.ls-footer-pills .ls-footer-pill:nth-child(4) { animation: pillGlow 3.5s 1.5s ease-in-out infinite; }
.ls-footer-pills .ls-footer-pill:nth-child(5) { animation: pillGlow 3.5s 2.0s ease-in-out infinite; }
.ls-footer-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,160,55,0.1), transparent);
    margin: 0 auto 1.2rem; max-width: 500px;
}
.ls-footer-meta {
    font-size: 0.64rem; color: rgba(80,120,80,0.4);
    letter-spacing: 0.06em; line-height: 1.9;
}
.ls-footer-meta strong { color: rgba(100,160,100,0.55); }
.ls-footer-leaf {
    font-size: 1.1rem; display: inline-block;
    animation: leafFloat 4s ease-in-out infinite;
    margin: 0.6rem 0 0;
}
.ls-footer-dots {
    display: flex; gap: 0.4rem; justify-content: center; margin-top: 1.2rem;
}
.ls-footer-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: rgba(61,255,128,0.25);
    animation: dotBounce 1.6s ease-in-out infinite;
}
.ls-footer-dot:nth-child(2) { animation-delay: 0.2s; }
.ls-footer-dot:nth-child(3) { animation-delay: 0.4s; }
.ls-footer-dot:nth-child(4) { animation-delay: 0.6s; }
.ls-footer-dot:nth-child(5) { animation-delay: 0.8s; }

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def show_splash():
    """Full-page animated splash — displays for ~2 seconds then advances."""
    st.markdown("""
    <div class="splash-screen">
        <div class="splash-icon">🌿</div>
        <div class="splash-name">Leaf<em>Scan</em> AI</div>
        <div class="splash-tag">AI · Plant Disease Detection</div>
        <div class="splash-dots">
            <div class="splash-dot"></div>
            <div class="splash-dot"></div>
            <div class="splash-dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    time.sleep(2)
    # Mark device as visited so we don't show the full splash again
    _set_cookie(VISITED_COOKIE, "1", days=9999)
    st.session_state.stage = "welcome"
    st.rerun()


def show_welcome():
    """Animated welcome / onboarding screen for first-time users."""
    _, col, _ = st.columns([1, 2.4, 1])
    with col:
        st.markdown("""
        <div class="welcome-wrap">
            <div class="welcome-eyebrow">🌿 Welcome to LeafScan AI</div>
            <div class="welcome-title">
                Diagnose your plants<br>with <em>AI precision</em>
            </div>
            <div class="welcome-sub">
                Take a photo of any plant leaf and get an instant AI-powered disease diagnosis —
                along with treatment recommendations, severity ratings, and a downloadable PDF report.
                Powered by EfficientNetV2B3 trained on 38 disease classes.
            </div>
            <div class="feature-grid">
                <div class="feature-card">
                    <span class="feature-icon">🔬</span>
                    <div class="feature-title">Instant Diagnosis</div>
                    <div class="feature-desc">Upload any leaf photo and get AI analysis in seconds — no expertise needed.</div>
                </div>
                <div class="feature-card">
                    <span class="feature-icon">💊</span>
                    <div class="feature-title">Treatment Plans</div>
                    <div class="feature-desc">Detailed, actionable treatment steps for every disease detected by the model.</div>
                </div>
                <div class="feature-card">
                    <span class="feature-icon">📄</span>
                    <div class="feature-title">PDF Reports</div>
                    <div class="feature-desc">Download a full diagnosis report with top-5 predictions and confidence scores.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🌿 Get Started →", use_container_width=True, type="primary"):
            st.session_state.stage = "login"
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)


def show_auth():
    """Login / Register / Forgot-Password screen."""
    # Centre the card
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.markdown("""
        <div class="auth-card">
            <span class="auth-logo">🌿</span>
            <div class="auth-title">Leaf<em>Scan</em> AI</div>
            <div class="auth-sub">AI · Plant Disease Detection</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Mode selector ──────────────────────────────────────────────────────
        mode = st.session_state.auth_mode
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🔐 Login",    use_container_width=True,
                         type="primary" if mode == "login"    else "secondary"):
                st.session_state.auth_mode = "login";    st.rerun()
        with c2:
            if st.button("✨ Register", use_container_width=True,
                         type="primary" if mode == "register" else "secondary"):
                st.session_state.auth_mode = "register"; st.rerun()
        with c3:
            if st.button("🔑 Forgot",   use_container_width=True,
                         type="primary" if mode == "forgot"   else "secondary"):
                st.session_state.auth_mode = "forgot";   st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── LOGIN ──────────────────────────────────────────────────────────────
        if mode == "login":
            with st.form("login_form", clear_on_submit=False):
                phone_in = st.text_input("📱 Phone Number",
                                         placeholder="e.g. 9876543210")
                pass_in  = st.text_input("🔒 Password",
                                         type="password",
                                         placeholder="Your password")
                submitted = st.form_submit_button("Login →", use_container_width=True)

            if submitted:
                phone_clean = phone_in.strip().replace(" ", "").replace("+91", "").replace("-", "")
                if not phone_clean or not pass_in:
                    st.error("⚠️ Please fill in both fields.")
                else:
                    user = login_user(phone_clean, pass_in)
                    if user:
                        token = make_session(user["id"])
                        st.session_state.user  = user
                        st.session_state.token = token
                        _set_cookie(SESSION_COOKIE, token)
                        st.session_state.stage = "app"
                        st.rerun()
                    else:
                        if phone_exists(phone_clean):
                            st.error("❌ Incorrect password. Please try again.")
                        else:
                            st.error("❌ No account found with this number.")
                            st.info("💡 New here? Switch to **Register** to create an account.")

        # ── REGISTER ──────────────────────────────────────────────────────────
        elif mode == "register":
            with st.form("register_form", clear_on_submit=False):
                name_in   = st.text_input("👤 Full Name",        placeholder="Your name")
                phone_in  = st.text_input("📱 Phone Number",     placeholder="10-digit mobile number")
                pass_in   = st.text_input("🔒 Password",         type="password", placeholder="Min 6 characters")
                cpass_in  = st.text_input("🔒 Confirm Password", type="password", placeholder="Repeat your password")
                submitted = st.form_submit_button("Create Account →", use_container_width=True)

            if submitted:
                phone_clean = phone_in.strip().replace(" ", "").replace("+91", "").replace("-", "")
                errors = []
                if not name_in.strip():
                    errors.append("Name is required.")
                if not phone_clean or len(phone_clean) < 8 or not phone_clean.isdigit():
                    errors.append("Enter a valid phone number (digits only).")
                if len(pass_in) < 6:
                    errors.append("Password must be at least 6 characters.")
                if pass_in != cpass_in:
                    errors.append("Passwords do not match.")

                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    ok, result = register_user(name_in.strip(), phone_clean, pass_in)
                    if ok:
                        token = make_session(result)
                        user  = {"id": result, "name": name_in.strip(), "phone": phone_clean}
                        st.session_state.user  = user
                        st.session_state.token = token
                        _set_cookie(SESSION_COOKIE, token)
                        st.session_state.stage = "app"
                        st.success(f"✅ Welcome, {name_in.strip()}! Account created.")
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error("📱 This phone number is already registered.")
                        st.info("💡 Already have an account? Use **Login** — or use **Forgot Password** if you can't remember it.")

        # ── FORGOT PASSWORD ────────────────────────────────────────────────────
        else:
            st.markdown("""
            <div style="font-size:0.82rem;color:rgba(120,160,120,0.65);
                        margin-bottom:0.8rem;line-height:1.65;text-align:center;">
                Enter your registered phone number and set a new password.
            </div>
            """, unsafe_allow_html=True)

            with st.form("forgot_form", clear_on_submit=False):
                phone_in  = st.text_input("📱 Registered Phone Number", placeholder="Your phone number")
                new_pw    = st.text_input("🔒 New Password",             type="password", placeholder="Min 6 characters")
                cnew_pw   = st.text_input("🔒 Confirm New Password",     type="password", placeholder="Repeat new password")
                submitted = st.form_submit_button("Reset Password →", use_container_width=True)

            if submitted:
                phone_clean = phone_in.strip().replace(" ", "").replace("+91", "").replace("-", "")
                if not phone_clean:
                    st.error("Enter your phone number.")
                elif len(new_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                elif new_pw != cnew_pw:
                    st.error("Passwords do not match.")
                else:
                    if reset_password(phone_clean, new_pw):
                        st.success("✅ Password reset successfully! Please log in.")
                        time.sleep(1.5)
                        st.session_state.auth_mode = "login"
                        st.rerun()
                    else:
                        st.error("❌ No account found with this number.")
                        st.info("💡 New to LeafScan? Switch to **Register** to create a free account.")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER — show the right screen and stop execution
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.stage == "splash":
    show_splash()
    st.stop()

if st.session_state.stage == "welcome":
    show_welcome()
    st.stop()

if st.session_state.stage == "login":
    show_auth()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# ▼▼▼  EVERYTHING BELOW RUNS ONLY WHEN stage == "app"  ▼▼▼
# ══════════════════════════════════════════════════════════════════════════════

keras_exists  = os.path.exists(KERAS_PATH)
tflite_exists = os.path.exists(TFLITE_PATH)

# ══════════════════════════════════════════════════════════════════════════════
# DISEASE DATABASE
# ══════════════════════════════════════════════════════════════════════════════
DISEASE_DB = {
    "Apple___Apple_scab": {
        "display": "Apple — Apple Scab",
        "treatment": (
            "1. Apply <strong>captan (0.2%)</strong> or <strong>myclobutanil</strong> fungicide every 7–10 days from bud break.<br>"
            "2. Remove and destroy all fallen infected leaves — spores overwinter in leaf litter.<br>"
            "3. Avoid overhead irrigation; use drip irrigation to keep foliage dry.<br>"
            "4. Prune for open canopy air circulation. Apply lime sulfur during dormancy."
        ),
        "severity": "moderate", "spread": "Rain-splash & wind-borne spores", "season": "Spring – Early Summer"
    },
    "Apple___Black_rot": {
        "display": "Apple — Black Rot",
        "treatment": (
            "1. Prune all infected wood <strong>15 cm below visible cankers</strong>; sterilise tools with 10% bleach between cuts.<br>"
            "2. Apply <strong>copper-based fungicide + mancozeb</strong> every 10 days during wet periods.<br>"
            "3. Remove and destroy all mummified fruits — primary inoculum source.<br>"
            "4. Improve tree vigor with balanced fertilisation; stressed trees are far more susceptible."
        ),
        "severity": "severe", "spread": "Rain-splash & insect vectors", "season": "Spring – Fall"
    },
    "Apple___Cedar_apple_rust": {
        "display": "Apple — Cedar Apple Rust",
        "treatment": (
            "1. Apply <strong>myclobutanil or propiconazole</strong> at bud break; repeat every 7 days for 3 applications.<br>"
            "2. Remove nearby juniper/cedar host trees within 300 m — they are required for the disease cycle.<br>"
            "3. Plant resistant apple varieties (Liberty, Redfree, Enterprise) for future seasons.<br>"
            "4. Apply copper spray during dormancy as preventive measure."
        ),
        "severity": "moderate", "spread": "Wind-borne spores from cedar/juniper trees", "season": "Spring"
    },
    "Apple___healthy":                    {"display": "Apple — Healthy",       "treatment": "", "severity": "healthy", "spread": "—", "season": "—"},
    "Blueberry___healthy":                {"display": "Blueberry — Healthy",   "treatment": "", "severity": "healthy", "spread": "—", "season": "—"},
    "Cherry_(including_sour)___Powdery_mildew": {
        "display": "Cherry — Powdery Mildew",
        "treatment": (
            "1. Apply <strong>sulfur (0.3%)</strong> or <strong>potassium bicarbonate</strong> spray weekly from first sign.<br>"
            "2. Remove and destroy all infected shoots — do not compost.<br>"
            "3. Improve air circulation through selective pruning of crossing branches.<br>"
            "4. Avoid excess nitrogen fertilisation which promotes soft, susceptible growth.<br>"
            "5. Apply <strong>neem oil (2%)</strong> as an organic alternative."
        ),
        "severity": "mild", "spread": "Wind-borne conidia", "season": "Summer – Fall"
    },
    "Cherry_(including_sour)___healthy":  {"display": "Cherry — Healthy",      "treatment": "", "severity": "healthy", "spread": "—", "season": "—"},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "display": "Corn — Gray Leaf Spot",
        "treatment": (
            "1. Apply <strong>strobilurin (azoxystrobin)</strong> or <strong>triazole (propiconazole)</strong> fungicide at VT/R1 stage.<br>"
            "2. Plant certified resistant hybrids — check GLS resistance ratings in seed catalogs.<br>"
            "3. Rotate crops strictly — <strong>never plant corn after corn</strong> in the same field.<br>"
            "4. Reduce tillage to decrease infected surface debris that harbours overwintering spores.<br>"
            "5. Scout from V6 stage; spray if 50% of plants show lesions below ear leaf."
        ),
        "severity": "moderate", "spread": "Wind & rain splash from infected residue", "season": "Mid-Summer – Harvest"
    },
    "Corn_(maize)___Common_rust_": {
        "display": "Corn — Common Rust",
        "treatment": (
            "1. Apply <strong>propiconazole or azoxystrobin</strong> at first rust appearance — timing is critical.<br>"
            "2. Most effective before R2 (blister) grain fill stage — do not delay treatment.<br>"
            "3. Plant rust-resistant hybrids next season; check seed catalog resistance ratings.<br>"
            "4. Scout weekly after tasseling; economic threshold is 5% leaf area affected on upper leaves."
        ),
        "severity": "moderate", "spread": "Wind-borne urediniospores from southern regions", "season": "Summer – Late Summer"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "display": "Corn — Northern Leaf Blight",
        "treatment": (
            "1. Apply <strong>strobilurin or triazole fungicide</strong> at tasseling (VT) if disease is present before silking.<br>"
            "2. Plant resistant hybrids with <strong>Ht1/Ht2/Htm1 resistance genes</strong>.<br>"
            "3. Rotate with soybean or wheat for at least 1 year to break the disease cycle.<br>"
            "4. Destroy infected crop residue after harvest — disk or plow under to accelerate decomposition."
        ),
        "severity": "severe", "spread": "Wind & rain from infected crop debris", "season": "Late Summer – Fall"
    },
    "Corn_(maize)___healthy":             {"display": "Corn — Healthy",        "treatment": "", "severity": "healthy", "spread": "—", "season": "—"},
    "Grape___Black_rot": {
        "display": "Grape — Black Rot",
        "treatment": (
            "1. Apply <strong>mancozeb or myclobutanil</strong> from bud break, every 10–14 days through early fruit development.<br>"
            "2. Remove and destroy ALL mummified berries and infected canes — primary overwintering source.<br>"
            "3. Improve trellis training for airflow; thin shoots to reduce canopy density.<br>"
            "4. Apply <strong>captan</strong> after wet periods as curative spray (within 72 hours of infection event)."
        ),
        "severity": "severe", "spread": "Rain-splash from mummified fruit", "season": "Spring – Summer"
    },
    "Grape___Esca_(Black_Measles)": {
        "display": "Grape — Black Measles (Esca)",
        "treatment": (
            "1. No chemical cure available for established Esca infections.<br>"
            "2. Apply <strong>Trichoderma-based biological wound sealant</strong> immediately after every pruning cut.<br>"
            "3. Remove and burn severely infected vines — do not leave in vineyard.<br>"
            "4. Delay pruning until <strong>late dormancy</strong> to minimise the infection window.<br>"
            "5. Never prune during wet or rainy weather — wounds are entry points."
        ),
        "severity": "severe", "spread": "Fungal infection through pruning wounds", "season": "Dormant pruning period"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "display": "Grape — Leaf Blight",
        "treatment": (
            "1. Apply <strong>copper oxychloride (0.3%)</strong> or mancozeb every 2 weeks from first symptom.<br>"
            "2. Remove infected leaves and improve canopy air circulation through shoot thinning.<br>"
            "3. Avoid overhead irrigation — switch to drip at vine base.<br>"
            "4. Apply protective fungicide spray before forecast rain events."
        ),
        "severity": "moderate", "spread": "Rain & wind", "season": "Late Summer"
    },
    "Grape___healthy":                    {"display": "Grape — Healthy",       "treatment": "", "severity": "healthy", "spread": "—", "season": "—"},
    "Orange___Haunglongbing_(Citrus_greening)": {
        "display": "Orange — Citrus Greening (HLB)",
        "treatment": (
            "⚠️ <strong>No cure exists. Notifiable disease in many countries.</strong><br>"
            "1. <strong>Remove and destroy infected trees immediately</strong> — burning is preferred over burial.<br>"
            "2. Control Asian citrus psyllid with <strong>imidacloprid soil drenches</strong> or foliar sprays.<br>"
            "3. Plant only <strong>certified disease-free nursery stock</strong> from licensed nurseries.<br>"
            "4. Install reflective mulch to deter psyllid landing. Report to agricultural authority."
        ),
        "severity": "severe", "spread": "Asian citrus psyllid (Diaphorina citri)", "season": "Year-round"
    },
    "Peach___Bacterial_spot": {
        "display": "Peach — Bacterial Spot",
        "treatment": (
            "1. Apply <strong>copper hydroxide + mancozeb tank mix</strong> from petal fall every 5–7 days during wet weather.<br>"
            "2. Avoid overhead irrigation — use drip or micro-sprinklers at the base.<br>"
            "3. Plant resistant varieties: <strong>Contender, Redhaven, Reliance</strong>.<br>"
            "4. Remove heavily infected shoots with sterilised pruners.<br>"
            "5. Apply <strong>oxytetracycline antibiotic spray</strong> during bloom for severe infections (where legally permitted)."
        ),
        "severity": "moderate", "spread": "Rain-splash & wind during wet periods", "season": "Spring – Summer"
    },
    "Peach___healthy":                    {"display": "Peach — Healthy",       "treatment": "", "severity": "healthy", "spread": "—", "season": "—"},
    "Pepper,_bell___Bacterial_spot": {
        "display": "Bell Pepper — Bacterial Spot",
        "treatment": (
            "1. Apply <strong>copper hydroxide (0.3%)</strong> spray every 5–7 days from first symptom.<br>"
            "2. Use only <strong>certified disease-free seeds</strong> and transplants from trusted nurseries.<br>"
            "3. Avoid working in the field when plants are wet — spreads easily on hands and tools.<br>"
            "4. Remove and destroy all infected plant debris at end of season.<br>"
            "5. Rotate crops for <strong>at least 2 years</strong> — do not replant peppers or tomatoes in same bed."
        ),
        "severity": "moderate", "spread": "Rain-splash, contaminated tools & infected seeds", "season": "Warm wet periods"
    },
    "Pepper,_bell___healthy":             {"display": "Bell Pepper — Healthy", "treatment": "", "severity": "healthy", "spread": "—", "season": "—"},
    "Potato___Early_blight": {
        "display": "Potato — Early Blight",
        "treatment": (
            "1. Apply <strong>chlorothalonil or mancozeb</strong> every 7–10 days from first lesion appearance.<br>"
            "2. Remove lower infected leaves immediately — do not leave on soil surface.<br>"
            "3. Mulch soil to reduce rain-splash dispersal of soil-borne spores.<br>"
            "4. Ensure adequate <strong>potassium nutrition</strong> — K deficiency dramatically increases susceptibility.<br>"
            "5. Rotate crops every 3 years. Use certified disease-free seed potatoes."
        ),
        "severity": "mild", "spread": "Rain-splash & wind from infected debris", "season": "Summer – Fall"
    },
    "Potato___Late_blight": {
        "display": "Potato — Late Blight",
        "treatment": (
            "🚨 <strong>URGENT — Late Blight destroys entire crops within days. Act immediately.</strong><br>"
            "1. Apply <strong>fluopicolide or mandipropamid</strong> fungicide immediately — do not wait.<br>"
            "2. Remove and <strong>bag all infected foliage</strong> — never compost; burn or bury deep.<br>"
            "3. Destroy all nearby cull piles and volunteer potato plants in the area.<br>"
            "4. Harvest early if infection is widespread to salvage tubers before they rot.<br>"
            "5. <strong>Never save infected tubers for seed</strong> — infection passes directly to next season."
        ),
        "severity": "severe", "spread": "Wind-borne sporangia in cool wet weather (15–20°C)", "season": "Cool wet periods"
    },
    "Potato___healthy":                   {"display": "Potato — Healthy",      "treatment": "", "severity": "healthy", "spread": "—", "season": "—"},
    "Raspberry___healthy":                {"display": "Raspberry — Healthy",   "treatment": "", "severity": "healthy", "spread": "—", "season": "—"},
    "Soybean___healthy":                  {"display": "Soybean — Healthy",     "treatment": "", "severity": "healthy", "spread": "—", "season": "—"},
    "Squash___Powdery_mildew": {
        "display": "Squash — Powdery Mildew",
        "treatment": (
            "1. Apply <strong>potassium bicarbonate or neem oil (2%)</strong> spray every 7 days.<br>"
            "2. Remove heavily infected leaves — do not compost.<br>"
            "3. Ensure plants receive <strong>6+ hours of direct sunlight</strong> — shade promotes this disease.<br>"
            "4. Avoid overhead watering; water at the base in the morning only.<br>"
            "5. Plant resistant varieties: <strong>Butternut, Waltham</strong> squash next season."
        ),
        "severity": "mild", "spread": "Wind-borne spores", "season": "Summer – Fall"
    },
    "Strawberry___Leaf_scorch": {
        "display": "Strawberry — Leaf Scorch",
        "treatment": (
            "1. Apply <strong>captan or thiram</strong> fungicide at first sign of leaf scorch.<br>"
            "2. Remove and destroy all infected leaves — do not leave debris in the bed.<br>"
            "3. Improve drainage — disease thrives in waterlogged conditions.<br>"
            "4. Renovate planting after harvest by mowing and thinning crowns.<br>"
            "5. Replace plants every <strong>3 years</strong> — older beds accumulate more disease pressure."
        ),
        "severity": "mild", "spread": "Rain-splash from infected plant debris", "season": "Spring – Early Summer"
    },
    "Strawberry___healthy":               {"display": "Strawberry — Healthy",  "treatment": "", "severity": "healthy", "spread": "—", "season": "—"},
    "Tomato___Bacterial_spot": {
        "display": "Tomato — Bacterial Spot",
        "treatment": (
            "1. Apply <strong>copper hydroxide + mancozeb tank mix</strong> every 5–7 days in wet weather.<br>"
            "2. Use only <strong>certified disease-free transplants</strong> from reputable nurseries.<br>"
            "3. Avoid overhead irrigation — switch to drip irrigation at base of plant.<br>"
            "4. Remove and destroy infected plant material; sterilise all garden tools.<br>"
            "5. Rotate crops for <strong>2+ years</strong>. Do not replant tomatoes or peppers in same location."
        ),
        "severity": "moderate", "spread": "Rain-splash & contaminated tools/hands", "season": "Warm wet periods"
    },
    "Tomato___Early_blight": {
        "display": "Tomato — Early Blight",
        "treatment": (
            "1. <strong>Remove infected lower leaves immediately</strong> — disease progresses upward through the plant.<br>"
            "2. Apply <strong>chlorothalonil or mancozeb</strong> every 7 days during humid weather.<br>"
            "3. Mulch heavily around base to prevent soil-splash dispersal of spores.<br>"
            "4. Ensure adequate calcium and potassium nutrition — deficiency worsens susceptibility.<br>"
            "5. Stake and prune for maximum airflow; never leave debris in the bed at end of season."
        ),
        "severity": "moderate", "spread": "Rain-splash & wind from infected debris", "season": "Mid-Summer – Fall"
    },
    "Tomato___Late_blight": {
        "display": "Tomato — Late Blight",
        "treatment": (
            "🚨 <strong>URGENT — Late Blight can destroy entire crops rapidly. Act immediately.</strong><br>"
            "1. Apply <strong>chlorothalonil or mancozeb</strong> immediately — timing is critical.<br>"
            "2. Remove all infected leaves and stems — bag immediately; never compost.<br>"
            "3. Destroy all infected plant material by burning or deep burial.<br>"
            "4. Avoid overhead irrigation; improve air circulation through staking and pruning.<br>"
            "5. <strong>Never save seed</strong> from infected fruit — infection can pass to next season."
        ),
        "severity": "severe", "spread": "Wind-borne sporangia in cool wet weather", "season": "Cool wet periods"
    },
    "Tomato___Leaf_Mold": {
        "display": "Tomato — Leaf Mold",
        "treatment": (
            "1. Reduce humidity below 85% — improve greenhouse/tunnel ventilation immediately.<br>"
            "2. Apply <strong>chlorothalonil or mancozeb</strong> at first sign of pale spots on upper leaves.<br>"
            "3. Remove and destroy all infected leaves — do not compost.<br>"
            "4. Avoid wetting foliage during irrigation; water at soil level in the morning.<br>"
            "5. Plant resistant varieties: <strong>Resistant Lola, Cobra F1</strong> for next season."
        ),
        "severity": "moderate", "spread": "Airborne spores in high-humidity conditions", "season": "Summer (greenhouse)"
    },
    "Tomato___Septoria_leaf_spot": {
        "display": "Tomato — Septoria Leaf Spot",
        "treatment": (
            "1. Apply <strong>chlorothalonil (0.2%)</strong> every 7–10 days from first symptom appearance.<br>"
            "2. Remove and destroy all infected lower leaves — do not touch healthy foliage afterward.<br>"
            "3. Mulch soil surface to prevent rain-splash of soil-borne spores onto lower leaves.<br>"
            "4. Stake plants and remove suckers to maximise airflow through the canopy.<br>"
            "5. Practice 3-year crop rotation. Remove all tomato debris at end of season."
        ),
        "severity": "moderate", "spread": "Rain-splash from infected soil & debris", "season": "Mid-Summer – Fall"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "display": "Tomato — Spider Mites",
        "treatment": (
            "1. Apply <strong>abamectin or bifenazate</strong> miticide — alternate chemical classes to prevent resistance.<br>"
            "2. Spray <strong>neem oil (2%) + insecticidal soap</strong> as a less toxic first-line treatment.<br>"
            "3. Increase air humidity — spider mites thrive in hot, dry conditions above 30°C.<br>"
            "4. Introduce predatory mite <strong>Phytoseiulus persimilis</strong> for biological control in greenhouses.<br>"
            "5. Remove and destroy heavily infested leaves; avoid water stress which attracts mites."
        ),
        "severity": "moderate", "spread": "Wind, clothing, infested transplants", "season": "Hot dry periods"
    },
    "Tomato___Target_Spot": {
        "display": "Tomato — Target Spot",
        "treatment": (
            "1. Apply <strong>chlorothalonil or azoxystrobin</strong> every 7–14 days from first symptom.<br>"
            "2. Improve air circulation — prune suckers and stake plants for an open canopy.<br>"
            "3. Avoid overhead irrigation; use drip irrigation at base of plant.<br>"
            "4. Remove and destroy all fallen infected leaf debris from soil surface.<br>"
            "5. Rotate away from tomato/pepper/potato for at least 2 years."
        ),
        "severity": "moderate", "spread": "Wind-borne spores from infected crop debris", "season": "Warm humid periods"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "display": "Tomato — Yellow Leaf Curl Virus",
        "treatment": (
            "⚠️ <strong>No cure — management is prevention-focused.</strong><br>"
            "1. Control whitefly vectors with <strong>imidacloprid soil drench</strong> at transplanting.<br>"
            "2. Apply <strong>reflective silver mulch</strong> — disorientstes whiteflies from landing on plants.<br>"
            "3. Install <strong>50-mesh insect-proof netting</strong> over seedling beds and nurseries.<br>"
            "4. Remove and destroy infected plants immediately — they are inoculum sources.<br>"
            "5. Plant resistant/tolerant varieties: <strong>TY Zak F1, Scarlet F1</strong> in future seasons."
        ),
        "severity": "severe", "spread": "Bemisia tabaci whitefly", "season": "Year-round (warm climates)"
    },
    "Tomato___Tomato_mosaic_virus": {
        "display": "Tomato — Mosaic Virus",
        "treatment": (
            "⚠️ <strong>No chemical cure — prevention and sanitation are critical.</strong><br>"
            "1. <strong>Remove infected plants immediately</strong> — bag them; do not compost or leave in field.<br>"
            "2. Wash hands thoroughly with soap for 20 seconds before handling plants — ToMV is highly contagious.<br>"
            "3. Disinfect all tools with 10% bleach solution or 70% isopropyl alcohol between plants.<br>"
            "4. Control aphid and thrips vectors with <strong>imidacloprid or spinosad</strong>.<br>"
            "5. Plant only <strong>certified virus-free seed</strong> or TMV-resistant varieties next season."
        ),
        "severity": "severe", "spread": "Mechanical contact, contaminated tools & seed", "season": "Year-round"
    },
    "Tomato___healthy": {"display": "Tomato — Healthy", "treatment": "", "severity": "healthy", "spread": "—", "season": "—"},
}

def get_disease_info(label: str) -> dict:
    if label in DISEASE_DB:
        return DISEASE_DB[label]
    display = label.replace("_", " ").replace("___", " — ")
    return {
        "display": display,
        "treatment": (
            "No specific treatment data available for this condition.<br>"
            "1. Consult your local agricultural extension service for advice.<br>"
            "2. Remove heavily infected tissue to reduce disease pressure.<br>"
            "3. Improve air circulation and avoid overhead irrigation.<br>"
            "4. Check nearby plants for similar symptoms."
        ),
        "severity": "moderate", "spread": "Consult specialist", "season": "Unknown"
    }

SEV_CONFIG = {
    "healthy":  ("✅ Healthy Plant",    "sev-healthy"),
    "mild":     ("🟡 Mild Severity",    "sev-mild"),
    "moderate": ("🟠 Moderate Risk",    "sev-moderate"),
    "severe":   ("🔴 Severe — Act Now", "sev-severe"),
}

PLANT_TIPS = {
    "Apple":      "🍎 Water deeply 1–2× per week. Thin fruit clusters in June for better sizing and disease prevention.",
    "Tomato":     "🍅 Stake plants and remove suckers weekly. Always water at the base — wet foliage invites disease.",
    "Corn":       "🌽 Side-dress with nitrogen at V6 stage. Scout for pests weekly from tasseling to silking.",
    "Grape":      "🍇 Prune aggressively every winter. Train on a trellis for maximum air circulation.",
    "Peach":      "🍑 Thin fruit to 15–20 cm apart for larger size. Apply dormant oil spray every winter.",
    "Potato":     "🥔 Hill soil around stems as they grow. Rotate field every 3 years to break disease cycles.",
    "Pepper":     "🌶 Maintain consistent soil moisture — drought stress followed by heavy watering causes blossom end rot.",
    "Cherry":     "🍒 Net trees to protect from birds. Never prune in wet weather. Seal all pruning wounds immediately.",
    "Orange":     "🍊 Fertilise 3× per year with citrus-specific fertiliser. Test soil pH annually (target: 6.0–7.0).",
    "Soybean":    "🌱 Inoculate seeds with Bradyrhizobium. Scout for Asian soybean rust from R1 flowering stage.",
    "Squash":     "🎃 Hand-pollinate early morning for better fruit set. Watch for powdery mildew from midsummer.",
    "Strawberry": "🍓 Replace bed every 3 years. Use straw mulch to keep fruit off soil and reduce splash disease.",
    "Raspberry":  "🫐 Remove floricanes immediately after harvest. Trellis primocanes for airflow.",
    "Blueberry":  "🫐 Maintain soil pH 4.5–5.5 with pine bark mulch. Prune oldest canes every 3–4 years.",
}

def get_plant_tip(label: str):
    for plant, tip in PLANT_TIPS.items():
        if plant.lower() in label.lower():
            return tip
    return None

HEALTHY_PRECAUTIONS = {
    "Apple": {
        "precautions": [
            "🌿 Prune dead or crossing branches every winter to maintain open canopy and airflow.",
            "💧 Water deeply at the base 2–3× per week in dry spells. Avoid wetting foliage.",
            "🐛 Scout for codling moth and aphids weekly from bud break onwards.",
            "🍎 Thin fruit clusters to 1 per spur in early June for larger, healthier apples.",
            "🌱 Apply balanced NPK fertiliser in early spring before bud break.",
        ],
        "farmer_tips": "Apply dormant oil spray every winter before buds open to smother overwintering pest eggs. Maintain a weed-free circle of 1 m around each trunk — weeds compete for nutrients and harbour pests."
    },
    "Tomato": {
        "precautions": [
            "🌿 Remove suckers weekly to direct energy into fruit production.",
            "💧 Use drip irrigation — wet foliage is the #1 cause of fungal disease.",
            "🪴 Stake or cage plants before they need it, not after they fall.",
            "🌱 Side-dress with calcium nitrate every 3 weeks to prevent blossom end rot.",
            "🔄 Never plant tomatoes in the same bed two years running — rotate crops.",
        ],
        "farmer_tips": "Mulch heavily (5–8 cm) around the base to retain moisture, regulate soil temperature, and prevent rain-splash of soil-borne pathogens onto lower leaves. This alone can reduce early blight pressure by up to 50%."
    },
    "Corn": {
        "precautions": [
            "🌱 Apply starter fertiliser (NPK 10-34-0) at planting for strong root establishment.",
            "💧 Ensure consistent moisture at V6–V8 stage — drought stress here permanently reduces yield.",
            "🐛 Scout for fall armyworm from seedling stage; treat early with spinosad or emamectin.",
            "🌿 Side-dress with nitrogen (urea) at V6 growth stage for maximum yield.",
            "🔄 Rotate with soybean or legume crop every alternate season to break pest cycles.",
        ],
        "farmer_tips": "Plant in blocks of at least 4 rows rather than single rows to ensure adequate wind pollination. Poor pollination is a common cause of incomplete ear fill that is often mistaken for disease."
    },
    "Grape": {
        "precautions": [
            "✂️ Prune aggressively every dormant season — leave only 2 buds per spur.",
            "🌿 Train shoots weekly onto trellis wires to maintain canopy structure.",
            "💧 Switch to drip irrigation at vine base — never overhead water.",
            "🌱 Apply potassium-rich fertiliser post-harvest to strengthen vines for next season.",
            "🔍 Inspect for powdery mildew on undersides of leaves every 7 days.",
        ],
        "farmer_tips": "Apply a copper-based fungicide spray at 5–10% bud burst as a preventive — this is the most cost-effective spray timing in the entire season for preventing downy mildew and black rot."
    },
    "Peach": {
        "precautions": [
            "✂️ Prune to an open vase shape to maximise sunlight penetration.",
            "🍑 Thin fruit to one every 15–20 cm by hand in late May for maximum size.",
            "💧 Apply drip irrigation — peaches are highly sensitive to overhead wetness.",
            "🌱 Fertilise with balanced NPK in early spring; avoid heavy nitrogen after June.",
            "🌿 Apply dormant copper spray every winter before bud swell.",
        ],
        "farmer_tips": "Peaches are highly susceptible to brown rot in wet summers. Keep fruit thinned so they do not touch — contact between fruit is one of the main routes for brown rot to spread."
    },
    "Potato": {
        "precautions": [
            "🥔 Use only certified disease-free seed potatoes — never save from infected stock.",
            "🌿 Hill soil around stems as they grow to prevent greening and support tubers.",
            "💧 Maintain even soil moisture — drought then heavy rain causes hollow heart.",
            "🔄 Rotate potato to a new bed every 3 years to break late blight cycles.",
            "🌱 Apply potassium sulphate fertiliser — potassium directly improves disease resistance.",
        ],
        "farmer_tips": "Scout for early blight symptoms (dark target-shaped spots on lower leaves) weekly from 40 days after planting. Early detection and a prompt fungicide application at first lesion sighting is far more effective than waiting until disease is widespread."
    },
    "Pepper": {
        "precautions": [
            "💧 Keep soil consistently moist — drought then heavy watering causes blossom end rot.",
            "🌿 Mulch with straw to maintain even moisture and soil temperature.",
            "🐛 Monitor for thrips and aphids — both transmit viral diseases.",
            "🌱 Feed with calcium-rich fertiliser every 2 weeks during fruiting.",
            "🌬 Space plants 45–60 cm apart for good airflow and reduced disease risk.",
        ],
        "farmer_tips": "Peppers grow best with a phosphorus-rich starter fertiliser at transplanting (e.g. 15-30-15). This promotes strong root development, making the plant far more resilient to both drought and disease throughout the season."
    },
    "Cherry": {
        "precautions": [
            "🍒 Net trees before fruit colours to protect from bird damage.",
            "✂️ Prune only in dry weather — cherry wounds in wet conditions invite fungal canker.",
            "💧 Reduce irrigation as fruit matures to prevent splitting.",
            "🌱 Apply sulphate of potash in late summer to harden wood for winter.",
            "🌿 Remove and burn any gummosis-affected branches immediately.",
        ],
        "farmer_tips": "Always seal pruning wounds over 2 cm with a wound sealant immediately after cutting. Cherry is extremely vulnerable to silver leaf disease entering through fresh cuts."
    },
    "Orange": {
        "precautions": [
            "🍊 Test soil pH annually — citrus requires 6.0–7.0 for optimal nutrient uptake.",
            "🌿 Fertilise 3× per year with citrus-specific NPK + trace elements.",
            "💧 Deep irrigate every 7–10 days in summer; reduce in winter.",
            "🔍 Inspect leaf undersides monthly for scale insects and Asian citrus psyllid.",
            "✂️ Remove water shoots (vigorous vertical growth) promptly — they waste energy.",
        ],
        "farmer_tips": "Citrus responds extremely well to a zinc and manganese foliar spray every 6 months. Micronutrient deficiency (visible as yellowing between leaf veins) is one of the most common and easily fixed causes of poor yield."
    },
    "Soybean": {
        "precautions": [
            "🌱 Inoculate seeds with Bradyrhizobium japonicum before planting for nitrogen fixation.",
            "💧 Ensure moisture at R1 (flowering) and R3 (pod fill) — the most yield-critical stages.",
            "🐛 Scout for soybean aphid and bean leaf beetle weekly from V4 stage.",
            "🔄 Rotate with a non-legume crop every alternate year.",
            "🌿 Apply boron foliar spray at flowering to improve pod set.",
        ],
        "farmer_tips": "Soybean yield is most sensitive to moisture stress during pod fill (R3–R6). If rainfall is insufficient during this 3–4 week window, supplemental irrigation of just 25 mm can prevent yield losses of 20–30%."
    },
    "Squash": {
        "precautions": [
            "🎃 Hand-pollinate early morning flowers with a soft brush if fruit set is poor.",
            "💧 Water at soil level — wet foliage promotes powdery mildew rapidly.",
            "🐛 Check undersides of leaves for squash bug eggs weekly; crush clusters on sight.",
            "🌿 Apply neem oil spray every 14 days as a preventive against powdery mildew.",
            "✂️ Remove old or yellowing leaves at the base to improve airflow.",
        ],
        "farmer_tips": "Squash vine borer moths lay eggs at the stem base — larvae bore inside before symptoms appear. Wrap the bottom 30 cm of each stem in aluminium foil at transplanting to physically block egg-laying."
    },
    "Strawberry": {
        "precautions": [
            "🍓 Mulch with straw to keep fruit off soil and reduce splash-borne disease.",
            "💧 Use drip irrigation — surface moisture promotes grey mould (Botrytis).",
            "🌱 Feed with high-potassium fertiliser during fruiting for sweetness and firmness.",
            "🔄 Renovate the bed immediately after harvest by mowing and thinning.",
            "🌿 Replace entire planting every 3 years — older beds accumulate disease pressure.",
        ],
        "farmer_tips": "Remove all runners you do not intend to propagate — each runner diverts energy from fruit. A well-managed runner-free plant can produce 30–40% more fruit than an unmanaged one."
    },
    "Raspberry": {
        "precautions": [
            "🫐 Remove floricanes (2-year-old canes) immediately after harvest — they won't fruit again.",
            "🌿 Tie primocanes onto trellis wires as they grow to prevent wind damage.",
            "💧 Maintain consistent moisture — irregular watering causes crumbly fruit.",
            "🌱 Apply sulphate of ammonia in early spring for strong cane growth.",
            "🔍 Check for raspberry beetle grubs in ripening fruit weekly.",
        ],
        "farmer_tips": "Raspberry canes need 6+ hours of direct sunlight. If your canes are getting less, reflective mulch between rows can increase light exposure, yield, and fruit colour significantly."
    },
    "Blueberry": {
        "precautions": [
            "🫐 Maintain soil pH strictly between 4.5–5.5 — outside this range plants cannot absorb nutrients.",
            "🌱 Use ericaceous (acidic) fertiliser only — never general-purpose fertilisers.",
            "💧 Blueberries have shallow roots — mulch 8–10 cm deep and water frequently.",
            "✂️ Prune out the oldest, thickest canes every 3–4 years to encourage new growth.",
            "🔍 Net bushes before fruit turns blue — birds can strip a bush overnight.",
        ],
        "farmer_tips": "Blueberries require cross-pollination between two different varieties for maximum yield. Adding a second compatible variety within 15 metres can increase yield by 30–60%."
    },
}

GENERAL_PRECAUTIONS = {
    "precautions": [
        "💧 Water at the base of plants in early morning — wet foliage overnight invites fungal disease.",
        "🌿 Maintain good spacing between plants to ensure airflow and reduce humidity.",
        "🌱 Test your soil every 2 years and amend with balanced NPK based on results.",
        "🔄 Rotate crops to a different bed or field each season to break disease and pest cycles.",
        "🐛 Scout your crop weekly — early detection of any pest or disease is the most effective control.",
    ],
    "farmer_tips": "Keep a simple farm diary: note planting dates, fertiliser applications, pest sightings, and weather events. Patterns over 2–3 seasons are invaluable for predicting and preventing problems before they occur."
}

def get_healthy_precautions(label: str) -> dict:
    for plant, data in HEALTHY_PRECAUTIONS.items():
        if plant.lower() in label.lower():
            return data
    return GENERAL_PRECAUTIONS

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🌿 Loading Keras model…")
def load_keras_model():
    if not TF_AVAILABLE:
        st.error("❌ TensorFlow is not installed. Run: `pip install tensorflow`")
        st.stop()
    if not os.path.exists(KERAS_PATH):
        st.error(f"❌ Keras model not found: `{KERAS_PATH}`")
        st.stop()
    m = tf.keras.models.load_model(KERAS_PATH)
    dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    m.predict(dummy, verbose=0)
    return m

@st.cache_resource(show_spinner="📱 Loading TFLite model…")
def load_tflite_interpreter():
    if not os.path.exists(TFLITE_PATH):
        st.error(f"❌ TFLite model not found: `{TFLITE_PATH}`")
        st.stop()
    if TFLITE_RUNTIME:
        interp = tflite.Interpreter(model_path=TFLITE_PATH)
    elif TF_AVAILABLE:
        interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
    else:
        st.error("❌ Neither TensorFlow nor tflite_runtime is installed.")
        st.stop()
    interp.allocate_tensors()
    return interp

@st.cache_resource(show_spinner="📋 Loading class names…")
def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        st.error(f"❌ class_names.json not found: `{CLASS_NAMES_PATH}`")
        st.stop()
    with open(CLASS_NAMES_PATH) as f:
        cn = json.load(f)
    if isinstance(cn, dict):
        cn = [cn[k] for k in sorted(cn, key=lambda x: int(x))]
    return cn

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# backend.ipynb builds the model with Rescaling(1/255) as first layer and
# include_preprocessing=False — so the model expects raw [0,255] float32.
# Do NOT apply any additional normalisation here.
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)   # raw [0,255] — model rescales inside
    return np.expand_dims(arr, axis=0)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def predict_keras(pil_image: Image.Image, model, use_tta: bool = True):
    arr    = preprocess(pil_image)
    tensor = tf.cast(arr, tf.float32)

    tta_aug_fn = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.05),
        tf.keras.layers.RandomContrast(0.1),
    ], name="tta_aug")

    if use_tta:
        acc = model(tensor, training=False).numpy()[0]
        for _ in range(TTA_STEPS - 1):
            aug  = tta_aug_fn(tensor, training=True)
            acc += model(aug, training=False).numpy()[0]
        probs = acc / TTA_STEPS
    else:
        probs = model.predict(arr, verbose=0)[0]

    idx  = int(np.argmax(probs))
    conf = float(probs[idx]) * 100
    return idx, conf, probs


def predict_tflite(pil_image: Image.Image, interpreter):
    arr = preprocess(pil_image)

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inp_dtype = input_details[0]["dtype"]
    # TFLite model was converted from the same Keras model which has
    # Rescaling(1/255) inside — pass raw [0,255] float32.
    # Only quantize if the TFLite model explicitly uses int types.
    if inp_dtype == np.uint8:
        arr = arr.clip(0, 255).astype(np.uint8)
    elif inp_dtype == np.int8:
        scale, zero_point = input_details[0]["quantization"]
        arr = (arr / scale + zero_point).clip(-128, 127).astype(np.int8)
    # float32 → pass through as-is (most common case)

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    out_dtype = output_details[0]["dtype"]
    if out_dtype in (np.uint8, np.int8):
        scale, zero_point = output_details[0]["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale

    if output.min() < 0 or output.max() > 1.01:
        output = np.exp(output - output.max())
        output = output / output.sum()

    idx  = int(np.argmax(output))
    conf = float(output[idx]) * 100
    return idx, conf, output

# ══════════════════════════════════════════════════════════════════════════════
# CHART GENERATOR (for PDF & inline display)
# ══════════════════════════════════════════════════════════════════════════════
def make_top5_chart(all_probs, class_names_list) -> bytes:
    """Return PNG bytes of a styled top-5 bar chart."""
    if not MPL_AVAILABLE:
        return b""
    top5 = np.argsort(all_probs)[::-1][:5]
    names = [get_disease_info(class_names_list[i])["display"][:28] for i in top5]
    probs = [float(all_probs[i]) * 100 for i in top5]
    colors_list = ["#3dff80" if i == 0 else "#1a4d1a" for i in range(5)]

    fig, ax = plt.subplots(figsize=(6, 2.8))
    fig.patch.set_facecolor("#0b1e0d")
    ax.set_facecolor("#050d06")

    bars = ax.barh(names[::-1], probs[::-1], color=colors_list[::-1],
                   height=0.55, edgecolor="none")
    for bar, prob in zip(bars, probs[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1f}%", va="center", ha="left",
                fontsize=8, color="#a0d4a0", fontweight="bold")

    ax.set_xlim(0, max(probs) + 12)
    ax.tick_params(colors="#7a9e7a", labelsize=7.5)
    ax.xaxis.label.set_color("#7a9e7a")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a3d1e")
    ax.set_xlabel("Confidence %", color="#7a9e7a", fontsize=8)
    ax.set_title("Top-5 Predictions", color="#3dff80", fontsize=9, fontweight="bold", pad=6)
    plt.tight_layout(pad=0.8)

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()

# ══════════════════════════════════════════════════════════════════════════════
# PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════
def _strip_html(text: str) -> str:
    text = _re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&#39;", "'")
    return text.strip()

def generate_pdf_report(
    label, confidence, info, is_healthy, sev_label,
    all_probs, class_names_list, pil_image,
    tta_enabled, num_classes, backend_name
) -> bytes:
    if not REPORTLAB_AVAILABLE:
        return b""

    buf = io.BytesIO()
    W, H = A4

    BG       = colors.HexColor("#050d06")
    CARD_BG  = colors.HexColor("#0b1e0d")
    ACCENT   = colors.HexColor("#3dff80")
    ACCENT2  = colors.HexColor("#00b84d")
    GOLD     = colors.HexColor("#ffc235")
    TEXT     = colors.HexColor("#ddeedd")
    SUBTEXT  = colors.HexColor("#7a9e7a")
    BORDER   = colors.HexColor("#1a3d1e")
    SEV_COLS = {
        "healthy":  colors.HexColor("#3dff80"),
        "mild":     colors.HexColor("#7ad4f5"),
        "moderate": colors.HexColor("#ffc235"),
        "severe":   colors.HexColor("#ff7a5a"),
    }
    sev_col = SEV_COLS.get(info.get("severity", "moderate"), GOLD)

    c = rl_canvas.Canvas(buf, pagesize=A4)

    def rounded_rect(x, y, w, h, r=6, fill_col=None, stroke_col=None, line_w=0.5):
        c.saveState()
        if fill_col:   c.setFillColor(fill_col)
        if stroke_col: c.setStrokeColor(stroke_col); c.setLineWidth(line_w)
        else:          c.setLineWidth(0)
        p = c.beginPath()
        p.roundRect(x, y, w, h, r)
        c.drawPath(p, fill=bool(fill_col), stroke=bool(stroke_col))
        c.restoreState()

    c.setFillColor(BG); c.rect(0, 0, W, H, fill=1, stroke=0)
    for i in range(60):
        alpha = (60 - i) / 60 * 0.35
        c.setFillColor(colors.HexColor("#004d18")); c.setFillAlpha(alpha)
        c.rect(0, H - i * 2, W, 2, fill=1, stroke=0)
    c.setFillAlpha(1.0)

    header_h = 72; header_y = H - header_h
    rounded_rect(20, header_y, W - 40, header_h, r=10, fill_col=CARD_BG, stroke_col=BORDER, line_w=0.8)
    c.setFillColor(ACCENT);  c.setFont("Helvetica-Bold", 22); c.drawString(36, H - 38, "LeafScan AI")
    c.setFillColor(ACCENT2); c.setFont("Helvetica", 8);       c.drawString(36, H - 52, f"AI-Powered Plant Disease Diagnostics  ·  Backend: {backend_name}")
    now_str = datetime.datetime.now().strftime("%d %b %Y  %H:%M")
    c.setFillColor(SUBTEXT); c.setFont("Helvetica", 8)
    c.drawRightString(W - 36, H - 38, f"Report generated: {now_str}")
    c.drawRightString(W - 36, H - 52, f"Model: EfficientNetV2B3  ·  {num_classes} classes")

    card_y   = header_y - 16
    card_h   = 130
    card_top = card_y
    rounded_rect(20, card_top - card_h, W - 40, card_h, r=10,
                 fill_col=CARD_BG, stroke_col=BORDER, line_w=0.8)

    img_x = W - 36 - 110; img_y = card_top - card_h + 10; img_w = 110; img_h = 110
    try:
        img_buf = io.BytesIO(); pil_image.convert("RGB").resize((200, 200)).save(img_buf, format="PNG")
        img_buf.seek(0)
        c.drawImage(ImageReader(img_buf), img_x, img_y, width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")
        rounded_rect(img_x - 2, img_y - 2, img_w + 4, img_h + 4, r=6, stroke_col=BORDER, line_w=0.5)
    except Exception:
        pass

    c.setFillColor(sev_col); c.setFont("Helvetica-Bold", 6.5)
    c.drawString(36, card_top - 18, info.get("severity", "").upper())
    c.setFillColor(TEXT); c.setFont("Helvetica-Bold", 17)
    disp_name = info["display"]
    if len(disp_name) > 38: disp_name = disp_name[:36] + "…"
    c.drawString(36, card_top - 36, disp_name)
    c.setFillColor(SUBTEXT); c.setFont("Helvetica", 8)
    c.drawString(36, card_top - 50, label[:50])
    bar_y2   = card_top - 72
    bar_full = (img_x - 46)
    bar_fill = bar_full * min(confidence, 100) / 100
    rounded_rect(36, bar_y2, bar_full, 8, r=4, fill_col=colors.HexColor("#0a1e0c"))
    if bar_fill > 0:
        rounded_rect(36, bar_y2, bar_fill, 8, r=4, fill_col=sev_col)
    c.setFillColor(sev_col); c.setFont("Helvetica-Bold", 9)
    c.drawString(36, bar_y2 - 14, f"Confidence: {confidence:.1f}%")
    sev_txt, _ = SEV_CONFIG.get(info.get("severity","moderate"), SEV_CONFIG["moderate"])
    c.setFillColor(sev_col); c.setFont("Helvetica-Bold", 8)
    c.drawString(36, bar_y2 - 28, sev_txt)
    c.setFillColor(SUBTEXT); c.setFont("Helvetica", 7.5)
    meta_y = bar_y2 - 44
    c.drawString(36, meta_y,      f"Spread:  {info.get('spread','—')}")
    c.drawString(36, meta_y - 12, f"Season:  {info.get('season','—')}")

    treat_y_start = card_top - card_h - 16
    treat_h = 145
    rounded_rect(20, treat_y_start - treat_h, W - 40, treat_h, r=8, fill_col=CARD_BG, stroke_col=BORDER, line_w=0.8)
    sec_y = treat_y_start
    c.setFillColor(ACCENT2); c.setFont("Helvetica-Bold", 6.5)
    if is_healthy:
        c.drawString(36, sec_y - 14, "PLANT STATUS")
        c.setFillColor(TEXT);    c.setFont("Helvetica", 8.5); c.drawString(36, sec_y - 28, "This plant appears healthy. No disease treatment required.")
        c.setFillColor(SUBTEXT); c.setFont("Helvetica", 8);   c.drawString(36, sec_y - 42, "Continue regular monitoring and maintain good growing conditions.")
    else:
        c.drawString(36, sec_y - 14, "RECOMMENDED TREATMENT")
        raw_treat = _strip_html(info.get("treatment", "No treatment data available."))
        lines = []
        for step in raw_treat.split("\n"):
            step = step.strip()
            if not step: continue
            while len(step) > 100:
                cut = step.rfind(" ", 0, 100); cut = cut if cut > 0 else 100
                lines.append(step[:cut]); step = step[cut:].strip()
            if step: lines.append(step)
        c.setFillColor(TEXT); c.setFont("Helvetica", 7.5)
        ty2 = sec_y - 28
        for line in lines[:8]:
            c.drawString(36, ty2, line); ty2 -= 11
            if ty2 < sec_y - treat_h + 10: break

    top5_y_start = treat_y_start - treat_h - 16
    top5_h = 130
    rounded_rect(20, top5_y_start - top5_h, W - 40, top5_h, r=8, fill_col=CARD_BG, stroke_col=BORDER, line_w=0.8)
    c.setFillColor(ACCENT2); c.setFont("Helvetica-Bold", 6.5); c.drawString(36, top5_y_start - 14, "TOP-5 PREDICTIONS")
    top5 = np.argsort(all_probs)[::-1][:5]
    bar_area_w = W - 80; bar_y = top5_y_start - 28
    for rank, i in enumerate(top5, 1):
        prob = float(all_probs[i]) * 100; name = get_disease_info(class_names_list[i])["display"]
        is_top = rank == 1
        lbl_col = ACCENT if is_top else SUBTEXT; bar_col = ACCENT2 if is_top else colors.HexColor("#1a4d1a")
        c.setFillColor(lbl_col); c.setFont("Helvetica-Bold" if is_top else "Helvetica", 7)
        c.drawString(36, bar_y + 4, f"{rank}. {name[:40] + ('…' if len(name) > 40 else '')}")
        c.setFillColor(SUBTEXT); c.setFont("Helvetica-Bold", 7); c.drawRightString(W - 36, bar_y + 4, f"{prob:.1f}%")
        track_y = bar_y - 4; track_h = 5
        rounded_rect(36, track_y, bar_area_w, track_h, r=2, fill_col=colors.HexColor("#0f2010"))
        fill_bar_w = max(2, bar_area_w * prob / 100)
        rounded_rect(36, track_y, fill_bar_w, track_h, r=2, fill_col=bar_col)
        bar_y -= 22

    c.setFillColor(SUBTEXT); c.setFont("Helvetica", 7)
    c.drawCentredString(W / 2, 28, f"LeafScan AI  ·  EfficientNetV2B3  ·  PlantVillage Dataset  ·  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    c.setStrokeColor(BORDER); c.setLineWidth(0.5); c.line(36, 38, W - 36, 38)

    # ── Chart page (page 2) ──────────────────────────────────────────────────
    if MPL_AVAILABLE:
        try:
            chart_png = make_top5_chart(all_probs, class_names_list)
            if chart_png:
                c.showPage()
                c.setFillColor(BG); c.rect(0, 0, W, H, fill=1, stroke=0)
                rounded_rect(20, H - 80, W - 40, 72, r=10, fill_col=CARD_BG, stroke_col=BORDER, line_w=0.8)
                c.setFillColor(ACCENT); c.setFont("Helvetica-Bold", 16)
                c.drawString(36, H - 42, "Analysis Chart")
                c.setFillColor(ACCENT2); c.setFont("Helvetica", 8)
                c.drawString(36, H - 58, "Top-5 prediction confidence scores — visual breakdown")

                chart_w = W - 80; chart_h = 220
                chart_y = H - 100 - chart_h
                rounded_rect(20, chart_y - 20, W - 40, chart_h + 40, r=10,
                             fill_col=CARD_BG, stroke_col=BORDER, line_w=0.8)
                c.drawImage(ImageReader(io.BytesIO(chart_png)),
                            40, chart_y, width=chart_w, height=chart_h,
                            preserveAspectRatio=True)

                c.setFillColor(SUBTEXT); c.setFont("Helvetica", 7)
                c.drawCentredString(W / 2, 28,
                    f"LeafScan AI  ·  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
                c.setStrokeColor(BORDER); c.setLineWidth(0.5); c.line(36, 38, W - 36, 38)
        except Exception:
            pass

    c.save()
    return buf.getvalue()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Profile · Language · History · Settings · Logout
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    user = st.session_state.user

    # ── Language selector ─────────────────────────────────────────────────────
    lang_choice = st.selectbox(T("lang_label"), ["English 🇬🇧", "हिंदी 🇮🇳"],
                               index=0 if st.session_state.lang == "en" else 1,
                               label_visibility="collapsed")
    new_lang = "en" if lang_choice.startswith("E") else "hi"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

    st.markdown('<hr style="border-color:rgba(0,160,55,0.12);margin:0.4rem 0 0.8rem;">', unsafe_allow_html=True)

    # ── App brand ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:0.2rem 0 0.5rem;">
        <div style="font-family:'Cormorant Garamond',serif;font-size:1.6rem;font-weight:600;color:#3dff80;">
            🌿 LeafScan AI
        </div>
        <div style="font-size:0.62rem;color:rgba(0,200,80,0.45);letter-spacing:0.12em;text-transform:uppercase;">
            Plant Disease Diagnostics
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(0,160,55,0.12);margin:0.5rem 0 0.8rem;">', unsafe_allow_html=True)

    # ── Profile section ───────────────────────────────────────────────────────
    if user:
        st.markdown(f"""
        <div style="background:rgba(0,180,65,0.07);border:1px solid rgba(0,170,58,0.18);
                    border-radius:14px;padding:1rem;margin-bottom:0.8rem;">
            <div style="font-size:2rem;text-align:center;margin-bottom:0.4rem;">👤</div>
            <div style="font-size:0.88rem;font-weight:600;color:rgba(200,230,200,0.9);text-align:center;">{user['name']}</div>
            <div style="font-size:0.68rem;color:rgba(100,148,100,0.55);text-align:center;margin-top:0.2rem;">📱 {user['phone']}</div>
        </div>
        """, unsafe_allow_html=True)

        col_lo, col_del = st.columns(2)
        with col_lo:
            if st.button(T("logout_btn"), use_container_width=True):
                end_session(st.session_state.token)
                _del_cookie(SESSION_COOKIE)
                for k in ["user","token","history","total_scans","healthy_count","scan_submitted","last_uploaded_name"]:
                    st.session_state[k] = _defaults.get(k)
                st.session_state.stage = "splash"
                st.rerun()
        with col_del:
            if st.button(T("delete_btn"), use_container_width=True,
                         help="Permanently delete your account and all data"):
                delete_account(user["id"])
                _del_cookie(SESSION_COOKIE)
                _del_cookie(VISITED_COOKIE)
                for k in ["user","token","history","total_scans","healthy_count","scan_submitted","last_uploaded_name"]:
                    st.session_state[k] = _defaults.get(k)
                st.session_state.stage = "splash"
                st.rerun()

    st.markdown('<hr style="border-color:rgba(0,160,55,0.12);margin:0.5rem 0 0.8rem;">', unsafe_allow_html=True)

    # ── Model backend ─────────────────────────────────────────────────────────
    st.markdown(f'<div style="font-size:0.62rem;letter-spacing:0.12em;text-transform:uppercase;color:rgba(0,200,80,0.5);margin-bottom:0.5rem;">{T("backend_hdr")}</div>', unsafe_allow_html=True)

    backend_options = []
    if keras_exists and TF_AVAILABLE:
        backend_options.append("🧠 Keras (.keras)  — Full model, TTA support")
    if tflite_exists:
        backend_options.append("📱 TFLite (.tflite) — Lite / mobile optimised")

    if not backend_options:
        st.error("No model files found! Place `plant_disease_model.keras` beside `app.py`.")
        st.stop()

    selected_backend_label = st.radio("backend", backend_options, label_visibility="collapsed")
    use_keras = selected_backend_label.startswith("🧠")

    if use_keras:
        st.markdown('<span class="model-badge badge-keras"><span class="badge-dot"></span>Keras Active</span>', unsafe_allow_html=True)
        model_obj   = load_keras_model()
        class_names = load_class_names()
        NUM_CLASSES = len(class_names)
        tta_enabled = st.toggle("Enable TTA (8-pass)", value=True, help="Test-Time Augmentation improves accuracy.")
        backend_name = "Keras (.keras)"
    else:
        st.markdown('<span class="model-badge badge-tflite"><span class="badge-dot"></span>TFLite Active</span>', unsafe_allow_html=True)
        model_obj   = load_tflite_interpreter()
        class_names = load_class_names()
        NUM_CLASSES = len(class_names)
        tta_enabled = False
        backend_name = "TFLite (.tflite)"
        st.info("ℹ️ TTA not supported in TFLite mode.", icon="📱")

    st.markdown('<hr style="border-color:rgba(0,160,55,0.12);margin:0.8rem 0;">', unsafe_allow_html=True)

    # ── Session stats ─────────────────────────────────────────────────────────
    st.markdown(f'<div style="font-size:0.62rem;letter-spacing:0.12em;text-transform:uppercase;color:rgba(0,200,80,0.5);margin-bottom:0.6rem;">{T("stats_hdr")}</div>', unsafe_allow_html=True)
    healthy_pct = (round(st.session_state.healthy_count / st.session_state.total_scans * 100)
                   if st.session_state.total_scans > 0 else 0)
    for num, lbl in [
        (st.session_state.total_scans, "Total Scans"),
        (st.session_state.healthy_count, "Healthy"),
        (st.session_state.total_scans - st.session_state.healthy_count, "Diseased"),
        (f"{healthy_pct}%", "Health Rate"),
    ]:
        st.markdown(
            f'<div class="stat-tile"><div class="stat-num">{num}</div><div class="stat-label">{lbl}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<hr style="border-color:rgba(0,160,55,0.12);margin:0.8rem 0;">', unsafe_allow_html=True)

    # ── My Plants shortcut ────────────────────────────────────────────────────
    st.markdown(f'<div style="font-size:0.62rem;letter-spacing:0.12em;text-transform:uppercase;color:rgba(0,200,80,0.5);margin-bottom:0.5rem;">🌱 My Plants</div>', unsafe_allow_html=True)
    if user:
        plants_sidebar = get_plants(user["id"])
        if plants_sidebar:
            for pl in plants_sidebar[:5]:
                st.markdown(
                    f'<div style="font-size:0.78rem;color:rgba(180,220,180,0.8);padding:0.25rem 0.5rem;'
                    f'border-left:2px solid rgba(0,180,70,0.3);margin-bottom:0.3rem;">🌿 {pl["name"]}</div>',
                    unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:0.75rem;color:rgba(100,140,100,0.45);padding:0.3rem 0;">No plants added yet.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:0.75rem;color:rgba(100,140,100,0.45);padding:0.3rem 0;">Log in to see plants.</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(0,160,55,0.12);margin:0.8rem 0;">', unsafe_allow_html=True)

    # ── Scan history in sidebar ───────────────────────────────────────────────
    st.markdown(f'<div style="font-size:0.62rem;letter-spacing:0.12em;text-transform:uppercase;color:rgba(0,200,80,0.5);margin-bottom:0.5rem;">{T("history_hdr")}</div>', unsafe_allow_html=True)
    uid = user["id"] if user else None
    db_scans = get_user_scans(uid) if uid else []

    if not db_scans:
        st.markdown(f'<div style="font-size:0.78rem;color:rgba(100,140,100,0.45);padding:0.5rem 0;">{T("no_history")}</div>', unsafe_allow_html=True)
    else:
        for s in db_scans[:8]:
            icon = "✅" if s["is_healthy"] else "⚠️"
            conf_col = "#3dff80" if s["is_healthy"] else "#ffc235"
            with st.expander(f"{icon} {s['display_label'][:22]}…" if len(s['display_label']) > 22 else f"{icon} {s['display_label']}"):
                # thumbnail
                if s["image_thumb"]:
                    try:
                        st.image(Image.open(io.BytesIO(s["image_thumb"])), use_container_width=True)
                    except Exception:
                        pass
                st.markdown(
                    f'<div style="font-size:0.7rem;color:rgba(100,148,100,0.6);">'
                    f'🕐 {s["scan_time"]}<br>'
                    f'<span style="color:{conf_col};font-weight:700;">{s["confidence"]:.1f}%</span>'
                    f' · {s.get("backend","")}</div>',
                    unsafe_allow_html=True
                )
                if REPORTLAB_AVAILABLE:
                    try:
                        scan_info = {"display":s["display_label"],"treatment":s["treatment"],
                                     "severity":s["severity"],"spread":s["spread"],"season":s["season"]}
                        scan_img = Image.open(io.BytesIO(s["image_thumb"])) if s["image_thumb"] else Image.new("RGB",(128,128),(10,30,12))
                        dummy_p = np.zeros(len(class_names))
                        try: dummy_p[class_names.index(s["raw_label"])] = s["confidence"]/100
                        except ValueError: pass
                        sev_lbl2, _ = SEV_CONFIG.get(s["severity"], SEV_CONFIG["moderate"])
                        pdf_b = generate_pdf_report(
                            label=s["raw_label"], confidence=s["confidence"],
                            info=scan_info, is_healthy=s["is_healthy"],
                            sev_label=sev_lbl2, all_probs=dummy_p,
                            class_names_list=class_names, pil_image=scan_img,
                            tta_enabled=False, num_classes=NUM_CLASSES,
                            backend_name=s.get("backend","")
                        )
                        st.download_button("⬇️ PDF", data=pdf_b,
                            file_name=f"LeafScan_{s['display_label'][:15]}_{s['scan_time'][:10]}.pdf",
                            mime="application/pdf", use_container_width=True,
                            key=f"sb_dl_{s['id']}")
                    except Exception:
                        pass
                if st.button("🗑 Delete", key=f"sb_del_{s['id']}", use_container_width=True):
                    delete_scan(s["id"], uid)
                    st.rerun()

    st.markdown('<hr style="border-color:rgba(0,160,55,0.12);margin:0.8rem 0;">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.65rem;color:rgba(80,130,80,0.5);line-height:1.7;">'
                f'<b style="color:rgba(0,200,80,0.5);">Classes:</b> {NUM_CLASSES}<br>'
                f'<b style="color:rgba(0,200,80,0.5);">Architecture:</b> EfficientNetV2B3<br>'
                f'<b style="color:rgba(0,200,80,0.5);">Dataset:</b> PlantVillage'
                f'</div>', unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════════════
# TOP NAVBAR
# ══════════════════════════════════════════════════════════════════════════════
user = st.session_state.user

# Build scan-history count for the badge
_nav_uid = user["id"] if user else None
_nav_scans = get_user_scans(_nav_uid) if _nav_uid else []
_nav_scan_count = len(_nav_scans)

if user:
    st.markdown(f"""
    <div class="ls-navbar">
        <div class="ls-navbar-brand"><span>🌿</span> Leaf<em>Scan</em> AI</div>
        <div class="ls-navbar-center">
            <div class="ls-nav-welcome">
                <div class="ls-nav-avatar">👤</div>
                <div class="ls-nav-user-text">
                    <span class="ls-nav-greeting">Welcome back</span>
                    <span class="ls-nav-name">{user['name']}</span>
                </div>
            </div>
            <div class="ls-nav-phone">📱 {user['phone']}</div>
            <div class="ls-nav-scan-badge">🔬 Scans&nbsp;<span class="ls-nav-scan-count">{_nav_scan_count}</span></div>
        </div>
        <div style="display:flex;align-items:center;gap:0.6rem;"></div>
    </div>
    """, unsafe_allow_html=True)
    # Logout button — right-aligned compact button
    _nb_col1, _nb_col2 = st.columns([8, 1])
    with _nb_col2:
        if st.button("⏻ Logout", key="navbar_logout", use_container_width=True,
                     help="Sign out of your account"):
            end_session(st.session_state.token)
            _del_cookie(SESSION_COOKIE)
            for k in ["user","token","history","total_scans","healthy_count",
                      "scan_submitted","last_uploaded_name"]:
                st.session_state[k] = _defaults.get(k)
            st.session_state.stage = "splash"
            st.rerun()

    # ── Scan History Panel (below navbar) ─────────────────────────────────────
    if _nav_scans:
        with st.expander(f"📋 Scan History  ({_nav_scan_count} scans) — click to view & download reports", expanded=False):
            for _s in _nav_scans[:20]:
                _icon = "✅" if _s["is_healthy"] else "⚠️"
                _conf_col = "#3dff80" if _s["is_healthy"] else "#ffc235"
                _scol1, _scol2, _scol3 = st.columns([0.7, 2.5, 1.2])
                with _scol1:
                    if _s["image_thumb"]:
                        try:
                            st.image(Image.open(io.BytesIO(_s["image_thumb"])), use_container_width=True)
                        except Exception:
                            st.markdown("🍃", unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="font-size:1.8rem;text-align:center;">🍃</div>', unsafe_allow_html=True)
                with _scol2:
                    st.markdown(
                        f'<div style="padding:0.25rem 0;">'
                        f'<div style="font-size:0.82rem;font-weight:600;color:#c5d8c5;">{_icon} {_s["display_label"]}</div>'
                        f'<div style="font-size:0.68rem;color:rgba(100,148,100,0.55);margin-top:0.15rem;">'
                        f'🕐 {_s["scan_time"]} &nbsp;·&nbsp; '
                        f'<span style="color:{_conf_col};font-weight:700;">{_s["confidence"]:.1f}%</span>'
                        f' &nbsp;·&nbsp; {_s.get("backend","")}'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )
                with _scol3:
                    if REPORTLAB_AVAILABLE:
                        try:
                            _scan_info = {
                                "display":   _s["display_label"],
                                "treatment": _s["treatment"],
                                "severity":  _s["severity"],
                                "spread":    _s["spread"],
                                "season":    _s["season"],
                            }
                            _scan_img = (
                                Image.open(io.BytesIO(_s["image_thumb"]))
                                if _s["image_thumb"]
                                else Image.new("RGB", (128, 128), (10, 30, 12))
                            )
                            _dummy_p = np.zeros(len(class_names))
                            try:
                                _dummy_p[class_names.index(_s["raw_label"])] = _s["confidence"] / 100
                            except (ValueError, IndexError):
                                pass
                            _sev_lbl2, _ = SEV_CONFIG.get(_s["severity"], SEV_CONFIG["moderate"])
                            _pdf_b = generate_pdf_report(
                                label=_s["raw_label"],
                                confidence=_s["confidence"],
                                info=_scan_info,
                                is_healthy=_s["is_healthy"],
                                sev_label=_sev_lbl2,
                                all_probs=_dummy_p,
                                class_names_list=class_names,
                                pil_image=_scan_img,
                                tta_enabled=False,
                                num_classes=NUM_CLASSES,
                                backend_name=_s.get("backend", ""),
                            )
                            st.download_button(
                                "⬇️ PDF",
                                data=_pdf_b,
                                file_name=f"LeafScan_{_s['display_label'][:15].replace(' ','_')}_{_s['scan_time'][:10]}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                key=f"nav_hist_dl_{_s['id']}",
                            )
                        except Exception:
                            pass
                    if st.button("🗑", key=f"nav_hist_del_{_s['id']}", help="Delete this scan"):
                        delete_scan(_s["id"], user["id"])
                        st.rerun()
                st.markdown('<hr style="border-color:rgba(0,100,28,0.1);margin:0.3rem 0;">', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="ls-navbar">
        <div class="ls-navbar-brand"><span>🌿</span> Leaf<em>Scan</em> AI</div>
        <div class="ls-nav-guest">🔒 Not logged in — sign in via the sidebar</div>
        <div></div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-eyebrow">🌿 AI Plant Pathology System</div>
    <div class="hero-title">Diagnose <em>any</em> leaf,<br>instantly.</div>
    <div class="hero-sub">
        Upload or photograph any plant leaf. EfficientNetV2B3 analyses it against 38 disease classes
        and returns an instant diagnosis with treatment recommendations.
    </div>
    <div class="hero-pills">
        <span class="hero-pill">🧠 {backend_name}</span>
        <span class="hero-pill">🌿 38 Classes</span>
        <span class="hero-pill">{'⚡ TTA · 8-Pass' if tta_enabled else '⚡ Single-Pass'}</span>
        <span class="hero-pill">📷 Camera Input</span>
        <span class="hero-pill">🌐 हिंदी Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_scan, tab_plants, tab_guide = st.tabs([T("scan_tab"), T("plants_tab"), T("guide_tab")])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCAN
# ══════════════════════════════════════════════════════════════════════════════
with tab_scan:
    st.markdown(f'<div class="section-hdr">{T("upload_hdr")}</div>', unsafe_allow_html=True)

    input_tab_file, input_tab_cam = st.tabs([T("file_lbl"), T("camera_lbl")])
    with input_tab_file:
        uploaded = st.file_uploader("Upload leaf image", type=["jpg","jpeg","png","webp"],
                                     label_visibility="collapsed")
    with input_tab_cam:
        camera_img = st.camera_input("Take a photo of the leaf", label_visibility="collapsed")

    # Determine active image source
    active_file = camera_img if camera_img else uploaded
    active_name = (camera_img.name if camera_img else (uploaded.name if uploaded else None))

    # Reset submitted state when new image arrives
    if active_name != st.session_state.last_uploaded_name:
        st.session_state.last_uploaded_name = active_name
        st.session_state.scan_submitted = False

    if active_file:
        image_pil = Image.open(active_file)

        # ── Plant selector ─────────────────────────────────────────────────────
        plants = get_plants(user["id"]) if user else []
        plant_options = ["— None —"] + [p["name"] for p in plants]
        sel_plant_name = st.selectbox("📌 Attach to plant profile (optional):",
                                       plant_options, label_visibility="visible")
        sel_plant_id = None
        if sel_plant_name != "— None —":
            for p in plants:
                if p["name"] == sel_plant_name:
                    sel_plant_id = p["id"]
                    break

        # ── Preview + Submit ────────────────────────────────────────────────────
        if not st.session_state.scan_submitted:
            prev_col, _ = st.columns([1, 1.5], gap="large")
            with prev_col:
                st.markdown(f'<div class="section-hdr">{T("image_hdr")}</div>', unsafe_allow_html=True)
                st.image(image_pil, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            _, btn_col, _ = st.columns([1, 1, 1])
            with btn_col:
                if st.button(T("analyse_btn"), use_container_width=True, type="primary"):
                    st.session_state.scan_submitted = True
                    st.rerun()

        if st.session_state.scan_submitted:
            try:
                col_img, col_res = st.columns([1, 1.5], gap="large")
                with col_img:
                    st.markdown(f'<div class="section-hdr">{T("image_hdr")}</div>', unsafe_allow_html=True)
                    st.image(image_pil, use_container_width=True)
                    w, h = image_pil.size
                    st.markdown(f'<div style="font-size:.7rem;color:rgba(100,140,100,0.5);text-align:center;">{w}×{h}px · {backend_name}</div>', unsafe_allow_html=True)

                with col_res:
                    st.markdown(f'<div class="section-hdr">{T("result_hdr")}</div>', unsafe_allow_html=True)
                    with st.spinner("🧠 Running TTA inference…" if tta_enabled else "⚡ Analysing…"):
                        t0 = time.time()
                        if use_keras:
                            idx, confidence, all_probs = predict_keras(image_pil, model_obj, use_tta=tta_enabled)
                        else:
                            idx, confidence, all_probs = predict_tflite(image_pil, model_obj)
                        elapsed = time.time() - t0

                    raw_label  = class_names[idx]
                    info       = get_disease_info(raw_label)
                    label      = info["display"]
                    is_healthy = info["severity"] == "healthy"
                    sev_label, sev_cls = SEV_CONFIG.get(info["severity"], SEV_CONFIG["moderate"])
                    card_cls  = "result-healthy" if is_healthy else "result-diseased"
                    meter_cls = "meter-fill-h"   if is_healthy else "meter-fill-d"

                    # ── Not-a-leaf guard ───────────────────────────────────────
                    # The model is trained only on plant leaves. If max confidence
                    # is very low the image is almost certainly not a leaf at all.
                    _NOT_LEAF_THRESHOLD = 28.0   # below this % → warn and stop
                    if confidence < _NOT_LEAF_THRESHOLD:
                        src_label = "camera photo" if camera_img else "uploaded image"
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, rgba(38,10,0,0.9), rgba(30,8,0,0.85));
                            border: 1px solid rgba(255,80,30,0.35);
                            border-radius: 18px; padding: 2rem 2rem 1.8rem;
                            text-align: center; margin-top: 1rem;
                            animation: cardReveal 0.5s cubic-bezier(0.16,1,0.3,1) both;">
                            <div style="font-size:3rem;margin-bottom:0.7rem;">🚫🍃</div>
                            <div style="font-family:'Cormorant Garamond',serif;font-size:1.7rem;
                                        font-weight:600;color:#ff7a5a;margin-bottom:0.5rem;">
                                That doesn't look like a leaf
                            </div>
                            <div style="font-size:0.88rem;color:rgba(200,150,130,0.75);
                                        line-height:1.75;max-width:440px;margin:0 auto 1.2rem;">
                                The AI couldn't find a plant leaf in your {src_label}
                                (max confidence was only <strong style="color:#ff7a5a;">{confidence:.1f}%</strong>).
                                <br>Please try again with a clear photo of a <em>single leaf</em>
                                on a plain background, well-lit and in focus.
                            </div>
                            <div style="display:flex;flex-wrap:wrap;gap:0.5rem;justify-content:center;margin-top:0.5rem;">
                                <span style="background:rgba(255,80,30,0.08);border:1px solid rgba(255,80,30,0.2);
                                             color:rgba(255,130,100,0.7);font-size:0.65rem;font-weight:600;
                                             letter-spacing:0.08em;padding:0.22rem 0.75rem;border-radius:999px;">
                                    💡 Single leaf filling the frame
                                </span>
                                <span style="background:rgba(255,80,30,0.08);border:1px solid rgba(255,80,30,0.2);
                                             color:rgba(255,130,100,0.7);font-size:0.65rem;font-weight:600;
                                             letter-spacing:0.08em;padding:0.22rem 0.75rem;border-radius:999px;">
                                    💡 Natural light, in focus
                                </span>
                                <span style="background:rgba(255,80,30,0.08);border:1px solid rgba(255,80,30,0.2);
                                             color:rgba(255,130,100,0.7);font-size:0.65rem;font-weight:600;
                                             letter-spacing:0.08em;padding:0.22rem 0.75rem;border-radius:999px;">
                                    💡 Plain / contrasting background
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button(T("scan_again"), use_container_width=True):
                            st.session_state.scan_submitted = False
                            st.session_state.last_uploaded_name = None
                            st.rerun()
                        # Skip all result rendering — do not save to DB
                        st.stop()
                    # ── End not-a-leaf guard ───────────────────────────────────

                    st.session_state.total_scans += 1
                    if is_healthy: st.session_state.healthy_count += 1
                    st.session_state.history.insert(0, {"label": label, "confidence": confidence,
                                                         "is_healthy": is_healthy, "time": datetime.datetime.now().strftime("%H:%M:%S"),
                                                         "backend": backend_name})
                    if len(st.session_state.history) > 20: st.session_state.history.pop()

                    # Save to DB
                    if user:
                        save_scan(user["id"], raw_label, label, confidence, is_healthy,
                                  info["severity"], info.get("spread",""), info.get("season",""),
                                  info.get("treatment",""), backend_name, image_pil, sel_plant_id)

                    st.markdown(
                        f'<div class="result-card {card_cls}">'
                        f'<div class="result-tag">Inference · {elapsed:.2f}s · {backend_name}</div>'
                        f'<div class="result-title">{label}</div>'
                        f'<div class="result-subtitle">{raw_label}</div>'
                        f'<div class="meter-label">Confidence<span class="meter-pct">{confidence:.1f}%</span></div>'
                        f'<div class="meter-wrap"><div class="{meter_cls}" style="width:{min(confidence,100):.1f}%;"></div></div>'
                        f'<span class="sev-badge {sev_cls}">{sev_label}</span>'
                        f'</div>', unsafe_allow_html=True)

                    if is_healthy:
                        st.markdown(
                            f'<div class="treatment-box" style="border-color:#5cb85c;">'
                            f'<span class="box-title">{T("healthy_status")}</span>'
                            f'This plant appears <strong>healthy</strong>. No treatment required.</div>',
                            unsafe_allow_html=True)
                        # Precautions & farmer tips for healthy crops
                        prec = get_healthy_precautions(raw_label)
                        items_html = "".join(
                            f'<div class="precaution-item">{p}</div>'
                            for p in prec["precautions"]
                        )
                        st.markdown(
                            f'<div class="precaution-box">'
                            f'<span class="box-title">🛡 Precautions to Keep It Healthy</span>'
                            f'{items_html}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<div class="farmer-tip-box">'
                            f'<span class="box-title">👨‍🌾 Farmer\'s Pro Tip</span>'
                            f'<span class="farmer-tip-icon">💡</span>'
                            f'<div class="farmer-tip-text">{prec["farmer_tips"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="treatment-box"><span class="box-title">{T("treatment_title")}</span>{info["treatment"]}</div>'
                            f'<div class="meta-grid">'
                            f'<div class="meta-tile"><div class="mt-label">{T("spread_lbl")}</div><div class="mt-value">{info["spread"]}</div></div>'
                            f'<div class="meta-tile"><div class="mt-label">{T("season_lbl")}</div><div class="mt-value">{info["season"]}</div></div>'
                            f'</div>', unsafe_allow_html=True)

                    tip = get_plant_tip(raw_label)
                    if tip:
                        st.markdown(f'<div class="tip-box"><span class="box-title">{T("tip_title")}</span>{tip}</div>', unsafe_allow_html=True)

                # Top-5 chart (inline matplotlib)
                st.markdown(f'<div class="section-hdr">{T("top5_hdr")}</div>', unsafe_allow_html=True)
                if MPL_AVAILABLE:
                    chart_png = make_top5_chart(all_probs, class_names)
                    if chart_png:
                        _, ch_col, _ = st.columns([0.05, 1, 0.05])
                        with ch_col:
                            st.image(chart_png, use_container_width=True)
                else:
                    top5 = np.argsort(all_probs)[::-1][:5]
                    for rank, i in enumerate(top5, 1):
                        prob = float(all_probs[i])*100
                        name = get_disease_info(class_names[i])["display"]
                        is_top = rank == 1
                        bar_c = "#7ec87e" if is_top else "#2d5e2d"
                        lbl_c = "#a8e6a3" if is_top else "#7a9e7a"
                        fw    = "700"     if is_top else "400"
                        st.markdown(
                            f'<div class="prob-row"><div class="prob-lbl">'
                            f'<span style="color:{lbl_c};font-weight:{fw};">{rank}. {name}</span>'
                            f'<span style="color:#a8e6a3;font-weight:{fw};">{prob:.1f}%</span></div>'
                            f'<div class="prob-bg"><div class="prob-fill" style="width:{prob:.1f}%;background:{bar_c};"></div></div></div>',
                            unsafe_allow_html=True)

                if confidence < 60:
                    st.warning(T("low_conf").format(c=confidence))

                if REPORTLAB_AVAILABLE:
                    st.markdown(f'<div class="section-hdr">{T("report_hdr")}</div>', unsafe_allow_html=True)
                    with st.spinner("Preparing PDF…"):
                        pdf_bytes = generate_pdf_report(
                            label=raw_label, confidence=confidence, info=info,
                            is_healthy=is_healthy, sev_label=sev_label,
                            all_probs=all_probs, class_names_list=class_names,
                            pil_image=image_pil, tta_enabled=tta_enabled,
                            num_classes=NUM_CLASSES, backend_name=backend_name)
                    safe_name = info["display"].replace(" ","_").replace("—","").replace("/","-")
                    st.download_button(label=T("download_btn"), data=pdf_bytes,
                        file_name=f"LeafScan_{safe_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf", use_container_width=True)
                else:
                    st.info("ℹ️ PDF export unavailable. Install: `pip install reportlab matplotlib`")

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(T("scan_again"), use_container_width=True):
                    st.session_state.scan_submitted = False
                    st.session_state.last_uploaded_name = None
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Error: `{e}`")
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MY PLANTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_plants:
    st.markdown(f'<div class="section-hdr">{T("plants_hdr")}</div>', unsafe_allow_html=True)

    if not user:
        st.info("Please log in to manage plant profiles.")
    else:
        uid = user["id"]
        plants = get_plants(uid)

        # ── Add new plant ────────────────────────────────────────────────────
        with st.expander(T("add_plant"), expanded=len(plants) == 0):
            with st.form("add_plant_form"):
                pname = st.text_input(T("plant_name"), placeholder="e.g. My Tomato #1")
                ptype = st.selectbox(T("plant_type"),
                    ["Tomato", "Apple", "Corn (Maize)", "Grape", "Potato", "Peach",
                     "Bell Pepper", "Cherry", "Orange", "Soybean", "Squash",
                     "Strawberry", "Raspberry", "Blueberry", "Other"])
                pnotes = st.text_input(T("plant_notes"), placeholder="Location, age, etc.")
                if st.form_submit_button(T("save_plant"), use_container_width=True):
                    if pname.strip():
                        add_plant(uid, pname.strip(), ptype, pnotes.strip())
                        st.success(f"✅ Plant '{pname}' added!")
                        st.rerun()
                    else:
                        st.error("Plant name is required.")

        st.markdown("<br>", unsafe_allow_html=True)

        if not plants:
            st.markdown(f'<div class="empty-state"><div class="icon">🌱</div><div class="etitle">{T("no_plants")}</div><div class="edesc">Add a plant above and attach future scans to it to track its health over time.</div></div>', unsafe_allow_html=True)
        else:
            for plant in plants:
                plant_scans = get_plant_scans(plant["id"], uid)
                healthy_cnt = sum(1 for s in plant_scans if s["is_healthy"])
                diseased_cnt = len(plant_scans) - healthy_cnt
                health_pct = round(healthy_cnt / len(plant_scans) * 100) if plant_scans else 0
                health_color = "#3dff80" if health_pct >= 70 else "#ffc235" if health_pct >= 40 else "#ff7a5a"

                with st.expander(f"🌱  {plant['name']}  ·  {plant['plant_type']}  ·  {len(plant_scans)} scans"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.markdown(f'<div class="stat-tile"><div class="stat-num">{len(plant_scans)}</div><div class="stat-label">Total Scans</div></div>', unsafe_allow_html=True)
                    with col_b:
                        st.markdown(f'<div class="stat-tile"><div class="stat-num" style="color:{health_color};">{health_pct}%</div><div class="stat-label">Health Rate</div></div>', unsafe_allow_html=True)
                    with col_c:
                        st.markdown(f'<div class="stat-tile"><div class="stat-num" style="color:#ff7a5a;">{diseased_cnt}</div><div class="stat-label">Diseased</div></div>', unsafe_allow_html=True)

                    if plant["notes"]:
                        st.markdown(f'<div style="font-size:0.78rem;color:rgba(120,160,120,0.6);margin:0.5rem 0;">📝 {plant["notes"]}</div>', unsafe_allow_html=True)

                    # Health trend chart
                    if MPL_AVAILABLE and len(plant_scans) >= 2:
                        dates = [s["scan_time"][:10] for s in reversed(plant_scans[-10:])]
                        healths = [1 if s["is_healthy"] else 0 for s in reversed(plant_scans[-10:])]
                        confs   = [s["confidence"] for s in reversed(plant_scans[-10:])]

                        fig, ax = plt.subplots(figsize=(5, 1.8))
                        fig.patch.set_facecolor("#0b1e0d")
                        ax.set_facecolor("#050d06")
                        colors_h = ["#3dff80" if h else "#ff7a5a" for h in healths]
                        ax.bar(range(len(dates)), confs, color=colors_h, width=0.6)
                        ax.set_xticks(range(len(dates)))
                        ax.set_xticklabels(dates, rotation=45, fontsize=6, color="#7a9e7a")
                        ax.set_yticks([0, 50, 100])
                        ax.tick_params(colors="#7a9e7a", labelsize=6)
                        ax.set_ylabel("Confidence %", color="#7a9e7a", fontsize=6)
                        ax.set_title(f"{plant['name']} — Health Trend", color="#3dff80", fontsize=7, pad=4)
                        for spine in ax.spines.values(): spine.set_edgecolor("#1a3d1e")
                        import matplotlib.patches as mpatches
                        h_patch = mpatches.Patch(color="#3dff80", label="Healthy")
                        d_patch = mpatches.Patch(color="#ff7a5a", label="Diseased")
                        ax.legend(handles=[h_patch, d_patch], fontsize=5.5, loc="upper right",
                                  facecolor="#0b1e0d", edgecolor="#1a3d1e", labelcolor="#7a9e7a")
                        plt.tight_layout(pad=0.5)
                        buf = io.BytesIO(); fig.savefig(buf, format="PNG", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
                        plt.close(fig)
                        st.image(buf.getvalue(), use_container_width=True)

                    # Recent scans for this plant
                    if plant_scans:
                        st.markdown('<div style="font-size:0.62rem;color:rgba(0,200,80,0.5);text-transform:uppercase;letter-spacing:0.12em;margin-top:0.8rem;">Recent Scans</div>', unsafe_allow_html=True)
                        for s in plant_scans[:5]:
                            icon = "✅" if s["is_healthy"] else "⚠️"
                            conf_col = "#3dff80" if s["is_healthy"] else "#ffc235"
                            st.markdown(
                                f'<div class="hist-card">'
                                f'<div><div class="hist-name">{icon} {s["display_label"]}</div>'
                                f'<div class="hist-meta">{s["scan_time"]}</div></div>'
                                f'<div style="color:{conf_col};font-weight:700;">{s["confidence"]:.1f}%</div>'
                                f'</div>', unsafe_allow_html=True)

                    if st.button(f"🗑 Delete '{plant['name']}'", key=f"del_plant_{plant['id']}", use_container_width=True):
                        delete_plant(plant["id"], uid)
                        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DISEASE GUIDE
# ══════════════════════════════════════════════════════════════════════════════
with tab_guide:
    st.markdown('<div class="section-hdr">📖 Full Disease Reference</div>', unsafe_allow_html=True)
    search = st.text_input("🔍 Search by plant or disease name…", placeholder="e.g. corn, blight, rust, tomato")
    for key, info in DISEASE_DB.items():
        if info["severity"] == "healthy":
            continue
        if search and search.lower() not in info["display"].lower() and search.lower() not in info["treatment"].lower():
            continue
        sev_label, _ = SEV_CONFIG.get(info["severity"], SEV_CONFIG["moderate"])
        with st.expander(f"{info['display']}  ·  {sev_label}"):
            st.markdown(
                f'<div style="font-size:.88rem;color:#c0d8c0;line-height:1.8;padding:.5rem 0;">'
                f'<strong style="color:#7ec87e;">💊 Treatment:</strong><br>{info["treatment"]}<br><br>'
                f'<div style="display:flex;gap:2rem;">'
                f'<div><strong style="color:#7ec87e;">🌬 Spread:</strong> {info["spread"]}</div>'
                f'<div><strong style="color:#7ec87e;">📅 Season:</strong> {info["season"]}</div>'
                f'</div></div>',
                unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="ls-footer-wrap">
    <div class="ls-footer-glow"></div>
    <div class="ls-footer">
        <div class="ls-footer-logo">🌿 Leaf<span>Scan</span> AI</div>
        <div class="ls-footer-tagline">AI-Powered Plant Disease Detection</div>
        <div class="ls-footer-pills">
            <span class="ls-footer-pill">🧠 EfficientNetV2B3</span>
            <span class="ls-footer-pill">🌿 PlantVillage Dataset</span>
            <span class="ls-footer-pill">⚡ 38 Disease Classes</span>
            <span class="ls-footer-pill">🔧 {backend_name}</span>
            <span class="ls-footer-pill">👤 {user['name'] if user else '—'}</span>
        </div>
        <div class="ls-footer-divider"></div>
        <div class="ls-footer-meta">
            <strong>LeafScan AI</strong> &nbsp;·&nbsp; Built with Streamlit &nbsp;·&nbsp; Keras &amp; TFLite inference
            <br>Plant health intelligence for farmers &amp; researchers &nbsp;·&nbsp; <strong>Ashutosh kashyap</strong>
        </div>
        <div class="ls-footer-leaf">🌿</div>
        <div class="ls-footer-dots">
            <div class="ls-footer-dot"></div>
            <div class="ls-footer-dot"></div>
            <div class="ls-footer-dot"></div>
            <div class="ls-footer-dot"></div>
            <div class="ls-footer-dot"></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)