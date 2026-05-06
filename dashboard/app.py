"""
dashboard/app.py  ·  EFREI M2 — Maintenance Prédictive Industrielle 2025-26
═══════════════════════════════════════════════════════════════════════════════
Dashboard décisionnel — Design industriel professionnel

Lancement :
    streamlit run dashboard/app.py
"""

import os, sys, time
from pathlib import Path
from datetime import datetime, timedelta

import yaml
import joblib
import warnings
import requests
import numpy as np
import pandas as pd
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PredictMaint Pro · Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH   = ROOT / "data" / "raw" / "predictive_maintenance_v3.csv"
MODELS_DIR  = ROOT / "models"
RESULTS_DIR = ROOT / "results"

FEATURES_NUM = [
    "vibration_rms", "temperature_motor", "current_phase_avg",
    "pressure_level", "rpm", "hours_since_maintenance",
]
TARGET = "failure_within_24h"

MODEL_LABELS = {
    "logistic_regression": "Régression Logistique",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
    "mlp":                 "MLP Deep Learning",
}

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION API — Architecture Front / API / Modèle (cf. sujet p.15)
# ──────────────────────────────────────────────────────────────────────────────
# Le dashboard fonctionne en deux modes contrôlés par variables d'environnement :
#   - USE_API=false (défaut) : prédiction locale via joblib.load (mode dev rapide)
#   - USE_API=true           : prédiction via POST /predict de l'API REST
# En Docker, API_URL=http://api:8000 (nom de service du compose).
# En local, API_URL=http://localhost:8000.
USE_API     = os.getenv("USE_API", "false").lower() == "true"
API_URL     = os.getenv("API_URL", "http://localhost:8000")
API_TIMEOUT = 5  # secondes — court pour ne pas bloquer le dashboard


@st.cache_data(ttl=10, show_spinner=False)
def api_is_alive() -> bool:
    """
    Health check de l'API. Cache 10s pour éviter de spammer /health
    à chaque rerender Streamlit. Renvoie True si l'API répond ET a un modèle chargé.
    """
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200 and r.json().get("model_loaded", False)
    except Exception:
        return False


def predict_via_api(vals: dict, model_name: str | None = None) -> dict | None:
    """
    Appelle POST /predict de l'API REST.

    Args:
        vals       : dict avec les capteurs bruts (mêmes clés que SensorInput côté API)
        model_name : nom du modèle ('xgboost', 'random_forest', etc.) ou None = défaut API

    Returns:
        dict JSON complet de l'API (probability_failure, risk_level, health_score, ...)
        ou None si l'API est injoignable / erreur.
    """
    payload = {
        "vibration_rms":           float(vals["vibration_rms"]),
        "temperature_motor":       float(vals["temperature_motor"]),
        "current_phase_avg":       float(vals["current_phase_avg"]),
        "pressure_level":          float(vals["pressure_level"]),
        "rpm":                     float(vals["rpm"]),
        "hours_since_maintenance": float(vals["hours_since_maintenance"]),
        "ambient_temp":            float(vals.get("ambient_temp", 22.0)),
        "operating_mode":          str(vals["operating_mode"]),
    }
    if model_name:
        payload["model_name"] = model_name

    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=API_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"⚠️ API injoignable ({type(e).__name__}) — fallback modèle local.")
        return None


def predict_proba(vals: dict, models: dict, model_key: str) -> float:
    """
    Wrapper d'inférence — point d'entrée unique du dashboard.

    Routage :
      - USE_API=true ET API up   → POST /predict
      - USE_API=true ET API down → fallback silencieux sur modèle local
      - USE_API=false            → modèle local directement (comportement initial)

    Garantit que l'UI ne crash JAMAIS si l'API tombe : un dashboard production-ready
    doit dégrader gracieusement. Renvoie toujours une probabilité de panne ∈ [0, 1].
    """
    if USE_API:
        result = predict_via_api(vals, model_name=model_key)
        if result is not None:
            return float(result["probability_failure"])
        # fallback silencieux

    # Mode local — comportement historique
    X = build_X(vals)
    return float(models[model_key].predict_proba(X)[0, 1])


# ──────────────────────────────────────────────────────────────────────────────
# THÈME CSS — DARK INDUSTRIAL
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: #0d1117; color: #e6edf3; }
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
      border-right: 1px solid #30363d;
  }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 1.2rem 2rem; }
  h1 { color: #58a6ff !important; font-weight: 800 !important; letter-spacing: -0.5px; }
  h2 { color: #e6edf3 !important; }
  .stMarkdown p { color: #c9d1d9; }

  .kpi-card {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d; border-radius: 14px;
    padding: 20px 22px; text-align: center;
    transition: transform .2s, box-shadow .2s;
    position: relative; overflow: hidden;
  }
  .kpi-card::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; background: var(--accent, #58a6ff);
    border-radius: 14px 14px 0 0;
  }
  .kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(88,166,255,.2); }
  .kpi-value { font-size: 2.4rem; font-weight: 800; color: var(--accent, #58a6ff); margin: 8px 0 4px; }
  .kpi-label { font-size: .85rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
  .kpi-delta { font-size: .75rem; margin-top: 4px; padding: 2px 8px; border-radius: 12px; display: inline-block; }
  .delta-up   { background: rgba(248,81,73,.15);  color: #f85149; }
  .delta-down { background: rgba(63,185,80,.15);  color: #3fb950; }
  .delta-ok   { background: rgba(88,166,255,.15); color: #58a6ff; }

  .alert-critical {
    background: rgba(248,81,73,.1); border: 1px solid rgba(248,81,73,.4);
    border-left: 4px solid #f85149; border-radius: 8px;
    padding: 14px 18px; margin: 8px 0; color: #ffa198;
  }
  .alert-high {
    background: rgba(210,153,34,.1); border: 1px solid rgba(210,153,34,.4);
    border-left: 4px solid #d29922; border-radius: 8px;
    padding: 14px 18px; margin: 8px 0; color: #e3b341;
  }
  .alert-medium {
    background: rgba(58,133,255,.1); border: 1px solid rgba(58,133,255,.4);
    border-left: 4px solid #3a85ff; border-radius: 8px;
    padding: 14px 18px; margin: 8px 0; color: #58a6ff;
  }
  .alert-ok {
    background: rgba(63,185,80,.1); border: 1px solid rgba(63,185,80,.4);
    border-left: 4px solid #3fb950; border-radius: 8px;
    padding: 14px 18px; margin: 8px 0; color: #3fb950;
  }
  .badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: .75rem; font-weight: 700; letter-spacing: .5px;
  }
  .badge-critical { background: rgba(248,81,73,.2);  color: #f85149; border: 1px solid #f85149; }
  .badge-warning  { background: rgba(210,153,34,.2); color: #d29922; border: 1px solid #d29922; }
  .badge-ok       { background: rgba(63,185,80,.2);  color: #3fb950; border: 1px solid #3fb950; }
  .section-title {
    font-size: 1.1rem; font-weight: 700; color: #58a6ff;
    text-transform: uppercase; letter-spacing: 2px;
    border-bottom: 1px solid #21262d; padding-bottom: 8px; margin: 24px 0 16px;
  }
  .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { color: #8b949e; background: transparent; border-radius: 6px; }
  .stTabs [aria-selected="true"] { background: #1f6feb !important; color: #ffffff !important; }
  [data-testid="metric-container"] {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 14px;
  }
  [data-testid="stMetricValue"] { color: #58a6ff !important; font-weight: 800; }
  hr { border-color: #21262d !important; }
  .fleet-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 12px; padding: 16px;
    transition: box-shadow .2s;
  }
  .fleet-card:hover { box-shadow: 0 4px 16px rgba(88,166,255,.15); }
  .fleet-machine-id { font-size: .75rem; color: #8b949e; margin-bottom: 4px; }
  .fleet-machine-name { font-size: 1rem; font-weight: 700; color: #e6edf3; }
  .fleet-metric { font-size: .8rem; color: #8b949e; margin-top: 6px; }
  .fleet-metric span { color: #c9d1d9; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement des données…")
def load_data():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df["date"]       = df["timestamp"].dt.date
    df["hour"]       = df["timestamp"].dt.hour
    df["dayofweek"]  = df["timestamp"].dt.dayofweek
    df["month"]      = df["timestamp"].dt.month
    df["temp_delta"] = df["temperature_motor"] - df["ambient_temp"]
    df["age_vibration"] = df["hours_since_maintenance"] * df["vibration_rms"]
    df["vibration_per_rpm"] = df["vibration_rms"] / df["rpm"].clip(lower=1)
    return df


@st.cache_resource(show_spinner="Chargement des modèles…")
def load_models():
    models = {}
    for key in MODEL_LABELS:
        p = MODELS_DIR / f"{key}.pkl"
        if p.exists():
            try:
                models[key] = joblib.load(p)
            except Exception:
                pass
    return models


@st.cache_data(show_spinner=False)
def load_metrics():
    p = RESULTS_DIR / "metrics_comparison.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_resource(show_spinner=False)
def load_auth_config():
    """Charge la configuration d'authentification depuis config/auth.yaml"""
    config_path = ROOT / "config" / "auth.yaml"
    if not config_path.exists():
        st.error(f"❌ Fichier de configuration manquant : {config_path}")
        st.stop()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        st.error(f"❌ Erreur chargement config : {e}")
        st.stop()


AUTH_CONFIG = load_auth_config()
ROLES = AUTH_CONFIG.get("roles", {})
AUTH_USERS = AUTH_CONFIG.get("users", {})
ROLE_LABELS = {role_key: role_info.get("label", role_key) for role_key, role_info in ROLES.items()}
ROLE_DATA_SCIENCE = "data_science"
ROLE_ENGINEER = "engineer"


def initialize_auth_state():
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("username", "")
    st.session_state.setdefault("role", ROLE_ENGINEER)


def authenticate(username: str, password: str) -> tuple[bool, str | None]:
    user = AUTH_USERS.get(username.strip().lower())
    if not user:
        return False, None
    
    password_hash = user.get("password_hash")
    if not password_hash:
        return False, None
    
    try:
        ph = PasswordHasher()
        ph.verify(password_hash, password)
        return True, user.get("role", ROLE_ENGINEER)
    except VerifyMismatchError:
        return False, None
    except Exception:
        return False, None


def show_login_page():
    st.markdown("<h1>🔐 Authentification Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8b949e">Connectez-vous pour accéder au tableau de bord PredictMaint Pro.</p>',
        unsafe_allow_html=True,
    )
    with st.form("login_form"):
        username = st.text_input("Nom d'utilisateur", value=st.session_state.get("username", ""))
        password = st.text_input("Mot de passe", type="password")
        submit = st.form_submit_button("Se connecter")
        if submit:
            ok, role = authenticate(username, password)
            if ok:
                st.session_state.authenticated = True
                st.session_state.username = username.strip().lower()
                st.session_state.role = role
                st.rerun()
            else:
                st.error("Identifiant ou mot de passe incorrect.")

    st.markdown("---")
    st.markdown("**Profils disponibles :**")
    for username, user_info in AUTH_USERS.items():
        role_key = user_info.get("role", ROLE_ENGINEER)
        role_info = ROLES.get(role_key, {})
        description = role_info.get("description", "")
        st.markdown(f"- `{username}` — {description}")
    st.stop()


def health_score(vals):
    def norm(v, lo, hi): return max(0., min(1., (v - lo) / (hi - lo + 1e-9)))
    v = norm(vals.get("vibration_rms",    2.5), 0.5,  6.0)
    t = norm(vals.get("temperature_motor",75.), 40,  120)
    p = norm(vals.get("pressure_level",   6.0), 2.0,  12)
    r = norm(vals.get("rpm",              1800), 500, 3500)
    return round(1 - (0.30*v + 0.25*t + 0.20*p + 0.15*r), 3)


def risk_label(proba):
    if proba < 0.30: return "LOW",      "#3fb950", "badge-ok"
    if proba < 0.55: return "MEDIUM",   "#d29922", "badge-warning"
    if proba < 0.78: return "HIGH",     "#f85149", "badge-critical"
    return               "CRITICAL",    "#ff6e6e", "badge-critical"


def build_X(vals):
    return pd.DataFrame([{
        "vibration_rms":         vals["vibration_rms"],
        "temperature_motor":     vals["temperature_motor"],
        "current_phase_avg":     vals["current_phase_avg"],
        "pressure_level":        vals["pressure_level"],
        "rpm":                   float(vals["rpm"]),
        "hours_since_maintenance": float(vals["hours_since_maintenance"]),
        "operating_mode":        vals["operating_mode"],
        "temp_delta":            vals["temperature_motor"] - vals.get("ambient_temp", 22.),
        "age_vibration":         vals["hours_since_maintenance"] * vals["vibration_rms"],
        "vibration_per_rpm":     vals["vibration_rms"] / max(vals["rpm"], 1),
        "vibration_rms_roll_mean":    np.nan,
        "vibration_rms_roll_std":     np.nan,
        "vibration_rms_diff":         np.nan,
        "temperature_motor_roll_mean":np.nan,
        "temperature_motor_roll_std": np.nan,
        "temperature_motor_diff":     np.nan,
        "current_phase_avg_roll_mean":np.nan,
        "current_phase_avg_diff":     np.nan,
    }])


def plotly_dark(fig, height=380):
    fig.update_layout(
        height=height,
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", size=12),
        margin=dict(t=30, b=20, l=20, r=20),
        xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    )
    return fig


def kpi(label, value, delta_html="", accent="#58a6ff"):
    return (
        f'<div class="kpi-card" style="--accent:{accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{delta_html}</div>'
    )


initialize_auth_state()
if not st.session_state.authenticated:
    show_login_page()

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:16px 0 8px">'
        '<div style="font-size:2.4rem">⚡</div>'
        '<div style="font-size:1.2rem;font-weight:800;color:#58a6ff;letter-spacing:-0.5px">'
        'PredictMaint<span style="color:#3fb950">Pro</span></div>'
        '<div style="font-size:.72rem;color:#8b949e;margin-top:2px">EFREI M2 · Data Science 2025-26</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        f'''<div style="padding:12px;margin-bottom:12px;border:1px solid #30363d;border-radius:10px;">
          <b>Utilisateur</b> : {st.session_state.username}<br>
          <b>Profil</b> : {ROLE_LABELS.get(st.session_state.role, "—")}
        </div>''',
        unsafe_allow_html=True,
    )
    if st.button("🔓 Se déconnecter"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.role = ROLE_ENGINEER
        st.rerun()

    common_pages = [
        "🏠  Vue d'ensemble",
        "🚨  Alertes en temps réel",
        "⚙️  Simulateur Machine",
        "📊  Analyse EDA",
    ]
    data_pages = common_pages + [
        "🤖  Modèles & Performances",
        "🔍  Explicabilité IA",
        "💼  Use Cases Métier",
    ]
    page_options = data_pages if st.session_state.role == ROLE_DATA_SCIENCE else common_pages

    page = st.radio(
        "Navigation",
        options=page_options,
        label_visibility="collapsed",
    )
    st.divider()

    if st.session_state.role == ROLE_ENGINEER:
        st.markdown(
            '<div class="alert-medium">Accès ingénieur : pages modèles et explicabilité masquées.</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div style="color:#484f58;font-size:.72rem;text-align:center">{datetime.now().strftime("%d/%m/%Y %H:%M")}</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    df      = load_data()
    models  = load_models()
    metrics = load_metrics()

    st.markdown(
        '<div style="font-size:.75rem;color:#8b949e;text-transform:uppercase;'
        'letter-spacing:1px;margin-bottom:6px">STATUT SYSTÈME</div>',
        unsafe_allow_html=True,
    )
    data_color = "#3fb950" if df is not None else "#f85149"
    data_text  = f"Données — <b>{len(df):,}</b> obs." if df is not None else "Données manquantes"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
        f'<span style="color:{data_color};font-size:1.2rem">●</span>'
        f'<span style="color:#c9d1d9;font-size:.85rem">{data_text}</span></div>',
        unsafe_allow_html=True,
    )
    for key, label in MODEL_LABELS.items():
        ok = key in models
        color = "#3fb950" if ok else "#484f58"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
            f'<span style="color:{color};font-size:.9rem">{"●" if ok else "○"}</span>'
            f'<span style="color:#8b949e;font-size:.78rem">{label}</span></div>',
            unsafe_allow_html=True,
        )

    # ── Indicateur du mode d'inférence (Front / API / Modèle) ────────────────
    if USE_API:
        api_ok = api_is_alive()
        mode_color = "#3fb950" if api_ok else "#d29922"
        mode_icon  = "🟢" if api_ok else "🟡"
        mode_label = (
            f"<b>API REST</b> · {mode_icon} active<br>"
            f"<span style='color:#8b949e;font-size:.7rem'>{API_URL}</span>"
            if api_ok else
            f"<b>API REST</b> · {mode_icon} indisponible<br>"
            f"<span style='color:#8b949e;font-size:.7rem'>fallback local actif</span>"
        )
    else:
        mode_color = "#58a6ff"
        mode_label = (
            "<b>Mode local</b> · 🔵 modèles en mémoire<br>"
            "<span style='color:#8b949e;font-size:.7rem'>USE_API=true pour activer l'API</span>"
        )

    st.markdown(
        f'<div style="margin-top:14px;padding:10px 12px;'
        f'border-left:3px solid {mode_color};background:rgba(255,255,255,0.02);'
        f'border-radius:4px;font-size:.78rem;color:#c9d1d9;line-height:1.4">'
        f'{mode_label}</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        f'<div style="color:#484f58;font-size:.72rem;text-align:center">'
        f'{datetime.now().strftime("%d/%m/%Y %H:%M")}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — VUE D'ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Vue d'ensemble":
    st.markdown('<h1>🏭 Centre de Pilotage — Maintenance Prédictive</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8b949e;margin-top:-10px">Tableau de bord décisionnel · Responsable Maintenance & DSI</p>',
        unsafe_allow_html=True,
    )

    if df is None:
        st.markdown(
            '<div class="alert-critical">⚠️ <b>Données introuvables.</b> '
            'Placez <code>predictive_maintenance_v3.csv</code> dans <code>data/raw/</code></div>',
            unsafe_allow_html=True,
        )
        st.stop()

    n_total  = len(df)
    n_pannes = int(df[TARGET].sum())
    pct_ok   = 100 - n_pannes / n_total * 100
    rul_med  = df["rul_hours"].median() if "rul_hours" in df.columns else 0
    n_mach   = df["machine_id"].nunique() if "machine_id" in df.columns else "—"
    n_types  = df["failure_type"].nunique() - 1 if "failure_type" in df.columns else 0

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.markdown(kpi("Observations",      f"{n_total:,}",          '<span class="kpi-delta delta-ok">Dataset complet</span>'), unsafe_allow_html=True)
    c2.markdown(kpi("Pannes détectées",  f"{n_pannes:,}",         f'<span class="kpi-delta delta-up">⬆ {n_pannes/n_total*100:.1f}%</span>', "#f85149"), unsafe_allow_html=True)
    c3.markdown(kpi("Fiabilité globale", f"{pct_ok:.1f}%",        '<span class="kpi-delta delta-ok">Score flotte</span>', "#3fb950"), unsafe_allow_html=True)
    c4.markdown(kpi("RUL médian",        f"{rul_med:.0f}h",       '<span class="kpi-delta delta-ok">Durée restante</span>', "#d29922"), unsafe_allow_html=True)
    c5.markdown(kpi("Machines",          f"{n_mach}",             '<span class="kpi-delta delta-ok">Parc surveillé</span>', "#bc8cff"), unsafe_allow_html=True)
    c6.markdown(kpi("Types défaillance", f"{n_types}",            '<span class="kpi-delta delta-up">Catalogués</span>', "#f0883e"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1.6])
    with col_l:
        st.markdown('<div class="section-title">Répartition des classes</div>', unsafe_allow_html=True)
        vc = df[TARGET].value_counts()
        fig = go.Figure(go.Pie(
            labels=["Normal", "Panne imminente"],
            values=vc.values,
            hole=0.6,
            marker=dict(colors=["#3fb950", "#f85149"], line=dict(color="#0d1117", width=3)),
            textinfo="percent+label",
            textfont=dict(color="#e6edf3", size=13),
        ))
        fig.update_layout(
            height=320, plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font=dict(color="#c9d1d9"), margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
            annotations=[dict(text=f"<b>{pct_ok:.0f}%</b><br>OK", x=0.5, y=0.5,
                               font_size=18, showarrow=False, font=dict(color="#3fb950"))],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Taux de panne par type de machine</div>', unsafe_allow_html=True)
        if "machine_type" in df.columns:
            rate_m = (df.groupby("machine_type")[TARGET].mean() * 100).reset_index()
            rate_m.columns = ["machine_type", "taux"]
            rate_m = rate_m.sort_values("taux", ascending=True)
            fig = go.Figure(go.Bar(
                y=rate_m["machine_type"], x=rate_m["taux"], orientation="h",
                marker=dict(color=rate_m["taux"],
                            colorscale=[[0,"#3fb950"],[0.5,"#d29922"],[1,"#f85149"]]),
                text=rate_m["taux"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside", textfont=dict(color="#c9d1d9"),
            ))
            plotly_dark(fig, 320)
            fig.update_layout(xaxis_title="Taux de panne (%)", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    # Évolution temporelle
    st.markdown('<div class="section-title">Évolution temporelle — Taux de panne journalier</div>', unsafe_allow_html=True)
    daily = (
        df.groupby("date")
          .agg(n=("machine_id" if "machine_id" in df.columns else TARGET, "count"),
               failures=(TARGET, "sum"))
          .reset_index()
    )
    daily["rate"] = daily["failures"] / daily["n"] * 100
    daily["date"] = pd.to_datetime(daily["date"])
    daily["ma7"]  = daily["rate"].rolling(7, center=True).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["rate"],
        fill="tozeroy", fillcolor="rgba(248,81,73,0.1)",
        line=dict(color="#f85149", width=2), name="Taux panne (%)",
        hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Taux : %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["ma7"],
        line=dict(color="#58a6ff", width=2, dash="dot"), name="Moyenne 7j",
    ))
    plotly_dark(fig, 300)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1.1, bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

    col_hl, col_hr = st.columns(2)
    with col_hl:
        st.markdown('<div class="section-title">Heatmap — Machine × Mode opératoire</div>', unsafe_allow_html=True)
        if "machine_type" in df.columns and "operating_mode" in df.columns:
            pivot = df.groupby(["machine_type", "operating_mode"])[TARGET].mean().unstack() * 100
            fig = px.imshow(
                pivot, text_auto=".1f",
                color_continuous_scale=[[0,"#3fb950"],[0.5,"#d29922"],[1,"#f85149"]],
                labels=dict(x="Mode", y="Machine", color="Taux (%)"),
            )
            fig.update_layout(height=300, plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                               font=dict(color="#c9d1d9"), margin=dict(t=10,b=10,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)

    with col_hr:
        st.markdown('<div class="section-title">Sunburst — Défaillances par type</div>', unsafe_allow_html=True)
        if "failure_type" in df.columns and "machine_type" in df.columns:
            ft_df = (
                df[df["failure_type"] != "none"]
                  .groupby(["machine_type","failure_type"])
                  .size().reset_index(name="count")
            )
            fig = px.sunburst(ft_df, path=["machine_type","failure_type"], values="count",
                               color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(height=300, plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                               font=dict(color="#c9d1d9"), margin=dict(t=10,b=10,l=10,r=10))
            fig.update_traces(textfont=dict(color="#e6edf3"))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ALERTES EN TEMPS RÉEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨  Alertes en temps réel":
    st.markdown("<h1>🚨 Centre d'Alertes — Monitoring Temps Réel</h1>", unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e;margin-top:-10px">Simulation live du flux de données capteurs</p>',
                unsafe_allow_html=True)

    MACHINES = {
        "M-001": {"name": "Compresseur A3",  "baseline": dict(vibration_rms=2.1, temperature_motor=72, rpm=1800, pressure_level=6.2, current_phase_avg=17, hours_since_maintenance=95)},
        "M-002": {"name": "Pompe P7",        "baseline": dict(vibration_rms=1.4, temperature_motor=65, rpm=1200, pressure_level=5.5, current_phase_avg=14, hours_since_maintenance=42)},
        "M-003": {"name": "Turbine T2",      "baseline": dict(vibration_rms=4.8, temperature_motor=98, rpm=3000, pressure_level=8.1, current_phase_avg=28, hours_since_maintenance=312)},
        "M-004": {"name": "Convoyeur K1",    "baseline": dict(vibration_rms=3.2, temperature_motor=85, rpm=900,  pressure_level=4.0, current_phase_avg=22, hours_since_maintenance=180)},
    }

    def simulate_reading(baseline, seed=42):
        rng = np.random.default_rng(seed)
        noise = lambda v, s: float(v + rng.normal(0, s))
        return {
            "vibration_rms":        max(0.1, noise(baseline["vibration_rms"], 0.4)),
            "temperature_motor":    max(30,  noise(baseline["temperature_motor"], 3)),
            "rpm":                  max(200, noise(baseline["rpm"], 80)),
            "pressure_level":       max(0.5, noise(baseline["pressure_level"], 0.3)),
            "current_phase_avg":    max(2,   noise(baseline["current_phase_avg"], 1)),
            "hours_since_maintenance": baseline["hours_since_maintenance"],
            "ambient_temp": 22.0,
            "operating_mode": "high_load" if baseline["hours_since_maintenance"] > 200 else "normal",
        }

    auto_refresh = st.toggle("🔄 Actualisation automatique (5s)", value=False)
    if auto_refresh:
        st.caption("Actualisation active…")

    seed_base = int(datetime.now().timestamp() / 5) if auto_refresh else 42

    st.markdown('<div class="section-title">État du Parc Machine</div>', unsafe_allow_html=True)
    fleet_cols = st.columns(4)
    fleet_data = {}

    for idx, (mid, minfo) in enumerate(MACHINES.items()):
        reading = simulate_reading(minfo["baseline"], seed=seed_base + idx)
        hs = health_score(reading)
        proba_sim = float(np.clip((1 - hs) * 1.2 + np.random.default_rng(seed_base+idx+100).normal(0, 0.05), 0, 1))
        rl, rc, rb = risk_label(proba_sim)
        fleet_data[mid] = {"reading": reading, "health": hs, "proba": proba_sim, "risk": rl}

        with fleet_cols[idx]:
            st.markdown(
                f'<div class="fleet-card">'
                f'<div class="fleet-machine-id">{mid}</div>'
                f'<div class="fleet-machine-name">{minfo["name"]}</div>'
                f'<div style="margin:8px 0"><span class="badge {rb}">{rl}</span></div>'
                f'<div class="fleet-metric">Santé : <span>{hs*100:.0f}%</span></div>'
                f'<div class="fleet-metric">Panne 24h : <span style="color:{rc}">{proba_sim*100:.0f}%</span></div>'
                f'<div class="fleet-metric">Vibration : <span>{reading["vibration_rms"]:.2f} mm/s</span></div>'
                f'<div class="fleet-metric">Temp. : <span>{reading["temperature_motor"]:.0f}°C</span></div>'
                f'<div class="fleet-metric">Maintenance : <span>{reading["hours_since_maintenance"]}h</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar comparatif
    st.markdown('<div class="section-title">Radar Comparatif — Capteurs normalisés</div>', unsafe_allow_html=True)
    radar_features = ["vibration_rms", "temperature_motor", "rpm", "pressure_level", "current_phase_avg"]
    radar_ranges   = {"vibration_rms":(0,8), "temperature_motor":(30,130), "rpm":(200,3500), "pressure_level":(0.5,12), "current_phase_avg":(2,35)}
    radar_labels   = ["Vibration","Température","RPM","Pression","Courant"]
    colors_r       = ["#58a6ff","#3fb950","#f85149","#d29922"]

    fig = go.Figure()
    for idx, (mid, minfo) in enumerate(MACHINES.items()):
        vals = fleet_data[mid]["reading"]
        normed = [(vals[f]-radar_ranges[f][0])/(radar_ranges[f][1]-radar_ranges[f][0]) for f in radar_features]
        normed += normed[:1]
        c = colors_r[idx]
        rgb = f"{int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)}"
        fig.add_trace(go.Scatterpolar(
            r=normed, theta=radar_labels+radar_labels[:1],
            fill="toself", name=minfo["name"],
            line=dict(color=c, width=2),
            fillcolor=f"rgba({rgb},0.15)",
        ))
    fig.update_layout(
        polar=dict(bgcolor="#161b22",
                   radialaxis=dict(visible=True, range=[0,1], gridcolor="#30363d", tickfont=dict(color="#8b949e")),
                   angularaxis=dict(gridcolor="#30363d", tickfont=dict(color="#c9d1d9"))),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"),
        height=380, legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(t=30,b=20,l=60,r=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Journal d'alertes
    st.markdown('<div class="section-title">Journal d\'alertes récentes</div>', unsafe_allow_html=True)
    alert_log = [
        {"time": datetime.now()-timedelta(minutes=2),   "machine":"M-003 Turbine T2",   "level":"CRITICAL","msg":"Vibration RMS = 5.1 mm/s · Seuil dépassé (>5.0)",            "cls":"alert-critical"},
        {"time": datetime.now()-timedelta(minutes=7),   "machine":"M-004 Convoyeur K1",  "level":"HIGH",    "msg":"Température moteur = 91°C · +6°C vs seuil nominal",          "cls":"alert-high"},
        {"time": datetime.now()-timedelta(minutes=18),  "machine":"M-001 Compresseur A3","level":"MEDIUM",  "msg":"Courant de phase = 19A · Légère surcharge détectée",          "cls":"alert-medium"},
        {"time": datetime.now()-timedelta(minutes=34),  "machine":"M-003 Turbine T2",    "level":"HIGH",    "msg":"RPM anormal : 3 280 tr/min (max nominal : 3 000)",            "cls":"alert-high"},
        {"time": datetime.now()-timedelta(minutes=52),  "machine":"M-002 Pompe P7",      "level":"LOW",     "msg":"Pression légèrement basse : 5.1 bar · Surveillance OK",       "cls":"alert-ok"},
        {"time": datetime.now()-timedelta(hours=1,minutes=15),"machine":"M-003 Turbine T2","level":"CRITICAL","msg":"hours_since_maintenance = 312h · Maintenance URGENTE",      "cls":"alert-critical"},
        {"time": datetime.now()-timedelta(hours=2),     "machine":"M-004 Convoyeur K1",  "level":"MEDIUM",  "msg":"Vibration en hausse : tendance +0.4 mm/s/h",                 "cls":"alert-medium"},
    ]
    dot_colors = {"CRITICAL":"#f85149","HIGH":"#d29922","MEDIUM":"#58a6ff","LOW":"#3fb950"}
    for al in alert_log:
        bc = dot_colors.get(al["level"], "#8b949e")
        st.markdown(
            f'<div class="{al["cls"]}" style="display:flex;align-items:flex-start;gap:12px">'
            f'<span style="color:{bc};font-size:1.3rem;flex-shrink:0">▲</span>'
            f'<div style="flex:1">'
            f'<div style="display:flex;justify-content:space-between">'
            f'<b style="color:#e6edf3">{al["machine"]}</b>'
            f'<span style="font-size:.72rem;color:#8b949e">{al["time"].strftime("%H:%M:%S")}</span>'
            f'</div>'
            f'<div style="font-size:.85rem;margin-top:3px">{al["msg"]}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    if auto_refresh:
        time.sleep(5)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODÈLES & PERFORMANCES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Modèles & Performances":
    st.markdown("<h1>🤖 Comparaison des Modèles ML</h1>", unsafe_allow_html=True)

    if metrics is None:
        st.markdown('<div class="alert-medium">💡 Entraînez les modèles (<code>python main.py</code>) pour les métriques réelles. Données simulées ci-dessous.</div>', unsafe_allow_html=True)
        metrics = pd.DataFrame([
            {"model":"Logistic Regression","recall_1":0.72,"precision_1":0.61,"f1_1":0.66,"pr_auc":0.71,"roc_auc":0.87,"fn_count":105,"fp_count":224},
            {"model":"Random Forest",      "recall_1":0.81,"precision_1":0.74,"f1_1":0.77,"pr_auc":0.84,"roc_auc":0.93,"fn_count":72, "fp_count":155},
            {"model":"XGBoost",            "recall_1":0.85,"precision_1":0.78,"f1_1":0.81,"pr_auc":0.88,"roc_auc":0.95,"fn_count":57, "fp_count":131},
            {"model":"MLP (Deep Learning)","recall_1":0.79,"precision_1":0.71,"f1_1":0.75,"pr_auc":0.82,"roc_auc":0.91,"fn_count":79, "fp_count":167},
        ])

    mdf = metrics.set_index("model") if "model" in metrics.columns else metrics
    metric_cols = [c for c in ["recall_1","precision_1","f1_1","pr_auc","roc_auc"] if c in mdf.columns]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Meilleur Recall",   f"{mdf['recall_1'].max():.3f}", f'<span class="kpi-delta delta-ok">{mdf["recall_1"].idxmax()}</span>', "#3fb950"), unsafe_allow_html=True)
    c2.markdown(kpi("Meilleur F1",       f"{mdf['f1_1'].max():.3f}",    f'<span class="kpi-delta delta-ok">{mdf["f1_1"].idxmax()}</span>',    "#58a6ff"), unsafe_allow_html=True)
    c3.markdown(kpi("Meilleur ROC-AUC",  f"{mdf['roc_auc'].max():.3f}", f'<span class="kpi-delta delta-ok">{mdf["roc_auc"].idxmax()}</span>', "#d29922"), unsafe_allow_html=True)
    if "fn_count" in mdf.columns:
        c4.markdown(kpi("FN minimum",    f"{int(mdf['fn_count'].min())}",f'<span class="kpi-delta delta-ok">{mdf["fn_count"].idxmin()}</span>',"#f0883e"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_table, tab_bar, tab_radar, tab_cost, tab_critique = st.tabs([
        "📋 Tableau", "📊 Barres", "🕸 Radar", "💰 Analyse Coût", "🔬 Critique"
    ])

    with tab_table:
        display_round = mdf[metric_cols].round(4)
        st.dataframe(
            display_round.style
                .background_gradient(cmap="YlGn", subset=[c for c in ["recall_1","f1_1","pr_auc","roc_auc"] if c in display_round.columns])
                .format("{:.4f}"),
            use_container_width=True, height=200,
        )
        st.caption("**Priorité métier :** `recall_1` > `pr_auc` > `roc_auc` — éviter l'accuracy seule (données déséquilibrées)")

    with tab_bar:
        df_melt = mdf[metric_cols].reset_index().melt(id_vars="model", var_name="Métrique", value_name="Score")
        fig = px.bar(df_melt, x="model", y="Score", color="Métrique", barmode="group", text_auto=".3f",
                      color_discrete_sequence=["#f85149","#58a6ff","#3fb950","#d29922","#bc8cff"])
        plotly_dark(fig, 450)
        fig.update_traces(textfont_size=11, textangle=0)
        fig.update_layout(xaxis_tickangle=-20, bargap=0.15, legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

    with tab_radar:
        fig = go.Figure()
        colors_r2 = ["#58a6ff","#3fb950","#f85149","#bc8cff"]
        for i, (name, row_r) in enumerate(mdf.iterrows()):
            vals = [row_r[m] for m in metric_cols]; vals += vals[:1]
            c = colors_r2[i % len(colors_r2)]
            rgb = f"{int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)}"
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=metric_cols+metric_cols[:1],
                fill="toself", name=str(name),
                line=dict(color=c, width=2),
                fillcolor=f"rgba({rgb},0.15)",
            ))
        fig.update_layout(
            polar=dict(bgcolor="#161b22",
                       radialaxis=dict(visible=True, range=[0.5,1], gridcolor="#30363d", tickfont=dict(color="#8b949e")),
                       angularaxis=dict(gridcolor="#30363d", tickfont=dict(color="#c9d1d9"))),
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"),
            height=450, legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(t=40,b=20,l=60,r=60),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_cost:
        COST_FN = st.slider("Coût d'un FN — panne manquée (€)", 1000, 20000, 4112, 100)
        COST_FP = st.slider("Coût d'un FP — fausse alerte (€)", 50, 2000, 200, 50)
        if "fn_count" in mdf.columns and "fp_count" in mdf.columns:
            cost_df = mdf[["fn_count","fp_count"]].copy()
            cost_df["Coût FN"] = cost_df["fn_count"] * COST_FN
            cost_df["Coût FP"] = cost_df["fp_count"] * COST_FP
            cost_df["Coût Total"] = cost_df["Coût FN"] + cost_df["Coût FP"]
            cost_df = cost_df.reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Pannes manquées (FN)", x=cost_df["model"], y=cost_df["Coût FN"],
                                  marker_color="#f85149", text=cost_df["Coût FN"].apply(lambda v: f"{v:,.0f}€"),
                                  textposition="inside", textfont=dict(color="white")))
            fig.add_trace(go.Bar(name="Fausses alertes (FP)", x=cost_df["model"], y=cost_df["Coût FP"],
                                  marker_color="#58a6ff", text=cost_df["Coût FP"].apply(lambda v: f"{v:,.0f}€"),
                                  textposition="inside", textfont=dict(color="white")))
            fig.update_layout(barmode="stack")
            plotly_dark(fig, 400)
            fig.update_layout(xaxis_tickangle=-15, legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, use_container_width=True)

            best_m = cost_df.set_index("model")["Coût Total"].idxmin()
            savings = cost_df.set_index("model")["Coût Total"].max() - cost_df.set_index("model")["Coût Total"].min()
            st.markdown(f'<div class="alert-ok">💰 <b>Modèle optimal :</b> {best_m} — économise <b>{savings:,.0f} €</b> vs pire modèle</div>', unsafe_allow_html=True)

    with tab_critique:
        st.markdown("""
### Analyse critique — Choix du modèle final

| Critère | LR (Baseline) | Random Forest | **XGBoost** | MLP (DL) |
|:--------|:---:|:---:|:---:|:---:|
| Recall classe 1 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| PR-AUC | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Interprétabilité | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Temps d'inférence | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Robustesse déséquilibre | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

**Conclusion :** XGBoost offre le meilleur compromis sur données tabulaires (~24k obs).
**Deep Learning ≠ toujours supérieur** sur des données de taille modérée (biais/variance).
La Régression Logistique reste précieuse comme baseline interprétable (transparence algorithmique RGPD Art. 22).
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SIMULATEUR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️  Simulateur Machine":
    st.markdown("<h1>⚙️ Simulateur de Prédiction — Temps Réel</h1>", unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e;margin-top:-10px">Saisissez les valeurs capteurs pour une prédiction instantanée</p>', unsafe_allow_html=True)

    if not models:
        st.markdown('<div class="alert-high">⚠️ Aucun modèle disponible. Lancez <code>python main.py</code> puis rechargez.</div>', unsafe_allow_html=True)
        st.stop()

    scenarios = {
        "🟢  Machine en bon état":   dict(vibration_rms=1.8, temperature_motor=68, current_phase_avg=15, pressure_level=5.8, rpm=1600, hours_since_maintenance=45,  ambient_temp=22, operating_mode="normal"),
        "🟡  Usure modérée":          dict(vibration_rms=3.2, temperature_motor=85, current_phase_avg=21, pressure_level=7.2, rpm=2100, hours_since_maintenance=210, ambient_temp=26, operating_mode="high_load"),
        "🔴  Panne imminente":        dict(vibration_rms=5.8, temperature_motor=112,current_phase_avg=30, pressure_level=9.5, rpm=2900, hours_since_maintenance=380, ambient_temp=35, operating_mode="peak"),
        "⚡  Surcharge thermique":     dict(vibration_rms=2.1, temperature_motor=128,current_phase_avg=35, pressure_level=6.1, rpm=1800, hours_since_maintenance=120, ambient_temp=42, operating_mode="peak"),
        "🔧  Déséquilibre mécanique": dict(vibration_rms=6.5, temperature_motor=78, current_phase_avg=19, pressure_level=5.9, rpm=2400, hours_since_maintenance=95,  ambient_temp=23, operating_mode="normal"),
    }

    st.markdown('<div class="section-title">Scénarios prédéfinis</div>', unsafe_allow_html=True)
    selected_sc = st.selectbox("Charger un scénario :", ["— Personnalisé —"] + list(scenarios.keys()))
    sc_vals = scenarios.get(selected_sc)

    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🔩 Mécaniques**")
        vibration_rms           = st.slider("Vibration RMS (mm/s)", 0.5, 8.0,  float(sc_vals["vibration_rms"])           if sc_vals else 2.5,  step=0.1)
        rpm                     = st.slider("RPM", 500, 3500,                   int(sc_vals["rpm"])                       if sc_vals else 1800, step=50)
        hours_since_maintenance = st.slider("Heures depuis maintenance", 0, 500, int(sc_vals["hours_since_maintenance"])  if sc_vals else 120,  step=5)
    with col2:
        st.markdown("**🌡️ Thermiques & Électriques**")
        temperature_motor = st.slider("Température moteur (°C)", 30.0, 140.0, float(sc_vals["temperature_motor"]) if sc_vals else 75.0, step=0.5)
        ambient_temp      = st.slider("Température ambiante (°C)", 10.0, 50.0, float(sc_vals["ambient_temp"])     if sc_vals else 22.0, step=0.5)
        current_phase_avg = st.slider("Courant de phase (A)", 5.0, 40.0,       float(sc_vals["current_phase_avg"]) if sc_vals else 18.0, step=0.5)
    with col3:
        st.markdown("**⚙️ Hydrauliques & Mode**")
        pressure_level = st.slider("Pression (bar)", 2.0, 12.0, float(sc_vals["pressure_level"]) if sc_vals else 6.0, step=0.1)
        operating_mode = st.selectbox("Mode opératoire", ["normal","high_load","peak"],
                                       index=["normal","high_load","peak"].index(sc_vals["operating_mode"]) if sc_vals else 0)
        if st.session_state.role == ROLE_ENGINEER:
            selected_model = list(models.keys())[0]
            st.markdown(
                f'''<div style="margin-top:14px; padding:10px; border:1px solid #30363d; border-radius:10px; background:rgba(255,255,255,0.03);">
                <b>Modèle par défaut :</b> {MODEL_LABELS.get(selected_model, selected_model)}</div>''',
                unsafe_allow_html=True,
            )
        else:
            selected_model = st.selectbox("Modèle ML", list(models.keys()), format_func=lambda k: MODEL_LABELS.get(k, k))

    input_vals = dict(vibration_rms=vibration_rms, temperature_motor=temperature_motor,
                      current_phase_avg=current_phase_avg, pressure_level=pressure_level,
                      rpm=rpm, hours_since_maintenance=hours_since_maintenance,
                      ambient_temp=ambient_temp, operating_mode=operating_mode)

    try:
        proba = predict_proba(input_vals, models, selected_model)
    except Exception as e:
        st.error(f"Erreur prédiction : {e}")
        st.stop()

    hs = health_score(input_vals)
    rl, rc, rb = risk_label(proba)
    temp_delta = temperature_motor - ambient_temp

    st.divider()
    st.markdown('<div class="section-title">Résultat de l\'analyse</div>', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    r1.markdown(kpi("Prob. de panne",   f"{proba*100:.1f}%",  f'<span class="kpi-delta {"delta-up" if proba>0.5 else "delta-ok"}">{rl}</span>', rc), unsafe_allow_html=True)
    r2.markdown(kpi("Score de santé",   f"{hs*100:.0f}%",     f'<span class="kpi-delta {"delta-up" if hs<0.4 else "delta-ok"}">{"Dégradé" if hs<0.4 else "Bon"}</span>', "#3fb950" if hs>0.6 else "#d29922" if hs>0.35 else "#f85149"), unsafe_allow_html=True)
    r3.markdown(kpi("Écart thermique",  f"{temp_delta:.0f}°C", f'<span class="kpi-delta {"delta-up" if temp_delta>65 else "delta-ok"}">{"Surchauffe" if temp_delta>65 else "Normal"}</span>', "#d29922"), unsafe_allow_html=True)
    r4.markdown(kpi("Modèle",           (MODEL_LABELS.get(selected_model,"—")[:16]+"…")[:18], '<span class="kpi-delta delta-ok">Prêt</span>', "#bc8cff"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    recs = {
        "CRITICAL": "🚨 <b>ACTION IMMÉDIATE.</b> Arrêtez la machine — risque d'arrêt non planifié dans les 24h. Inspection complète requise.",
        "HIGH":     "⚠️ <b>Intervention prioritaire.</b> Planifiez une maintenance dans les 4-6h. Réduisez la charge.",
        "MEDIUM":   "🔔 <b>Surveillance renforcée.</b> Planifiez un contrôle dans les 48h.",
        "LOW":      "✅ <b>Machine en bon état.</b> Continuez le plan de maintenance préventive standard.",
    }
    alert_cls = "alert-critical" if rl == "CRITICAL" else "alert-high" if rl == "HIGH" else "alert-medium" if rl == "MEDIUM" else "alert-ok"
    st.markdown(
        f'<div class="{alert_cls}">{recs.get(rl,"")}'
        f'<br><small>Probabilité : <b>{proba*100:.1f}%</b> · Santé : <b>{hs*100:.0f}%</b></small></div>',
        unsafe_allow_html=True,
    )

    col_g, col_b = st.columns([1.2, 1])
    with col_g:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "font": {"size": 42, "color": rc}},
            title={"text": "Probabilité de panne", "font": {"size": 15, "color": "#8b949e"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e", "tickfont": {"color": "#8b949e"}},
                "bar":  {"color": rc, "thickness": 0.25},
                "bgcolor": "#161b22", "bordercolor": "#30363d",
                "steps": [
                    {"range": [0, 30],  "color": "rgba(63,185,80,0.15)"},
                    {"range": [30, 55], "color": "rgba(210,153,34,0.15)"},
                    {"range": [55, 78], "color": "rgba(248,81,73,0.15)"},
                    {"range": [78, 100],"color": "rgba(248,81,73,0.25)"},
                ],
                "threshold": {"line": {"color": "#ffffff", "width": 3}, "value": proba * 100},
            },
        ))
        fig_g.update_layout(height=320, plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                             font=dict(color="#c9d1d9"), margin=dict(t=40,b=0,l=40,r=40))
        st.plotly_chart(fig_g, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Contribution des capteurs</div>', unsafe_allow_html=True)
        contribs = pd.DataFrame({
            "Capteur": ["vibration_rms","temperature_motor","hours_maint.","pressure","rpm"],
            "Risque":  [vibration_rms/8., (temperature_motor-30)/110, hours_since_maintenance/500, pressure_level/12., rpm/3500],
        })
        fig_c = px.bar(contribs, x="Risque", y="Capteur", orientation="h",
                        color="Risque", color_continuous_scale=[[0,"#3fb950"],[0.5,"#d29922"],[1,"#f85149"]])
        plotly_dark(fig_c, 260)
        fig_c.update_layout(coloraxis_showscale=False, margin=dict(t=5,b=5,l=10,r=10))
        st.plotly_chart(fig_c, use_container_width=True)

        with st.expander("Features dérivées"):
            st.json({
                "temp_delta":        round(temp_delta, 2),
                "age_vibration":     round(hours_since_maintenance * vibration_rms, 2),
                "vibration_per_rpm": round(vibration_rms / max(rpm, 1), 5),
                "health_score":      hs,
            })


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Analyse EDA":
    st.markdown("<h1>📊 Analyse Exploratoire des Données</h1>", unsafe_allow_html=True)

    if df is None:
        st.markdown('<div class="alert-critical">Données introuvables.</div>', unsafe_allow_html=True)
        st.stop()

    tab_dist, tab_corr, tab_target, tab_3d, tab_qual = st.tabs([
        "Distributions", "Corrélations", "Cibles", "3D Scatter", "Qualité"
    ])

    with tab_dist:
        feature = st.selectbox("Variable", FEATURES_NUM, key="dist_feat")
        fig = go.Figure()
        for cls, color, label in [(0,"#3fb950","Normal"),(1,"#f85149","Panne")]:
            data = df[df[TARGET]==cls][feature].dropna()
            fig.add_trace(go.Histogram(x=data, name=label, nbinsx=60,
                                        marker_color=color, opacity=0.7, histnorm="probability density"))
        plotly_dark(fig, 420)
        fig.update_layout(barmode="overlay", xaxis_title=feature, yaxis_title="Densité",
                           legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = make_subplots(rows=2, cols=3, subplot_titles=FEATURES_NUM)
        for i, col_name in enumerate(FEATURES_NUM):
            r, c = divmod(i, 3)
            for cls, color, nm in [(0,"#3fb950","Normal"),(1,"#f85149","Panne")]:
                data = df[df[TARGET]==cls][col_name].dropna()
                fig2.add_trace(go.Violin(y=data, name=nm, box_visible=True, meanline_visible=True,
                                          opacity=0.75, fillcolor=color, line_color=color,
                                          showlegend=(i==0)), row=r+1, col=c+1)
        fig2.update_layout(height=560, violinmode="group", plot_bgcolor="#0d1117",
                            paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"),
                            legend=dict(bgcolor="rgba(0,0,0,0)"))
        for ax in fig2.layout:
            if "xaxis" in ax or "yaxis" in ax:
                fig2.layout[ax]["gridcolor"] = "#21262d"
        st.plotly_chart(fig2, use_container_width=True)

    with tab_corr:
        num_cols = FEATURES_NUM + [TARGET]
        if "rul_hours" in df.columns: num_cols.append("rul_hours")
        if "estimated_repair_cost" in df.columns: num_cols.append("estimated_repair_cost")
        corr = df[[c for c in num_cols if c in df.columns]].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                         color_continuous_scale=[[0,"#f85149"],[0.5,"#21262d"],[1,"#3fb950"]],
                         color_continuous_midpoint=0)
        fig.update_layout(height=520, plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                           font=dict(color="#c9d1d9"), margin=dict(t=30,b=10,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        x_var = c1.selectbox("Axe X", FEATURES_NUM, index=0, key="px")
        y_var = c2.selectbox("Axe Y", FEATURES_NUM, index=1, key="py")
        sample = df[[x_var, y_var, TARGET]].dropna().sample(min(4000,len(df)), random_state=42)
        fig3 = px.scatter(sample, x=x_var, y=y_var,
                           color=sample[TARGET].map({0:"Normal",1:"Panne"}),
                           color_discrete_map={"Normal":"#3fb950","Panne":"#f85149"}, opacity=0.5)
        plotly_dark(fig3, 420)
        fig3.update_layout(legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig3, use_container_width=True)

    with tab_target:
        col_a, col_b = st.columns(2)
        with col_a:
            if "failure_type" in df.columns:
                ft = df["failure_type"].value_counts().drop("none", errors="ignore")
                fig = px.bar(x=ft.values, y=ft.index, orientation="h",
                              color=ft.values, color_continuous_scale=[[0,"#58a6ff"],[1,"#f85149"]])
                plotly_dark(fig, 340)
                fig.update_layout(coloraxis_showscale=False, xaxis_title="Nombre")
                st.plotly_chart(fig, use_container_width=True)
        with col_b:
            if "rul_hours" in df.columns:
                fig = go.Figure(go.Histogram(x=df["rul_hours"], nbinsx=60, marker_color="#bc8cff"))
                mean_rul = df["rul_hours"].mean()
                fig.add_vline(x=mean_rul, line_dash="dash", line_color="#d29922",
                               annotation_text=f"µ = {mean_rul:.0f}h", annotation_font_color="#d29922")
                plotly_dark(fig, 340)
                fig.update_layout(xaxis_title="RUL (h)", yaxis_title="Fréquence")
                st.plotly_chart(fig, use_container_width=True)

        if "failure_type" in df.columns and "estimated_repair_cost" in df.columns:
            cost_data = (
                df[df["failure_type"]!="none"]
                  .groupby("failure_type")["estimated_repair_cost"]
                  .agg(["mean","std"]).reset_index()
            )
            cost_data.columns = ["Type","Coût moyen","Écart-type"]
            cost_data = cost_data.sort_values("Coût moyen", ascending=True)
            fig = go.Figure(go.Bar(
                y=cost_data["Type"], x=cost_data["Coût moyen"], orientation="h",
                marker=dict(color=cost_data["Coût moyen"], colorscale=[[0,"#3fb950"],[1,"#f85149"]]),
                error_x=dict(type="data", array=cost_data["Écart-type"], color="#8b949e"),
                text=cost_data["Coût moyen"].apply(lambda v: f"{v:,.0f} €"),
                textposition="outside", textfont=dict(color="#c9d1d9"),
            ))
            plotly_dark(fig, 360)
            fig.update_layout(xaxis_title="Coût moyen (€)", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab_3d:
        axes = st.columns(3)
        x3 = axes[0].selectbox("Axe X", FEATURES_NUM, index=0, key="3dx")
        y3 = axes[1].selectbox("Axe Y", FEATURES_NUM, index=1, key="3dy")
        z3 = axes[2].selectbox("Axe Z", FEATURES_NUM, index=2, key="3dz")
        samp3d = df[[x3,y3,z3,TARGET]].dropna().sample(min(3000,len(df)), random_state=42)
        fig3d = px.scatter_3d(samp3d, x=x3, y=y3, z=z3,
                               color=samp3d[TARGET].map({0:"Normal",1:"Panne"}),
                               color_discrete_map={"Normal":"#3fb950","Panne":"#f85149"},
                               opacity=0.6, size_max=4)
        fig3d.update_layout(
            height=560, paper_bgcolor="#0d1117",
            scene=dict(bgcolor="#0d1117",
                       xaxis=dict(backgroundcolor="#161b22",gridcolor="#30363d",color="#c9d1d9"),
                       yaxis=dict(backgroundcolor="#161b22",gridcolor="#30363d",color="#c9d1d9"),
                       zaxis=dict(backgroundcolor="#161b22",gridcolor="#30363d",color="#c9d1d9")),
            font=dict(color="#c9d1d9"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=20,b=0,l=0,r=0),
        )
        st.plotly_chart(fig3d, use_container_width=True)

    with tab_qual:
        missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        missing = missing[missing > 0]
        if missing.empty:
            st.markdown('<div class="alert-ok">✅ Aucune valeur manquante — Dataset de qualité optimale.</div>', unsafe_allow_html=True)
        else:
            fig = px.bar(x=missing.values, y=missing.index, orientation="h",
                          color=missing.values, color_continuous_scale="Reds")
            plotly_dark(fig, 350)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Statistiques descriptives</div>', unsafe_allow_html=True)
        st.dataframe(df[FEATURES_NUM].describe().T.round(3), use_container_width=True)

        iqr_rows = []
        for col_n in FEATURES_NUM:
            q1, q3 = df[col_n].quantile(0.25), df[col_n].quantile(0.75)
            iqr = q3 - q1
            n_out = ((df[col_n] < q1-1.5*iqr) | (df[col_n] > q3+1.5*iqr)).sum()
            iqr_rows.append({"Variable":col_n,"Q1":round(q1,2),"Q3":round(q3,2),
                              "IQR":round(iqr,2),"Outliers":int(n_out),"% Outliers":round(n_out/len(df)*100,2)})
        st.dataframe(pd.DataFrame(iqr_rows), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — EXPLICABILITÉ
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Explicabilité IA":
    st.markdown("<h1>🔍 Explicabilité — IA Responsable & Transparente</h1>", unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e;margin-top:-10px">SHAP · Feature Importance · Analyse des erreurs · RGPD Art. 22</p>', unsafe_allow_html=True)

    tab_fi, tab_shap, tab_local, tab_fn = st.tabs([
        "Feature Importance", "SHAP — Théorie & Graphiques", "Explication locale", "Analyse FN/FP"
    ])

    with tab_fi:
        feat_imp = None
        if models:
            for key in ["xgboost","random_forest","logistic_regression"]:
                if key not in models: continue
                m = models[key]
                try:
                    clf  = m.named_steps["classifier"]
                    prep = m.named_steps["preprocessor"]
                    names = [n.replace("num__","").replace("cat__","") for n in prep.get_feature_names_out()]
                    imp = (clf.feature_importances_ if hasattr(clf,"feature_importances_")
                            else np.abs(clf.coef_[0]) if hasattr(clf,"coef_") else None)
                    if imp is None: continue
                    feat_imp = pd.DataFrame({"Feature":names,"Importance":imp}).sort_values("Importance",ascending=False).head(15)
                    st.caption(f"Source : {MODEL_LABELS.get(key,key)}")
                    break
                except Exception: continue

        if feat_imp is None:
            feat_imp = pd.DataFrame({
                "Feature":    ["vibration_rms","temperature_motor","age_vibration","hours_since_maintenance","vibration_per_rpm","temp_delta","rpm","pressure_level","current_phase_avg"],
                "Importance": [0.28,0.22,0.18,0.11,0.09,0.05,0.03,0.02,0.01],
            })
            st.markdown('<div class="alert-medium">💡 Importances simulées — Entraînez les modèles pour les valeurs réelles.</div>', unsafe_allow_html=True)

        feat_imp_s = feat_imp.sort_values("Importance", ascending=True)
        fig = go.Figure(go.Bar(
            y=feat_imp_s["Feature"], x=feat_imp_s["Importance"], orientation="h",
            marker=dict(color=feat_imp_s["Importance"],
                        colorscale=[[0,"#21262d"],[0.5,"#58a6ff"],[1,"#f85149"]]),
            text=feat_imp_s["Importance"].apply(lambda v: f"{v:.3f}"),
            textposition="outside", textfont=dict(color="#c9d1d9"),
        ))
        plotly_dark(fig, max(400, len(feat_imp)*28))
        fig.update_layout(xaxis_title="Importance relative", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
**Interprétation métier :**
- **vibration_rms** → principal indicateur de dégradation mécanique (roulements, balourd)
- **temperature_motor** → surchauffe = usure accélérée → pannes thermiques
- **age_vibration** → interaction `heures × vibration` : machine âgée + forte vibration = risque maximal
- **hours_since_maintenance** → risque croissant exponentiellement sans maintenance
- **vibration_per_rpm** → déséquilibre mécanique normalisé par vitesse
        """)

    with tab_shap:
        st.markdown("""
### SHAP — SHapley Additive exPlanations
> Attribution de la contribution de chaque feature à la prédiction, basée sur la théorie des jeux coopératifs.

| Graphique | Utilité |
|:---|:---|
| **Summary plot** | Vue globale : quel feature impacte le plus ? Sens positif/négatif ? |
| **Beeswarm** | Distribution des valeurs SHAP pour toutes les observations |
| **Waterfall** | Décomposition d'une prédiction individuelle |
| **Dependence plot** | Effet d'un feature en fonction de sa valeur |

```bash
python main.py --shap
```
        """)

        shap_imgs = list(RESULTS_DIR.glob("shap_*.png")) if RESULTS_DIR.exists() else []
        if shap_imgs:
            for img_path in sorted(shap_imgs):
                st.image(str(img_path), caption=img_path.stem, use_container_width=True)
        else:
            st.markdown('<div class="section-title">Waterfall SHAP simulé — Scénario CRITIQUE</div>', unsafe_allow_html=True)
            shap_ex = pd.DataFrame({
                "Feature": ["vibration_rms=5.8","temperature_motor=112","age_vibration=2204","hours_maint.=380","operating_mode=peak","rpm=2900","pressure_level=9.5"],
                "SHAP":    [+0.52,+0.38,+0.29,+0.21,+0.14,-0.07,-0.04],
            })
            fig = go.Figure(go.Bar(
                y=shap_ex["Feature"], x=shap_ex["SHAP"], orientation="h",
                marker=dict(color=shap_ex["SHAP"].apply(lambda v: "#f85149" if v>0 else "#3fb950")),
                text=shap_ex["SHAP"].apply(lambda v: f"{v:+.3f}"),
                textposition="outside", textfont=dict(color="#c9d1d9"),
            ))
            plotly_dark(fig, 380)
            fig.add_vline(x=0, line_dash="dash", line_color="#8b949e")
            fig.update_layout(xaxis_title="Contribution SHAP", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab_local:
        if df is None:
            st.error("Données introuvables.")
        else:
            sample_idx = st.slider("Index observation", 0, min(999,len(df)-1), 0)
            row_d = df.iloc[sample_idx]
            st.markdown(f"**Classe réelle :** `{'🔴 PANNE' if row_d[TARGET]==1 else '🟢 NORMAL'}`")
            col_a, col_b, col_c = st.columns(3)
            for i, f in enumerate(FEATURES_NUM):
                [col_a,col_b,col_c][i%3].metric(f, f"{float(row_d.get(f,0)):.2f}")

            if models:
                sel_loc = st.selectbox("Modèle", list(models.keys()),
                                        format_func=lambda k: MODEL_LABELS.get(k,k), key="sel_loc")
                loc_vals = {f: float(row_d.get(f, 0)) for f in FEATURES_NUM}
                loc_vals["ambient_temp"]   = float(row_d.get("ambient_temp", 22.))
                loc_vals["operating_mode"] = str(row_d.get("operating_mode", "normal"))
                try:
                    p_loc = predict_proba(loc_vals, models, sel_loc)
                    rl_loc, rc_loc, rb_loc = risk_label(p_loc)
                    st.metric("Probabilité prédite", f"{p_loc*100:.1f}%")
                    st.markdown(f'<span class="badge {rb_loc}">{rl_loc}</span>', unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Prédiction non disponible : {e}")

    with tab_fn:
        st.markdown("""
### Analyse des erreurs — FN et FP

#### Faux Négatifs (FN) — Pannes manquées
> Impact critique : arrêt non planifié, sécurité, coût ~4 112 €/événement

**Causes typiques :** pannes rares hors distribution · capteur défaillant · panne très précoce

#### Faux Positifs (FP) — Fausses alertes
> Impact modéré : intervention inutile (~200 €), fatigue d'alerte opérateur

**Causes typiques :** pic de vibration transitoire · démarrage à froid · capteur en étalonnage

#### Optimisation du seuil de décision
        """)
        thresholds = np.arange(0.1, 1.0, 0.05)
        th_df = pd.DataFrame({
            "Seuil": thresholds,
            "Recall estimé":    np.clip(1 - (thresholds * 0.6), 0, 1),
            "Précision estimée": np.clip(thresholds * 0.8 + 0.2, 0, 1),
        })
        th_df["F1 estimé"] = (2 * th_df["Recall estimé"] * th_df["Précision estimée"] /
                               (th_df["Recall estimé"] + th_df["Précision estimée"]).clip(lower=1e-9))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=th_df["Seuil"], y=th_df["Recall estimé"],
                                  name="Recall", line=dict(color="#f85149", width=2)))
        fig.add_trace(go.Scatter(x=th_df["Seuil"], y=th_df["Précision estimée"],
                                  name="Précision", line=dict(color="#3fb950", width=2)))
        fig.add_trace(go.Scatter(x=th_df["Seuil"], y=th_df["F1 estimé"],
                                  name="F1-Score", line=dict(color="#58a6ff", width=2, dash="dot")))
        fig.add_vline(x=0.5, line_dash="dash", line_color="#8b949e",
                       annotation_text="Seuil 0.5", annotation_font_color="#8b949e")
        fig.add_vline(x=0.35, line_dash="dot", line_color="#d29922",
                       annotation_text="Recommandé 0.35", annotation_font_color="#d29922")
        plotly_dark(fig, 360)
        fig.update_layout(xaxis_title="Seuil de décision", yaxis_title="Score",
                           legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Recommandation :** Seuil ~0.35 pour la maintenance industrielle — maximise le recall (moins de pannes manquées).")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — USE CASES MÉTIER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💼  Use Cases Métier":
    st.markdown("<h1>💼 Use Cases Métier — Applications Industrielles</h1>", unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e;margin-top:-10px">Valeur métier · ROI · Scénarios d\'usage réel · Architecture déploiement</p>', unsafe_allow_html=True)

    tab_uc, tab_roi, tab_arch, tab_api = st.tabs([
        "Use Cases", "Calcul ROI", "Architecture", "Démo API"
    ])

    with tab_uc:
        use_cases = [
            {
                "icon":"🔧","color":"#f85149","title":"Maintenance préventive intelligente","who":"Responsable Maintenance",
                "problem":"Les pannes non planifiées coûtent ~4 112 €/événement et causent des arrêts de production.",
                "solution":"Le modèle prédit une panne 24h à l'avance → intervention préventive planifiée.",
                "impact":"Réduction de 40-60% des arrêts non planifiés · ROI : 180 000 €/an (parc 50 machines)",
                "api":"POST /predict + webhook alerte",
            },
            {
                "icon":"📊","color":"#58a6ff","title":"Monitoring de flotte en temps réel","who":"DSI / Ingénieur IoT",
                "problem":"Impossible de surveiller manuellement 100+ machines simultanément.",
                "solution":"Flux capteurs → API /predict/batch → tableau de bord centralisé.",
                "impact":"Surveillance 24/7 automatisée · Détection anomalie en < 30 secondes",
                "api":"POST /predict/batch (jusqu'à 1000 obs.)",
            },
            {
                "icon":"💰","color":"#3fb950","title":"Optimisation des stocks pièces","who":"Responsable Supply Chain",
                "problem":"Sur-stockage de pièces de rechange vs ruptures lors de pannes.",
                "solution":"Prédiction de défaillance type → anticipation commande pièce spécifique.",
                "impact":"Réduction stock de 30% · Zéro rupture critique sur pièces surveillées",
                "api":"POST /predict + failure_type prédit",
            },
            {
                "icon":"🛡️","color":"#d29922","title":"Conformité & traçabilité (ISO/RGPD)","who":"Responsable Qualité / Auditeur",
                "problem":"Obligation de justifier les décisions de maintenance (ISO 55001, RGPD Art.22).",
                "solution":"Explications SHAP + logs horodatés de chaque prédiction via l'API.",
                "impact":"Traçabilité 100% · Rapport d'audit automatique · Conformité réglementaire",
                "api":"GET /model-info + POST /predict (timestamp inclus)",
            },
            {
                "icon":"🏭","color":"#bc8cff","title":"Intégration SCADA/MES","who":"Automaticien / Intégrateur",
                "problem":"Les automates SCADA ne disposent pas de capacité de prédiction ML.",
                "solution":"API REST intégrée dans la boucle SCADA → prédiction déclenchée à chaque cycle.",
                "impact":"Prédiction en < 50ms · Compatible OPC-UA, Modbus TCP, MQTT",
                "api":"POST /predict (latence ~15ms avec XGBoost)",
            },
        ]

        for uc in use_cases:
            st.markdown(
                f'<div style="background:#161b22;border:1px solid #30363d;border-left:4px solid {uc["color"]};'
                f'border-radius:10px;padding:20px;margin:12px 0">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">'
                f'<span style="font-size:1.8rem">{uc["icon"]}</span>'
                f'<div><div style="font-size:1.05rem;font-weight:700;color:#e6edf3">{uc["title"]}</div>'
                f'<div style="font-size:.78rem;color:#8b949e">👤 {uc["who"]}</div></div></div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">'
                f'<div><div style="font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px">Problème</div>'
                f'<div style="font-size:.85rem;color:#c9d1d9;margin-top:4px">{uc["problem"]}</div></div>'
                f'<div><div style="font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px">Solution IA</div>'
                f'<div style="font-size:.85rem;color:#c9d1d9;margin-top:4px">{uc["solution"]}</div></div>'
                f'<div><div style="font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px">Impact</div>'
                f'<div style="font-size:.85rem;color:{uc["color"]};margin-top:4px;font-weight:600">{uc["impact"]}</div>'
                f'<div style="font-size:.72rem;color:#484f58;margin-top:4px"><code>{uc["api"]}</code></div>'
                f'</div></div></div>',
                unsafe_allow_html=True,
            )

    with tab_roi:
        st.markdown('<div class="section-title">Calculateur ROI — Maintenance Prédictive</div>', unsafe_allow_html=True)
        col_i, col_o = st.columns(2)
        with col_i:
            n_machines_roi = st.slider("Nombre de machines surveillées", 10, 500, 50)
            pannes_par_an  = st.slider("Pannes/machine/an (sans système)", 2, 20, 6)
            cout_panne     = st.slider("Coût moyen d'une panne non planifiée (€)", 1000, 30000, 4112, 100)
            cout_fp        = st.slider("Coût d'une intervention préventive inutile (€)", 100, 2000, 200, 50)
            taux_detection = st.slider("Taux de détection (Recall %)", 50, 95, 82)
            taux_fp_rate   = st.slider("Taux de fausses alertes (FP rate %)", 5, 40, 15)
            cout_deploy    = st.slider("Coût déploiement annuel (€)", 5000, 100000, 25000, 1000)

        with col_o:
            pannes_tot      = n_machines_roi * pannes_par_an
            pannes_evitees  = pannes_tot * (taux_detection / 100)
            pannes_manquees = pannes_tot * (1 - taux_detection / 100)
            fausses_alertes = pannes_tot * (taux_fp_rate / 100)
            gain_brut       = pannes_evitees * cout_panne
            cout_fn         = pannes_manquees * cout_panne
            cout_fp_total   = fausses_alertes * cout_fp
            roi_net         = gain_brut - cout_fn - cout_fp_total - cout_deploy
            roi_pct         = (roi_net / cout_deploy * 100) if cout_deploy > 0 else 0

            r1c, r2c = st.columns(2)
            r1c.metric("Pannes évitées / an",  f"{pannes_evitees:.0f}")
            r2c.metric("Économies brutes",      f"{gain_brut:,.0f} €")
            r1c.metric("Pannes manquées (FN)",  f"{pannes_manquees:.0f}")
            r2c.metric("Coût résiduel",         f"{cout_fn+cout_fp_total:,.0f} €")
            r1c.metric("Fausses alertes / an",  f"{fausses_alertes:.0f}")
            r2c.metric("ROI Net annuel",        f"{roi_net:,.0f} €",
                        delta=f"{roi_pct:.0f}% sur investissement",
                        delta_color="normal" if roi_net > 0 else "inverse")

        fig_roi = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative","relative","relative","relative","total"],
            x=["Gains (pannes évitées)","Coût FN (manquées)","Coût FP (fausses alertes)","Coût déploiement","ROI Net"],
            y=[gain_brut,-cout_fn,-cout_fp_total,-cout_deploy,0],
            connector={"line":{"color":"#30363d"}},
            increasing={"marker":{"color":"#3fb950"}},
            decreasing={"marker":{"color":"#f85149"}},
            totals={"marker":{"color":"#58a6ff"}},
            text=[f"{gain_brut:,.0f}€",f"-{cout_fn:,.0f}€",f"-{cout_fp_total:,.0f}€",f"-{cout_deploy:,.0f}€",f"{roi_net:,.0f}€"],
            textposition="outside", textfont=dict(color="#c9d1d9"),
        ))
        plotly_dark(fig_roi, 400)
        fig_roi.update_layout(yaxis_title="Montant (€)", showlegend=False)
        st.plotly_chart(fig_roi, use_container_width=True)

    with tab_arch:
        st.markdown("""
### Architecture de déploiement recommandée

```
┌──────────────────────────────────────────────────┐
│              TERRAIN (Edge / IoT)                │
│  [Capteurs IoT] → [Gateway/PLC] → MQTT/OPC-UA   │
└─────────────────────┬────────────────────────────┘
                      │ données capteurs temps réel
┌─────────────────────▼────────────────────────────┐
│             COUCHE INFÉRENCE ML                  │
│  FastAPI REST  ·  /predict  ·  /predict/batch    │
│  XGBoost Pipeline  ·  ~15ms  ·  Docker           │
└──────────┬──────────────────┬────────────────────┘
           │ résultats        │ logs + métriques
┌──────────▼──────┐  ┌────────▼──────────────────┐
│ ALERTES/ACTIONS │  │  MONITORING & STOCKAGE    │
│ Webhook SCADA   │  │  MinIO S3 · MLflow        │
│ Email / SMS     │  │  Dérive modèle            │
│ GMAO ticket     │  │  Logs horodatés           │
└─────────────────┘  └───────────────────────────┘
                                  │
┌─────────────────────────────────▼──────────────┐
│           DASHBOARD DÉCISIONNEL                │
│  Streamlit · Vue flotte · Alertes · Simulation │
└────────────────────────────────────────────────┘
```

| Couche | Technologie | Rôle |
|:---|:---|:---|
| ML | XGBoost + scikit-learn Pipeline | Inférence anti-leakage |
| API | FastAPI + Pydantic v2 | REST, validation, Swagger |
| Dashboard | Streamlit + Plotly | Décisionnel interactif |
| Stockage | MinIO S3-compatible | Modèles, données, résultats |
| Déploiement | Docker + uvicorn | Production containerisée |
| Explainability | SHAP | Transparence algorithmique |
        """)

    with tab_api:
        st.markdown('<div class="section-title">Exemples d\'appels API — Swagger : http://localhost:8000/docs</div>', unsafe_allow_html=True)

        examples = {
            "✅ Machine normale (prediction=0)": {
                "curl": 'curl -X POST http://localhost:8000/predict \\\n  -H "Content-Type: application/json" \\\n  -d \'{"vibration_rms":1.8,"temperature_motor":68,"current_phase_avg":15,"pressure_level":5.8,"rpm":1600,"hours_since_maintenance":45,"ambient_temp":22,"operating_mode":"normal"}\'',
                "expected": '{"prediction":0,"probability_failure":0.08,"risk_level":"LOW","health_score":0.87,"model_used":"xgboost"}',
            },
            "🔴 Panne imminente (prediction=1)": {
                "curl": 'curl -X POST http://localhost:8000/predict \\\n  -H "Content-Type: application/json" \\\n  -d \'{"vibration_rms":5.8,"temperature_motor":112,"current_phase_avg":30,"pressure_level":9.5,"rpm":2900,"hours_since_maintenance":380,"ambient_temp":35,"operating_mode":"peak"}\'',
                "expected": '{"prediction":1,"probability_failure":0.91,"risk_level":"CRITICAL","health_score":0.12,"model_used":"xgboost"}',
            },
            "📋 Batch — flotte 2 machines": {
                "curl": 'curl -X POST http://localhost:8000/predict/batch \\\n  -H "Content-Type: application/json" \\\n  -d \'{"observations":[{"vibration_rms":1.8,"temperature_motor":68,"current_phase_avg":15,"pressure_level":5.8,"rpm":1600,"hours_since_maintenance":45,"ambient_temp":22,"operating_mode":"normal"},{"vibration_rms":5.8,"temperature_motor":112,"current_phase_avg":30,"pressure_level":9.5,"rpm":2900,"hours_since_maintenance":380,"ambient_temp":35,"operating_mode":"peak"}]}\'',
                "expected": '{"count":2,"errors":0,"results":[...]}',
            },
            "🏥 Health check": {
                "curl": "curl http://localhost:8000/health",
                "expected": '{"status":"ok","model_loaded":true,"model_name":"xgboost","uptime_seconds":42.3}',
            },
            "ℹ️ Infos modèle actif": {
                "curl": "curl http://localhost:8000/model-info",
                "expected": '{"active_model":"xgboost","task":"binary_classification","target":"failure_within_24h","threshold":0.5}',
            },
            "💡 Use cases disponibles": {
                "curl": "curl http://localhost:8000/use-cases",
                "expected": '{"use_cases":[{"id":"maintenance_preventive","title":"Maintenance préventive intelligente",...}]}',
            },
            "🔬 Recommandations par niveau": {
                "curl": "curl http://localhost:8000/recommendations/CRITICAL",
                "expected": '{"risk_level":"CRITICAL","action":"Arrêt immédiat de la machine","delay":"< 1h","cost_estimate":4112}',
            },
        }

        for name, ex in examples.items():
            with st.expander(name, expanded=(name == "🔴 Panne imminente (prediction=1)")):
                st.code(ex["curl"], language="bash")
                st.code(ex["expected"], language="json")

        st.markdown('<div class="alert-ok">📖 Documentation Swagger interactive : <b>http://localhost:8000/docs</b> · ReDoc : <b>http://localhost:8000/redoc</b></div>', unsafe_allow_html=True)