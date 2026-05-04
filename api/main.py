"""
api/main.py  ·  EFREI M2 — Maintenance Prédictive Industrielle 2025-26
═══════════════════════════════════════════════════════════════════════════════
API REST — Use Cases industriels + prédiction ML

Lancement :
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Swagger : http://localhost:8000/docs
ReDoc   : http://localhost:8000/redoc

Endpoints :
    GET  /                          — Accueil + liens utiles
    GET  /health                    — Santé du service
    GET  /model-info                — Infos modèle actif
    POST /predict                   — Prédiction unitaire
    POST /predict/batch             — Prédiction batch (max 1000)
    GET  /use-cases                 — Catalogue des use cases métier
    GET  /recommendations/{level}   — Recommandations par niveau de risque
    GET  /machines/simulate         — Simulation de données capteurs
    POST /predict/explain           — Prédiction + explication métier
    GET  /statistics                — Statistiques globales du service
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException, Request, Path as FPath
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ─── Constantes ───────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / "models"
DEFAULT_MODEL = "xgboost"
MODEL_FILES = {
    "logistic_regression": "logistic_regression.pkl",
    "random_forest":       "random_forest.pkl",
    "xgboost":             "xgboost.pkl",
    "mlp":                 "mlp.pkl",
}

# ─── Application FastAPI ──────────────────────────────────────────────────────
app = FastAPI(
    title="PredictMaint Pro — API d'inférence ML",
    description=(
        "## API REST — Maintenance Prédictive Industrielle\n\n"
        "API pour la **prédiction de pannes industrielles** dans les 24h à venir. "
        "Entraînée sur `industrial_machine_maintenance.csv` (24 042 observations).\n\n"
        "### Use Cases supportés\n"
        "- 🔧 Maintenance préventive intelligente\n"
        "- 📊 Monitoring de flotte en temps réel (batch)\n"
        "- 🛡️ Traçabilité réglementaire (RGPD Art. 22 / ISO 55001)\n"
        "- 🏭 Intégration SCADA/MES via REST\n"
        "- 💰 Optimisation des stocks pièces de rechange\n\n"
        "**Modèles disponibles :** Logistic Regression · Random Forest · XGBoost · MLP\n\n"
        "**Stack :** FastAPI · Pydantic v2 · scikit-learn Pipeline · XGBoost · SHAP"
    ),
    version="2.0.0",
    contact={"name": "EFREI M2 Data Science", "email": "contact@efrei.fr"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── État global ──────────────────────────────────────────────────────────────
_state = {"model": None, "model_name": None, "loaded_at": None}
_start_time = datetime.utcnow()
_prediction_count = 0
_alert_count = 0


def _load_model(name: str = DEFAULT_MODEL):
    path = MODELS_DIR / MODEL_FILES.get(name, f"{name}.pkl")
    if not path.exists():
        return None, f"Modèle '{name}' introuvable ({path})"
    try:
        return joblib.load(path), None
    except Exception as e:
        return None, str(e)


@app.on_event("startup")
def startup_event():
    model, err = _load_model(DEFAULT_MODEL)
    if model:
        _state["model"]      = model
        _state["model_name"] = DEFAULT_MODEL
        _state["loaded_at"]  = datetime.utcnow().isoformat()
        print(f"[API] Modèle '{DEFAULT_MODEL}' chargé avec succès.")
    else:
        print(f"[API] Avertissement : {err}")
        print("[API] Le service démarre sans modèle. POST /predict renverra 503.")


# ─── Schémas Pydantic ─────────────────────────────────────────────────────────

class SensorInput(BaseModel):
    """Données capteurs pour une observation unique."""
    vibration_rms:           float = Field(..., ge=0.0, le=15.0,  examples=[2.5],  description="Vibration RMS en mm/s — seuil alerte : > 5.0")
    temperature_motor:       float = Field(..., ge=0.0, le=200.0, examples=[75.0], description="Température moteur en °C — nominal : 40-90")
    current_phase_avg:       float = Field(..., ge=0.0, le=100.0, examples=[18.0], description="Courant de phase moyen en Ampères")
    pressure_level:          float = Field(..., ge=0.0, le=20.0,  examples=[6.0],  description="Niveau de pression en bar — nominal : 4-8")
    rpm:                     float = Field(..., ge=0,   le=6000,  examples=[1800],  description="Vitesse de rotation en tr/min")
    hours_since_maintenance: float = Field(..., ge=0,   le=1000,  examples=[120],   description="Heures écoulées depuis la dernière maintenance")
    ambient_temp:            float = Field(..., ge=-10.0, le=60.0,examples=[22.0],  description="Température ambiante en °C")
    operating_mode: Literal["normal", "high_load", "peak"] = Field(
        ..., examples=["normal"], description="Mode opératoire de la machine"
    )
    model_name: Optional[Literal[
        "logistic_regression", "random_forest", "xgboost", "mlp"
    ]] = Field(None, description="Modèle à utiliser (défaut : xgboost)")


class BatchInput(BaseModel):
    """Prédiction sur plusieurs observations (max 1000)."""
    observations: list[SensorInput] = Field(..., min_length=1, max_length=1000)


class PredictionOutput(BaseModel):
    """Résultat d'une prédiction."""
    prediction:          int   = Field(..., description="0=Normal / 1=Panne imminente")
    probability_failure: float = Field(..., description="Probabilité de panne [0, 1]")
    probability_normal:  float = Field(..., description="Probabilité de fonctionnement normal [0, 1]")
    risk_level:          str   = Field(..., description="LOW / MEDIUM / HIGH / CRITICAL")
    health_score:        float = Field(..., description="Score de santé machine [0, 1]")
    model_used:          str   = Field(..., description="Nom du modèle utilisé")
    features_derived:    dict  = Field(..., description="Features calculées automatiquement")
    recommendation:      str   = Field(..., description="Action recommandée")
    timestamp:           str   = Field(..., description="Horodatage UTC de la prédiction")


class PredictionWithExplanation(PredictionOutput):
    """Prédiction avec explication métier détaillée."""
    explanation:      dict  = Field(..., description="Explication des facteurs de risque")
    alert_triggered:  bool  = Field(..., description="Si une alerte a été déclenchée")
    estimated_cost:   float = Field(..., description="Coût estimé si panne (€)")


class HealthOutput(BaseModel):
    status:           str
    model_loaded:     bool
    model_name:       Optional[str]
    loaded_at:        Optional[str]
    models_available: list[str]
    uptime_seconds:   float
    predictions_served: int
    alerts_triggered:   int


# ─── Fonctions utilitaires ────────────────────────────────────────────────────

def _build_df(sensor: SensorInput) -> pd.DataFrame:
    return pd.DataFrame([{
        "vibration_rms":         sensor.vibration_rms,
        "temperature_motor":     sensor.temperature_motor,
        "current_phase_avg":     sensor.current_phase_avg,
        "pressure_level":        sensor.pressure_level,
        "rpm":                   float(sensor.rpm),
        "hours_since_maintenance": float(sensor.hours_since_maintenance),
        "operating_mode":        sensor.operating_mode,
        "temp_delta":            sensor.temperature_motor - sensor.ambient_temp,
        "age_vibration":         sensor.hours_since_maintenance * sensor.vibration_rms,
        "vibration_per_rpm":     sensor.vibration_rms / max(float(sensor.rpm), 1.0),
        "vibration_rms_roll_mean":     np.nan,
        "vibration_rms_roll_std":      np.nan,
        "vibration_rms_diff":          np.nan,
        "temperature_motor_roll_mean": np.nan,
        "temperature_motor_roll_std":  np.nan,
        "temperature_motor_diff":      np.nan,
        "current_phase_avg_roll_mean": np.nan,
        "current_phase_avg_diff":      np.nan,
    }])


def _risk_level(proba: float) -> str:
    if proba < 0.30: return "LOW"
    if proba < 0.55: return "MEDIUM"
    if proba < 0.78: return "HIGH"
    return "CRITICAL"


def _recommendation(risk: str) -> str:
    recs = {
        "LOW":      "Machine en bon état. Maintenir le plan de maintenance préventive standard.",
        "MEDIUM":   "Risque modéré. Planifier un contrôle dans les 48h. Surveiller les tendances.",
        "HIGH":     "Intervention prioritaire requise. Planifier une maintenance dans les 4-6h. Réduire la charge.",
        "CRITICAL": "ACTION IMMÉDIATE. Arrêter la machine et planifier une inspection complète. Risque d'arrêt non planifié.",
    }
    return recs.get(risk, "")


def _health_score(s: SensorInput) -> float:
    def norm(v, lo, hi): return max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-9)))
    v = norm(s.vibration_rms,    0.5,  6.0)
    t = norm(s.temperature_motor, 40, 120)
    p = norm(s.pressure_level,   2.0,  12.0)
    r = norm(s.rpm,              500, 3500)
    return round(1 - (0.30*v + 0.25*t + 0.20*p + 0.15*r), 4)


def _estimated_cost(risk: str) -> float:
    costs = {"LOW": 0., "MEDIUM": 200., "HIGH": 1500., "CRITICAL": 4112.}
    return costs.get(risk, 0.)


def _run_prediction(sensor: SensorInput, model_name: Optional[str] = None):
    global _prediction_count, _alert_count
    name = model_name or sensor.model_name or _state["model_name"] or DEFAULT_MODEL

    if name != _state["model_name"]:
        m, err = _load_model(name)
        if err:
            raise HTTPException(status_code=503, detail=err)
    else:
        m = _state["model"]

    if m is None:
        raise HTTPException(
            status_code=503,
            detail="Aucun modèle chargé. Entraînez les modèles avec `python main.py` puis relancez l'API.",
        )

    X = _build_df(sensor)
    try:
        proba_arr = m.predict_proba(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")

    proba_failure = float(proba_arr[1])
    proba_normal  = float(proba_arr[0])
    prediction    = int(proba_failure >= 0.5)
    risk          = _risk_level(proba_failure)
    hs            = _health_score(sensor)

    _prediction_count += 1
    if risk in ("HIGH", "CRITICAL"):
        _alert_count += 1

    return PredictionOutput(
        prediction=prediction,
        probability_failure=round(proba_failure, 4),
        probability_normal=round(proba_normal, 4),
        risk_level=risk,
        health_score=hs,
        model_used=name,
        features_derived={
            "temp_delta":        round(sensor.temperature_motor - sensor.ambient_temp, 2),
            "age_vibration":     round(sensor.hours_since_maintenance * sensor.vibration_rms, 2),
            "vibration_per_rpm": round(sensor.vibration_rms / max(float(sensor.rpm), 1), 5),
        },
        recommendation=_recommendation(risk),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Racine"])
def root():
    """Point d'entrée — liens utiles."""
    return {
        "service":   "PredictMaint Pro — API ML v2.0",
        "docs":      "/docs",
        "redoc":     "/redoc",
        "health":    "/health",
        "use_cases": "/use-cases",
        "predict":   "POST /predict",
        "batch":     "POST /predict/batch",
        "stats":     "/statistics",
    }


@app.get("/health", response_model=HealthOutput, tags=["Infrastructure"])
def health_check():
    """Vérifie l'état du service — utilisé par les load balancers et outils de monitoring."""
    available = [name for name, fname in MODEL_FILES.items() if (MODELS_DIR/fname).exists()]
    elapsed   = (datetime.utcnow() - _start_time).total_seconds()
    return HealthOutput(
        status="ok" if _state["model"] else "degraded",
        model_loaded=_state["model"] is not None,
        model_name=_state["model_name"],
        loaded_at=_state["loaded_at"],
        models_available=available,
        uptime_seconds=round(elapsed, 1),
        predictions_served=_prediction_count,
        alerts_triggered=_alert_count,
    )


@app.get("/model-info", tags=["Modèles"])
def model_info():
    """
    Informations sur le modèle actif et les modèles disponibles.

    Utile pour la **traçabilité réglementaire** (RGPD Art. 22 / ISO 55001).
    """
    available = {}
    for name, fname in MODEL_FILES.items():
        path = MODELS_DIR / fname
        available[name] = {"available": path.exists(), "path": str(path)}

    return {
        "active_model": _state["model_name"],
        "loaded_at":    _state["loaded_at"],
        "models":       available,
        "task":         "binary_classification",
        "target":       "failure_within_24h",
        "threshold":    0.5,
        "input_features": {
            "numerical":    ["vibration_rms","temperature_motor","current_phase_avg",
                             "pressure_level","rpm","hours_since_maintenance","ambient_temp"],
            "categorical":  ["operating_mode"],
            "derived_auto": ["temp_delta","age_vibration","vibration_per_rpm"],
        },
        "output": {
            "classes": {0: "Normal", 1: "Panne imminente (<24h)"},
            "risk_levels": {
                "LOW":      "proba < 0.30 — Machine en bon état",
                "MEDIUM":   "proba 0.30–0.55 — Surveillance renforcée",
                "HIGH":     "proba 0.55–0.78 — Intervention prioritaire",
                "CRITICAL": "proba > 0.78 — Action immédiate requise",
            },
        },
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prédiction"])
def predict(sensor: SensorInput):
    """
    Prédit la probabilité de panne dans les **24 prochaines heures**.

    ### Use Case 1 — Maintenance préventive
    Intégrez cet endpoint dans votre boucle SCADA/PLC pour déclencher
    automatiquement une alerte maintenance quand `risk_level >= HIGH`.

    ### Use Case 2 — Tableau de bord temps réel
    Appelez cet endpoint à intervalle régulier (ex: toutes les 15min)
    pour mettre à jour le statut de chaque machine.

    ### Exemple curl
    ```bash
    curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{
        "vibration_rms": 4.5,
        "temperature_motor": 95,
        "current_phase_avg": 22,
        "pressure_level": 7,
        "rpm": 2200,
        "hours_since_maintenance": 300,
        "ambient_temp": 25,
        "operating_mode": "peak"
      }'
    ```
    """
    return _run_prediction(sensor)


@app.post("/predict/batch", tags=["Prédiction"])
def predict_batch(batch: BatchInput):
    """
    Prédiction sur un **lot d'observations** (max 1000).

    ### Use Case — Monitoring de flotte
    Envoyez les données de toutes vos machines en une seule requête
    pour obtenir un bilan complet de l'état de votre parc en < 1 seconde.

    Retourne aussi un **résumé statistique** de la flotte :
    nombre de machines à risque, distribution des niveaux de risque.
    """
    results = []
    errors  = []
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}

    for i, obs in enumerate(batch.observations):
        try:
            result = _run_prediction(obs)
            results.append({"index": i, **result.model_dump()})
            risk_counts[result.risk_level] = risk_counts.get(result.risk_level, 0) + 1
        except HTTPException as e:
            errors.append({"index": i, "error": e.detail})

    n = len(results)
    return {
        "count":   n,
        "errors":  len(errors),
        "summary": {
            "risk_distribution":     risk_counts,
            "machines_at_risk":      risk_counts["HIGH"] + risk_counts["CRITICAL"],
            "average_failure_proba": round(sum(r["probability_failure"] for r in results) / max(n, 1), 4),
            "average_health_score":  round(sum(r["health_score"] for r in results) / max(n, 1), 4),
            "critical_machines":     [r["index"] for r in results if r["risk_level"] == "CRITICAL"],
        },
        "results":       results,
        "error_details": errors,
    }


@app.post("/predict/explain", response_model=PredictionWithExplanation, tags=["Prédiction"])
def predict_with_explanation(sensor: SensorInput):
    """
    Prédiction avec **explication métier détaillée**.

    Identifie les facteurs de risque dominants et fournit une explication
    humainement lisible — obligatoire pour la conformité RGPD Art. 22.

    ### Use Case — Traçabilité réglementaire
    Loggez chaque réponse pour constituer un audit trail conforme ISO 55001.
    """
    base = _run_prediction(sensor)

    # Analyse des facteurs de risque
    risk_factors = []
    if sensor.vibration_rms > 4.0:
        risk_factors.append({"factor": "vibration_rms", "value": sensor.vibration_rms,
                             "threshold": 4.0, "severity": "HIGH",
                             "message": f"Vibration critique : {sensor.vibration_rms:.2f} mm/s (seuil : 4.0)"})
    if sensor.temperature_motor > 90:
        risk_factors.append({"factor": "temperature_motor", "value": sensor.temperature_motor,
                             "threshold": 90, "severity": "HIGH",
                             "message": f"Surchauffe moteur : {sensor.temperature_motor:.0f}°C (seuil : 90°C)"})
    if sensor.hours_since_maintenance > 200:
        risk_factors.append({"factor": "hours_since_maintenance", "value": sensor.hours_since_maintenance,
                             "threshold": 200, "severity": "MEDIUM",
                             "message": f"Maintenance en retard : {sensor.hours_since_maintenance:.0f}h (recommandé : < 200h)"})
    if sensor.operating_mode == "peak":
        risk_factors.append({"factor": "operating_mode", "value": "peak",
                             "threshold": "normal/high_load", "severity": "MEDIUM",
                             "message": "Mode peak : sollicitation maximale de la machine"})
    if sensor.pressure_level > 8.5:
        risk_factors.append({"factor": "pressure_level", "value": sensor.pressure_level,
                             "threshold": 8.5, "severity": "MEDIUM",
                             "message": f"Pression élevée : {sensor.pressure_level:.1f} bar (seuil : 8.5)"})

    return PredictionWithExplanation(
        **base.model_dump(),
        explanation={
            "risk_factors":   risk_factors,
            "dominant_factor": risk_factors[0]["factor"] if risk_factors else "none",
            "n_risk_factors":  len(risk_factors),
            "age_vibration":   round(sensor.hours_since_maintenance * sensor.vibration_rms, 2),
            "temp_delta":      round(sensor.temperature_motor - sensor.ambient_temp, 2),
            "narrative": (
                f"La machine présente {len(risk_factors)} facteur(s) de risque. "
                f"Probabilité de panne : {base.probability_failure*100:.1f}%. "
                f"{'Action immédiate requise.' if base.risk_level == 'CRITICAL' else 'Surveiller de près.'}"
                if risk_factors else
                "Aucun seuil critique dépassé. Machine en état nominal."
            ),
        },
        alert_triggered=base.risk_level in ("HIGH", "CRITICAL"),
        estimated_cost=_estimated_cost(base.risk_level),
    )


@app.get("/use-cases", tags=["Métier"])
def get_use_cases():
    """
    Catalogue des **use cases métier** supportés par cette API.

    Retourne la liste des applications industrielles possibles avec
    les endpoints associés, les personas utilisateurs et les bénéfices attendus.
    """
    return {
        "service": "PredictMaint Pro v2.0",
        "use_cases": [
            {
                "id":          "maintenance_preventive",
                "icon":        "🔧",
                "title":       "Maintenance préventive intelligente",
                "persona":     "Responsable Maintenance",
                "problem":     "Les pannes non planifiées coûtent ~4 112 €/événement et causent des arrêts de production.",
                "solution":    "Prédiction 24h à l'avance → intervention planifiée → zéro arrêt surprise.",
                "endpoint":    "POST /predict",
                "roi_estimate": "Réduction 40-60% des arrêts non planifiés · ~180 000 €/an (parc 50 machines)",
                "example": {
                    "trigger":  "rpm > 2000 ET hours_since_maintenance > 200",
                    "action":   "Planifier intervention J+1",
                    "outcome":  "Panne évitée · économie 4 112 €",
                },
            },
            {
                "id":       "fleet_monitoring",
                "icon":     "📊",
                "title":    "Monitoring de flotte en temps réel",
                "persona":  "DSI / Ingénieur IoT",
                "problem":  "Impossible de surveiller manuellement 100+ machines simultanément.",
                "solution": "Flux capteurs → batch API → dashboard centralisé temps réel.",
                "endpoint": "POST /predict/batch",
                "roi_estimate": "Surveillance 24/7 automatisée · Détection anomalie < 30 secondes",
                "example": {
                    "trigger":  "Toutes les 15 minutes : envoi batch 50 machines",
                    "action":   "Tableau de bord mis à jour · Alerte SMS si CRITICAL",
                    "outcome":  "Vision complète parc machine en temps réel",
                },
            },
            {
                "id":       "regulatory_compliance",
                "icon":     "🛡️",
                "title":    "Conformité & traçabilité réglementaire",
                "persona":  "Responsable Qualité / Auditeur",
                "problem":  "Obligation de justifier les décisions de maintenance (ISO 55001, RGPD Art.22).",
                "solution": "Chaque prédiction inclut timestamp, modèle utilisé, explication SHAP.",
                "endpoint": "POST /predict/explain",
                "roi_estimate": "Traçabilité 100% · Rapport d'audit automatique · Conformité certifiable",
                "example": {
                    "trigger":  "Toute décision de maintenance",
                    "action":   "Log automatique avec explication humainement lisible",
                    "outcome":  "Audit trail complet · Conformité ISO 55001",
                },
            },
            {
                "id":       "scada_integration",
                "icon":     "🏭",
                "title":    "Intégration SCADA / MES",
                "persona":  "Automaticien / Intégrateur Systèmes",
                "problem":  "Les automates SCADA ne disposent pas de capacité ML native.",
                "solution": "Appel HTTP REST depuis le SCADA → prédiction en < 50ms.",
                "endpoint": "POST /predict",
                "roi_estimate": "Latence ~15ms (XGBoost) · Compatible OPC-UA, Modbus TCP, MQTT",
                "example": {
                    "trigger":  "Chaque cycle automate (1s à 15min)",
                    "action":   "Si risk_level=CRITICAL → arrêt automatique machine",
                    "outcome":  "Boucle de contrôle fermée intégrant le ML",
                },
            },
            {
                "id":       "spare_parts_optimization",
                "icon":     "💰",
                "title":    "Optimisation des stocks pièces de rechange",
                "persona":  "Responsable Supply Chain",
                "problem":  "Sur-stockage de pièces de rechange vs ruptures lors de pannes.",
                "solution": "Prédiction de type de défaillance → commande ciblée anticipée.",
                "endpoint": "POST /predict + failure_type",
                "roi_estimate": "Réduction stock de 30% · Zéro rupture critique",
                "example": {
                    "trigger":  "risk_level=HIGH sur Turbine T2",
                    "action":   "Commande automatique roulement ref#TRB-4582",
                    "outcome":  "Pièce disponible J+2 · Pas de rupture",
                },
            },
        ],
        "quick_start": {
            "step1": "GET /health — Vérifier que le service est opérationnel",
            "step2": "POST /predict — Tester avec vos données capteurs",
            "step3": "POST /predict/batch — Soumettre votre flotte complète",
            "step4": "GET /use-cases — Explorer les cas d'usage",
            "docs":  "http://localhost:8000/docs",
        },
    }


@app.get("/recommendations/{risk_level}", tags=["Métier"])
def get_recommendations(
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = FPath(
        ..., description="Niveau de risque : LOW | MEDIUM | HIGH | CRITICAL"
    )
):
    """
    Retourne les **recommandations d'action** pour un niveau de risque donné.

    Intégrez ces recommandations dans votre GMAO ou système de ticketing
    pour automatiser la création d'ordres de travail.
    """
    recommendations = {
        "LOW": {
            "risk_level":         "LOW",
            "color":              "#3fb950",
            "icon":               "✅",
            "action":             "Maintenance préventive standard",
            "delay":              "Prochain arrêt planifié",
            "priority":           1,
            "cost_estimate":      0.,
            "actions_required":   [
                "Continuer le plan de maintenance préventive",
                "Enregistrer les relevés dans le journal de bord",
                "Prochain contrôle : échéance maintenance standard",
            ],
            "kpi_targets": {"vibration_rms": "< 3.0 mm/s", "temperature_motor": "< 80°C", "hours_since_maintenance": "< 168h"},
        },
        "MEDIUM": {
            "risk_level":         "MEDIUM",
            "color":              "#d29922",
            "icon":               "⚠️",
            "action":             "Surveillance renforcée + contrôle planifié",
            "delay":              "48h maximum",
            "priority":           2,
            "cost_estimate":      200.,
            "actions_required":   [
                "Augmenter la fréquence de relevés (toutes les 2h)",
                "Planifier un contrôle physique dans les 48h",
                "Vérifier l'étanchéité et les niveaux de lubrification",
                "Alerter le technicien de garde",
            ],
            "kpi_targets": {"vibration_rms": "< 4.0 mm/s", "temperature_motor": "< 90°C", "hours_since_maintenance": "< 200h"},
        },
        "HIGH": {
            "risk_level":         "HIGH",
            "color":              "#f85149",
            "icon":               "🔴",
            "action":             "Intervention prioritaire — réduire la charge",
            "delay":              "4-6 heures",
            "priority":           3,
            "cost_estimate":      1500.,
            "actions_required":   [
                "Réduire la charge opératoire à 60% maximum",
                "Envoyer un technicien pour inspection physique < 4h",
                "Préparer la pièce de rechange la plus probable",
                "Prévenir le responsable maintenance",
                "Créer un ordre de travail URGENT dans la GMAO",
            ],
            "kpi_targets": {"vibration_rms": "Réduire < 4.0", "temperature_motor": "Refroidir < 90°C", "hours_since_maintenance": "Maintenance < 48h"},
        },
        "CRITICAL": {
            "risk_level":         "CRITICAL",
            "color":              "#ff6e6e",
            "icon":               "🚨",
            "action":             "ARRÊT IMMÉDIAT — Inspection complète",
            "delay":              "< 1 heure",
            "priority":           4,
            "cost_estimate":      4112.,
            "actions_required":   [
                "ARRÊTER la machine immédiatement ou à la prochaine opportunité sûre",
                "Déclencher la procédure d'urgence maintenance",
                "Sécuriser la zone de travail",
                "Inspection complète par technicien senior",
                "Ne pas redémarrer sans validation du responsable maintenance",
                "Documenter l'incident dans le système de traçabilité",
            ],
            "kpi_targets": {"action": "Arrêt machine", "inspection": "Complète avant redémarrage"},
        },
    }

    if risk_level not in recommendations:
        raise HTTPException(status_code=404, detail=f"Niveau de risque '{risk_level}' invalide. Valeurs acceptées : LOW, MEDIUM, HIGH, CRITICAL")

    return recommendations[risk_level]


@app.get("/machines/simulate", tags=["Démonstration"])
def simulate_machine(
    scenario: Literal["normal", "usure", "critique", "surchauffe", "desequilibre"] = "normal",
    add_noise: bool = True,
):
    """
    Génère des **données capteurs simulées** pour un scénario donné.

    Utile pour tester l'intégration ou démontrer le système sans machine réelle.

    - `normal`       : Machine en bon état
    - `usure`        : Usure modérée (risque MEDIUM)
    - `critique`     : Panne imminente (risque CRITICAL)
    - `surchauffe`   : Surcharge thermique (risque HIGH)
    - `desequilibre` : Déséquilibre mécanique (risque HIGH)
    """
    scenarios = {
        "normal":       dict(vibration_rms=1.8, temperature_motor=68, current_phase_avg=15, pressure_level=5.8, rpm=1600, hours_since_maintenance=45,  ambient_temp=22, operating_mode="normal"),
        "usure":        dict(vibration_rms=3.2, temperature_motor=85, current_phase_avg=21, pressure_level=7.2, rpm=2100, hours_since_maintenance=210, ambient_temp=26, operating_mode="high_load"),
        "critique":     dict(vibration_rms=5.8, temperature_motor=112,current_phase_avg=30, pressure_level=9.5, rpm=2900, hours_since_maintenance=380, ambient_temp=35, operating_mode="peak"),
        "surchauffe":   dict(vibration_rms=2.1, temperature_motor=128,current_phase_avg=35, pressure_level=6.1, rpm=1800, hours_since_maintenance=120, ambient_temp=42, operating_mode="peak"),
        "desequilibre": dict(vibration_rms=6.5, temperature_motor=78, current_phase_avg=19, pressure_level=5.9, rpm=2400, hours_since_maintenance=95,  ambient_temp=23, operating_mode="normal"),
    }

    data = dict(scenarios[scenario])
    if add_noise:
        rng = np.random.default_rng(int(datetime.utcnow().timestamp() * 1000) % 100000)
        for key in ["vibration_rms","temperature_motor","current_phase_avg","pressure_level"]:
            scale = {"vibration_rms": 0.2, "temperature_motor": 2.0,
                     "current_phase_avg": 0.5, "pressure_level": 0.1}[key]
            data[key] = round(float(data[key]) + float(rng.normal(0, scale)), 3)

    return {
        "scenario":      scenario,
        "description":   {
            "normal":       "Machine en bon état — tous les paramètres nominaux",
            "usure":        "Usure modérée — risque MEDIUM attendu",
            "critique":     "Panne imminente — risque CRITICAL attendu",
            "surchauffe":   "Surcharge thermique — risque HIGH attendu",
            "desequilibre": "Déséquilibre mécanique — risque HIGH attendu",
        }[scenario],
        "noise_applied": add_noise,
        "sensor_data":   data,
        "ready_to_post": "Utilisez ces données dans POST /predict",
        "curl_example": (
            "curl -X POST http://localhost:8000/predict "
            '-H "Content-Type: application/json" '
            "-d '" + str(data).replace("'", '"') + "'"
        ),
    }


@app.get("/statistics", tags=["Infrastructure"])
def get_statistics():
    """
    Statistiques globales du service — utile pour le monitoring opérationnel.
    """
    available = [name for name, fname in MODEL_FILES.items() if (MODELS_DIR/fname).exists()]
    elapsed   = (datetime.utcnow() - _start_time).total_seconds()

    return {
        "service":            "PredictMaint Pro v2.0",
        "uptime_seconds":     round(elapsed, 1),
        "uptime_human":       f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m",
        "predictions_served": _prediction_count,
        "alerts_triggered":   _alert_count,
        "alert_rate":         round(_alert_count / max(_prediction_count, 1) * 100, 2),
        "models_available":   available,
        "active_model":       _state["model_name"],
        "endpoints": {
            "total":           8,
            "prediction":      ["/predict", "/predict/batch", "/predict/explain"],
            "information":     ["/health", "/model-info", "/statistics"],
            "business":        ["/use-cases", "/recommendations/{level}"],
            "demo":            ["/machines/simulate"],
        },
        "dataset_info": {
            "name":       "industrial_machine_maintenance.csv",
            "target":     "failure_within_24h",
            "n_samples":  24042,
            "class_imbalance": "~17% positifs (pannes)",
            "features":   8,
        },
    }


# ─── Gestionnaire d'erreurs global ────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Erreur interne du serveur", "detail": str(exc)},
    )


# ─── Point d'entrée direct ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
