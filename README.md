# Système Intelligent Multi-Modèles — Maintenance Prédictive Industrielle

**Projet Data Science M2 — EFREI | Sarah Malaeb | 2025-26**  
**RNCP36739 — Bloc 4 : Implémenter des méthodes d'intelligence artificielle**

---

## Structure du projet

```
maintenance_failure_prediction/
├── main.py                    # Orchestrateur principal (entraînement + évaluation)
├── requirements.txt           # Dépendances Python
├── data/
│   └── raw/
│       └── predictive_maintenance_v3.csv   # ← Télécharger depuis Kaggle
├── src/
│   ├── preprocessing.py       # Pipeline anti-leakage (rolling features, scaling)
│   ├── model_logistic.py      # Modèle 1 — Régression Logistique (baseline)
│   ├── model_random_forest.py # Modèle 2 — Random Forest
│   ├── model_xgboost.py       # Modèle 3 — XGBoost (Gradient Boosting)
│   ├── model_mlp.py           # Modèle 4 — MLP (Deep Learning)
│   ├── evaluation.py          # Métriques, courbes ROC/PR, comparaison
│   ├── explainability.py      # SHAP — Feature Importance + waterfall
│   └── minio_loader.py        # Intégration MinIO (critère RNCP C3 — cloud)
├── dashboard/
│   └── app.py                 # Dashboard Streamlit (EF4 — OBLIGATOIRE)
├── api/
│   └── main.py                # API REST FastAPI (EF5 — optionnel)
├── models/                    # Modèles sérialisés (.pkl) — généré par main.py
├── results/                   # Métriques, graphiques — généré par main.py
└── notebooks/
    └── EDA_Maintenance_Predictive.ipynb
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Données

Télécharger le dataset sur Kaggle :  
https://www.kaggle.com/datasets/tatheerabbas/industrial-machine-predictive-maintenance

Placer le fichier `predictive_maintenance_v3.csv` dans `data/raw/`.

**Caractéristiques :** 24 042 lignes × 15 colonnes | 4 tâches ML possibles

---

## Utilisation

### 1. Entraîner tous les modèles

```bash
python main.py
```

### 2. Entraîner un modèle spécifique + optimisation

```bash
python main.py --model xgboost --optimize
python main.py --model rf --optimize
```

### 3. Pipeline complet (entraînement + SHAP + MinIO)

```bash
python main.py --shap --upload-minio
```

### 4. Charger les données depuis MinIO (critère RNCP C3)

```bash
# Démarrer MinIO localement (Docker)
docker run -p 9000:9000 -p 9001:9001 \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  minio/minio server /data --console-address ":9001"

# Uploader le dataset
python src/minio_loader.py --action upload-data

# Entraîner en chargeant depuis MinIO
python main.py --from-minio
```

### 5. Lancer le Dashboard Streamlit (EF4)

```bash
streamlit run dashboard/app.py
```

### 6. Lancer l'API REST FastAPI (EF5)

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Documentation Swagger : http://localhost:8000/docs

### 7. Générer les graphiques SHAP seuls

```bash
python src/explainability.py --model models/xgboost.pkl
```

---

## Architecture technique

### Modèles implémentés

| # | Modèle | Famille | Anti-déséquilibre |
|---|--------|---------|-------------------|
| 1 | Régression Logistique | Linéaire (baseline) | `class_weight='balanced'` |
| 2 | Random Forest | Ensemble / Bagging | `class_weight='balanced_subsample'` |
| 3 | XGBoost | Gradient Boosting | `scale_pos_weight` |
| 4 | MLP | Deep Learning | `sample_weight` |

### Métriques prioritaires

- **Recall** (classe 1) — minimiser les pannes manquées (FN)
- **PR-AUC** — robustesse sur données déséquilibrées (85%/15%)
- **F1-score** — compromis Precision/Recall
- **ROC-AUC** — performance globale

### Anti-leakage

Le pipeline sklearn est `fit()` **uniquement sur le train set**.  
Les features rolling utilisent `shift(1)` — jamais la valeur courante.

### Critères RNCP validés

| Critère | Réalisé |
|---------|---------|
| C1 — Environnements logiciels adaptés | scikit-learn, XGBoost, Streamlit, FastAPI |
| C2 — Logiciels ETL mobilisés | Pipeline sklearn (imputation, scaling, encoding) |
| C3 — Plateformes cloud data management | **MinIO** (S3-compatible, local + cloud) |
| C4 — Nettoyage avancé | Rolling features, anti-leakage, IQR outliers |
| C5 — Données prêtes pour ML | Pipeline complet train/test stratifié |

---

## Exemple de requête API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "vibration_rms": 4.5,
    "temperature_motor": 95.0,
    "current_phase_avg": 22.0,
    "pressure_level": 7.0,
    "rpm": 2200,
    "hours_since_maintenance": 300,
    "ambient_temp": 25.0,
    "operating_mode": "peak"
  }'
```

Réponse :
```json
{
  "prediction": 1,
  "probability_failure": 0.872,
  "risk_level": "CRITICAL",
  "health_score": 0.312,
  "model_used": "xgboost"
}
```

---

## Livrables

- [x] Code source structuré en modules
- [x] Pipeline complet anti-leakage (sklearn Pipeline)
- [x] 4 modèles comparés (LR, RF, XGBoost, MLP)
- [x] Évaluation rigoureuse (Recall, F1, PR-AUC, ROC-AUC)
- [x] Explicabilité SHAP (global + local)
- [x] Dashboard Streamlit interactif (EF4)
- [x] API REST FastAPI avec Swagger (EF5)
- [x] Intégration MinIO (critère RNCP C3)
- [x] Notebook EDA documenté
