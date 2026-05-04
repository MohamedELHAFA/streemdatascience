# Système Intelligent Multi-Modèles — Maintenance Prédictive Industrielle

**Projet Data Science M2 — EFREI | Sarah Malaeb | 2025-26**
**RNCP36739 — Bloc 4 : Implémenter des méthodes d'intelligence artificielle**

---

## Structure du projet

```
maintenance_failure_prediction/
├── main.py                       # Orchestrateur principal (entraînement + évaluation)
├── requirements.txt              # Dépendances Python
├── Dockerfile                    # Image conteneurisée (training + dashboard + api)
├── docker-compose.yml            # Stack complète (compatible Docker & Podman)
├── docker-compose.podman.yml     # Override Podman/Fedora (SELinux + rootless)
├── .env                          # Credentials MinIO (à gitignorer)
├── .dockerignore                 # Exclusions du contexte de build
├── data/
│   └── raw/
│       └── predictive_maintenance_v3.csv   # ← Télécharger depuis Kaggle
├── src/
│   ├── preprocessing.py          # Pipeline anti-leakage (rolling features, scaling)
│   ├── model_logistic.py         # Modèle 1 — Régression Logistique (baseline)
│   ├── model_random_forest.py    # Modèle 2 — Random Forest
│   ├── model_xgboost.py          # Modèle 3 — XGBoost (Gradient Boosting)
│   ├── model_mlp.py              # Modèle 4 — MLP (Deep Learning)
│   ├── evaluation.py             # Métriques, courbes ROC/PR, comparaison
│   ├── explainability.py         # SHAP — Feature Importance + waterfall
│   └── minio_loader.py           # Intégration MinIO (critère RNCP C3 — cloud)
├── dashboard/
│   └── app.py                    # Dashboard Streamlit (EF4 — OBLIGATOIRE)
├── api/
│   └── main.py                   # API REST FastAPI (EF5 — optionnel)
├── models/                       # Modèles sérialisés (.pkl) — généré par main.py
├── results/                      # Métriques, graphiques — généré par main.py
└── notebooks/
    └── EDA_Maintenance_Predictive.ipynb
```

---

## Données

Télécharger le dataset sur Kaggle :
https://www.kaggle.com/datasets/tatheerabbas/industrial-machine-predictive-maintenance

Placer le fichier `predictive_maintenance_v3.csv` dans `data/raw/`.

**Caractéristiques :** 24 042 lignes × 15 colonnes | 4 tâches ML possibles

---

## Deux modes d'exécution

Le projet supporte **deux modes** au choix selon le contexte :

| Mode                              | Usage recommandé                          |
|-----------------------------------|-------------------------------------------|
| 🐍 **Local** (venv Python)        | Développement, itération rapide           |
| 🐳 **Containerisé** (Docker/Podman)| Démo, soutenance, reproductibilité totale|

Le code utilise des **variables d'environnement** (`MINIO_ENDPOINT`, etc.) pour fonctionner identiquement dans les deux modes — même base de code, deux runtimes.

---

# 🐍 Mode 1 — Local (venv)

## Installation

```bash
pip install -r requirements.txt
```

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

### 4. Charger les données depuis MinIO

```bash
# Démarrer MinIO localement (Docker / Podman one-shot)
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

# 🐳 Mode 2 — Containerisé (Docker / Podman)

Stack complète orchestrée : **MinIO persistant + bucket auto-créé + Training + Dashboard + API**.
Compatible **Docker** (Linux/Mac/Windows) et **Podman** (Fedora/RHEL).

## Pourquoi cette stack ?

- **Reproductibilité** : un seul `up` et tout l'environnement est prêt (critère RNCP C1)
- **MinIO persistant** : volume Docker dédié, données conservées entre redémarrages (critère RNCP C3)
- **Portabilité** : pattern multi-fichier compose pour gérer Docker ET Podman sans modification du code
- **Isolation** : chaque service (MinIO, training, dashboard, API) dans son propre container, communication via réseau Docker

## Démarrage selon ta plateforme

### 🟦 Sur Docker (Linux Ubuntu/Debian, Mac, Windows)

```bash
docker compose up -d --build minio dashboard api
```

### 🟥 Sur Podman (Fedora, RHEL, Rocky)

```bash
podman-compose -f docker-compose.yml -f docker-compose.podman.yml up -d --build minio dashboard api
```

L'override Podman ajoute :
- `:Z` sur les bind mounts → relabel SELinux automatique
- `userns_mode: keep-id` → mapping correct UID/GID en mode rootless

💡 **Astuce** : crée un alias pour ne pas retaper la commande à chaque fois :
```bash
echo 'alias mlcompose="podman-compose -f docker-compose.yml -f docker-compose.podman.yml"' >> ~/.bashrc
source ~/.bashrc

# Puis :
mlcompose up -d --build minio dashboard api
```

## Accéder aux services

| Service       | URL                          | Identifiants               |
|---------------|------------------------------|----------------------------|
| Console MinIO | http://localhost:9001        | minioadmin / minioadmin123 |
| Dashboard     | http://localhost:8501        | —                          |
| API Swagger   | http://localhost:8000/docs   | —                          |

## Lancer l'entraînement (one-shot)

Le service `training` est dans un profile dédié (ne démarre pas avec `up`). On le lance manuellement :

**Docker :**
```bash
docker compose run --rm training
```

**Podman :**
```bash
podman-compose -f docker-compose.yml -f docker-compose.podman.yml run --rm training
# ou avec l'alias :
mlcompose run --rm training
```

Variantes :
```bash
# Modèle spécifique avec optimisation
... run --rm training python main.py --model xgboost --optimize

# Charger les données depuis MinIO
... run --rm training python main.py --from-minio
```

À la fin, les modèles `.pkl` apparaissent dans `./models/` sur la machine hôte (volume monté).

## Persistance MinIO

| Donnée                          | Emplacement                         | Persistant |
|---------------------------------|-------------------------------------|------------|
| Datasets/modèles dans MinIO     | volume `maintenance_minio_data`     | ✅         |
| CSV source                      | `./data/raw/` (host)                | ✅         |
| Modèles `.pkl` entraînés        | `./models/` (host)                  | ✅         |

Le volume `maintenance_minio_data` survit à `down`, aux reboots, etc. Il n'est supprimé **que** par `down -v`.

### Backup du volume

**Docker :**
```bash
docker run --rm \
  -v maintenance_minio_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/minio_$(date +%Y%m%d).tar.gz -C /data .
```

**Podman :**
```bash
podman run --rm \
  -v maintenance_minio_data:/data:Z \
  -v ./backups:/backup:Z \
  alpine tar czf /backup/minio_$(date +%Y%m%d).tar.gz -C /data .
```

## Arrêter la stack

```bash
docker compose down            # arrêt propre, données conservées
docker compose down -v         # ⚠️ supprime aussi le volume MinIO
```

## Commandes utiles

```bash
# Logs en direct
docker compose logs -f
docker compose logs -f dashboard

# État des services
docker compose ps

# Shell dans un container
docker compose exec dashboard bash

# Rebuild après modif du code
docker compose up -d --build dashboard api
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
| C1 — Environnements logiciels adaptés | scikit-learn, XGBoost, Streamlit, FastAPI, **stack containerisée Docker/Podman** |
| C2 — Logiciels ETL mobilisés | Pipeline sklearn (imputation, scaling, encoding) |
| C3 — Plateformes cloud data management | **MinIO** (S3-compatible) avec persistance volume + bucket auto-créé |
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

## Pièges fréquents et solutions

| Symptôme                                   | Plateforme | Solution                                    |
|--------------------------------------------|------------|---------------------------------------------|
| `Connection refused localhost:9000` dans l'app | Docker/Podman | Utiliser `minio:9000` (variable `MINIO_ENDPOINT`) |
| `Permission denied` sur les volumes        | Podman     | Vérifier que `-f docker-compose.podman.yml` est chargé |
| `unknown field "userns_mode"`              | Docker     | Ne pas charger l'override Podman           |
| Modèles introuvables côté API              | Tous       | Lancer `... run --rm training` d'abord    |
| Port déjà utilisé (9000/8000/8501)         | Tous       | `ss -tlnp \| grep <port>` puis libérer    |
| Données perdues après `down`               | Tous       | Tu as fait `down -v` — ne pas le faire    |

---

## Livrables

- [x] Code source structuré en modules
- [x] Pipeline complet anti-leakage (sklearn Pipeline)
- [x] 4 modèles comparés (LR, RF, XGBoost, MLP)
- [x] Évaluation rigoureuse (Recall, F1, PR-AUC, ROC-AUC)
- [x] Explicabilité SHAP (global + local)
- [x] Dashboard Streamlit interactif (EF4)
- [x] API REST FastAPI avec Swagger (EF5)
- [x] Intégration MinIO avec persistance (critère RNCP C3)
- [x] **Stack containerisée portable (Docker + Podman)**
- [x] Notebook EDA documenté