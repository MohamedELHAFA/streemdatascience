# PredictMaint Pro — Système Intelligent de Maintenance Prédictive Industrielle

> Projet Data Science M2 · EFREI · Data Engineering & AI · 2025–2026
> Certification **RNCP36739 — Bloc 4** : Implémenter des méthodes d'intelligence artificielle pour modéliser et prédire de nouveaux comportements

---

## Sommaire

1. [Présentation](#presentation)
2. [Aperçu architectural](#apercu-architectural)
3. [Stack technique](#stack-technique)
4. [Démarrage rapide](#demarrage-rapide)
5. [Installation détaillée par OS](#installation-detaillee-par-os)
   - [Windows 10/11 (Docker Desktop)](#windows-1011-docker-desktop)
   - [Ubuntu / Debian (Docker Engine)](#ubuntu--debian-docker-engine)
   - [Fedora / RHEL / Rocky (Podman rootless)](#fedora--rhel--rocky-podman-rootless)
   - [macOS (Docker Desktop)](#macos-docker-desktop)
   - [Mode local Python (sans conteneurs)](#mode-local-python-sans-conteneurs)
6. [Utilisation](#utilisation)
7. [Architecture du code](#architecture-du-code)
8. [Modélisation et résultats](#modelisation-et-resultats)
9. [API REST](#api-rest)
10. [Dashboard](#dashboard)
11. [Persistance et MinIO](#persistance-et-minio)
12. [Variables d'environnement](#variables-denvironnement)
13. [Dépannage](#depannage)
14. [Conformité RNCP](#conformite-rncp)
15. [Auteur](#auteur)

---

## Présentation

Ce projet conçoit une **plateforme intelligente de maintenance prédictive** capable d'exploiter les signaux de capteurs industriels (vibration, température, pression, RPM, courant, mode opératoire) pour anticiper les défaillances d'équipements **dans les 24 heures à venir**.

Le dataset contient **24 042 observations** issues de machines industrielles, avec un déséquilibre de classes réaliste (≈85 % de fonctionnement normal, ≈15 % de pannes). Cette asymétrie impose une démarche méthodologique rigoureuse : métriques adaptées (Recall, PR-AUC), stratégies anti-déséquilibre par modèle, optimisation du seuil de décision selon le coût métier.

**Tâche prédictive retenue :** classification binaire `failure_within_24h` (0 = normal, 1 = panne imminente).

La solution n'est pas un simple notebook : c'est un **système complet** comprenant pipeline de preprocessing anti-leakage, comparaison de quatre modèles (dont un Deep Learning), explicabilité SHAP, dashboard décisionnel, API REST de production et stockage objet S3-compatible.

---

## Aperçu architectural

```
┌──────────────────────────────────────────────────────────────────┐
│                         UTILISATEUR FINAL                        │
│              (Responsable Maintenance · Ingénieur)               │
└────────────────────┬─────────────────────────┬───────────────────┘
                     │                         │
                     ▼                         ▼
            ┌────────────────┐        ┌────────────────┐
            │   Dashboard    │   ───► │   API REST     │
            │   Streamlit    │  HTTP  │    FastAPI     │
            │   :8501        │        │    :8000       │
            └────────────────┘        └────────┬───────┘
                                               │
                                       ┌───────▼────────┐
                                       │  Modèles .pkl  │
                                       │ LR · RF · XGB  │
                                       │ · MLP (DL)     │
                                       └───────┬────────┘
                                               │
                                       ┌───────▼────────┐
                                       │   MinIO S3     │
                                       │  :9000 / :9001 │
                                       │ datasets +     │
                                       │ artefacts ML   │
                                       └────────────────┘
```

**Principe directeur : séparation Front / API / Modèle.**
Le dashboard ne charge plus directement les fichiers `.pkl` — il consomme l'endpoint `POST /predict` de l'API exactement comme le ferait un client externe (mobile, ERP, SCADA). Cette architecture est conforme à la recommandation page 15 du sujet : *« le dashboard devra idéalement appeler l'API pour obtenir les prédictions, afin de reproduire une architecture réaliste »*.

Un mécanisme de **fallback local** garantit que l'UI ne crashe pas si l'API tombe : le dashboard bascule alors silencieusement sur les modèles montés en read-only, avec un indicateur visuel dans la sidebar.

---

## Stack technique

| Couche | Technologie | Rôle |
|---|---|---|
| Langage | Python 3.11 | Cœur applicatif |
| ML classique | scikit-learn 1.4+ | Régression Logistique, Random Forest, MLP, pipelines |
| Gradient Boosting | XGBoost 2.x | Modèle final retenu |
| Explicabilité | SHAP 0.44+ | Importance globale et locale |
| Dashboard | Streamlit 1.35+ + Plotly 5.x | Interface décisionnelle interactive |
| API REST | FastAPI 0.110+ + Pydantic 2.x + Uvicorn | Service d'inférence + Swagger auto |
| Sérialisation | Joblib 1.3+ | Modèles `.pkl` |
| Storage | MinIO 7.2+ (compatible S3) | Datasets + artefacts ML versionnés |
| Conteneurisation | Docker 24+ ou Podman 4.x+ | Reproductibilité et déploiement |
| Orchestration | docker-compose / podman-compose | Stack multi-services |

---

## Démarrage rapide

Si tu as déjà Docker (ou Podman) installé et que tu veux juste voir le système tourner :

```bash
git clone <repo>
cd maintenance_failure_prediction

# 1. Configurer les credentials MinIO
cp .env.example .env

# 2. Placer le dataset
#    Télécharger predictive_maintenance_v3.csv depuis Kaggle
#    https://www.kaggle.com/datasets/tatheerabbas/industrial-machine-predictive-maintenance
#    puis :
mv ~/Downloads/predictive_maintenance_v3.csv data/raw/

# 3. Lancer la stack (Docker)
docker compose up -d --build

# 4. Entraîner les modèles (one-shot)
docker compose --profile training run --rm training
```

Puis ouvrir :

| Service | URL |
|---|---|
| Dashboard | http://localhost:8501 |
| API Swagger | http://localhost:8000/docs |
| Console MinIO | http://localhost:9001 |

---

## Installation détaillée par OS

### Windows 10/11 (Docker Desktop)

**Prérequis :**
- Windows 10 version 2004+ ou Windows 11
- WSL2 activé (`wsl --install` dans PowerShell admin si besoin)
- [Docker Desktop pour Windows](https://www.docker.com/products/docker-desktop/) ≥ v4.20

**Installation :**

1. Installer Docker Desktop, le lancer une fois pour vérifier que le moteur démarre. Dans Settings → General, cocher *Use the WSL 2 based engine*.

2. Ouvrir PowerShell ou Windows Terminal dans le dossier du projet :
   ```powershell
   cd C:\Users\<toi>\projects\maintenance_failure_prediction
   ```

3. Configurer l'environnement :
   ```powershell
   Copy-Item .env.example .env
   ```

4. Placer `predictive_maintenance_v3.csv` dans `data\raw\` (drag & drop dans l'Explorateur fonctionne).

5. Lancer la stack :
   ```powershell
   docker compose up -d --build
   ```

6. Entraîner les modèles :
   ```powershell
   docker compose --profile training run --rm training
   ```

**Spécificités Windows :**
- Les chemins de volumes utilisent automatiquement la translation WSL2 — pas d'ajustement requis.
- Sous PowerShell, certaines commandes `bash` du README utilisent `\` comme continuation : remplacer par `` ` `` (backtick) ou tout mettre sur une ligne.
- Docker Desktop doit être *running* dans la barre des tâches avant chaque session.

### Ubuntu / Debian (Docker Engine)

**Prérequis :** Ubuntu 22.04+ ou Debian 12+, accès `sudo`.

**Installation de Docker Engine :**

```bash
# Pré-requis
sudo apt update && sudo apt install -y ca-certificates curl gnupg

# Clé GPG officielle Docker
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Installation
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Permettre l'usage sans sudo
sudo usermod -aG docker $USER
newgrp docker  # ou se déconnecter/reconnecter
```

**Lancement du projet :**

```bash
git clone <repo> && cd maintenance_failure_prediction
cp .env.example .env
# placer predictive_maintenance_v3.csv dans data/raw/

docker compose up -d --build
docker compose --profile training run --rm training
```

### Fedora / RHEL / Rocky (Podman rootless)

**Recommandation :** Podman est préinstallé sur Fedora et fonctionne en rootless par défaut, ce qui est plus sûr que Docker. SELinux est aussi actif — d'où l'override `docker-compose.podman.yml`.

**Prérequis :**
```bash
sudo dnf install -y podman podman-compose
```

Vérifier la version :
```bash
podman --version          # ≥ 4.x
podman-compose --version  # ≥ 1.0
```

**Lancement du projet :**

Toujours utiliser **les deux fichiers compose** (le principal + l'override Podman) :

```bash
git clone <repo> && cd maintenance_failure_prediction
cp .env.example .env
# placer predictive_maintenance_v3.csv dans data/raw/

podman-compose -f docker-compose.yml -f docker-compose.podman.yml up -d --build
```

**Astuce — créer un alias pour éviter de retaper les deux fichiers :**

```bash
echo 'alias mlcompose="podman-compose -f docker-compose.yml -f docker-compose.podman.yml"' >> ~/.bashrc
source ~/.bashrc

# Puis simplement :
mlcompose up -d --build
mlcompose ps
mlcompose logs -f dashboard
```

**Particularités Podman traitées automatiquement par l'override :**
- `:Z` sur les bind mounts → relabelisation SELinux automatique
- `userns_mode: keep-id` → mapping UID/GID correct en mode rootless

**Si tu vois cette erreur :**
```
? Please select an image:
  ▸ registry.fedoraproject.org/...
```
C'est `podman-compose` qui ne trouve pas l'image locale. Ne sélectionne **rien**, fais `Ctrl+C`, et tagge l'image manuellement :

```bash
podman tag localhost/maintenance-ml:latest \
           localhost/maintenance_failure_prediction_dashboard:latest
podman tag localhost/maintenance-ml:latest \
           localhost/maintenance_failure_prediction_api:latest
mlcompose up -d
```

### macOS (Docker Desktop)

**Prérequis :** macOS 12+ Intel ou Apple Silicon, [Docker Desktop pour Mac](https://www.docker.com/products/docker-desktop/).

```bash
# Installation via Homebrew (optionnel)
brew install --cask docker

# Une fois Docker Desktop démarré :
git clone <repo> && cd maintenance_failure_prediction
cp .env.example .env
# placer le CSV dans data/raw/

docker compose up -d --build
docker compose --profile training run --rm training
```

**Sur Apple Silicon (M1/M2/M3) :** XGBoost 2.x supporte nativement arm64. Les images se construisent en 3-5 min à la première utilisation.

### Mode local Python (sans conteneurs)

Pour itérer rapidement pendant le développement (pas de rebuild entre chaque modification de code) :

**Linux/macOS :**
```bash
python3.11 -m venv maintenance_failure_venv
source maintenance_failure_venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell) :**
```powershell
python -m venv maintenance_failure_venv
.\maintenance_failure_venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Lancement séparé des composants :**
```bash
# Entraîner tous les modèles
python main.py

# API REST
uvicorn api.main:app --reload --port 8000

# Dashboard (mode local — modèles chargés en mémoire)
streamlit run dashboard/app.py

# Dashboard (mode API — appelle uvicorn ci-dessus)
USE_API=true API_URL=http://localhost:8000 streamlit run dashboard/app.py
```

Sur Windows PowerShell, utiliser `$env:USE_API="true"` puis lancer la commande sur la ligne suivante.

---

## Utilisation

### Lancer la stack complète

| Action | Docker | Podman (Fedora) |
|---|---|---|
| Démarrer | `docker compose up -d --build` | `mlcompose up -d --build` |
| État | `docker compose ps` | `mlcompose ps` |
| Logs en direct | `docker compose logs -f dashboard` | `mlcompose logs -f dashboard` |
| Redémarrer un service | `docker compose restart api` | `mlcompose restart api` |
| Reconstruire | `docker compose up -d --build dashboard` | `mlcompose up -d --build dashboard` |
| Arrêter (données conservées) | `docker compose down` | `mlcompose down` |
| Tout supprimer (⚠️ MinIO inclus) | `docker compose down -v` | `mlcompose down -v` |

### Entraîner les modèles

Le service `training` est dans un *profile* dédié — il ne démarre pas avec `up`. À lancer manuellement :

```bash
# Tous les modèles (LR + RF + XGBoost + MLP)
docker compose --profile training run --rm training

# Un modèle spécifique avec optimisation hyperparamètres
docker compose --profile training run --rm training python main.py --model xgboost --optimize

# Avec génération SHAP et upload MinIO
docker compose --profile training run --rm training python main.py --shap --upload-minio

# Charger les données depuis MinIO (au lieu du fichier local)
docker compose --profile training run --rm training python main.py --from-minio
```

Les artefacts (`.pkl`, métriques, graphiques SHAP) apparaissent dans `models/` et `results/` côté hôte.

### Accéder aux services

| Service | URL | Identifiants |
|---|---|---|
| Dashboard Streamlit | http://localhost:8501 | — |
| API Swagger UI | http://localhost:8000/docs | — |
| API ReDoc | http://localhost:8000/redoc | — |
| Console MinIO | http://localhost:9001 | `minioadmin` / `minioadmin123` (cf. `.env`) |
| Endpoint S3 MinIO | http://localhost:9000 | idem |

### Tester l'API

```bash
# Health check
curl http://localhost:8000/health

# Prédiction unitaire
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

Réponse attendue :
```json
{
  "prediction": 1,
  "probability_failure": 0.872,
  "probability_normal": 0.128,
  "risk_level": "CRITICAL",
  "health_score": 0.312,
  "model_used": "xgboost",
  "features_derived": { ... },
  "recommendation": "ACTION IMMÉDIATE...",
  "timestamp": "2026-05-06T08:39:36.194Z"
}
```

---

## Architecture du code

```
maintenance_failure_prediction/
├── main.py                      # Orchestrateur d'entraînement
├── requirements.txt             # Dépendances Python pinnées
├── Dockerfile                   # Image multi-services (Python 3.11-slim)
├── docker-compose.yml           # Stack principale (Docker + Podman)
├── docker-compose.podman.yml    # Override Podman (SELinux + rootless)
├── .env.example                 # Template credentials MinIO
│
├── data/
│   └── raw/
│       └── predictive_maintenance_v3.csv   # Dataset (à télécharger)
│
├── src/                         # Code source modulaire
│   ├── preprocessing.py         # Pipeline anti-leakage (rolling features, scaling)
│   ├── model_logistic.py        # Modèle 1 — Régression Logistique (baseline)
│   ├── model_random_forest.py   # Modèle 2 — Random Forest
│   ├── model_xgboost.py         # Modèle 3 — XGBoost (Gradient Boosting)
│   ├── model_mlp.py             # Modèle 4 — MLP (Deep Learning)
│   ├── evaluation.py            # Métriques + courbes ROC/PR + comparaison
│   ├── explainability.py        # SHAP — feature importance + waterfall + dependence
│   └── minio_loader.py          # Client MinIO (upload/download dataset & artefacts)
│
├── dashboard/
│   └── app.py                   # Streamlit — 7 pages décisionnelles
│
├── api/
│   └── main.py                  # FastAPI — 11 endpoints + Swagger auto
│
├── models/                      # Modèles sérialisés (.pkl) — généré
├── results/                     # Métriques + graphiques — généré
└── notebooks/
    └── EDA_Maintenance_Predictive.ipynb   # Analyse exploratoire
```

### Anti data leakage — point méthodologique critique

Le pipeline scikit-learn est **`fit()` exclusivement sur le train set**. Les rolling features (moyenne, écart-type, différence sur fenêtre glissante de 3 mesures) utilisent un `shift(1)` qui exclut la valeur courante du calcul — ainsi, à l'instant *t*, le modèle ne voit jamais la mesure de *t* dans son propre historique. Détail dans `src/preprocessing.py` lignes 76–121.

---

## Modélisation et résultats

### Quatre modèles comparés

| # | Modèle | Famille | Stratégie anti-déséquilibre |
|---|---|---|---|
| 1 | Régression Logistique | Linéaire (baseline) | `class_weight='balanced'` |
| 2 | Random Forest | Ensemble / Bagging | `class_weight='balanced_subsample'` |
| 3 | XGBoost | Gradient Boosting | `scale_pos_weight ≈ 5.75` |
| 4 | MLP (3 couches : 128→64→32) | Deep Learning | `sample_weight` + early stopping |

### Métriques prioritaires

Sur ce problème industriel déséquilibré (15 % de pannes), l'**accuracy est trompeuse** : un modèle qui prédit toujours « normal » obtiendrait 85 % d'accuracy tout en étant inutile. Les métriques choisies en conséquence :

- **Recall classe 1** — taux de pannes correctement détectées (priorité absolue, un FN coûte ~4 112 €)
- **PR-AUC** — robustesse globale sur classes déséquilibrées
- **F1 classe 1** — compromis Precision/Recall
- **ROC-AUC** — performance globale (complémentaire)

### Résultats sur le test set (4 809 observations)

| Modèle | Accuracy | Recall (1) | Precision (1) | F1 (1) | PR-AUC | ROC-AUC | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.931 | 0.772 | 0.766 | 0.769 | 0.839 | 0.957 | 162 | 168 |
| Random Forest | 0.967 | 0.903 | 0.875 | 0.889 | 0.953 | 0.991 | 69 | 92 |
| **XGBoost** ⭐ | **0.975** | 0.899 | **0.933** | **0.916** | **0.973** | **0.995** | 72 | **46** |
| MLP (Deep Learning) | 0.959 | 0.847 | 0.870 | 0.858 | 0.927 | 0.985 | 109 | 90 |

### Modèle retenu : XGBoost

XGBoost obtient le meilleur **PR-AUC** (0.973) et la meilleure **Precision** sur la classe positive (0.933) — soit moitié moins de fausses alertes que Random Forest pour un Recall équivalent. C'est ce compromis Precision/Recall qui le distingue.

**Point pédagogique sur le Deep Learning :** le MLP n'a pas dépassé XGBoost. Sur des données tabulaires de cette taille (~24 k observations, ~14 features), les modèles d'arbres restent souvent supérieurs — un résultat cohérent avec la littérature et explicitement attendu par le sujet.

### Explicabilité (SHAP)

Les features les plus déterminantes selon SHAP sur le modèle XGBoost final :

1. `temperature_motor` (1.22)
2. `vibration_per_rpm` (1.21) — feature dérivée du feature engineering
3. `rpm` (1.10)
4. `current_phase_avg` (0.81)
5. `pressure_level` (0.60)
6. `hours_since_maintenance` (0.60)

Cette hiérarchie est **cohérente avec l'expertise métier** : la température moteur et le ratio vibration/régime sont les indicateurs précoces les plus fiables d'une dégradation.

---

## API REST

L'API expose **11 endpoints** documentés automatiquement via Swagger.

| Méthode | Endpoint | Rôle |
|---|---|---|
| GET | `/` | Page d'accueil avec liens utiles |
| GET | `/health` | Santé du service + modèle chargé |
| GET | `/model-info` | Métadonnées du modèle actif |
| POST | **`/predict`** | Prédiction unitaire |
| POST | `/predict/batch` | Prédictions multiples (max 1000) |
| POST | `/predict/explain` | Prédiction + explication métier |
| GET | `/use-cases` | Catalogue des use cases industriels |
| GET | `/recommendations/{risk_level}` | Recommandations par niveau de risque |
| GET | `/machines/simulate` | Simulation de données capteurs |
| GET | `/statistics` | Compteurs globaux du service |

**Validation des entrées via Pydantic v2** : bornes physiques sur chaque capteur (`vibration_rms ∈ [0, 15]`, `temperature_motor ∈ [0, 200]`, etc.), opérateur `Literal["normal", "high_load", "peak"]` pour le mode opératoire. Les erreurs renvoient un code HTTP 422 avec un message structuré.

---

## Dashboard

Le dashboard Streamlit comprend **7 pages thématiques** orientées utilisateur métier :

1. **🏠 Vue d'ensemble** — KPIs flotte, état global
2. **🚨 Alertes en temps réel** — Machines à risque immédiat
3. **🤖 Modèles & Performances** — Comparaison interactive, courbes ROC/PR, analyse coût FN/FP
4. **⚙️ Simulateur Machine** — Saisie de scénarios, prédiction live, switch entre modèles
5. **📊 Analyse EDA** — Distributions, corrélations, qualité des données
6. **🔍 Explicabilité IA** — SHAP global et local, analyse FN/FP
7. **💼 Use Cases Métier** — Application industrielle, ROI, intégration

**Mode d'inférence configurable :**

| `USE_API` | Comportement |
|---|---|
| `false` (défaut) | Modèles `.pkl` chargés en mémoire (mode dev rapide) |
| `true` | Appels HTTP vers l'API REST (architecture production) |

En mode `true`, si l'API tombe, un fallback local prend le relais automatiquement avec un warning visuel — le dashboard ne crashe jamais.

---

## Persistance et MinIO

MinIO est un serveur de stockage objet **compatible API S3** utilisé ici comme alternative locale à AWS S3. Il satisfait le critère RNCP **C3** : *« Les plateformes de data management sont mises en œuvre à partir du cloud »*.

### Ce qui est persistant

| Donnée | Emplacement | Survit à `down` ? |
|---|---|:---:|
| Datasets et modèles uploadés sur MinIO | volume `maintenance_minio_data` | ✅ |
| CSV source | `./data/raw/` (host) | ✅ |
| Modèles `.pkl` entraînés | `./models/` (host) | ✅ |
| Résultats et graphiques | `./results/` (host) | ✅ |

Le volume Docker/Podman `maintenance_minio_data` n'est supprimé **que** par `down -v`.

### Backup du volume MinIO

```bash
# Docker
docker run --rm \
  -v maintenance_minio_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/minio_$(date +%Y%m%d).tar.gz -C /data .

# Podman
podman run --rm \
  -v maintenance_minio_data:/data:Z \
  -v ./backups:/backup:Z \
  alpine tar czf /backup/minio_$(date +%Y%m%d).tar.gz -C /data .
```

### Workflow MinIO

```bash
# Uploader le dataset depuis l'hôte vers MinIO
docker compose --profile training run --rm training \
  python src/minio_loader.py --action upload-data

# Entraîner en chargeant le dataset depuis MinIO
docker compose --profile training run --rm training \
  python main.py --from-minio

# Pipeline complet : entraînement + SHAP + upload de tous les artefacts
docker compose --profile training run --rm training \
  python main.py --shap --upload-minio
```

---

## Variables d'environnement

À configurer dans `.env` (à partir de `.env.example`) :

| Variable | Défaut | Rôle |
|---|---|---|
| `MINIO_ROOT_USER` | `minioadmin` | Login admin MinIO |
| `MINIO_ROOT_PASSWORD` | `minioadmin123` | Mot de passe admin MinIO |
| `MINIO_BUCKET` | `maintenance` | Nom du bucket auto-créé |

Variables internes utilisées par le code (déjà configurées dans le `docker-compose.yml`) :

| Variable | Service | Rôle |
|---|---|---|
| `USE_API` | dashboard | `true` = appelle l'API, `false` = modèle local |
| `API_URL` | dashboard | URL de l'API (`http://api:8000` en compose) |
| `MINIO_ENDPOINT` | training, dashboard, api | `minio:9000` en compose, `localhost:9000` en local |
| `MINIO_ACCESS_KEY` | training, api | = `MINIO_ROOT_USER` |
| `MINIO_SECRET_KEY` | training, api | = `MINIO_ROOT_PASSWORD` |
| `MINIO_SECURE` | training, api | `false` (HTTP) en dev local |
| `MPLCONFIGDIR` | tous | `/tmp/mpl` — supprime un warning matplotlib en rootless |

---

## Dépannage

| Symptôme | Plateforme | Cause / Solution |
|---|---|---|
| `Connection refused localhost:9000` | Docker/Podman | Dans le code dans un container, utiliser `minio:9000` (variable `MINIO_ENDPOINT`) — `localhost` ne pointe pas vers MinIO. |
| `Permission denied` sur les volumes | Podman | L'override `docker-compose.podman.yml` n'est pas chargé — repasser avec `-f docker-compose.yml -f docker-compose.podman.yml`. |
| `unknown field "userns_mode"` | Docker | Ne PAS charger l'override Podman avec Docker — utiliser uniquement le compose principal. |
| `Modèles introuvables` côté API | Tous | Lancer d'abord `docker compose --profile training run --rm training`. |
| `address already in use :8000` | Tous | Un autre service occupe le port. `ss -tlnp \| grep :8000` (Linux), `netstat -ano \| findstr :8000` (Windows). |
| `address already in use :8501` | Tous | Idem pour Streamlit. |
| `Données introuvables — predictive_maintenance_v3.csv` | Tous | Le CSV n'est pas dans `data/raw/` côté hôte, ou le volume `./data:/app/data` n'est pas monté pour le service concerné. |
| `Please select an image` (menu interactif podman-compose) | Podman | `Ctrl+C` puis tagger l'image : `podman tag localhost/maintenance-ml:latest localhost/<projet>_dashboard:latest`. |
| YAML `expected <block end>` | Tous | Indentation cassée. Valider avec `python3 -c "import yaml; yaml.safe_load(open('docker-compose.yml'))"`. |
| Données perdues après `down` | Tous | Tu as fait `down -v` qui supprime les volumes. Faire juste `down` la prochaine fois. |
| Build très lent (Apple Silicon) | macOS M1/M2/M3 | Première build = 3-5 min normales (compilation native). Les builds suivants utilisent le cache. |

### Logs utiles en cas de problème

```bash
# Tous les services
docker compose logs -f

# Un service spécifique avec les 50 dernières lignes
docker compose logs --tail=50 dashboard

# Inspecter l'état d'un container
docker inspect ml_dashboard | grep -A 5 State

# Shell dans un container vivant
docker compose exec dashboard bash
```

---

## Conformité RNCP

Ce projet valide explicitement les compétences du **Bloc 4** du référentiel RNCP36739 :

| Critère | Réalisation dans le projet |
|---|---|
| **C1** — Environnements logiciels adaptés | Stack containerisée Docker + Podman, scikit-learn, XGBoost, Streamlit, FastAPI |
| **C2** — Outils ETL mobilisés | Pipeline scikit-learn (imputation, scaling, encoding, ColumnTransformer) |
| **C3** — Plateforme cloud data management | MinIO S3-compatible avec bucket auto-créé, persistance via volume |
| **C4** — Nettoyage avancé | Rolling features, anti-leakage temporel via `shift(1)`, gestion des outliers |
| **C5** — Données prêtes pour ML | Pipeline train/test stratifié, preprocessing fit uniquement sur train |
| **Bloc 4 — Modèle prédictif** | 4 modèles comparés (LR, RF, XGBoost, MLP), CV stratifiée, optimisation hyperparamètres, métriques adaptées au déséquilibre |

### Livrables couverts

- ✅ Code source structuré en modules (pas de notebook monolithique)
- ✅ Pipeline complet anti-leakage
- ✅ Comparaison rigoureuse multi-modèles
- ✅ Évaluation par métriques adaptées (Recall, F1, PR-AUC, ROC-AUC)
- ✅ Explicabilité SHAP (globale + locale + dependence plots + waterfalls)
- ✅ Dashboard Streamlit interactif (EF4 obligatoire)
- ✅ API REST FastAPI avec Swagger (EF5 optionnel implémenté)
- ✅ Architecture Front / API / Modèle conforme à la recommandation page 15
- ✅ Containerisation portable (Docker + Podman) — reproductibilité totale
- ✅ Intégration MinIO (critère cloud C3)

---

## Auteur

Projet Data Science M2 — EFREI · Data Engineering & AI · 2025–2026
Encadrante : **Sarah Malaeb**

Dataset source : [Industrial Machine Predictive Maintenance — Kaggle](https://www.kaggle.com/datasets/tatheerabbas/industrial-machine-predictive-maintenance)

---

> *« Un modèle performant n'est qu'une partie du système. Ce qui crée de la valeur en entreprise, c'est son intégration dans un pipeline complet. »* — Sujet du projet, page 3