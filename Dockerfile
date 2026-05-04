# ============================================================
# Dockerfile — Maintenance Prédictive Industrielle
# ✅ Compatible Docker ET Podman (Linux/Mac/Windows)
# ============================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dépendances système (XGBoost, OpenMP, healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python (cache préservé si requirements.txt inchangé)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copie du projet
COPY . .

# Création des dossiers de sortie
RUN mkdir -p /app/models /app/results /app/data/raw

CMD ["python", "main.py"]
