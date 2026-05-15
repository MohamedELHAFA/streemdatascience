"""
tunnel.py — Expose le dashboard Streamlit sur internet via ngrok
Usage : python tunnel.py
"""
import os
import sys
import subprocess
from pathlib import Path

from pyngrok import ngrok, conf

ROOT = Path(__file__).parent

# ── Lancer Streamlit en arrière-plan ────────────────────────────────────────
PORT = 8501

print(f"[1/2] Démarrage du dashboard Streamlit sur le port {PORT}...")

env = os.environ.copy()
env.update({
    "USE_API":        "false",
    "MINIO_ENDPOINT": os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    "MINIO_ACCESS_KEY": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    "MINIO_SECRET_KEY": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    "MINIO_BUCKET":    os.getenv("MINIO_BUCKET", "maintenance-predictive"),
})

streamlit_proc = subprocess.Popen(
    [
        sys.executable, "-m", "streamlit", "run",
        str(ROOT / "dashboard" / "app.py"),
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--server.address", "0.0.0.0",
    ],
    env=env,
    cwd=str(ROOT),
)

import time; time.sleep(4)  # laisser Streamlit démarrer

# ── Ouvrir le tunnel ngrok ───────────────────────────────────────────────────
print("[2/2] Ouverture du tunnel ngrok...")
tunnel = ngrok.connect(PORT, "http")

print("\n" + "=" * 60)
print("  ✅ DASHBOARD ACCESSIBLE PUBLIQUEMENT")
print("=" * 60)
print(f"\n  🌐 URL publique  : {tunnel.public_url}")
print(f"  🏠 URL locale    : http://localhost:{PORT}")
print("\n  Partage cette URL avec les autres personnes.")
print("  Appuie sur Ctrl+C pour arrêter.\n")
print("=" * 60)

try:
    streamlit_proc.wait()
except KeyboardInterrupt:
    print("\nArrêt...")
    ngrok.kill()
    streamlit_proc.terminate()
