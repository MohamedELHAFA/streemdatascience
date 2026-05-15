"""
src/minio_loader.py
───────────────────
Intégration MinIO — Critère RNCP C3 :
  "Les plateformes de data management sont mises en oeuvre à partir du cloud"

MinIO est un serveur de stockage objet open-source compatible S3.
Il s'utilise localement (ou en cloud) pour gérer le dataset et les artefacts ML
de la même façon qu'AWS S3, GCP GCS ou Azure Blob Storage.

Usage :
  # Démarrer MinIO localement (Docker)
  docker run -p 9000:9000 -p 9001:9001 \\
    -e "MINIO_ROOT_USER=minioadmin" \\
    -e "MINIO_ROOT_PASSWORD=minioadmin" \\
    minio/minio server /data --console-address ":9001"

  # Depuis Python
  from src.minio_loader import MinIOClient
  client = MinIOClient()
  client.upload_dataset("data/raw/predictive_maintenance_v3.csv")
  df = client.load_dataset_to_df()

Console MinIO : http://localhost:9001
"""

import io
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import urllib3

# MinIO est optionnel — le module fonctionne en mode dégradé si absent
try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False

# ─── Configuration par défaut ─────────────────────────────────────────────────
MINIO_ENDPOINT  = os.getenv("MINIO_ENDPOINT",  "localhost:9000")
MINIO_ACCESS    = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET    = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET    = os.getenv("MINIO_BUCKET",    "maintenance-predictive")
MINIO_SECURE    = os.getenv("MINIO_SECURE",    "false").lower() == "true"

DATASET_OBJECT  = "raw/predictive_maintenance_v3.csv"
MODELS_PREFIX   = "models/"


class MinIOClient:
    """
    Client MinIO pour la gestion du dataset et des artefacts ML.

    En l'absence de MinIO (non installé ou non démarré), toutes les méthodes
    tombent en fallback sur le système de fichiers local.
    """

    def __init__(
        self,
        endpoint:   str = MINIO_ENDPOINT,
        access_key: str = MINIO_ACCESS,
        secret_key: str = MINIO_SECRET,
        bucket:     str = MINIO_BUCKET,
        secure:     bool = MINIO_SECURE,
    ):
        self.bucket     = bucket
        self._connected = False
        self._client    = None

        if not MINIO_AVAILABLE:
            print("[MinIO] Package 'minio' non installé. Mode local activé.")
            print("        pip install minio")
            return

        try:
            _http = urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=3, read=5),
                retries=urllib3.Retry(total=1, backoff_factor=0),
            )
            self._client = Minio(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
                http_client=_http,
            )
            # Test de connexion : lister les buckets
            self._client.list_buckets()
            self._connected = True
            self._ensure_bucket()
            print(f"[MinIO] Connecté à {endpoint} — bucket : {bucket}")
        except Exception as e:
            print(f"[MinIO] Connexion impossible ({e}). Mode local activé.")

    # ── Bucket ────────────────────────────────────────────────────────────────

    def _ensure_bucket(self):
        """Crée le bucket s'il n'existe pas."""
        if not self._connected:
            return
        if not self._client.bucket_exists(self.bucket):
            self._client.make_bucket(self.bucket)
            print(f"[MinIO] Bucket '{self.bucket}' créé.")

    # ── Dataset ───────────────────────────────────────────────────────────────

    def upload_dataset(self, local_path: str | Path) -> bool:
        """
        Upload le dataset CSV vers MinIO.
        Retourne True si succès, False sinon.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            print(f"[MinIO] Fichier local introuvable : {local_path}")
            return False

        if not self._connected:
            print(f"[MinIO] Mode local — upload ignoré ({local_path})")
            return False

        try:
            self._client.fput_object(
                self.bucket, DATASET_OBJECT, str(local_path),
                content_type="text/csv",
            )
            size_mb = local_path.stat().st_size / 1_048_576
            print(f"[MinIO] Dataset uploadé : {DATASET_OBJECT} ({size_mb:.1f} MB)")
            return True
        except S3Error as e:
            print(f"[MinIO] Erreur upload : {e}")
            return False

    def download_dataset(self, local_path: str | Path) -> bool:
        """
        Télécharge le dataset depuis MinIO vers un fichier local.
        Retourne True si succès, False sinon.
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not self._connected:
            if local_path.exists():
                print(f"[MinIO] Mode local — fichier déjà présent : {local_path}")
                return True
            print(f"[MinIO] Mode local — fichier manquant : {local_path}")
            return False

        try:
            self._client.fget_object(self.bucket, DATASET_OBJECT, str(local_path))
            print(f"[MinIO] Dataset téléchargé : {local_path}")
            return True
        except S3Error as e:
            print(f"[MinIO] Erreur download : {e}")
            return False

    def load_dataset_to_df(
        self,
        local_fallback: Optional[str | Path] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Charge le dataset dans un DataFrame Pandas.

        Stratégie :
          1. Tente de charger depuis MinIO (streaming, sans écrire sur disque)
          2. Si échec → essaie le fichier local (local_fallback)
          3. Si tout échoue → retourne None
        """
        # Tentative MinIO (streaming)
        if self._connected:
            try:
                response = self._client.get_object(self.bucket, DATASET_OBJECT)
                df = pd.read_csv(io.BytesIO(response.read()))
                print(f"[MinIO] Dataset chargé depuis MinIO ({len(df):,} lignes)")
                response.close()
                response.release_conn()
                return df
            except S3Error as e:
                print(f"[MinIO] Lecture impossible depuis MinIO : {e}")

        # Fallback local
        if local_fallback:
            lp = Path(local_fallback)
            if lp.exists():
                df = pd.read_csv(lp)
                print(f"[MinIO] Fallback local : {lp} ({len(df):,} lignes)")
                return df

        return None

    # ── Modèles ML ────────────────────────────────────────────────────────────

    def upload_model(self, local_path: str | Path, model_name: str) -> bool:
        """Upload un modèle sérialisé vers MinIO."""
        local_path = Path(local_path)
        if not local_path.exists():
            print(f"[MinIO] Modèle introuvable : {local_path}")
            return False

        if not self._connected:
            print(f"[MinIO] Mode local — upload modèle ignoré.")
            return False

        object_name = f"{MODELS_PREFIX}{model_name}"
        try:
            self._client.fput_object(
                self.bucket, object_name, str(local_path),
                content_type="application/octet-stream",
            )
            print(f"[MinIO] Modèle uploadé : {object_name}")
            return True
        except S3Error as e:
            print(f"[MinIO] Erreur upload modèle : {e}")
            return False

    def upload_all_models(self, models_dir: str | Path) -> int:
        """Upload tous les fichiers .pkl du répertoire models/."""
        models_dir = Path(models_dir)
        count = 0
        for pkl in models_dir.glob("*.pkl"):
            if self.upload_model(pkl, pkl.name):
                count += 1
        print(f"[MinIO] {count} modèle(s) uploadé(s) vers '{self.bucket}/{MODELS_PREFIX}'")
        return count

    # ── Résultats ─────────────────────────────────────────────────────────────

    def upload_results(self, results_dir: str | Path) -> int:
        """Upload les fichiers de résultats (CSV, PNG) vers MinIO."""
        results_dir = Path(results_dir)
        count = 0
        if not self._connected:
            return count
        for f in results_dir.glob("*"):
            if f.suffix in (".csv", ".json", ".png"):
                object_name = f"results/{f.name}"
                try:
                    self._client.fput_object(self.bucket, object_name, str(f))
                    count += 1
                except S3Error:
                    pass
        print(f"[MinIO] {count} résultat(s) uploadé(s).")
        return count

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Retourne l'état de la connexion MinIO."""
        info = {
            "connected":    self._connected,
            "endpoint":     MINIO_ENDPOINT,
            "bucket":       self.bucket,
            "minio_available": MINIO_AVAILABLE,
        }
        if self._connected:
            try:
                objects = list(self._client.list_objects(self.bucket))
                info["objects_count"] = len(objects)
                info["objects"] = [o.object_name for o in objects[:20]]
            except Exception:
                pass
        return info


# ─── Point d'entrée CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Gestionnaire MinIO")
    parser.add_argument("--action", choices=["status", "upload-data", "upload-models",
                                              "upload-all", "download-data"],
                        default="status")
    args = parser.parse_args()

    client = MinIOClient()

    if args.action == "status":
        import json
        print(json.dumps(client.status(), indent=2))

    elif args.action == "upload-data":
        local = ROOT / "data" / "raw" / "predictive_maintenance_v3.csv"
        client.upload_dataset(local)

    elif args.action == "upload-models":
        client.upload_all_models(ROOT / "models")

    elif args.action == "upload-all":
        local = ROOT / "data" / "raw" / "predictive_maintenance_v3.csv"
        client.upload_dataset(local)
        client.upload_all_models(ROOT / "models")
        client.upload_results(ROOT / "results")

    elif args.action == "download-data":
        local = ROOT / "data" / "raw" / "predictive_maintenance_v3.csv"
        client.download_dataset(local)
