"""
main.py
───────
Orchestrateur principal — Maintenance Prédictive Industrielle

Usage :
  # Entraîner tous les modèles
  python main.py

  # Entraîner un modèle spécifique
  python main.py --model logistic
  python main.py --model rf
  python main.py --model xgboost
  python main.py --model mlp

  # Avec optimisation des hyperparamètres (rf / xgboost uniquement)
  python main.py --model rf --optimize
  python main.py --model xgboost --optimize

  # Sans validation croisée (plus rapide)
  python main.py --no-cv

  # Chemin personnalisé vers les données
  python main.py --data path/to/data.csv
"""

import argparse
import json
from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # backend non-interactif — avant tout import pyplot

# ─── Imports locaux ───────────────────────────────────────────────────────────
from src.preprocessing import load_and_split, build_preprocessor
from src import model_logistic, model_random_forest, model_xgboost, model_mlp
from src.evaluation import plot_comparison_table
from src.minio_loader import MinIOClient

# ─── Chemins ──────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent
DATA_PATH    = ROOT / "data" / "raw" / "predictive_maintenance_v3.csv"
MODELS_DIR   = ROOT / "models"
RESULTS_DIR  = ROOT / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)


# ─── Pipeline complet pour un modèle ─────────────────────────────────────────

def run_logistic(X_train, X_test, y_train, y_test, preprocessor,
                 run_cv: bool = True) -> dict:
    print("\n" + "=" * 60)
    print("  MODÈLE 1 — RÉGRESSION LOGISTIQUE (Baseline)")
    print("=" * 60)

    pipeline = model_logistic.train(X_train, y_train, preprocessor)

    if run_cv:
        model_logistic.cross_validate_model(pipeline, X_train, y_train)

    metrics = model_logistic.evaluate(pipeline, X_test, y_test,
                                      results_dir=RESULTS_DIR)

    print("\n── Interprétabilité (coefficients) ──")
    model_logistic.get_feature_importance(pipeline)

    model_logistic.save_model(pipeline, MODELS_DIR / "logistic_regression.pkl")

    return metrics


def run_random_forest(X_train, X_test, y_train, y_test, preprocessor,
                      run_cv: bool = True, optimize: bool = False) -> dict:
    print("\n" + "=" * 60)
    print("  MODÈLE 2 — RANDOM FOREST")
    print("=" * 60)

    pipeline = model_random_forest.train(X_train, y_train, preprocessor,
                                         optimize=optimize)

    if run_cv:
        model_random_forest.cross_validate_model(pipeline, X_train, y_train)

    metrics = model_random_forest.evaluate(pipeline, X_test, y_test,
                                           results_dir=RESULTS_DIR)

    print("\n── Interprétabilité (feature importance Gini) ──")
    model_random_forest.get_feature_importance(pipeline)

    model_random_forest.save_model(pipeline, MODELS_DIR / "random_forest.pkl")

    return metrics


def run_xgboost(X_train, X_test, y_train, y_test, preprocessor,
                run_cv: bool = True, optimize: bool = False) -> dict:
    print("\n" + "=" * 60)
    print("  MODÈLE 3 — XGBOOST (Gradient Boosting)")
    print("=" * 60)

    pipeline = model_xgboost.train(X_train, y_train, preprocessor,
                                   optimize=optimize)

    if run_cv:
        model_xgboost.cross_validate_model(pipeline, X_train, y_train)

    metrics = model_xgboost.evaluate(pipeline, X_test, y_test,
                                     results_dir=RESULTS_DIR)

    print("\n── Interprétabilité (feature importance gain) ──")
    model_xgboost.get_feature_importance(pipeline)

    model_xgboost.save_model(pipeline, MODELS_DIR / "xgboost.pkl")

    return metrics


def run_mlp(X_train, X_test, y_train, y_test, preprocessor,
            run_cv: bool = True) -> dict:
    print("\n" + "=" * 60)
    print("  MODÈLE 4 — MLP (Deep Learning)")
    print("=" * 60)

    # MLP nécessite un preprocessor frais (non-fitté) pour la CV
    from src.preprocessing import build_preprocessor
    extra_num = ["temp_delta", "age_vibration", "vibration_per_rpm"]
    preprocessor_fresh = build_preprocessor(extra_num_features=extra_num)

    pipeline, history = model_mlp.train(X_train, y_train, preprocessor_fresh)

    print("\n── Courbe d'apprentissage ──")
    model_mlp.plot_learning_curve(history)

    if run_cv:
        preprocessor_cv = build_preprocessor(extra_num_features=extra_num)
        model_mlp.cross_validate_model(preprocessor_cv, X_train, y_train, n_splits=3)

    metrics = model_mlp.evaluate(pipeline, X_test, y_test,
                                 results_dir=RESULTS_DIR)

    model_mlp.save_model(pipeline, MODELS_DIR / "mlp.pkl")

    return metrics


# ─── Sauvegarde des métriques ─────────────────────────────────────────────────

def save_metrics(all_metrics: list[dict]) -> None:
    df = pd.DataFrame(all_metrics)
    df.to_csv(RESULTS_DIR / "metrics_comparison.csv", index=False)
    print(f"\n💾 Métriques sauvegardées : {RESULTS_DIR / 'metrics_comparison.csv'}")

    with open(RESULTS_DIR / "metrics_comparison.json", "w") as f:
        json.dump(all_metrics, f, indent=2)


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Maintenance Prédictive — Pipeline de modélisation"
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["all", "logistic", "rf", "xgboost", "mlp"],
        help="Modèle à entraîner (défaut: all)"
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="Activer la recherche d'hyperparamètres (rf / xgboost)"
    )
    parser.add_argument(
        "--no-cv", action="store_true",
        help="Désactiver la validation croisée (plus rapide)"
    )
    parser.add_argument(
        "--data", type=str, default=str(DATA_PATH),
        help="Chemin vers le CSV"
    )
    parser.add_argument(
        "--from-minio", action="store_true",
        help="Charger le dataset depuis MinIO (critère RNCP C3)"
    )
    parser.add_argument(
        "--shap", action="store_true",
        help="Générer les graphiques SHAP après entraînement"
    )
    parser.add_argument(
        "--upload-minio", action="store_true",
        help="Uploader modèles et résultats vers MinIO après entraînement"
    )
    args = parser.parse_args()

    run_cv = not args.no_cv

    print("=" * 60)
    print("  MAINTENANCE PRÉDICTIVE — MODÉLISATION")
    print("=" * 60)
    print(f"  Modèle(s)    : {args.model}")
    print(f"  Optimisation : {'Oui' if args.optimize else 'Non'}")
    print(f"  Cross-val    : {'Oui' if run_cv else 'Non'}")
    print(f"  Données      : {args.data}")

    # ── Chargement & split ──
    data_path = args.data

    if args.from_minio:
        print("\n[MinIO] Tentative de chargement depuis MinIO...")
        minio = MinIOClient()
        df_minio = minio.load_dataset_to_df(local_fallback=data_path)
        if df_minio is not None:
            # Sauvegarder localement pour load_and_split
            Path(data_path).parent.mkdir(parents=True, exist_ok=True)
            df_minio.to_csv(data_path, index=False)
            print(f"[MinIO] Dataset sauvegardé localement : {data_path}")
        else:
            print("[MinIO] Échec MinIO — utilisation fichier local.")

    X_train, X_test, y_train, y_test, preprocessor = load_and_split(data_path)

    all_metrics = []
    model_choice = args.model

    # ── Exécution des modèles ──
    if model_choice in ("all", "logistic"):
        from src.preprocessing import build_preprocessor
        extra_num = ["temp_delta", "age_vibration", "vibration_per_rpm"]
        prep = build_preprocessor(extra_num_features=extra_num)
        m = run_logistic(X_train, X_test, y_train, y_test, prep, run_cv)
        all_metrics.append(m)

    if model_choice in ("all", "rf"):
        from src.preprocessing import build_preprocessor
        extra_num = ["temp_delta", "age_vibration", "vibration_per_rpm"]
        prep = build_preprocessor(extra_num_features=extra_num)
        m = run_random_forest(X_train, X_test, y_train, y_test, prep,
                              run_cv, args.optimize)
        all_metrics.append(m)

    if model_choice in ("all", "xgboost"):
        from src.preprocessing import build_preprocessor
        extra_num = ["temp_delta", "age_vibration", "vibration_per_rpm"]
        prep = build_preprocessor(extra_num_features=extra_num)
        m = run_xgboost(X_train, X_test, y_train, y_test, prep,
                        run_cv, args.optimize)
        all_metrics.append(m)

    if model_choice in ("all", "mlp"):
        m = run_mlp(X_train, X_test, y_train, y_test, preprocessor, run_cv)
        all_metrics.append(m)

    # ── Comparaison finale ──
    if len(all_metrics) > 1:
        print("\n" + "=" * 60)
        print("  COMPARAISON FINALE")
        print("=" * 60)
        df_final = plot_comparison_table(
            all_metrics,
            save_path=str(RESULTS_DIR / "comparison_finale.png")
        )
        save_metrics(all_metrics)

        # Recommandation automatique
        best = df_final["pr_auc"].idxmax()
        print(f"\n[Résultat] Meilleur modèle (PR-AUC) : {best}")
        print("   → Recommandé pour le déploiement en production")
        print("   → Vérifier aussi Recall et interprétabilité avant décision finale")

    # ── SHAP (explicabilité) ──
    if args.shap and (MODELS_DIR / "xgboost.pkl").exists():
        print("\n" + "=" * 60)
        print("  ANALYSE SHAP — EXPLICABILITÉ")
        print("=" * 60)
        try:
            from src.explainability import run_full_analysis
            run_full_analysis(
                model_path=str(MODELS_DIR / "xgboost.pkl"),
                data_path=data_path,
                results_dir=str(RESULTS_DIR),
            )
        except ImportError:
            import subprocess, sys
            subprocess.run([sys.executable, "src/explainability.py",
                            "--model", str(MODELS_DIR / "xgboost.pkl"),
                            "--data", str(data_path)], check=False)

    # ── Upload MinIO ──
    if args.upload_minio:
        print("\n" + "=" * 60)
        print("  UPLOAD MINIO — CLOUD DATA MANAGEMENT")
        print("=" * 60)
        minio = MinIOClient()
        minio.upload_dataset(data_path)
        minio.upload_all_models(MODELS_DIR)
        minio.upload_results(RESULTS_DIR)
        print("[MinIO] Upload complet.")
        print(json.dumps(minio.status(), indent=2))


if __name__ == "__main__":
    main()
