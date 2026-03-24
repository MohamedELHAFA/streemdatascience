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

# ─── Imports locaux ───────────────────────────────────────────────────────────
from src.preprocessing import load_and_split, build_preprocessor
from src import model_logistic, model_random_forest, model_xgboost, model_mlp
from src.evaluation import plot_comparison_table
import matplotlib
matplotlib.use('Agg')  # backend non-interactif, pas de fenêtre

# ─── Chemins ──────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent
DATA_PATH    = ROOT / "data" / "raw" / "predictive_maintenance_v3.csv"
MODELS_DIR   = ROOT / "models"
RESULTS_DIR  = ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


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
    X_train, X_test, y_train, y_test, preprocessor = load_and_split(args.data)

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
        print(f"\n🏆 Meilleur modèle (PR-AUC) : {best}")
        print("   → Recommandé pour le déploiement en production")
        print("   → Vérifier aussi Recall et interprétabilité avant décision finale")


if __name__ == "__main__":
    main()
