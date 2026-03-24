"""
model_random_forest.py
──────────────────────
Modèle 2 : Random Forest

Rôle dans le projet :
  - Premier modèle non-linéaire, référence ensemble
  - Hypothèse : combinaison de décisions arborescentes = meilleure
                que chaque arbre seul (bagging)
  - Forces    : gère la multicolinéarité (current_phase_avg/vibration),
                feature importance native, peu sensible aux outliers,
                pas besoin de normalisation (mais on la garde pour homogénéité)
  - Limites   : peut sur-apprendre si n_estimators trop faible,
                moins interprétable que la LR

Stratégie anti-déséquilibre :
  - class_weight='balanced_subsample' : équilibre dans chaque bootstrap
  - Plus robuste que 'balanced' pour les ensembles de méthodes
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV

from src.evaluation import (
    compute_metrics, print_metrics,
    find_optimal_threshold, plot_model_evaluation,
)


# ─── Entraînement ─────────────────────────────────────────────────────────────

def train(X_train, y_train, preprocessor,
          optimize: bool = False) -> Pipeline:
    """
    Construit et entraîne le pipeline Random Forest.

    Args:
        optimize : si True, lance une recherche d'hyperparamètres
                   (RandomizedSearchCV, ~2-3 min)
    """
    if optimize:
        print("🔍 Optimisation des hyperparamètres (RandomizedSearchCV)...")
        pipeline = _train_with_search(X_train, y_train, preprocessor)
    else:
        print("⏳ Entraînement Random Forest (paramètres par défaut optimisés)...")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=5,
                min_samples_split=10,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            )),
        ])
        pipeline.fit(X_train, y_train)

    print("✅ Entraînement terminé.")
    return pipeline


def _train_with_search(X_train, y_train, preprocessor) -> Pipeline:
    """
    Recherche aléatoire d'hyperparamètres sur le Random Forest.
    Scoring = PR-AUC (adapté au déséquilibre).
    """
    param_grid = {
        "classifier__n_estimators":    [100, 200, 300],
        "classifier__max_depth":       [10, 15, 20, None],
        "classifier__min_samples_leaf":[2, 5, 10],
        "classifier__min_samples_split":[5, 10, 20],
        "classifier__max_features":    ["sqrt", "log2"],
    }

    base_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_grid,
        n_iter=20,
        scoring="average_precision",  # PR-AUC
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"   Meilleurs params : {search.best_params_}")
    print(f"   Meilleur PR-AUC  : {search.best_score_:.4f}")
    return search.best_estimator_


# ─── Validation croisée ───────────────────────────────────────────────────────

def cross_validate_model(pipeline, X_train, y_train, n_splits: int = 5) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring={
            "recall":  "recall",
            "f1":      "f1",
            "roc_auc": "roc_auc",
            "pr_auc":  "average_precision",
        },
        return_train_score=True,   # pour détecter l'overfitting
        n_jobs=-1,
    )

    print(f"\n── Validation croisée ({n_splits} folds stratifiés) ──")
    for metric in ["recall", "f1", "pr_auc"]:
        train_s = cv_results[f"train_{metric}"]
        test_s  = cv_results[f"test_{metric}"]
        gap = train_s.mean() - test_s.mean()
        overfit = "⚠️ overfit" if gap > 0.1 else "✅"
        print(f"  {metric:<10} train={train_s.mean():.4f} | "
              f"test={test_s.mean():.4f} ± {test_s.std():.4f}  {overfit}")

    return cv_results


# ─── Évaluation sur le test set ───────────────────────────────────────────────

def evaluate(pipeline, X_test, y_test,
             results_dir: Path = None) -> dict:
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    opt_thresh = find_optimal_threshold(y_test, y_proba, metric="f1")
    y_pred = (y_proba >= opt_thresh).astype(int)

    print(f"\n  Seuil de décision optimisé : {opt_thresh:.3f}")

    metrics = compute_metrics(y_test, y_pred, y_proba, "Random Forest")
    print_metrics(metrics)

    save_path = str(results_dir / "rf_evaluation.png") if results_dir else None
    plot_model_evaluation(y_test, y_pred, y_proba,
                          "Random Forest",
                          save_path=save_path)
    return metrics


# ─── Interprétabilité ─────────────────────────────────────────────────────────

def get_feature_importance(pipeline, top_n: int = 15) -> pd.DataFrame:
    """
    Feature importance native du Random Forest (réduction d'impureté Gini).
    Plus stable que les coefficients LR mais attention au biais
    vers les variables continues et à forte cardinalité.
    """
    rf = pipeline.named_steps["classifier"]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

    imp_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
        "std":        std,
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1],
            xerr=imp_df["std"][::-1],
            color="#27ae60", alpha=0.8, capsize=3)
    ax.set_title(f"Random Forest — Feature Importance (top {top_n})\n"
                 "Basée sur la réduction d'impureté Gini (± std entre arbres)",
                 fontsize=11)
    ax.set_xlabel("Importance moyenne")
    plt.tight_layout()
    plt.close()

    return imp_df


# ─── Sauvegarde ───────────────────────────────────────────────────────────────

def save_model(pipeline, path: Path) -> None:
    joblib.dump(pipeline, path)
    print(f"💾 Modèle sauvegardé : {path}")


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)
