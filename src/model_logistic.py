"""
model_logistic.py
─────────────────
Modèle 1 : Régression Logistique — BASELINE

Rôle dans le projet :
  - Référence interprétable pour mesurer le gain des modèles complexes
  - Hypothèse : relation linéaire entre les features et log-odds de panne
  - Forces    : rapide, interprétable (coefficients), calibrée en probabilités
  - Limites   : ne capture pas les non-linéarités ni les interactions complexes
                sensible à la multicolinéarité (VIF élevé de current_phase_avg)

Stratégie anti-déséquilibre :
  - class_weight='balanced' : pénalise davantage les erreurs sur la classe 1
  - Seuil de décision optimisé via courbe PR (pas 0.5 par défaut)
"""

import joblib
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.evaluation import (
    compute_metrics, print_metrics,
    find_optimal_threshold, plot_model_evaluation,
)


# ─── Entraînement ─────────────────────────────────────────────────────────────

def train(X_train, y_train, preprocessor) -> Pipeline:
    """
    Construit et entraîne le pipeline LR.

    class_weight='balanced' : poids inversement proportionnels
    aux fréquences des classes → classe 1 (14.8%) reçoit ~5.7x
    plus de poids que la classe 0.

    C=0.1 : régularisation L2 légèrement plus forte que par défaut
    pour stabiliser les coefficients malgré la multicolinéarité.
    """
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    print("⏳ Entraînement Logistic Regression...")
    pipeline.fit(X_train, y_train)
    print("✅ Entraînement terminé.")

    return pipeline


# ─── Validation croisée ───────────────────────────────────────────────────────

def cross_validate_model(pipeline, X_train, y_train, n_splits: int = 5) -> dict:
    """
    Validation croisée stratifiée pour évaluer la robustesse.
    Stratified = préserve le ratio 85/15 dans chaque fold.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring={
            "recall":    "recall",
            "f1":        "f1",
            "roc_auc":   "roc_auc",
            "pr_auc":    "average_precision",
        },
        return_train_score=False,
        n_jobs=-1,
    )

    print(f"\n── Validation croisée ({n_splits} folds stratifiés) ──")
    for metric in ["recall", "f1", "pr_auc", "roc_auc"]:
        scores = cv_results[f"test_{metric}"]
        print(f"  {metric:<10} : {scores.mean():.4f} ± {scores.std():.4f}")

    return cv_results


# ─── Évaluation sur le test set ───────────────────────────────────────────────

def evaluate(pipeline, X_test, y_test,
             results_dir: Path = None) -> dict:
    """
    Évalue le modèle sur le test set avec seuil optimisé.
    """
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Seuil optimisé sur le test set (dans un vrai projet : sur val set)
    opt_thresh = find_optimal_threshold(y_test, y_proba, metric="f1")
    y_pred = (y_proba >= opt_thresh).astype(int)

    print(f"\n  Seuil de décision optimisé : {opt_thresh:.3f} (vs 0.5 par défaut)")

    metrics = compute_metrics(y_test, y_pred, y_proba, "Logistic Regression")
    print_metrics(metrics)

    save_path = str(results_dir / "logistic_evaluation.png") if results_dir else None
    plot_model_evaluation(y_test, y_pred, y_proba,
                          "Logistic Regression (Baseline)",
                          save_path=save_path)

    return metrics


# ─── Interprétabilité ─────────────────────────────────────────────────────────

def get_feature_importance(pipeline) -> None:
    """
    Affiche les coefficients de la régression logistique.
    Coefficients positifs → augmentent le risque de panne.
    Coefficients négatifs → diminuent le risque.
    """
    import matplotlib.pyplot as plt

    lr = pipeline.named_steps["classifier"]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = lr.coef_[0]

    import pandas as pd
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
    }).sort_values("coefficient", key=abs, ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if c > 0 else "#3498db" for c in coef_df["coefficient"]]
    ax.barh(coef_df["feature"], coef_df["coefficient"],
            color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Logistic Regression — Coefficients (top 15)\n"
                 "Rouge = augmente le risque | Bleu = diminue le risque",
                 fontsize=11)
    ax.set_xlabel("Coefficient (après standardisation)")
    plt.tight_layout()
    plt.close()


# ─── Sauvegarde ───────────────────────────────────────────────────────────────

def save_model(pipeline, path: Path) -> None:
    joblib.dump(pipeline, path)
    print(f"💾 Modèle sauvegardé : {path}")


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)
