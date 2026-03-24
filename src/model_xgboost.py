"""
model_xgboost.py
────────────────
Modèle 3 : XGBoost (Gradient Boosting)

Rôle dans le projet :
  - Modèle le plus performant attendu sur données tabulaires
  - Hypothèse : les arbres s'entraînent séquentiellement pour corriger
                les erreurs des précédents (boosting)
  - Forces    : excellent sur données tabulaires hétérogènes,
                gère nativement les valeurs manquantes,
                régularisation L1/L2 intégrée (réduit overfitting),
                feature importance + compatible SHAP
  - Limites   : plus lent que RF, plus d'hyperparamètres à tuner,
                peut sur-apprendre si learning_rate trop élevé

Stratégie anti-déséquilibre :
  - scale_pos_weight = n_négatifs / n_positifs ≈ 5.75
    → équivalent à class_weight pour XGBoost
    → la classe minoritaire (pannes) reçoit un poids ~5.75x plus élevé
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV

from src.evaluation import (
    compute_metrics, print_metrics,
    find_optimal_threshold, plot_model_evaluation,
)


# ─── Calcul du scale_pos_weight ───────────────────────────────────────────────

def _get_scale_pos_weight(y_train) -> float:
    """
    scale_pos_weight = n_négatifs / n_positifs
    Compense le déséquilibre des classes dans XGBoost.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = n_neg / n_pos
    print(f"   scale_pos_weight = {n_neg}/{n_pos} = {spw:.2f}")
    return spw


# ─── Entraînement ─────────────────────────────────────────────────────────────

def train(X_train, y_train, preprocessor,
          optimize: bool = False) -> Pipeline:
    """
    Construit et entraîne le pipeline XGBoost.

    Paramètres clés :
    - n_estimators=300      : nombre d'arbres (avec early stopping idéalement)
    - max_depth=6           : profondeur standard XGBoost (évite overfitting)
    - learning_rate=0.05    : faible LR + plus d'arbres = meilleure généralisation
    - subsample=0.8         : sous-échantillonnage des lignes (comme RF)
    - colsample_bytree=0.8  : sous-échantillonnage des features par arbre
    - reg_alpha=0.1         : régularisation L1 (sparsité)
    - reg_lambda=1.0        : régularisation L2 (lissage)
    """
    spw = _get_scale_pos_weight(y_train)

    if optimize:
        print("🔍 Optimisation des hyperparamètres XGBoost...")
        pipeline = _train_with_search(X_train, y_train, preprocessor, spw)
    else:
        print("⏳ Entraînement XGBoost...")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=spw,
                eval_metric="aucpr",       # PR-AUC comme métrique interne
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
            )),
        ])
        pipeline.fit(X_train, y_train)

    print("✅ Entraînement terminé.")
    return pipeline


def _train_with_search(X_train, y_train, preprocessor, spw) -> Pipeline:
    param_grid = {
        "classifier__n_estimators":   [200, 300, 500],
        "classifier__max_depth":      [4, 6, 8],
        "classifier__learning_rate":  [0.01, 0.05, 0.1],
        "classifier__subsample":      [0.7, 0.8, 0.9],
        "classifier__colsample_bytree": [0.7, 0.8, 1.0],
        "classifier__reg_alpha":      [0, 0.1, 0.5],
        "classifier__reg_lambda":     [0.5, 1.0, 2.0],
    }

    base_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            scale_pos_weight=spw,
            eval_metric="aucpr",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_grid,
        n_iter=25,
        scoring="average_precision",
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
        return_train_score=True,
        n_jobs=-1,
    )

    print(f"\n── Validation croisée XGBoost ({n_splits} folds) ──")
    for metric in ["recall", "f1", "pr_auc"]:
        train_s = cv_results[f"train_{metric}"]
        test_s  = cv_results[f"test_{metric}"]
        gap = train_s.mean() - test_s.mean()
        overfit = "⚠️ overfit" if gap > 0.1 else "✅"
        print(f"  {metric:<10} train={train_s.mean():.4f} | "
              f"test={test_s.mean():.4f} ± {test_s.std():.4f}  {overfit}")

    return cv_results


# ─── Évaluation ───────────────────────────────────────────────────────────────

def evaluate(pipeline, X_test, y_test,
             results_dir: Path = None) -> dict:
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    opt_thresh = find_optimal_threshold(y_test, y_proba, metric="f1")
    y_pred = (y_proba >= opt_thresh).astype(int)

    print(f"\n  Seuil de décision optimisé : {opt_thresh:.3f}")

    metrics = compute_metrics(y_test, y_pred, y_proba, "XGBoost")
    print_metrics(metrics)

    save_path = str(results_dir / "xgboost_evaluation.png") if results_dir else None
    plot_model_evaluation(y_test, y_pred, y_proba, "XGBoost",
                          save_path=save_path)
    return metrics


# ─── Interprétabilité ─────────────────────────────────────────────────────────

def get_feature_importance(pipeline, top_n: int = 15) -> pd.DataFrame:
    """
    Feature importance XGBoost + SHAP si disponible.
    XGBoost propose 3 types d'importance :
    - weight   : nombre de fois qu'une feature est utilisée pour splitter
    - gain     : amélioration moyenne de la perte (plus fiable)
    - cover    : nombre moyen d'observations couvertes
    """
    xgb = pipeline.named_steps["classifier"]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    # Importance par gain (la plus informative)
    imp_gain = xgb.get_booster().get_score(importance_type="gain")

    imp_df = pd.DataFrame([
        {"feature": feature_names[int(k.replace("f", ""))], "importance_gain": v}
        for k, v in imp_gain.items()
    ]).sort_values("importance_gain", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp_df["feature"][::-1], imp_df["importance_gain"][::-1],
            color="#e67e22", alpha=0.85)
    ax.set_title(f"XGBoost — Feature Importance (gain, top {top_n})\n"
                 "Gain = amélioration moyenne de la perte lors des splits",
                 fontsize=11)
    ax.set_xlabel("Importance (gain)")
    plt.tight_layout()
    plt.close()

    # SHAP si disponible
    try:
        import shap
        print("\n🔍 Calcul des valeurs SHAP...")
        explainer = shap.TreeExplainer(xgb)
        preprocessor = pipeline.named_steps["preprocessor"]
        # Appliquer le preprocessing sur un échantillon
        X_sample = preprocessor.transform(
            pipeline.named_steps["preprocessor"]._validate_data
        )
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample,
                          feature_names=feature_names, show=True)
    except Exception:
        print("   (SHAP non disponible — pip install shap pour l'activer)")

    return imp_df


# ─── Sauvegarde ───────────────────────────────────────────────────────────────

def save_model(pipeline, path: Path) -> None:
    joblib.dump(pipeline, path)
    print(f"💾 Modèle sauvegardé : {path}")


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)
