"""
model_mlp.py
────────────
Modèle 4 : MLP — Multi-Layer Perceptron (Deep Learning)

Rôle dans le projet :
  - Modèle Deep Learning obligatoire (exigence du sujet)
  - Hypothèse : les couches cachées apprennent des représentations
                hiérarchiques des interactions entre capteurs
  - Forces    : capture les non-linéarités complexes sans feature engineering,
                flexible (architecture ajustable)
  - Limites   : nécessite une normalisation stricte (déjà faite dans le pipeline),
                plus lent à entraîner, moins interprétable (boîte noire),
                risque d'overfitting sur données tabulaires limitées,
                pas forcément supérieur à XGBoost sur ce type de données

Point pédagogique important :
  Sur des données tabulaires (~24k obs, ~10 features), le MLP ne dépasse
  pas toujours XGBoost. C'est précisément ce que nous allons montrer et
  argumenter dans le rapport (compromis performance / complexité).

Stratégie anti-déséquilibre :
  - class_weight dans sklearn MLP n'est pas supporté nativement
  - On utilise sample_weight dans fit() pour compenser
  - Alternative : ajustement du seuil de décision
"""

import matplotlib
matplotlib.use('Agg')  # backend non-interactif — avant tout import de pyplot

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.evaluation import (
    compute_metrics, print_metrics,
    find_optimal_threshold, plot_model_evaluation,
)


# ─── Entraînement ─────────────────────────────────────────────────────────────

def train(X_train, y_train, preprocessor):
    """
    Construit et entraîne le pipeline MLP.

    Architecture choisie :
    - 3 couches cachées : (128, 64, 32)
      → Largeur décroissante = compression progressive des représentations
      → Pas trop profond pour éviter l'overfitting sur ~19k exemples
    - Activation ReLU : standard, rapide, évite le vanishing gradient
    - Régularisation L2 via alpha=0.001
    - early_stopping=True : arrêt si val_loss ne s'améliore plus
      → Protection contre l'overfitting (critique ici)

    Pourquoi pas plus profond ?
    - Avec ~19k observations et ~12 features, un réseau trop profond
      va mémoriser le train set (overfitting)
    - Le compromis biais/variance favorise une architecture modeste
    """
    print("Entraînement MLP (Deep Learning)...")

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    X_train_prep = preprocessor.fit_transform(X_train)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False,
    )

    mlp.fit(X_train_prep, y_train, sample_weight=sample_weights)

    # best_loss_ peut être None avec certaines versions sklearn
    # on utilise la dernière valeur de loss_curve_ comme fallback
    best_loss = (
        mlp.best_loss_
        if mlp.best_loss_ is not None
        else mlp.loss_curve_[-1]
    )

    history = {
        "train_loss": mlp.loss_curve_,
        "val_loss":   getattr(mlp, "validation_scores_", None),
        "n_iter":     mlp.n_iter_,
        "best_loss":  best_loss,
    }

    print(f"Entraînement terminé après {mlp.n_iter_} itérations.")
    print(f"   Meilleure loss : {best_loss:.4f}")

    pipeline = _build_fitted_pipeline(preprocessor, mlp)
    return pipeline, history


# Classe définie au niveau du module pour permettre la sérialisation joblib
class FittedMLPPipeline:
    def __init__(self, prep, model):
        self.preprocessor = prep
        self.classifier   = model
        self.named_steps  = {"preprocessor": prep, "classifier": model}

    def predict_proba(self, X):
        X_prep = self.preprocessor.transform(X)
        return self.classifier.predict_proba(X_prep)

    def predict(self, X):
        X_prep = self.preprocessor.transform(X)
        return self.classifier.predict(X_prep)


def _build_fitted_pipeline(preprocessor, mlp):
    """
    Wrapper pipeline-like pour compatibilité avec les fonctions d'évaluation.
    Utilise FittedMLPPipeline défini au niveau du module pour permettre joblib.
    """
    return FittedMLPPipeline(preprocessor, mlp)


# ─── Courbe d'apprentissage ───────────────────────────────────────────────────

def plot_learning_curve(history: dict, save_path: str = None) -> None:
    """
    Visualise la convergence du MLP.

    Points clés à analyser :
    - Les courbes train/val convergent-elles ? → pas d'overfitting
    - Y a-t-il un gap train/val ?             → overfitting détecté
    - La courbe est-elle stable ?             → learning_rate adapté
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(history["train_loss"], color="#3498db", linewidth=2,
            label="Loss entraînement")

    if history["val_loss"] is not None:
        ax.plot(history["val_loss"], color="#e74c3c", linewidth=2,
                linestyle="--", label="Score validation")

    ax.axvline(history["n_iter"] - 1, color="green", linestyle=":",
               linewidth=1.5, label=f"Arrêt (iter {history['n_iter']})")

    ax.set_xlabel("Itération")
    ax.set_ylabel("Loss / Score")
    ax.set_title(
        f"MLP — Courbe d'apprentissage ({history['n_iter']} itérations)\n"
        "Convergence train vs validation — détection overfitting",
        fontsize=11
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   Courbe sauvegardée : {save_path}")
    plt.close()

    # Analyse biais / variance
    best_loss = history["best_loss"]
    best_loss_str = f"{best_loss:.4f}" if best_loss is not None else "N/A"

    print("\n── Analyse Biais / Variance ──")
    print(f"  Itérations effectuées : {history['n_iter']}")
    print(f"  Meilleure loss        : {best_loss_str}")

    if history["n_iter"] < 50:
        print("  Peu d'itérations → possible underfitting (augmenter max_iter)")
    elif history["n_iter"] >= 180:
        print("  Convergence lente → vérifier learning_rate ou architecture")
    else:
        print("  Convergence normale")


# ─── Validation croisée ───────────────────────────────────────────────────────

def cross_validate_model(preprocessor, X_train, y_train,
                          n_splits: int = 3) -> dict:
    """
    Validation croisée stratifiée sur le MLP.
    3 folds (MLP plus lent que les autres modèles).

    Le preprocessing est refitté sur chaque fold train uniquement
    pour éviter tout data leakage entre folds.
    """
    from sklearn.metrics import (
        recall_score, f1_score, roc_auc_score, average_precision_score
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {"recall": [], "f1": [], "roc_auc": [], "pr_auc": []}

    print(f"\n── Validation croisée MLP ({n_splits} folds stratifiés) ──")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr  = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr  = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        # Preprocessor refitté sur le fold train uniquement
        from sklearn.base import clone
        prep_fold = clone(preprocessor)
        X_tr_prep  = prep_fold.fit_transform(X_tr)
        X_val_prep = prep_fold.transform(X_val)

        sw = compute_sample_weight(class_weight="balanced", y=y_tr)

        mlp_fold = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=256,
            learning_rate="adaptive",
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=42,
            verbose=False,
        )
        mlp_fold.fit(X_tr_prep, y_tr, sample_weight=sw)

        y_proba_val = mlp_fold.predict_proba(X_val_prep)[:, 1]
        y_pred_val  = (y_proba_val >= 0.5).astype(int)

        results["recall"].append(
            recall_score(y_val, y_pred_val, zero_division=0))
        results["f1"].append(
            f1_score(y_val, y_pred_val, zero_division=0))
        results["roc_auc"].append(
            roc_auc_score(y_val, y_proba_val))
        results["pr_auc"].append(
            average_precision_score(y_val, y_proba_val))

        print(f"  Fold {fold+1} : recall={results['recall'][-1]:.4f} | "
              f"f1={results['f1'][-1]:.4f} | "
              f"pr_auc={results['pr_auc'][-1]:.4f}")

    print("\n  Moyennes :")
    for metric, scores in results.items():
        arr = np.array(scores)
        print(f"  {metric:<10} : {arr.mean():.4f} ± {arr.std():.4f}")

    return results


# ─── Évaluation sur le test set ───────────────────────────────────────────────

def evaluate(pipeline, X_test, y_test,
             results_dir: Path = None) -> dict:
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    opt_thresh = find_optimal_threshold(y_test, y_proba, metric="f1")
    y_pred = (y_proba >= opt_thresh).astype(int)

    print(f"\n  Seuil de décision optimisé : {opt_thresh:.3f}")

    metrics = compute_metrics(y_test, y_pred, y_proba, "MLP")
    print_metrics(metrics)

    save_path = str(results_dir / "mlp_evaluation.png") if results_dir else None
    plot_model_evaluation(y_test, y_pred, y_proba,
                          "MLP (Deep Learning)",
                          save_path=save_path)
    return metrics


# ─── Sauvegarde ───────────────────────────────────────────────────────────────

def save_model(pipeline, path: Path) -> None:
    joblib.dump(pipeline, path)
    print(f"Modèle sauvegardé : {path}")


def load_model(path: Path):
    return joblib.load(path)