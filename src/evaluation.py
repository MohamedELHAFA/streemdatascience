"""
evaluation.py
─────────────
Fonctions d'évaluation communes à tous les modèles.

Nouveautés v2 :
  - find_optimal_threshold supporte 3 stratégies : f1, recall, recall_constrained
  - plot_threshold_analysis : visualise le compromis Recall/Precision/FN selon le seuil
  - Emojis retirés des titres matplotlib (compatibilité DejaVu Sans)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    recall_score,
    precision_score,
)


# ─── Métriques synthétiques ───────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba, model_name: str = "") -> dict:
    return {
        "model":       model_name,
        "accuracy":    (y_true == y_pred).mean(),
        "recall_1":    recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_1":        f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_macro":    f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc":     roc_auc_score(y_true, y_proba),
        "pr_auc":      average_precision_score(y_true, y_proba),
        "fn_count":    int(((y_true == 1) & (y_pred == 0)).sum()),
        "fp_count":    int(((y_true == 0) & (y_pred == 1)).sum()),
    }


def print_metrics(metrics: dict):
    fn  = metrics.get("fn_count", "?")
    fp  = metrics.get("fp_count", "?")
    print(f"\n{'─'*55}")
    print(f"  {metrics['model']}")
    print(f"{'─'*55}")
    print(f"  {'Accuracy':<25} : {metrics['accuracy']:.4f}  (attention : trompeuse)")
    print(f"  {'Recall (classe 1)':<25} : {metrics['recall_1']:.4f}  << PRIORITAIRE")
    print(f"  {'Precision (classe 1)':<25} : {metrics['precision_1']:.4f}")
    print(f"  {'F1 (classe 1)':<25} : {metrics['f1_1']:.4f}")
    print(f"  {'F1 macro':<25} : {metrics['f1_macro']:.4f}")
    print(f"  {'ROC-AUC':<25} : {metrics['roc_auc']:.4f}")
    print(f"  {'PR-AUC':<25} : {metrics['pr_auc']:.4f}  << PRIORITAIRE")
    print(f"  {'Pannes manquees (FN)':<25} : {fn}")
    print(f"  {'Fausses alertes (FP)':<25} : {fp}")
    cost_fn = fn * 4112 if isinstance(fn, int) else "?"
    cost_fp = fp * 200  if isinstance(fp, int) else "?"
    print(f"  Cout estime FN       : ~{cost_fn:,} EUR")
    print(f"  Cout estime FP       : ~{cost_fp:,} EUR")


# ─── Seuil de décision ────────────────────────────────────────────────────────

def find_optimal_threshold(y_true, y_proba,
                            metric: str = "f1",
                            min_precision: float = 0.50) -> float:
    """
    Trouve le seuil optimal selon la stratégie choisie.

    Args:
        metric :
          "f1"                → maximise le F1 (équilibré)
          "recall"            → maximise le Recall sans contrainte
          "recall_constrained"→ maximise le Recall avec Precision >= min_precision

    Returns:
        seuil optimal (float)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    if metric == "f1":
        f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
        best_idx = np.argmax(f1_scores[:-1])

    elif metric == "recall":
        # Recall pur — seuil le plus bas possible
        best_idx = len(thresholds) - 1  # seuil minimum → recall maximum

    elif metric == "recall_constrained":
        # Maximise Recall avec Precision >= min_precision
        # Compromis métier : acceptable x20 FN/FP mais pas infini
        valid = precision[:-1] >= min_precision
        if valid.any():
            best_idx = np.argmax(recall[:-1] * valid)
        else:
            # Fallback si aucun seuil ne satisfait la contrainte
            best_idx = np.argmax(recall[:-1])
            print(f"  Attention : aucun seuil avec Precision >= {min_precision:.0%}")
            print(f"  Fallback sur Recall pur.")

    return float(thresholds[best_idx])


def apply_threshold(y_proba, threshold: float):
    return (y_proba >= threshold).astype(int)


# ─── Analyse de l'impact du seuil ────────────────────────────────────────────

def plot_threshold_analysis(y_true, y_proba, model_name: str,
                             save_path: str = None):
    """
    Visualise comment le seuil affecte Recall, Precision, F1 et le
    nombre de pannes manquées (FN) — le plus important métriquement.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)

    # Calcul des FN pour chaque seuil
    n_pannes = (y_true == 1).sum()
    fn_counts = [int(n_pannes * (1 - r)) for r in recall[:-1]]
    fp_counts = []
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        fp_counts.append(int(((y_true == 0) & (y_pred_t == 1)).sum()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Impact du seuil de décision — {model_name}", fontsize=13,
                 fontweight="bold")

    # Graphique 1 : Precision / Recall / F1 vs seuil
    ax1 = axes[0]
    ax1.plot(thresholds, precision[:-1], color="#3498db", lw=2, label="Precision")
    ax1.plot(thresholds, recall[:-1],    color="#e74c3c", lw=2, label="Recall")
    ax1.plot(thresholds, f1_scores,      color="#2ecc71", lw=2,
             linestyle="--", label="F1-score")

    # Marquer les 3 seuils clés
    thresh_f1  = find_optimal_threshold(y_true, y_proba, "f1")
    thresh_rec = find_optimal_threshold(y_true, y_proba, "recall_constrained",
                                         min_precision=0.50)

    for thresh, color, label in [
        (0.50,      "gray",    "Defaut (0.50)"),
        (thresh_f1, "#e67e22", f"Optimal F1 ({thresh_f1:.2f})"),
        (thresh_rec,"#9b59b6", f"Max Recall P>=50% ({thresh_rec:.2f})"),
    ]:
        ax1.axvline(thresh, color=color, linestyle=":", lw=1.5, label=label)

    ax1.set_xlabel("Seuil de décision")
    ax1.set_ylabel("Score")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8, loc="center left")
    ax1.set_title("Precision / Recall / F1 selon le seuil")

    # Graphique 2 : FN et coût selon le seuil
    ax2 = axes[1]
    color_fn = "#e74c3c"
    color_fp = "#3498db"
    ax2.plot(thresholds, fn_counts, color=color_fn, lw=2,
             label="Pannes manquees (FN)")
    ax2.plot(thresholds, fp_counts, color=color_fp, lw=2,
             linestyle="--", label="Fausses alertes (FP)")

    for thresh, color, label in [
        (0.50,      "gray",    "Defaut (0.50)"),
        (thresh_f1, "#e67e22", f"Optimal F1 ({thresh_f1:.2f})"),
        (thresh_rec,"#9b59b6", f"Max Recall ({thresh_rec:.2f})"),
    ]:
        ax2.axvline(thresh, color=color, linestyle=":", lw=1.5, label=label)

    ax2.set_xlabel("Seuil de décision")
    ax2.set_ylabel("Nombre d'observations")
    ax2.set_title("Pannes manquees (FN) et fausses alertes (FP) selon le seuil")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   Graphique seuil sauvegarde : {save_path}")
    plt.close()

    # Tableau comparatif des 3 seuils
    print("\n── Comparaison des seuils de decision ──")
    print(f"  {'Strategie':<28} {'Seuil':>7} {'Recall':>8} {'Precision':>10} "
          f"{'F1':>7} {'FN':>6} {'FP':>6} {'Cout FN':>10}")
    print("  " + "-" * 85)

    for thresh, label in [
        (0.50,      "Defaut (0.50)"),
        (thresh_f1, "Optimal F1"),
        (thresh_rec,"Max Recall (P>=50%)"),
    ]:
        y_pred = apply_threshold(y_proba, thresh)
        r  = recall_score(y_true, y_pred, zero_division=0)
        p  = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        cout = fn * 4112
        print(f"  {label:<28} {thresh:>7.3f} {r:>8.3f} {p:>10.3f} "
              f"{f1:>7.3f} {fn:>6} {fp:>6} {cout:>10,}")


# ─── Évaluation complète d'un modèle ─────────────────────────────────────────

def plot_model_evaluation(y_true, y_pred, y_proba, model_name: str,
                           save_path: str = None):
    """4 graphiques d'évaluation : confusion, ROC, PR, distribution proba."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Evaluation — {model_name}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Matrice de confusion
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    labels = np.array([[f"{v}\n({p:.1f}%)" for v, p in zip(row_v, row_p)]
                       for row_v, row_p in zip(cm, cm_pct)])
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", ax=ax1,
                xticklabels=["Predit Normal", "Predit Panne"],
                yticklabels=["Reel Normal", "Reel Panne"],
                linewidths=1, cbar=False)
    fn = cm[1, 0]
    ax1.add_patch(plt.Rectangle((0, 1), 1, 1, fill=False,
                                 edgecolor="#e74c3c", lw=3))
    ax1.set_title(f"Matrice de confusion  (FN={fn} pannes manquees)")

    # 2. Courbe ROC
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    ax2.plot(fpr, tpr, color="#3498db", lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatoire")
    ax2.fill_between(fpr, tpr, alpha=0.1, color="#3498db")
    ax2.set_xlabel("Taux de Faux Positifs")
    ax2.set_ylabel("Recall")
    ax2.set_title("Courbe ROC")
    ax2.legend(loc="lower right")

    # 3. Courbe Precision-Recall
    ax3 = fig.add_subplot(gs[1, 0])
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    baseline = y_true.mean()
    ax3.plot(recall, precision, color="#e74c3c", lw=2,
             label=f"PR-AUC = {pr_auc:.3f}")
    ax3.axhline(baseline, color="gray", linestyle="--", lw=1,
                label=f"Baseline ({baseline:.2f})")
    ax3.fill_between(recall, precision, alpha=0.1, color="#e74c3c")
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.set_title("Courbe Precision-Recall (metrique prioritaire)")
    ax3.legend()

    # 4. Distribution des probabilités
    ax4 = fig.add_subplot(gs[1, 1])
    for cls, color, label in [(0, "#2ecc71", "Normal"), (1, "#e74c3c", "Panne")]:
        mask = y_true == cls
        ax4.hist(y_proba[mask], bins=50, alpha=0.5, color=color,
                 label=f"{label} (n={mask.sum():,})", density=True)
    opt_thresh = find_optimal_threshold(y_true, y_proba, "f1")
    ax4.axvline(0.5,        color="gray",    linestyle="--", lw=1.5,
                label="Seuil 0.5")
    ax4.axvline(opt_thresh, color="#e67e22", linestyle="--", lw=1.5,
                label=f"Seuil optimal F1 ({opt_thresh:.2f})")
    ax4.set_xlabel("Probabilite predite (classe 1)")
    ax4.set_ylabel("Densite")
    ax4.set_title("Distribution des probabilites")
    ax4.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   Graphique sauvegarde : {save_path}")
    plt.close()


# ─── Tableau comparatif final ─────────────────────────────────────────────────

def plot_comparison_table(all_metrics: list, save_path: str = None):
    df_metrics = pd.DataFrame(all_metrics).set_index("model")
    df_metrics = df_metrics.sort_values("pr_auc", ascending=False)

    print("\n" + "=" * 70)
    print("  TABLEAU COMPARATIF — TOUS LES MODELES")
    print("=" * 70)
    cols_display = ["recall_1", "precision_1", "f1_1", "pr_auc",
                    "roc_auc", "fn_count", "fp_count"]
    print(df_metrics[[c for c in cols_display if c in df_metrics.columns]]
          .round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Comparaison des modeles", fontsize=14, fontweight="bold")

    metrics_plot = ["recall_1", "f1_1", "pr_auc", "roc_auc"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    x = np.arange(len(df_metrics))
    width = 0.2

    for i, (metric, color) in enumerate(zip(metrics_plot, colors)):
        if metric in df_metrics.columns:
            axes[0].bar(x + i * width, df_metrics[metric],
                        width, label=metric, color=color, alpha=0.85)

    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(df_metrics.index, rotation=15, ha="right")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=9)
    axes[0].set_title("Metriques par modele")

    metrics_heat = [m for m in metrics_plot if m in df_metrics.columns]
    sns.heatmap(df_metrics[metrics_heat].T, annot=True, fmt=".3f",
                cmap="YlGn", ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title("Heatmap comparative")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return df_metrics
