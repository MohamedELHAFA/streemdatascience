"""
explainability.py
─────────────────
Analyse d'explicabilité SHAP pour le modèle XGBoost final.

SHAP (SHapley Additive exPlanations) répond à deux questions :
  - Globale  : quelles features influencent le plus le modèle en général ?
  - Locale   : pourquoi le modèle a-t-il prédit X pour CETTE observation ?

Graphiques générés :
  1. Summary plot    — importance globale + direction d'impact
  2. Beeswarm plot   — distribution des valeurs SHAP par feature
  3. Waterfall plot  — explication d'une prédiction individuelle
  4. Dependence plot — effet d'une feature en fonction de sa valeur
  5. Force plot      — visualisation locale HTML (optionnel)

Usage :
  python src/explainability.py
  python src/explainability.py --model models/xgboost.pkl
  python src/explainability.py --sample 42   # expliquer l'obs n°42 du test set
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

# ─── Chemins par défaut ───────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
MODEL_PATH  = ROOT / "models" / "xgboost.pkl"
DATA_PATH   = ROOT / "data" / "raw" / "predictive_maintenance_v3.csv"
RESULTS_DIR = ROOT / "results"


# ─── Nettoyage des noms de features ──────────────────────────────────────────

def clean_feature_names(names: list) -> list:
    """
    Retire les préfixes sklearn ('num__', 'cat__') pour
    des graphiques SHAP lisibles.

    Ex: 'num__vibration_rms' → 'vibration_rms'
        'cat__operating_mode_peak' → 'operating_mode_peak'
    """
    cleaned = []
    for name in names:
        name = name.replace("num__", "").replace("cat__", "")
        cleaned.append(name)
    return cleaned


# ─── Calcul des valeurs SHAP ─────────────────────────────────────────────────

def compute_shap_values(pipeline, X_test: pd.DataFrame,
                        sample_size: int = 1000):
    """
    Calcule les valeurs SHAP sur un échantillon du test set.

    Utilise TreeExplainer — optimisé pour XGBoost/RF, exact et rapide.
    Contrairement à KernelExplainer (modèle-agnostique), TreeExplainer
    exploite la structure des arbres pour un calcul polynomial.

    Args:
        sample_size : nombre d'observations à expliquer
                      (1000 suffisent pour une analyse globale fiable)
    Returns:
        shap_values   : array (n_samples, n_features)
        X_test_prep   : données préprocessées
        feature_names : noms de features nettoyés
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    xgb_model    = pipeline.named_steps["classifier"]

    # Préprocessing
    X_test_prep = preprocessor.transform(X_test)
    feature_names_raw = list(preprocessor.get_feature_names_out())
    feature_names = clean_feature_names(feature_names_raw)

    # Échantillon aléatoire reproductible
    n = min(sample_size, len(X_test_prep))
    idx = np.random.RandomState(42).choice(len(X_test_prep), n, replace=False)
    X_sample = X_test_prep[idx]

    print(f"Calcul SHAP sur {n} observations...")
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)
    expected_value = explainer.expected_value
    print(f"  Valeur de base (expected value) : {expected_value:.4f}")
    print(f"  Shape SHAP values               : {shap_values.shape}")

    # DataFrame SHAP pour analyse
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    return shap_values, X_sample, feature_names, expected_value, idx, shap_df


# ─── 1. Summary Plot ─────────────────────────────────────────────────────────

def plot_summary(shap_values, X_sample, feature_names: list,
                 save_path: str = None):
    """
    Summary plot — importance globale des features.

    Chaque point = une observation.
    Position X = valeur SHAP (impact sur la prédiction).
    Couleur    = valeur de la feature (rouge=élevée, bleu=faible).

    Lecture :
    - Features en haut = plus importantes globalement
    - Points rouges à droite = valeur élevée augmente le risque de panne
    - Points bleus à droite  = valeur faible augmente le risque de panne
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        show=False,
        max_display=15,
        plot_type="dot",
    )
    plt.title("SHAP Summary Plot — Impact global des features\n"
              "Chaque point = 1 observation | Rouge = valeur élevée | "
              "Droite = augmente le risque de panne",
              fontsize=11, pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Sauvegarde : {save_path}")
    plt.close()


# ─── 2. Beeswarm Plot ────────────────────────────────────────────────────────

def plot_beeswarm(shap_values, X_sample, feature_names: list,
                  save_path: str = None):
    """
    Beeswarm plot — distribution complète des valeurs SHAP.

    Plus dense que le summary plot : montre la distribution
    des impacts pour chaque feature sur toutes les observations.
    Permet de voir si l'impact est concentré ou dispersé.
    """
    shap_explanation = shap.Explanation(
        values=shap_values,
        data=X_sample,
        feature_names=feature_names,
    )
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_explanation, max_display=15, show=False)
    plt.title("SHAP Beeswarm Plot — Distribution des impacts\n"
              "Densité des points = fréquence | "
              "Largeur = variabilité de l'impact",
              fontsize=11, pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Sauvegarde : {save_path}")
    plt.close()


# ─── 3. Waterfall Plot ───────────────────────────────────────────────────────

def plot_waterfall(shap_values, X_sample, feature_names: list,
                   expected_value: float,
                   obs_idx: int = 0,
                   y_true=None, y_proba=None,
                   save_path: str = None):
    """
    Waterfall plot — explication d'UNE prédiction individuelle.

    Montre comment chaque feature pousse la prédiction
    depuis la valeur de base (expected_value) vers la prédiction finale.

    Barres rouges  = features qui augmentent le risque de panne
    Barres bleues  = features qui diminuent le risque de panne
    E[f(x)]        = prédiction moyenne du modèle (baseline)
    f(x)           = prédiction pour cette observation

    Args:
        obs_idx : index dans X_sample de l'observation à expliquer
    """
    shap_obs = shap.Explanation(
        values=shap_values[obs_idx],
        base_values=expected_value,
        data=X_sample[obs_idx],
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_obs, max_display=12, show=False)

    # Titre informatif avec le vrai label et la probabilité
    title = f"SHAP Waterfall — Explication observation #{obs_idx}"
    if y_true is not None and y_proba is not None:
        label = "PANNE" if y_true[obs_idx] == 1 else "Normal"
        proba = y_proba[obs_idx]
        title += f"\nRéalité : {label} | P(panne) prédite : {proba:.3f}"
    plt.title(title, fontsize=11, pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Sauvegarde : {save_path}")
    plt.close()


# ─── 4. Dependence Plot ──────────────────────────────────────────────────────

def plot_dependence(shap_values, X_sample, feature_names: list,
                    feature: str = "temperature_motor",
                    interaction_feature: str = "auto",
                    save_path: str = None):
    """
    Dependence plot — effet d'une feature sur les prédictions.

    Axe X     = valeur de la feature
    Axe Y     = valeur SHAP (impact sur la prédiction)
    Couleur   = feature d'interaction (détectée automatiquement)

    Permet de voir :
    - La relation entre la valeur d'une feature et son impact
    - Les interactions avec d'autres features (via la couleur)
    - Les seuils critiques (ex: à partir de quelle température le risque monte)
    """
    if feature not in feature_names:
        print(f"  Feature '{feature}' non trouvée. Features disponibles :")
        print(f"  {feature_names}")
        return

    feat_idx = feature_names.index(feature)

    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feat_idx, shap_values, X_sample,
        feature_names=feature_names,
        interaction_index=interaction_feature,
        show=False,
        alpha=0.5,
    )
    plt.title(f"SHAP Dependence Plot — {feature}\n"
              "Axe Y = impact sur P(panne) | "
              "Couleur = feature d'interaction détectée automatiquement",
              fontsize=11, pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Sauvegarde : {save_path}")
    plt.close()


# ─── 5. Bar Plot (importance agrégée) ────────────────────────────────────────

def plot_bar_importance(shap_values, feature_names: list,
                        save_path: str = None):
    """
    Bar plot — importance globale agrégée (|SHAP| moyen).

    Version simplifiée du summary plot :
    montre la moyenne des valeurs SHAP absolues par feature.
    Plus lisible pour une présentation ou un rapport.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn_r(
        np.linspace(0.2, 0.8, len(imp_df))
    )
    bars = ax.barh(imp_df["feature"], imp_df["mean_abs_shap"],
                   color=colors, alpha=0.85, edgecolor="white")

    # Annoter les valeurs
    for bar, val in zip(bars, imp_df["mean_abs_shap"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Valeur SHAP absolue moyenne")
    ax.set_title("Importance globale des features (SHAP)\n"
                 "|SHAP| moyen = contribution moyenne à la prédiction",
                 fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Sauvegarde : {save_path}")
    plt.close()

    # Affichage textuel
    print("\n── Top features (SHAP |moyen|) ──")
    for _, row in imp_df.iloc[::-1].iterrows():
        bar = "█" * int(row["mean_abs_shap"] * 200)
        print(f"  {row['feature']:<35} {row['mean_abs_shap']:.4f}  {bar}")

    return imp_df


# ─── Analyse des cas critiques ────────────────────────────────────────────────

def analyze_false_negatives(shap_values, X_sample, feature_names: list,
                             expected_value: float,
                             y_true_sample, y_proba_sample,
                             threshold: float = 0.5,
                             n_cases: int = 3,
                             results_dir: Path = None):
    """
    Analyse SHAP des faux négatifs — pannes manquées.

    Identifie les FN dans l'échantillon et explique pourquoi
    le modèle les a ratés. Utile pour comprendre les limites
    et orienter l'amélioration du modèle.
    """
    y_pred_sample = (y_proba_sample >= threshold).astype(int)
    fn_mask = (y_true_sample == 1) & (y_pred_sample == 0)
    fn_indices = np.where(fn_mask)[0]

    print(f"\n── Analyse des Faux Négatifs dans l'échantillon ──")
    print(f"  FN trouvés dans l'échantillon : {len(fn_indices)}")

    if len(fn_indices) == 0:
        print("  Aucun FN dans cet échantillon — essaie avec un seuil plus élevé")
        return

    for i, fn_idx in enumerate(fn_indices[:n_cases]):
        print(f"\n  FN #{i+1} (obs index {fn_idx}) :")
        print(f"    P(panne) prédite : {y_proba_sample[fn_idx]:.4f}  "
              f"(sous le seuil {threshold:.2f})")
        print(f"    Réalité          : PANNE (manquée)")

        # Top features qui ont tiré la prédiction vers le bas
        shap_obs = shap_values[fn_idx]
        neg_impact = [(feature_names[j], shap_obs[j])
                      for j in range(len(shap_obs)) if shap_obs[j] < 0]
        neg_impact.sort(key=lambda x: x[1])

        print(f"    Features ayant réduit le score de panne :")
        for feat, val in neg_impact[:5]:
            print(f"      {feat:<35} SHAP = {val:.4f}")

        # Waterfall pour ce FN
        if results_dir:
            save_path = str(results_dir / f"shap_fn_{i+1}_waterfall.png")
            plot_waterfall(
                shap_values, X_sample, feature_names,
                expected_value, obs_idx=fn_idx,
                y_true=y_true_sample,
                y_proba=y_proba_sample,
                save_path=save_path,
            )


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_shap_analysis(model_path: Path = MODEL_PATH,
                      data_path: Path = DATA_PATH,
                      results_dir: Path = RESULTS_DIR,
                      sample_obs: int = 0):
    """
    Lance l'analyse SHAP complète sur le modèle XGBoost final.

    Args:
        sample_obs : index de l'observation à expliquer individuellement
                     (waterfall plot). 0 = première obs du test set.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from src.preprocessing import load_and_split, FEATURES_DERIVED, FEATURES_ROLLING

    results_dir.mkdir(exist_ok=True)

    print("=" * 55)
    print("  ANALYSE SHAP — XGBoost Maintenance Prédictive")
    print("=" * 55)

    # ── Chargement modèle & données ──
    print(f"\nChargement du modèle : {model_path}")
    pipeline = joblib.load(model_path)

    print("Chargement des données...")
    X_train, X_test, y_train, y_test, _ = load_and_split(
        data_path, use_rolling=True
    )

    # ── Calcul SHAP ──
    print("\n── Calcul des valeurs SHAP ──")
    shap_values, X_sample, feature_names, expected_value, idx, shap_df = \
        compute_shap_values(pipeline, X_test, sample_size=1000)

    # Labels et probas pour l'échantillon
    y_true_sample  = y_test.values[idx]
    y_proba_sample = pipeline.predict_proba(X_test)[:, 1][idx]

    # ── Graphiques ──
    print("\n── Génération des graphiques SHAP ──")

    print("\n1. Bar plot (importance agrégée)...")
    plot_bar_importance(
        shap_values, feature_names,
        save_path=str(results_dir / "shap_bar_importance.png"),
    )

    print("\n2. Summary plot...")
    plot_summary(
        shap_values, X_sample, feature_names,
        save_path=str(results_dir / "shap_summary.png"),
    )

    print("\n3. Beeswarm plot...")
    plot_beeswarm(
        shap_values, X_sample, feature_names,
        save_path=str(results_dir / "shap_beeswarm.png"),
    )

    print("\n4. Waterfall plot (observation individuelle)...")
    plot_waterfall(
        shap_values, X_sample, feature_names,
        expected_value, obs_idx=sample_obs,
        y_true=y_true_sample,
        y_proba=y_proba_sample,
        save_path=str(results_dir / "shap_waterfall.png"),
    )

    print("\n5. Dependence plots (features clés)...")
    for feature in ["temperature_motor", "vibration_rms",
                    "hours_since_maintenance"]:
        plot_dependence(
            shap_values, X_sample, feature_names,
            feature=feature,
            save_path=str(results_dir / f"shap_dependence_{feature}.png"),
        )

    print("\n6. Analyse des faux négatifs...")
    analyze_false_negatives(
        shap_values, X_sample, feature_names,
        expected_value,
        y_true_sample, y_proba_sample,
        threshold=0.831,   # seuil optimal XGBoost
        n_cases=3,
        results_dir=results_dir,
    )

    # ── Résumé textuel ──
    print("\n" + "=" * 55)
    print("  RÉSUMÉ SHAP")
    print("=" * 55)
    mean_abs = np.abs(shap_values).mean(axis=0)
    top3_idx = np.argsort(mean_abs)[::-1][:3]
    print("\nTop 3 features les plus influentes :")
    for rank, i in enumerate(top3_idx, 1):
        pos_impact = (shap_values[:, i] > 0).mean() * 100
        print(f"  {rank}. {feature_names[i]:<35} "
              f"|SHAP| moy = {mean_abs[i]:.4f} | "
              f"Impact positif (risque) : {pos_impact:.0f}% des cas")

    print(f"\nGraphiques sauvegardés dans : {results_dir}")
    print("Fichiers générés :")
    for f in sorted(results_dir.glob("shap_*.png")):
        print(f"  {f.name}")

    return shap_df, feature_names


# ─── Alias pour compatibilité main.py ────────────────────────────────────────

def run_full_analysis(model_path: str, data_path: str,
                      results_dir: str, sample_obs: int = 0):
    """
    Alias de run_shap_analysis pour appel depuis main.py.
    Accepte des str en plus de Path.
    """
    return run_shap_analysis(
        model_path=Path(model_path),
        data_path=Path(data_path),
        results_dir=Path(results_dir),
        sample_obs=sample_obs,
    )


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse SHAP — XGBoost Maintenance Prédictive"
    )
    parser.add_argument("--model", type=str, default=str(MODEL_PATH),
                        help="Chemin vers le modèle .pkl")
    parser.add_argument("--data", type=str, default=str(DATA_PATH),
                        help="Chemin vers le CSV")
    parser.add_argument("--sample", type=int, default=0,
                        help="Index de l'observation pour le waterfall plot")
    args = parser.parse_args()

    run_shap_analysis(
        model_path=Path(args.model),
        data_path=Path(args.data),
        results_dir=RESULTS_DIR,
        sample_obs=args.sample,
    )