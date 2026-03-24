"""
preprocessing.py
────────────────
Pipeline de préparation des données — version améliorée.

Nouveautés v2 :
  - Features de tendance temporelle (rolling mean, diff, std)
  - Tri chronologique par machine avant calcul rolling
  - Shift(1) pour éviter le data leakage temporel

Règle fondamentale anti-leakage :
  - Le pipeline est fit() UNIQUEMENT sur X_train
  - Les rolling features sont calculées row-wise avant le split
    mais utilisent shift(1) → ne voient jamais la valeur courante
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ─── Constantes ───────────────────────────────────────────────────────────────

FEATURES_NUM = [
    "vibration_rms",
    "temperature_motor",
    "current_phase_avg",
    "pressure_level",
    "rpm",
    "hours_since_maintenance",
]
FEATURES_CAT = ["operating_mode"]
TARGET       = "failure_within_24h"

FEATURES_DERIVED = [
    "temp_delta",
    "age_vibration",
    "vibration_per_rpm",
]

# Features de tendance — calculées avec shift(1), jamais la valeur courante
FEATURES_ROLLING = [
    "vibration_rms_roll_mean",
    "vibration_rms_roll_std",
    "vibration_rms_diff",
    "temperature_motor_roll_mean",
    "temperature_motor_roll_std",
    "temperature_motor_diff",
    "current_phase_avg_roll_mean",
    "current_phase_avg_diff",
]

TEST_SIZE      = 0.2
RANDOM_STATE   = 42
ROLLING_WINDOW = 3   # ~9 minutes d'historique (mesures toutes les 3 min)


# ─── Feature engineering statique ────────────────────────────────────────────

def build_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features dérivées row-wise — aucun risque de leakage."""
    df = df.copy()
    df["temp_delta"]        = df["temperature_motor"] - df["ambient_temp"]
    df["age_vibration"]     = df["hours_since_maintenance"] * df["vibration_rms"]
    df["vibration_per_rpm"] = df["vibration_rms"] / (df["rpm"].clip(lower=1))
    return df


# ─── Features de tendance temporelle ─────────────────────────────────────────

def build_rolling_features(df: pd.DataFrame,
                            window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Features de tendance sur l'historique passé de chaque machine.

    Anti-leakage temporel via shift(1) :
      Chaque rolling ne voit que les observations PRECEDENTES.
      La valeur courante n'est jamais incluse dans son propre calcul.

    Exemple vibration_rms, window=3, shift(1) :
      t=1: 1.1  → roll_mean = NaN   (pas d'historique)
      t=2: 1.8  → roll_mean = 1.1
      t=3: 2.5  → roll_mean = 1.45
      t=4: 3.2  → roll_mean = 1.80  ← jamais t4 lui-même

    Les NaN en début de série sont gérés par SimpleImputer(median).
    """
    df = df.copy()

    # Tri chronologique par machine — OBLIGATOIRE
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    for col in ["vibration_rms", "temperature_motor", "current_phase_avg"]:
        # shift(1) : exclut la valeur courante du calcul
        shifted = df.groupby("machine_id")[col].shift(1)

        # Rolling mean sur l'historique passé
        df[f"{col}_roll_mean"] = (
            shifted.groupby(df["machine_id"])
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

        # Rolling std → instabilité/irrégularité de la machine
        df[f"{col}_roll_std"] = (
            shifted.groupby(df["machine_id"])
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )

        # Différence t-1 vs t-2 → vitesse de dégradation
        # Positif = dégradation, Négatif = amélioration, 0 = stable
        df[f"{col}_diff"] = (
            df.groupby("machine_id")[col].shift(1) -
            df.groupby("machine_id")[col].shift(2)
        )

    return df


# ─── Pipeline sklearn ─────────────────────────────────────────────────────────

def build_preprocessor(extra_num_features: list = None) -> ColumnTransformer:
    """Construit le ColumnTransformer — à fit() sur X_train uniquement."""
    num_features = FEATURES_NUM.copy()
    if extra_num_features:
        num_features += extra_num_features

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,     num_features),
            ("cat", categorical_pipeline, FEATURES_CAT),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


# ─── Chargement & split ───────────────────────────────────────────────────────

def load_and_split(data_path: str | Path, use_rolling: bool = True):
    """
    Charge, prépare et splitte les données.

    Args:
        use_rolling : active les features de tendance temporelle.
                      Mettre False pour comparer avant/après.
    """
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 1. Features statiques (row-wise, safe avant le split)
    df = build_static_features(df)

    # 2. Features rolling (shift anti-leakage, safe avant le split)
    if use_rolling:
        df = build_rolling_features(df)
        extra_num = FEATURES_DERIVED + FEATURES_ROLLING
    else:
        extra_num = FEATURES_DERIVED

    all_features = [f for f in FEATURES_NUM + extra_num + FEATURES_CAT
                    if f in df.columns]

    X = df[all_features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"{'Avec' if use_rolling else 'Sans'} features rolling")
    print(f"  Nombre de features    : {len(all_features)}")
    print(f"  Train : {len(X_train):,} obs  |  Panne : {y_train.mean()*100:.1f}%")
    print(f"  Test  : {len(X_test):,} obs   |  Panne : {y_test.mean()*100:.1f}%")

    preprocessor = build_preprocessor(extra_num_features=extra_num)

    return X_train, X_test, y_train, y_test, preprocessor
