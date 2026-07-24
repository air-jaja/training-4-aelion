"""Chargement et split temporel du `gold_dataset` — Machine Learning.

Extrait de `01_maintenance_ml.ipynb` (chargement + exclusions anti-fuite) et de
`prepare_features_ml_cross_validation_calendaire.ipynb` (split calendaire).
Aucun chemin en dur : tout passe par `indusense.common.config`.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import pandas as pd

from indusense.common.config import RAW_DIR, load_config


def load_gold_dataset(horizon: Optional[str] = None) -> tuple[pd.DataFrame, list[str], str]:
    """Charge `gold_dataset.parquet` et calcule la liste des features (anti-fuite exclue).

    Retourne `(df, feats, horizon)` — `df` est trié par machine puis par temps,
    `feats` exclut les identifiants, les colonnes de fuite temporelle et les labels
    des autres horizons (liste définie dans `configs/preprocessing.yaml`).
    """
    cfg = load_config("preprocessing", domain="ml")
    horizon = horizon or cfg["horizon"]

    gold_path = RAW_DIR / cfg["gold_dataset_filename"]
    df = pd.read_parquet(gold_path)
    df = df.sort_values(["machine_id_std", "window_start"]).reset_index(drop=True)

    exclude = set(cfg["exclude_columns"])
    all_horizon_labels = [c for c in df.columns if c.startswith("label_failure_next_")]
    feats = [c for c in df.columns if c not in exclude and c not in all_horizon_labels]

    return df, feats, horizon


def temporal_split(
    df: pd.DataFrame, feats: list[str], horizon: str
) -> tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]:
    """Split positionnel via la colonne `split_set` déjà fournie dans le Gold (train/validation/test).

    Retourne `(X_train, y_train), (X_val, y_val), (X_test, y_test)`.
    """
    train_df = df[df.split_set == "train"].sort_values("window_start").reset_index(drop=True)
    val_df = df[df.split_set == "validation"].reset_index(drop=True)
    test_df = df[df.split_set == "test"].reset_index(drop=True)

    X_train, y_train = train_df[feats], train_df[horizon].astype(int)
    X_val, y_val = val_df[feats], val_df[horizon].astype(int)
    X_test, y_test = test_df[feats], test_df[horizon].astype(int)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def calendar_cv_folds(
    df: pd.DataFrame, feats: list[str], horizon: str, n_splits: int = 5
) -> Iterator[dict]:
    """Validation croisée par découpage **calendaire** du train (indépendant de l'ordre des lignes).

    Contrairement à `TimeSeriesSplit` (positionnel), chaque fold est défini par une date de
    coupure sur `window_start` — train/validation restent strictement séparés dans le temps,
    quel que soit l'ordre des machines dans le DataFrame. Fenêtre expansive : le train de
    chaque fold s'étend du début de la période jusqu'à sa date de coupure.

    Génère, pour chaque fold, un dict `{fold, train_end, val_end, X_train, y_train, X_val, y_val}`
    — à charge de l'appelant d'entraîner/évaluer un modèle par fold (pas de modèle imposé ici,
    contrairement au notebook source qui étalonnait un `RandomForestClassifier`).
    """
    train_mask = df["split_set"] == "train"
    train_dates = df.loc[train_mask, "window_start"]
    date_min, date_max = train_dates.min(), train_dates.max()

    n_chunks = n_splits + 1
    boundaries = pd.date_range(date_min, date_max, periods=n_chunks + 1)

    for fold in range(n_splits):
        cutoff_train_end = boundaries[fold + 1]
        cutoff_val_end = boundaries[fold + 2]

        fold_train_mask = train_mask & (df["window_start"] < cutoff_train_end)
        fold_val_mask = train_mask & (df["window_start"] >= cutoff_train_end) & (df["window_start"] < cutoff_val_end)

        if fold_train_mask.sum() == 0 or fold_val_mask.sum() == 0:
            raise ValueError(
                f"Fold {fold + 1} vide (n_train={fold_train_mask.sum()}, n_val={fold_val_mask.sum()}) "
                f"— ajuster n_splits ou vérifier la couverture calendaire des données."
            )

        yield {
            "fold": fold + 1,
            "train_end": cutoff_train_end,
            "val_end": cutoff_val_end,
            "X_train": df.loc[fold_train_mask, feats],
            "y_train": df.loc[fold_train_mask, horizon].astype(int),
            "X_val": df.loc[fold_val_mask, feats],
            "y_val": df.loc[fold_val_mask, horizon].astype(int),
            "n_machines_train": df.loc[fold_train_mask, "machine_id_std"].nunique(),
            "n_machines_val": df.loc[fold_val_mask, "machine_id_std"].nunique(),
        }
