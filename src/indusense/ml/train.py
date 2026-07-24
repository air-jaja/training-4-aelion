"""Entraînement XGBoost — baseline et modèle final (Machine Learning).

Extrait de `pipeline_optuna_xgboost_predictive_maintenance.ipynb` (baseline) et
`pipeleine_model_card_generation.ipynb` (modèle final réentraîné sur train+validation).
Hyperparamètres et `n_jobs` lus depuis `configs/train.yaml`, jamais codés en dur.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb

from indusense.common.config import MODELS_DIR, RANDOM_SEED, load_config


def build_baseline_model(n_jobs: int | None = None) -> xgb.XGBClassifier:
    """Construit un XGBoost avec des hyperparamètres par défaut raisonnables (sans tuning).

    Sert de point de comparaison pour juger si un modèle réglé (Optuna) apporte un gain réel.
    """
    cfg = load_config("train", domain="ml")
    params = cfg["baseline_hyperparameters"]
    n_jobs = n_jobs if n_jobs is not None else cfg.get("n_jobs", -1)

    return xgb.XGBClassifier(
        **params, tree_method="hist", n_jobs=n_jobs, eval_metric="aucpr", random_state=RANDOM_SEED,
    )


def build_tuned_model(n_jobs: int | None = None) -> xgb.XGBClassifier:
    """Construit un XGBoost avec les hyperparamètres retenus (issus de la recherche Optuna).

    Les valeurs viennent de `configs/train.yaml` (clé `hyperparameters`) — à mettre à jour
    manuellement si une nouvelle étude Optuna change le modèle retenu (cf. `indusense.ml.tuning`).
    """
    cfg = load_config("train", domain="ml")
    params = cfg["hyperparameters"]
    n_jobs = n_jobs if n_jobs is not None else cfg.get("n_jobs", -1)

    return xgb.XGBClassifier(
        **params, tree_method="hist", n_jobs=n_jobs, eval_metric="aucpr", random_state=RANDOM_SEED,
    )


def train_final_model(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
) -> xgb.XGBClassifier:
    """Entraîne le modèle final sur train + validation réunis (toutes les données avant le test)."""
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)

    model = build_tuned_model()
    model.fit(X_trainval, y_trainval)
    return model


def save_model(model: xgb.XGBClassifier, model_id: str) -> Path:
    """Sauvegarde le booster natif (`get_booster().save_model()`), pas le wrapper sklearn.

    Évite une incompatibilité connue entre certaines versions de XGBoost et scikit-learn >= 1.6
    (`_estimator_type` non défini) — indépendant des versions installées.
    """
    model_path = MODELS_DIR / f"{model_id}.json"
    model.get_booster().save_model(str(model_path))
    return model_path
