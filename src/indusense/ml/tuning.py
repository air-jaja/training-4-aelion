"""Recherche d'hyperparamètres Optuna — XGBoost (Machine Learning).

Extrait de `pipeline_optuna_xgboost_predictive_maintenance.ipynb`. Espace de recherche,
budget (essais/timeout), sampler et pruner lus depuis `configs/tune.yaml`.
"""
from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

from indusense.common.config import RANDOM_SEED, load_config


def _suggest_from_space(trial: optuna.Trial, name: str, spec: dict):
    if "low" in spec and isinstance(spec["low"], int) and "log" not in spec:
        return trial.suggest_int(name, spec["low"], spec["high"])
    return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))


def make_objective(X_train: pd.DataFrame, y_train: pd.Series, n_jobs: int = 1):
    """Construit la fonction objectif Optuna : PR-AUC moyenne en validation croisée temporelle.

    `n_jobs=1` par défaut à l'intérieur de chaque essai — la parallélisation se fait au niveau
    des essais eux-mêmes (`study.optimize(..., n_jobs=...)`), pas des threads XGBoost internes.
    """
    cfg = load_config("tune", domain="ml")
    search_space = cfg["search_space"]
    n_splits = cfg["n_splits"]

    stride = max(1, round(1 / cfg["search_sample_fraction"]))
    X_arr = X_train.values[::stride]
    y_arr = y_train.values[::stride]

    neg, pos = (y_arr == 0).sum(), (y_arr == 1).sum()
    scale_pos_weight_max = neg / pos

    def objective(trial: optuna.Trial) -> float:
        params = {name: _suggest_from_space(trial, name, spec) for name, spec in search_space.items()}
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 1.0, scale_pos_weight_max)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_scores = []
        for fold_i, (tr_idx, va_idx) in enumerate(tscv.split(X_arr)):
            model = xgb.XGBClassifier(
                **params, tree_method="hist", n_jobs=n_jobs, eval_metric="aucpr",
                early_stopping_rounds=20, random_state=RANDOM_SEED,
            )
            model.fit(X_arr[tr_idx], y_arr[tr_idx], eval_set=[(X_arr[va_idx], y_arr[va_idx])], verbose=False)
            proba = model.predict_proba(X_arr[va_idx])[:, 1]
            fold_scores.append(average_precision_score(y_arr[va_idx], proba))

            trial.report(float(np.mean(fold_scores)), step=fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


def run_optuna_study(X_train: pd.DataFrame, y_train: pd.Series, n_jobs_search: int = 1, optuna_n_jobs: int = 1) -> optuna.Study:
    """Lance l'étude Optuna complète (TPE + MedianPruner, budget borné, seed fixée)."""
    cfg = load_config("tune", domain="ml")

    sampler = TPESampler(seed=RANDOM_SEED)
    pruner = MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(study_name="xgboost_gold_ml", direction="maximize", sampler=sampler, pruner=pruner)

    objective = make_objective(X_train, y_train, n_jobs=n_jobs_search)
    study.optimize(objective, n_trials=cfg["n_trials"], timeout=cfg["timeout_seconds"], n_jobs=optuna_n_jobs)

    return study
