"""Évaluation ML — calibration de seuil, métriques test (Machine Learning).

Centralise ce qui était dupliqué inline dans plusieurs notebooks (`pipeline_shap_explainability_ml`,
`pipeleine_model_card_generation`) : calibration du seuil de décision sur la validation, métriques
au seuil retenu sur train/val/test. Miroir de `indusense.vision.evaluate` côté DL.

Percentile de seuil lu depuis `configs/evaluate.yaml` — jamais codé en dur.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from indusense.common.config import load_config


def calibrate_threshold(model: xgb.XGBClassifier, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """Calibre le seuil de décision sur les scores des machines **saines** de validation uniquement.

    Jamais sur le test — le seuil ne doit voir aucune donnée qu'il n'a pas déjà vue à l'entraînement.
    """
    cfg = load_config("evaluate", domain="ml")
    val_proba = model.predict_proba(X_val)[:, 1]
    return float(np.percentile(val_proba[y_val == 0], cfg["threshold_percentile"]))


def evaluate_at_threshold(
    y: pd.Series, threshold: float,
    model: Optional[xgb.XGBClassifier] = None, X: Optional[pd.DataFrame] = None,
    proba: Optional[np.ndarray] = None,
) -> dict:
    """PR-AUC, AUC-ROC, rappel, précision et faux positifs à un seuil donné.

    Accepte soit `(model, X)` (calcule les probabilités), soit directement `proba` déjà calculé
    (évite un second passage forward si les scores existent déjà — même logique d'optimisation
    que côté DL, éviter les calculs redondants).
    """
    if proba is None:
        if model is None or X is None:
            raise ValueError("Fournir soit `proba`, soit `(model, X)`.")
        proba = model.predict_proba(X)[:, 1]

    pred = (proba > threshold).astype(int)
    y_arr = np.asarray(y)

    n_pos = max((y_arr == 1).sum(), 1)
    n_neg = max((y_arr == 0).sum(), 1)
    n_pred_pos = max((pred == 1).sum(), 1)

    return {
        "prauc": float(average_precision_score(y_arr, proba)),
        "auroc": float(roc_auc_score(y_arr, proba)),
        "recall": float(((pred == 1) & (y_arr == 1)).sum() / n_pos),
        "precision": float(((pred == 1) & (y_arr == 1)).sum() / n_pred_pos),
        "fpr": float(((pred == 1) & (y_arr == 0)).sum() / n_neg),
    }
