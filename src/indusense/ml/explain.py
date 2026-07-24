"""Explicabilité SHAP — XGBoost (Machine Learning).

Extrait de `pipeline_shap_explainability_ml.ipynb`. Taille d'échantillon, seuil d'alerte
anti-fuite et nombre de features affichées lus depuis `configs/explain.yaml`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from indusense.common.config import RANDOM_SEED, load_config


def compute_shap_values(model: xgb.XGBClassifier, X: pd.DataFrame) -> tuple[shap.Explanation, pd.DataFrame]:
    """Calcule les valeurs SHAP sur un échantillon de `X` (TreeExplainer, tree_path_dependent).

    Retourne `(shap_values, X_sample)` — l'échantillon effectivement utilisé, pour que
    l'appelant puisse générer les graphiques (summary_plot, waterfall, dependence) sans
    avoir à le recalculer.
    """
    cfg = load_config("explain", domain="ml")
    sample_size = min(cfg["sample_size"], len(X))
    X_sample = X.sample(n=sample_size, random_state=RANDOM_SEED)

    explainer = shap.TreeExplainer(model, feature_perturbation=cfg["feature_perturbation"])
    shap_values = explainer(X_sample)

    return shap_values, X_sample


def top_features_with_direction(shap_values: shap.Explanation, X_sample: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Top N features par impact SHAP moyen, avec lecture de direction (valeur haute -> panne/sain)."""
    cfg = load_config("explain", domain="ml")
    top_n = cfg["top_n_features"]

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.array([
            np.corrcoef(X_sample.iloc[:, i], shap_values.values[:, i])[0, 1] if X_sample.iloc[:, i].std() > 0 else np.nan
            for i in range(X_sample.shape[1])
        ])

    top_idx = np.argsort(mean_abs_shap)[-top_n:][::-1]

    rows = []
    for i in top_idx:
        c = corr[i]
        if np.isnan(c):
            direction = "non déterminée (feature constante)"
        elif c > 0.1:
            direction = "valeur haute -> pousse vers PANNE"
        elif c < -0.1:
            direction = "valeur haute -> pousse vers SAIN"
        else:
            direction = "effet non-monotone / mixte"
        rows.append({"feature": feats[i], "|SHAP| moyen": mean_abs_shap[i], "corr(valeur, SHAP)": c, "direction": direction})

    return pd.DataFrame(rows)


def concentration_share(shap_values: shap.Explanation, top_n: int = 10) -> float:
    """Part de l'impact total |SHAP| couverte par les `top_n` features (concentré vs diffus)."""
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    sorted_shap = np.sort(mean_abs_shap)[::-1]
    n = min(top_n, len(sorted_shap))
    return float(sorted_shap[:n].sum() / sorted_shap.sum())


def leakage_check(top_features: list[str], train_df: pd.DataFrame, future_cols: list[str]) -> pd.DataFrame:
    """Corrélation des features top vs colonnes futures explicitement exclues de l'entraînement.

    Une corrélation élevée (> seuil de `configs/explain.yaml`) signalerait qu'une feature
    censée être disponible au moment de la prédiction capte en réalité de l'information
    sur le futur — une fuite qui aurait échappé à la checklist d'exclusion.
    """
    cfg = load_config("explain", domain="ml")
    threshold = cfg["leakage_correlation_threshold"]

    rows = []
    for feat in top_features:
        corrs = {fc: train_df[feat].corr(train_df[fc]) for fc in future_cols}
        max_abs_corr = max(abs(v) for v in corrs.values())
        rows.append({"feature": feat, **corrs, "max_abs_corr": max_abs_corr, "à_vérifier": max_abs_corr > threshold})

    return pd.DataFrame(rows)
