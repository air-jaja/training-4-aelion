"""Évaluation de l'autoencodeur — Deep Learning (score d'anomalie, seuil, IoU).

Extrait de `pipeline_bottle_full.ipynb` (étapes 19-22 : calibration du seuil, AUC-ROC/AUC-PR,
segmentation pixel via IoU). Percentiles lus depuis `configs/evaluate.yaml`.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score

from indusense.common.config import load_config
from indusense.vision.dataset import BottlePipeline
from indusense.vision.model import error_heatmap


def reconstruction_errors(dataset: tf.data.Dataset, model: tf.keras.Model) -> np.ndarray:
    """Erreur de reconstruction (MSE) par image, sur tout un dataset `tf.data`."""
    errors = []
    for batch in dataset:
        recon = model(batch)
        err = tf.reduce_mean(tf.square(batch - recon), axis=[1, 2, 3])
        errors.append(err.numpy())
    return np.concatenate(errors)


def calibrate_threshold(model: tf.keras.Model, pipeline: BottlePipeline) -> float:
    """Calibre le seuil de décision au percentile configuré, sur les scores de `val_ds` (saines).

    Jamais sur le test — le seuil ne doit être informé que par des images saines déjà
    connues du split de validation, pas par le jeu d'évaluation final.
    """
    cfg = load_config("evaluate", domain="dl")
    val_scores = reconstruction_errors(pipeline.val_ds, model)
    return float(np.percentile(val_scores, cfg["threshold_percentile"]))


def evaluate_image_level(model: tf.keras.Model, pipeline: BottlePipeline, threshold: float) -> dict:
    """Rappel, faux positifs, AUC-ROC et AUC-PR au niveau image (saine vs défectueuse)."""
    good_scores = reconstruction_errors(pipeline.test_good_ds, model)
    defect_scores = np.concatenate([reconstruction_errors(ds, model) for ds in pipeline.test_defect_ds.values()])

    y_true = np.concatenate([np.zeros(len(good_scores)), np.ones(len(defect_scores))])
    y_score = np.concatenate([good_scores, defect_scores])

    recall = float((defect_scores > threshold).mean())
    fpr = float((good_scores > threshold).mean())

    return {
        "threshold": threshold,
        "recall": recall,
        "fpr": fpr,
        "auroc_image": float(roc_auc_score(y_true, y_score)),
        "aucpr_image": float(average_precision_score(y_true, y_score)),
        "good_scores": good_scores,
        "defect_scores": defect_scores,
    }


def evaluate_pixel_level(model: tf.keras.Model, pipeline: BottlePipeline) -> dict:
    """AUC-ROC/AUC-PR pixel et IoU moyen (segmentation via seuil pixel calibré sur `val_ds`)."""
    cfg = load_config("evaluate", domain="dl")

    def _collect(dataset, masks=None):
        heatmaps = []
        for batch in dataset:
            recon = model(batch)
            heatmaps.append(error_heatmap(batch, recon).numpy())
        heatmaps = np.concatenate(heatmaps, axis=0)
        if masks is None:
            masks = np.zeros_like(heatmaps, dtype=np.uint8)
        return heatmaps, masks

    val_heatmaps, _ = _collect(pipeline.val_ds)
    pixel_threshold = float(np.percentile(val_heatmaps.flatten(), cfg["pixel_threshold_percentile"]))

    hm_good, mask_good = _collect(pipeline.test_good_ds)
    hm_defect_list, mask_defect_list = [], []
    for name in pipeline.defect_names:
        hm_d, _ = _collect(pipeline.test_defect_ds[name])
        hm_defect_list.append(hm_d)
        mask_defect_list.append(pipeline.test_defect_masks[name])

    all_heatmaps = np.concatenate([hm_good] + hm_defect_list, axis=0)
    all_masks = np.concatenate([mask_good] + mask_defect_list, axis=0)

    auroc_pixel = float(roc_auc_score(all_masks.flatten(), all_heatmaps.flatten()))
    aucpr_pixel = float(average_precision_score(all_masks.flatten(), all_heatmaps.flatten()))

    defect_indices = [i for i in range(len(all_masks)) if all_masks[i].sum() > 0]
    ious = []
    for i in defect_indices:
        pred_mask = (all_heatmaps[i] > pixel_threshold).astype(np.uint8)
        intersection = np.logical_and(pred_mask, all_masks[i]).sum()
        union = np.logical_or(pred_mask, all_masks[i]).sum()
        ious.append(intersection / union if union > 0 else 1.0)

    return {
        "pixel_threshold": pixel_threshold,
        "auroc_pixel": auroc_pixel,
        "aucpr_pixel": aucpr_pixel,
        "mean_iou": float(np.mean(ious)),
        "n_defect_images": len(defect_indices),
    }
