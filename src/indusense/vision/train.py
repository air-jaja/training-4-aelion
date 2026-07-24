"""Entraînement de l'autoencodeur convolutionnel — Deep Learning.

Extrait de `pipeline_bottle_full.ipynb` (algorithme SSIM retenu). Chemins, taille d'image,
batch size, epochs et patience lus depuis `configs/preprocessing.yaml` et `configs/train.yaml`
— aucun chemin ni hyperparamètre codé en dur.
"""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from indusense.common.config import RANDOM_SEED, RAW_DIR, load_config
from indusense.vision.augment import build_train_augmentations, make_augment_fn
from indusense.vision.dataset import BottlePipeline, prepare_bottle_pipeline
from indusense.vision.model import build_autoencoder, mse_metric, ssim_loss


def load_bottle_pipeline() -> BottlePipeline:
    """Construit le pipeline `bottle` complet (train/val/test) à partir de la config."""
    cfg = load_config("preprocessing", domain="dl")
    data_dir = RAW_DIR / "vision"  # racine des datasets vision, distincte de RAW_DIR ML

    transform = build_train_augmentations()
    augment_fn = make_augment_fn(transform)

    return prepare_bottle_pipeline(
        data_dir=data_dir,
        category=cfg["bottle_dirname"],
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        val_fraction=cfg["val_fraction"],
        seed=RANDOM_SEED,
        augment_fn=augment_fn,
    )


def train_autoencoder(pipeline: BottlePipeline) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Entraîne l'autoencodeur (perte SSIM retenue) sur le pipeline fourni.

    Retourne `(model, history)` — le modèle entraîné et l'historique Keras (loss/metric
    par epoch), pour que l'appelant produise ses propres courbes/rendus si besoin.
    """
    cfg_preproc = load_config("preprocessing", domain="dl")
    cfg_train = load_config("train", domain="dl")

    tf.random.set_seed(RANDOM_SEED)

    model = build_autoencoder(img_size=cfg_preproc["img_size"], base_filters=cfg_train["base_filters"])
    model.compile(optimizer=cfg_train["optimizer"], loss=ssim_loss, metrics=[mse_metric])

    train_ds_xy = pipeline.train_ds.map(lambda x: (x, x))
    val_ds_xy = pipeline.val_ds.map(lambda x: (x, x))

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=cfg_train["patience"], restore_best_weights=True,
    )

    history = model.fit(
        train_ds_xy, validation_data=val_ds_xy,
        epochs=cfg_train["epochs"], callbacks=[early_stopping], verbose=0,
    )

    return model, history


def save_autoencoder(model: tf.keras.Model, model_id: str) -> Path:
    """Sauvegarde le modèle au format Keras natif, dans `MODELS_DIR` (jamais un chemin en dur)."""
    from indusense.common.config import MODELS_DIR

    model_path = MODELS_DIR / f"{model_id}.keras"
    model.save(model_path)
    return model_path
