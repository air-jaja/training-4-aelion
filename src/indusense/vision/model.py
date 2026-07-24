"""Architecture de l'autoencodeur convolutionnel — Deep Learning (détection d'anomalies).

Extrait de `pipeline_bottle_full.ipynb`. `base_filters` et la taille d'image restent des
paramètres explicites (pas de config ici : c'est une architecture, pas une exécution) —
les valeurs d'exécution réelles viennent de `configs/preprocessing.yaml`/`train.yaml`
au niveau de `indusense.vision.train`.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers


def build_autoencoder(img_size: int = 128, base_filters: int = 32) -> Model:
    """Autoencodeur convolutionnel symétrique.

    Encodeur : 4 Conv2D stridées (downsampling x2 à chaque étage).
    Bottleneck : représentation (img_size/16, img_size/16, base_filters*8).
    Décodeur : 4 Conv2DTranspose stridées (upsampling x2 à chaque étage), miroir exact.
    """
    inputs = layers.Input(shape=(img_size, img_size, 3), name="input_image")

    x = layers.Conv2D(base_filters, 3, strides=2, padding="same", activation="relu", name="enc_conv1")(inputs)
    x = layers.Conv2D(base_filters * 2, 3, strides=2, padding="same", activation="relu", name="enc_conv2")(x)
    x = layers.Conv2D(base_filters * 4, 3, strides=2, padding="same", activation="relu", name="enc_conv3")(x)
    x = layers.Conv2D(base_filters * 8, 3, strides=2, padding="same", activation="relu", name="enc_conv4")(x)

    latent = x  # bottleneck

    x = layers.Conv2DTranspose(base_filters * 4, 3, strides=2, padding="same", activation="relu", name="dec_conv1")(latent)
    x = layers.Conv2DTranspose(base_filters * 2, 3, strides=2, padding="same", activation="relu", name="dec_conv2")(x)
    x = layers.Conv2DTranspose(base_filters, 3, strides=2, padding="same", activation="relu", name="dec_conv3")(x)
    outputs = layers.Conv2DTranspose(3, 3, strides=2, padding="same", activation="sigmoid", name="dec_output")(x)

    return Model(inputs, outputs, name="conv_autoencoder")


def ssim_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Perte SSIM : 1 - SSIM, minimisée quand la reconstruction est structurellement fidèle."""
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def ssim_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """SSIM moyen sur le batch, en métrique de suivi (pour un modèle entraîné avec loss MSE)."""
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


ssim_metric.__name__ = "ssim"


def mse_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """MSE en métrique de suivi (pour un modèle entraîné avec la loss SSIM)."""
    return tf.reduce_mean(tf.square(y_true - y_pred))


mse_metric.__name__ = "mse"


def error_heatmap(original: tf.Tensor, reconstruction: tf.Tensor) -> tf.Tensor:
    """Carte d'erreur par pixel : moyenne du carré de l'écart sur les canaux -> (H, W)."""
    return tf.reduce_mean(tf.square(original - reconstruction), axis=-1)
