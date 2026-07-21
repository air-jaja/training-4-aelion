"""
indusense.vision.augment
==========================

Pipeline d'augmentation (Albumentations) destiné aux images **saines**
d'entraînement uniquement (`train/good`), pour la détection d'anomalies.

Transformations volontairement légères et cohérentes avec un contexte
industriel : la notion de "normal" que le modèle doit apprendre ne doit pas
être dénaturée par l'augmentation (pas de distorsions fortes type
`ElasticTransform`/`GridDistortion`, pas de crop agressif).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import albumentations as A
import cv2
import numpy as np


def build_train_augmentations(
    rotate_limit: int = 15,
    shift_limit: float = 0.05,
    scale_limit: float = 0.1,
    brightness_limit: float = 0.15,
    contrast_limit: float = 0.15,
    flip_p: float = 0.5,
    rotate_p: float = 0.5,
    shift_scale_p: float = 0.5,
    brightness_contrast_p: float = 0.5,
) -> A.Compose:
    """Construit le pipeline Albumentations pour les images saines d'entraînement.

    - `HorizontalFlip` : symétrie horizontale.
    - `Rotate` : rotation légère (angle de prise de vue variable).
    - `ShiftScaleRotate` : translation + mise à l'échelle légère (centrage/zoom
      imparfait de la caméra) ; la rotation de cette transformation est
      désactivée (`rotate_limit=0`) puisqu'elle est déjà gérée par `Rotate`.
    - `RandomBrightnessContrast` : variations d'éclairage.
    """
    return A.Compose([
        A.HorizontalFlip(p=flip_p),
        A.Rotate(limit=rotate_limit, p=rotate_p, border_mode=cv2.BORDER_REPLICATE),
        A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=0,
            border_mode=cv2.BORDER_REPLICATE,
            p=shift_scale_p,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=brightness_contrast_p,
        ),
    ])


def augment_image(image: np.ndarray, transform: A.Compose) -> np.ndarray:
    """Applique le pipeline d'augmentation à une image (uint8, RGB, HxWx3)."""
    return transform(image=image)["image"]


def make_augment_fn(transform: A.Compose) -> Callable[[np.ndarray], np.ndarray]:
    """Encapsule un `A.Compose` en callable `np.ndarray -> np.ndarray`,
    directement injectable dans `indusense.vision.dataset.build_dataset`
    (paramètre `augment_fn`) ou `prepare_bottle_pipeline`.
    """
    def _fn(image: np.ndarray) -> np.ndarray:
        return augment_image(image, transform)
    return _fn


def compare_original_vs_augmented(
    image_paths: list[Path],
    transform: A.Compose,
    load_fn: Callable[[Path], np.ndarray],
    n_draws: int = 3,
    save_path: Optional[str | Path] = None,
):
    """Génère une grille comparant, pour chaque image de `image_paths`,
    l'image originale (ligne du haut) et `n_draws` tirages augmentés
    (lignes suivantes).

    `load_fn` : fonction de chargement + resize (ex.
    `indusense.vision.dataset.load_and_resize_image`), injectée pour ne pas
    coupler ce module au format de fichier ou à la taille cible.

    Retourne la figure matplotlib (et l'enregistre dans `save_path` si fourni).
    """
    import matplotlib.pyplot as plt

    n_images = len(image_paths)
    fig, axes = plt.subplots(n_draws + 1, n_images, figsize=(3.2 * n_images, 3.2 * (n_draws + 1)))
    if n_images == 1:
        axes = axes.reshape(-1, 1)

    for col, path in enumerate(image_paths):
        original = load_fn(path)

        axes[0, col].imshow(original)
        axes[0, col].set_title(f"Original\n{Path(path).name}")
        axes[0, col].axis("off")

        for row in range(1, n_draws + 1):
            augmented = augment_image(original, transform)
            axes[row, col].imshow(augmented)
            axes[row, col].set_title(f"Augmenté #{row}")
            axes[row, col].axis("off")

    fig.suptitle("Comparaison original (haut) vs tirages augmentés (bas)")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130)

    return fig
