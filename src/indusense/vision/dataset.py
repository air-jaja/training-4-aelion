"""
indusense.vision.dataset
=========================

Utilitaires réutilisables pour préparer un dataset de détection d'anomalies au
format MVTec-AD (ex. catégorie `bottle`) :

- récupération de l'archive (URL directe ou Hugging Face Hub) et extraction ;
- inventaire et vérification (correspondance image/masque, fichiers corrompus) ;
- chargement + redimensionnement des images (OpenCV) et des masques (scikit-image,
  binarité préservée) ;
- normalisation des pixels dans [0, 1] ;
- split train/validation sur les images saines uniquement ;
- construction de pipelines `tf.data` pour l'entraînement, la validation et le test.

Ce module ne fait aucune hypothèse sur la stratégie d'augmentation : les
fonctions de construction de dataset acceptent un `augment_fn` optionnel
(callable `np.ndarray -> np.ndarray`), fourni par exemple par
`indusense.vision.augment`, pour rester découplé de la bibliothèque
d'augmentation utilisée.

Note sur le téléchargement
---------------------------
MVTec-AD est distribué sous licence CC BY-NC-SA 4.0 par MVTec Software GmbH.
Il n'existe pas d'URL publique unique garantie stable dans le temps pour
récupérer automatiquement chaque catégorie : selon le contexte, la source sera
soit l'URL officielle (nécessite d'accepter les conditions sur mvtec.com),
soit un mirroir interne/personnel (ex. stockage cloud d'équipe), soit un repo
Hugging Face Hub. Les deux mécanismes de téléchargement ci-dessous sont donc
génériques et paramétrables — à toi de fournir l'URL ou le repo_id pertinent
pour ton contexte plutôt que de dépendre d'une valeur codée en dur.
"""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from skimage.transform import resize as sk_resize
from sklearn.model_selection import train_test_split

DEFAULT_IMG_SIZE = 128
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# Téléchargement et extraction
# ---------------------------------------------------------------------------

def download_from_url(url: str, dest_path: str | Path, chunk_size: int = 1 << 16) -> Path:
    """Télécharge un fichier depuis une URL HTTP(S) directe, en streaming.

    Paramètres
    ----------
    url : URL directe du fichier (archive .tar.xz ou .zip attendue en aval).
    dest_path : chemin de destination du fichier téléchargé.
    """
    import requests  # import différé : dépendance optionnelle, seulement si on télécharge par URL

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return dest_path


def download_from_hf(
    repo_id: str,
    filename: str,
    dest_dir: str | Path,
    repo_type: str = "dataset",
) -> Path:
    """Télécharge un fichier depuis un repo Hugging Face Hub (`huggingface_hub`).

    Utile si l'archive du dataset est hébergée sur un repo HF (public ou privé,
    avec `huggingface-cli login` au préalable pour un repo privé).
    """
    from huggingface_hub import hf_hub_download

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=dest_dir
    )
    return Path(downloaded_path)


def extract_archive(archive_path: str | Path, dest_dir: str | Path) -> Path:
    """Extrait une archive `.tar.xz`/`.tar.gz`/`.tar`/`.zip` vers `dest_dir`.

    Gère le cas rencontré en pratique où une archive `.zip` contient elle-même
    une archive `.tar.xz` (ex. fichier renommé/reconditionné) : dans ce cas,
    on extrait le zip, puis on détecte et extrait l'éventuelle archive tar
    trouvée à l'intérieur.
    """
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
        # Cas archive imbriquée : un .tar.xz/.tar.gz à l'intérieur du zip
        for inner in dest_dir.glob("**/*"):
            if inner.is_file() and (
                inner.suffixes[-2:] == [".tar", ".xz"]
                or inner.suffixes[-2:] == [".tar", ".gz"]
                or inner.suffix == ".tar"
            ):
                with tarfile.open(inner) as tf_archive:
                    tf_archive.extractall(dest_dir)
                break
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf_archive:
            tf_archive.extractall(dest_dir)
    else:
        raise ValueError(f"Format d'archive non reconnu : {archive_path}")

    return dest_dir


def ensure_mvtec_category(
    data_dir: str | Path,
    category: str = "bottle",
    source_url: Optional[str] = None,
    hf_repo_id: Optional[str] = None,
    hf_filename: Optional[str] = None,
) -> Path:
    """Garantit que `data_dir/<category>` existe (structure MVTec-AD complète).

    Si le dossier existe déjà (avec `train/good`), ne fait rien. Sinon,
    télécharge l'archive (via `source_url` ou `hf_repo_id`/`hf_filename`) et
    l'extrait dans `data_dir`.

    Lève une erreur explicite si aucune source n'est fournie et que les
    données sont absentes, plutôt que d'échouer silencieusement plus tard.
    """
    data_dir = Path(data_dir)
    category_dir = data_dir / category

    if (category_dir / "train" / "good").is_dir():
        return category_dir

    if source_url is None and hf_repo_id is None:
        raise FileNotFoundError(
            f"Dataset introuvable dans {category_dir} et aucune source de "
            "téléchargement fournie (source_url ou hf_repo_id/hf_filename)."
        )

    data_dir.mkdir(parents=True, exist_ok=True)

    if source_url is not None:
        archive_path = data_dir / f"{category}_download.archive"
        download_from_url(source_url, archive_path)
    else:
        archive_path = download_from_hf(hf_repo_id, hf_filename, data_dir)

    extract_archive(archive_path, data_dir)

    if not (category_dir / "train" / "good").is_dir():
        raise RuntimeError(
            f"Extraction terminée mais structure attendue absente dans {category_dir}. "
            "Vérifier le contenu de l'archive téléchargée."
        )
    return category_dir


# ---------------------------------------------------------------------------
# Inventaire et vérification
# ---------------------------------------------------------------------------

@dataclass
class CategoryInventory:
    train_good: list[Path]
    test_good: list[Path]
    defect_names: list[str]
    test_defect_paths: dict[str, list[Path]]
    ground_truth_paths: dict[str, list[Path]]


def inventory(category_dir: str | Path) -> CategoryInventory:
    """Liste les fichiers du dataset et vérifie la correspondance image/masque."""
    category_dir = Path(category_dir)

    train_good = sorted((category_dir / "train" / "good").glob("*.png"))
    test_good = sorted((category_dir / "test" / "good").glob("*.png"))
    defect_dirs = [d for d in (category_dir / "test").iterdir() if d.is_dir() and d.name != "good"]
    defect_names = sorted(d.name for d in defect_dirs)

    test_defect_paths, ground_truth_paths = {}, {}
    for name in defect_names:
        imgs = sorted((category_dir / "test" / name).glob("*.png"))
        masks = sorted((category_dir / "ground_truth" / name).glob("*_mask.png"))
        if len(imgs) != len(masks):
            raise ValueError(
                f"Mismatch images/masques pour '{name}': {len(imgs)} images vs {len(masks)} masques"
            )
        test_defect_paths[name] = imgs
        ground_truth_paths[name] = masks

    return CategoryInventory(train_good, test_good, defect_names, test_defect_paths, ground_truth_paths)


def verify_images(paths: list[Path]) -> list[Path]:
    """Retourne la liste des fichiers illisibles/corrompus parmi `paths`."""
    bad = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:
            bad.append(p)
    return bad


# ---------------------------------------------------------------------------
# Chargement, resize, normalisation
# ---------------------------------------------------------------------------

def load_and_resize_image(path: str | Path, size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    """Charge une image et la redimensionne en (size, size, 3), RGB, uint8."""
    img = Image.open(path).convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def load_and_resize_mask(path: str | Path, size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    """Charge un masque binaire et le redimensionne sans casser la binarité."""
    mask = Image.open(path).convert("L")
    mask = np.array(mask)
    mask = sk_resize(mask, (size, size), order=0, preserve_range=True, anti_aliasing=False)
    mask = (mask > 127).astype(np.uint8)
    return mask


def normalize(img: np.ndarray) -> np.ndarray:
    """Normalise une image uint8 [0,255] en float32 [0,1]."""
    return img.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Split train / validation
# ---------------------------------------------------------------------------

def train_val_split(
    train_good_paths: list[Path],
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> tuple[list[Path], list[Path]]:
    """Split aléatoire des chemins d'images saines (train/good) en train/validation."""
    return train_test_split(train_good_paths, test_size=val_fraction, random_state=seed)


# ---------------------------------------------------------------------------
# Pipelines tf.data
# ---------------------------------------------------------------------------

def _make_loader(img_size: int, augment_fn: Optional[Callable[[np.ndarray], np.ndarray]]):
    def _load(path):
        path = path.numpy().decode("utf-8")
        img = load_and_resize_image(path, size=img_size)
        if augment_fn is not None:
            img = augment_fn(img)
        img = normalize(img)
        return img.astype(np.float32)

    def _tf_wrapper(path):
        img = tf.py_function(_load, [path], tf.float32)
        img.set_shape((img_size, img_size, 3))
        return img

    return _tf_wrapper


def build_dataset(
    paths: list[Path],
    img_size: int = DEFAULT_IMG_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    augment_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    shuffle: bool = False,
    cache: bool = True,
    seed: int = DEFAULT_SEED,
) -> tf.data.Dataset:
    """Construit un pipeline `tf.data` (chargement + resize + normalisation [+ augmentation]).

    `augment_fn`, si fourni, est appliqué à l'image redimensionnée (uint8) avant
    normalisation. Ne jamais passer `augment_fn` pour des datasets de
    validation/test.
    """
    paths_str = [str(p) for p in paths]
    ds = tf.data.Dataset.from_tensor_slices(paths_str)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths_str), seed=seed)
    ds = ds.map(_make_loader(img_size, augment_fn), num_parallel_calls=tf.data.AUTOTUNE)
    if cache and augment_fn is None:
        # pas de cache si augmentation : on veut un tirage différent à chaque epoch
        ds = ds.cache()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


@dataclass
class BottlePipeline:
    train_paths: list[Path]
    val_paths: list[Path]
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_good_ds: tf.data.Dataset
    test_defect_ds: dict[str, tf.data.Dataset]
    test_defect_masks: dict[str, np.ndarray]
    defect_names: list[str]


def prepare_bottle_pipeline(
    data_dir: str | Path,
    category: str = "bottle",
    img_size: int = DEFAULT_IMG_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
    augment_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    source_url: Optional[str] = None,
    hf_repo_id: Optional[str] = None,
    hf_filename: Optional[str] = None,
) -> BottlePipeline:
    """Orchestration complète : téléchargement (si besoin) → inventaire →
    split → pipelines `tf.data` train/val/test, prêts à l'emploi.

    `augment_fn` doit être fourni par l'appelant (ex.
    `indusense.vision.augment.build_train_augmentations()` encapsulé dans un
    callable) — ce module reste volontairement agnostique de la bibliothèque
    d'augmentation utilisée.
    """
    category_dir = ensure_mvtec_category(
        data_dir, category=category, source_url=source_url, hf_repo_id=hf_repo_id, hf_filename=hf_filename
    )
    inv = inventory(category_dir)

    train_paths, val_paths = train_val_split(inv.train_good, val_fraction=val_fraction, seed=seed)

    train_ds = build_dataset(
        train_paths, img_size=img_size, batch_size=batch_size,
        augment_fn=augment_fn, shuffle=True, cache=(augment_fn is None), seed=seed,
    )
    val_ds = build_dataset(val_paths, img_size=img_size, batch_size=batch_size, shuffle=False, cache=True)
    test_good_ds = build_dataset(inv.test_good, img_size=img_size, batch_size=batch_size, shuffle=False, cache=False)

    test_defect_ds, test_defect_masks = {}, {}
    for name in inv.defect_names:
        test_defect_ds[name] = build_dataset(
            inv.test_defect_paths[name], img_size=img_size, batch_size=batch_size, shuffle=False, cache=False
        )
        test_defect_masks[name] = np.stack(
            [load_and_resize_mask(m, size=img_size) for m in inv.ground_truth_paths[name]]
        )

    return BottlePipeline(
        train_paths=train_paths,
        val_paths=val_paths,
        train_ds=train_ds,
        val_ds=val_ds,
        test_good_ds=test_good_ds,
        test_defect_ds=test_defect_ds,
        test_defect_masks=test_defect_masks,
        defect_names=inv.defect_names,
    )
