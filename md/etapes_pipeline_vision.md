# Intégration du pipeline vision dans le projet — étapes

Ce document décrit comment le travail fait sur le notebook `bottle_preprocessing.ipynb` a été transformé en code réutilisable, et comment le brancher sur `main.py`.

## 1. Nouvelle arborescence de package : `src/indusense/vision/`

```
src/
└── indusense/
    ├── __init__.py
    └── vision/
        ├── __init__.py
        ├── dataset.py    # téléchargement, chargement, normalisation, split, masques GT
        └── augment.py    # pipeline Albumentations sur les images saines
```

Le nom de package `indusense` reprend celui déjà utilisé dans `main.py` ("Data Processing Pipeline for InduSense Dataset"). Le sous-module `vision` isole tout ce qui concerne la détection d'anomalies visuelles, à côté du dossier `modules/` existant (télémétrie, base de données, etc.).

### Rendre le package importable

Ce projet utilise actuellement une disposition « flat » (`modules/` à la racine), pas encore de layout `src/`. Deux options :

**Option A — déclarer le layout `src` dans `pyproject.toml`** (recommandé, propre) :
```toml
[tool.setuptools.packages.find]
where = ["src"]
```
puis `uv sync` pour que `indusense` soit installé en mode editable dans le venv, importable depuis n'importe où (y compris `main.py`) sans manipulation supplémentaire.

**Option B — ajout ponctuel au `PYTHONPATH`** (dépannage rapide, pas pour la durée) :
```bash
set PYTHONPATH=%PYTHONPATH%;src   # cmd Windows
$env:PYTHONPATH += ";src"          # PowerShell
```

L'option A est celle à utiliser une fois le prototype validé.

---

## 2. `dataset.py` — ce qu'il fait

Reprend et généralise la logique du notebook, sans dépendre d'Albumentations (découplage) :

| Fonction | Rôle |
|---|---|
| `download_from_url(url, dest_path)` | Téléchargement HTTP direct en streaming (ex. lien mydrive.ch ou tout autre mirroir) |
| `download_from_hf(repo_id, filename, dest_dir)` | Téléchargement via Hugging Face Hub (`huggingface_hub`, déjà dans les dépendances du projet) |
| `extract_archive(archive_path, dest_dir)` | Extraction `.tar.xz`/`.tar.gz`/`.zip`, gère aussi le cas d'un `.tar.xz` imbriqué dans un `.zip` (rencontré avec le fichier `bottle_tar_zip.00X` reçu précédemment) |
| `ensure_mvtec_category(data_dir, category, source_url\|hf_repo_id)` | Télécharge + extrait seulement si la catégorie n'est pas déjà présente sur disque |
| `inventory(category_dir)` | Liste `train/good`, `test/good`, classes de défauts, vérifie la correspondance image ↔ masque |
| `verify_images(paths)` | Détecte les fichiers illisibles |
| `load_and_resize_image` / `load_and_resize_mask` / `normalize` | Identiques au notebook (OpenCV pour les images, scikit-image `order=0` pour les masques binaires, normalisation `[0,1]` en dernière étape) |
| `train_val_split(train_good_paths, val_fraction, seed)` | Split sur les images saines uniquement |
| `build_dataset(paths, ..., augment_fn=None)` | Pipeline `tf.data` générique ; `augment_fn` optionnel (callable `np.ndarray -> np.ndarray`) pour rester agnostique de la bibliothèque d'augmentation |
| `prepare_bottle_pipeline(...)` | Fonction tout-en-un : télécharge si besoin, inventorie, split, construit train/val/test (+ masques), retourne un objet `BottlePipeline` |

**Point de conception important** : `dataset.py` n'importe jamais `augment.py`. L'augmentation est injectée depuis l'extérieur via `augment_fn`, pour que le module de chargement de données n'ait aucune dépendance dure à Albumentations.

**Note sur le téléchargement** : MVTec-AD est sous licence CC BY-NC-SA 4.0 (MVTec Software GmbH). Il n'existe pas d'URL publique unique garantie stable — `source_url`/`hf_repo_id` sont donc des paramètres à fournir selon le contexte (URL officielle après acceptation des conditions, mirroir interne, ou repo Hugging Face Hub), plutôt qu'une valeur codée en dur dans le module.

---

## 3. `augment.py` — ce qu'il fait

| Fonction | Rôle |
|---|---|
| `build_train_augmentations(...)` | Construit le pipeline Albumentations (`HorizontalFlip`, `Rotate`, `ShiftScaleRotate`, `RandomBrightnessContrast`), paramètres exposés en arguments |
| `augment_image(image, transform)` | Applique le pipeline à une image |
| `make_augment_fn(transform)` | Encapsule le `A.Compose` en callable `np.ndarray -> np.ndarray`, directement injectable dans `dataset.build_dataset(..., augment_fn=...)` |
| `compare_original_vs_augmented(image_paths, transform, load_fn, n_draws, save_path)` | Génère et sauvegarde la grille de comparaison originale vs augmentée (même logique que la cellule notebook 5bis) |

**Note** : Albumentations affiche un avertissement de dépréciation sur `ShiftScaleRotate` ("special case of Affine transform, use Affine instead") — fonctionnel pour l'instant, à surveiller lors d'une future montée de version d'Albumentations (migration vers `A.Affine` à prévoir).

---

## 4. Mise à jour de `main.py`

Deux nouvelles options ajoutées, sur le même modèle que les options existantes (`argparse` + `run_module`) :

```python
parser.add_argument("--dl-dataset", action="store_true",
    help="Prépare le dataset de vision (téléchargement si besoin, inventaire, split train/val, pipelines tf.data)")
parser.add_argument("--dl-albumentation", action="store_true",
    help="Génère un aperçu comparatif original vs augmenté (Albumentations) pour le dataset de vision")
```

Deux nouvelles variables d'environnement (mêmes conventions que `INPUT_DATA_DIR`/`OUTPUT_DIR` déjà présentes) :

```python
os.environ["VISION_DATA_DIR"] = os.getenv("VISION_DATA_DIR", "./data")
os.environ["VISION_OUTPUT_DIR"] = os.getenv("VISION_OUTPUT_DIR", "./artifacts/outputs/vision")
```

Deux fonctions ajoutées, appelées via `run_module(...)` comme les autres étapes :

- `run_dl_dataset()` : appelle `prepare_bottle_pipeline`, affiche un résumé (nb images train/val, classes de défauts, tailles des masques).
- `run_dl_albumentation()` : construit le pipeline d'augmentation, génère et sauvegarde l'aperçu comparatif dans `VISION_OUTPUT_DIR/augmentation_check.png`.

Branchées à la fin du bloc d'exécution, **volontairement exclues de `--all`** (le flux `--all` concerne le pipeline IT-ops existant — anonymisation/bronze/gold/analyses — pas la partie vision, qui reste à déclencher explicitement) :

```python
if args.dl_dataset:
    run_module("Préparation du dataset de vision (dl-dataset)", run_dl_dataset)

if args.dl_albumentation:
    run_module("Aperçu d'augmentation (dl-albumentation)", run_dl_albumentation)
```

### Utilisation

```bash
uv run python main.py --dl-dataset
uv run python main.py --dl-albumentation
```

---

## 5. Validation effectuée

Les deux modules ont été testés de bout en bout sur le vrai dataset `bottle` (MVTec-AD, 209 images train, 3 classes de défauts) :

- `inventory()` : correspondance image/masque vérifiée sur les 63 paires défectueuses.
- `train_val_split()` : 177 train / 32 validation (85/15).
- `prepare_bottle_pipeline()` avec `augment_fn` injecté depuis `augment.build_train_augmentations()` : batch `(8, 128, 128, 3)`, `float32`, plage `[0, 1]` correcte.
- `compare_original_vs_augmented()` : grille générée et sauvegardée, variations visibles (rotation, translation/échelle, luminosité) sur plusieurs tirages.
- La logique des fonctions `run_dl_dataset()`/`run_dl_albumentation()` de `main.py` a été rejouée de façon isolée (mêmes appels, mêmes variables d'environnement) — comportement conforme.

---

## 6. Prochaine étape suggérée

Une fois `dl_dataset`/`dl_albumentation` validés dans ton environnement Windows :
- ajouter un groupe de dépendances `dl` complet dans `requirements` de CI si applicable ;
- envisager un `--dl-train` pour lancer l'entraînement du modèle d'anomalie (autoencodeur/PaDiM) en réutilisant `prepare_bottle_pipeline`.
