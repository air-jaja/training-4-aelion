# Concevoir un autoencodeur convolutionnel (Keras/TensorFlow) et lire son `summary()`

## Contexte

Ce guide accompagne `conv_autoencoder.ipynb`, construit sur le pipeline `indusense.vision.dataset` (images `bottle`, 128×128×3, normalisées `[0,1]`). L'autoencodeur n'apprend qu'à reconstruire des images **saines** (`train/good`) ; l'erreur de reconstruction sur une image inconnue devient le score d'anomalie.

---

## 1. Import du package `indusense` depuis le notebook

Tant que le layout `src/` n'est pas déclaré dans `pyproject.toml` (voir `etapes_pipeline_vision.md`), `indusense` n'est pas installé dans le venv — il faut l'ajouter au `sys.path` à la main. Un chemin relatif (`sys.path.insert(0, "src")` ou même `os.path.abspath("src")`) est fragile : il suppose que le kernel Jupyter a démarré avec le `cwd` sur la racine du projet, ce qui n'est pas garanti (dépend d'où VS Code/Jupyter a été lancé, ou de l'emplacement du notebook).

**Solution robuste** — remonter dynamiquement depuis le `cwd` du kernel jusqu'à trouver `pyproject.toml` (marqueur fiable de la racine du projet) :

```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # réduit les logs TensorFlow

import sys
from pathlib import Path

_project_root = Path.cwd()
while not (_project_root / "pyproject.toml").exists() and _project_root != _project_root.parent:
    _project_root = _project_root.parent
sys.path.insert(0, str(_project_root / "src"))

import tensorflow as tf
from tensorflow.keras import layers, Model

from indusense.vision.dataset import prepare_bottle_pipeline
from indusense.vision.augment import build_train_augmentations, make_augment_fn
```

Fonctionne quel que soit le `cwd` du kernel (racine du projet, sous-dossier `notebooks/`, etc.), tant que le notebook reste **dans l'arborescence du projet** (sous la racine contenant `pyproject.toml`).

**Solution définitive** (recommandée à terme) : déclarer le layout `src/` dans `pyproject.toml` (`[tool.setuptools.packages.find] where = ["src"]`) puis `uv sync` — `indusense` devient alors importable nativement, sans bricolage de `sys.path`.

---

## 2. Principe de l'architecture

Un autoencodeur convolutionnel a deux moitiés symétriques :

- **Encodeur** : réduit progressivement la résolution spatiale (`strides=2` ou `MaxPooling2D`) tout en augmentant le nombre de filtres — échange l'information spatiale contre de l'information sémantique.
- **Bottleneck** : la représentation la plus compacte, au milieu du réseau. C'est la contrainte qui force le modèle à apprendre une structure du "normal" plutôt que de simplement copier l'image (sans bottleneck, le réseau pourrait apprendre l'identité et l'erreur de reconstruction ne serait plus informative).
- **Décodeur** : miroir inverse (`Conv2DTranspose` ou `UpSampling2D` + `Conv2D`), remonte en résolution jusqu'à retrouver la taille et le nombre de canaux de l'image d'origine.

---

## 3. Exemple d'implémentation (Keras, API fonctionnelle)

```python
from tensorflow.keras import layers, Model

def build_autoencoder(img_size=128, base_filters=32):
    inputs = layers.Input(shape=(img_size, img_size, 3), name="input_image")

    # Encodeur : downsampling progressif
    x = layers.Conv2D(base_filters,     3, strides=2, padding="same", activation="relu", name="enc_conv1")(inputs)  # 128->64
    x = layers.Conv2D(base_filters * 2, 3, strides=2, padding="same", activation="relu", name="enc_conv2")(x)       # 64->32
    x = layers.Conv2D(base_filters * 4, 3, strides=2, padding="same", activation="relu", name="enc_conv3")(x)       # 32->16
    x = layers.Conv2D(base_filters * 8, 3, strides=2, padding="same", activation="relu", name="enc_conv4")(x)       # 16->8

    latent = x  # bottleneck : (8, 8, 256)

    # Décodeur : upsampling symétrique
    x = layers.Conv2DTranspose(base_filters * 4, 3, strides=2, padding="same", activation="relu", name="dec_conv1")(latent)  # 8->16
    x = layers.Conv2DTranspose(base_filters * 2, 3, strides=2, padding="same", activation="relu", name="dec_conv2")(x)       # 16->32
    x = layers.Conv2DTranspose(base_filters,     3, strides=2, padding="same", activation="relu", name="dec_conv3")(x)       # 32->64
    outputs = layers.Conv2DTranspose(3, 3, strides=2, padding="same", activation="sigmoid", name="dec_output")(x)            # 64->128

    return Model(inputs, outputs, name="conv_autoencoder")
```

## 4. Choix de conception à justifier

| Choix | Pourquoi |
|---|---|
| `strides=2` plutôt que `MaxPooling2D` | La convolution stridée apprend elle-même comment sous-échantillonner, plutôt qu'un max fixe |
| Filtres qui doublent à chaque étage (32→64→128→256) | Compense la perte de résolution spatiale par plus de capacité de représentation |
| `padding="same"` partout | Simplifie le calcul des tailles, garantit la symétrie exacte encodeur/décodeur |
| Activation `sigmoid` en sortie | Cohérente avec la normalisation `[0,1]` du pipeline — un `tanh`/`linear` demanderait de renormaliser en `[-1,1]` ou d'ajuster la loss |
| Loss `MeanSquaredError` (ou `binary_crossentropy`) | Pas de label : l'image d'entrée est aussi la cible (`fit(x, x)`) |

---

## 5. Lire le `summary()`

Exemple de sortie (128×128×3, `base_filters=32`) :

```
Model: "conv_autoencoder"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_image (InputLayer)        │ (None, 128, 128, 3)    │             0 │
│ enc_conv1 (Conv2D)              │ (None, 64, 64, 32)     │           896 │
│ enc_conv2 (Conv2D)              │ (None, 32, 32, 64)     │        18,496 │
│ enc_conv3 (Conv2D)              │ (None, 16, 16, 128)    │        73,856 │
│ enc_conv4 (Conv2D)              │ (None, 8, 8, 256)      │       295,168 │
│ dec_conv1 (Conv2DTranspose)     │ (None, 16, 16, 128)    │       295,040 │
│ dec_conv2 (Conv2DTranspose)     │ (None, 32, 32, 64)     │        73,792 │
│ dec_conv3 (Conv2DTranspose)     │ (None, 64, 64, 32)     │        18,464 │
│ dec_output (Conv2DTranspose)    │ (None, 128, 128, 3)    │           867 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 776,579 (2.96 MB)
```

### Colonnes

- **Layer (type)** : nom donné (`name=...`) + type de couche Keras.
- **Output Shape** : `(None, H, W, C)` — `None` = taille de batch (variable). Vérifie ici que le downsampling/upsampling se comporte comme prévu (`128→64→32→16→8` puis `8→16→32→64→128`).
- **Param #** : nombre de poids entraînables. Pour une `Conv2D` :
  ```
  params = (kernel_h × kernel_w × canaux_entrée + 1) × canaux_sortie
  ```
  Le `+1` est le biais par filtre. Vérification : `enc_conv1` = `(3×3×3 + 1) × 32 = 896` ; `enc_conv2` = `(3×3×32 + 1) × 64 = 18 496`.
- **Total params** : somme de toutes les couches, avec équivalent mémoire (poids en `float32` → `total × 4 octets`).
- **Trainable / Non-trainable** : ici tout est entraînable ; des couches non-trainable apparaîtraient avec un encodeur pré-entraîné figé (`layer.trainable = False`).

### Points à vérifier systématiquement

1. **Output Shape de la dernière couche == Output Shape de l'entrée** — condition nécessaire pour comparer reconstruction et original pixel à pixel.
2. **Symétrie du nombre de paramètres** entre encodeur et décodeur — un déséquilibre marqué signale souvent un bottleneck mal dimensionné.
3. **Total params vs volume de données d'entraînement** : à mettre en regard (ex. 776k paramètres pour 177 images est un ratio tendu, risque de sur-apprentissage — réduire `base_filters` ou renforcer la régularisation si la loss de validation stagne/remonte).

### Ratio de compression

Le bottleneck ne se juge pas qu'au nombre de paramètres — ce qui compte pour la détection d'anomalie, c'est combien de **valeurs** il faut pour représenter une image une fois passée dans l'encodeur, comparé au nombre de valeurs de l'image d'origine :

```
ratio de compression = (nb de valeurs en entrée) / (nb de valeurs dans le latent)
                      = (H × W × C entrée) / (H' × W' × C' latent)
```

Sur ce modèle (`128×128×3` → bottleneck `8×8×256`) :

```python
input_values = 128 * 128 * 3    # 49 152
latent_values = 8 * 8 * 256     # 16 384
ratio = input_values / latent_values   # 3.0x
```

**Lecture** : un ratio élevé (bottleneck très compact) force le modèle à apprendre une représentation plus abstraite du "normal" — utile pour la détection d'anomalie, mais risque de sous-apprentissage (reconstruction dégradée même sur des images saines) si le ratio est trop agressif. Un ratio trop faible (bottleneck presque aussi grand que l'entrée) risque à l'inverse de laisser le modèle apprendre une quasi-identité, peu informative pour distinguer sain/défectueux. Un ratio de 3x ici est modéré — à ajuster (`base_filters`, nombre d'étages) selon que la reconstruction sous-apprend ou sur-apprend en pratique.

---

## 6. Compilation (MSE + SSIM)

Pas de label : l'image d'entrée est aussi la cible (`fit(x, x)`, target = entrée = reconstruction).

- **Loss = MSE** (`mean_squared_error`) : pénalise l'erreur de reconstruction pixel à pixel, c'est elle qui pilote l'optimisation.
- **SSIM** (Structural Similarity Index) suivi en métrique additionnelle, pas comme loss principale : contrairement au MSE, elle est sensible à la structure locale (luminance, contraste, texture) plutôt qu'à l'écart pixel à pixel brut — plus proche de la façon dont un défaut structurel dégraderait perceptuellement la reconstruction. Keras n'a pas de métrique SSIM native, on l'encapsule via `tf.image.ssim` :

```python
def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))  # max_val=1.0 car images normalisées [0,1]

ssim_metric.__name__ = "ssim"  # nom utilisé dans history.history et les logs

autoencoder.compile(optimizer="adam", loss="mse", metrics=[ssim_metric])

train_ds_xy = pipeline.train_ds.map(lambda x: (x, x))
val_ds_xy = pipeline.val_ds.map(lambda x: (x, x))
```

- **Optimiseur = Adam** : choix par défaut robuste pour ce type de modèle, pas de réglage fin du learning rate nécessaire pour démarrer.

---

## 7. Suivi MLflow

Même convention que le reste du projet (SQLite locale, `mlflow/mlflow.db`) :

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
mlflow.set_experiment("bottle_autoencoder")
```

- **Params** loggés une fois par run (`img_size`, `batch_size`, `epochs`, `base_filters`, `optimizer`, `loss`).
- **Metrics** loggées à chaque epoch (`train_loss`, `val_loss`, `train_ssim`, `val_ssim`) — permet de comparer plusieurs runs dans l'UI MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db`).
- **Artifacts** : courbes d'apprentissage et exemples de reconstruction, sauvegardés en figures attachées au run (`mlflow.log_figure`).

---

## 8. Entraînement

```python
EPOCHS = 30

with mlflow.start_run(run_name="conv_autoencoder_bottle") as run:
    mlflow.log_params({
        "img_size": IMG_SIZE, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "base_filters": 32, "optimizer": "adam", "loss": "mse",
    })

    history = autoencoder.fit(train_ds_xy, validation_data=val_ds_xy, epochs=EPOCHS, verbose=2)

    for epoch in range(len(history.history["loss"])):
        mlflow.log_metrics({
            "train_loss": history.history["loss"][epoch],
            "val_loss": history.history["val_loss"][epoch],
            "train_ssim": history.history["ssim"][epoch],
            "val_ssim": history.history["val_ssim"][epoch],
        }, step=epoch)

    run_id = run.info.run_id
```

`EPOCHS` volontairement bas au départ pour une première validation rapide du pipeline (ex. 3) — à augmenter une fois confirmé que loss et SSIM évoluent dans le bon sens.

---

## 9. Courbes d'apprentissage

Deux courbes pertinentes à suivre, chacune en train **et** validation (l'écart train/validation renseigne sur le sur-apprentissage) :

- **Loss (MSE)** : doit décroître sur les deux courbes ; un `val_loss` qui remonte alors que `train_loss` continue de baisser signale un sur-apprentissage.
- **SSIM** : doit croître vers 1.0 (reconstruction structurellement fidèle) ; à lire en complément de la loss, pas à sa place — les deux métriques peuvent diverger légèrement (le MSE pénalise l'écart brut, le SSIM la structure).

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(history.history["loss"], label="train")
axes[0].plot(history.history["val_loss"], label="validation")
axes[0].set_title("Loss (MSE)")
axes[1].plot(history.history["ssim"], label="train")
axes[1].plot(history.history["val_ssim"], label="validation")
axes[1].set_title("SSIM")

with mlflow.start_run(run_id=run_id):
    mlflow.log_figure(fig, "curves.png")
```

---

## 10. Exemples de reconstruction après entraînement

Comparaison visuelle sur un batch de **validation** (jamais vu en augmentation, jamais utilisé pour la loss d'entraînement) : contrairement à l'aperçu de la section 5 du notebook (poids aléatoires, avant tout entraînement), la reconstruction doit maintenant ressembler à l'original si l'entraînement a convergé.

```python
for batch in val_ds_xy.take(1):
    x_val, _ = batch
    reconstruction_trained = autoencoder(x_val)

# ... affichage original vs reconstruction (voir notebook) ...

with mlflow.start_run(run_id=run_id):
    mlflow.log_figure(fig, "reconstructions.png")
```

---

## 11. Prochaine étape

Une fois le run MLflow validé (loss qui décroît, SSIM qui croît, reconstructions visuellement correctes sur les images saines) : établir un score d'anomalie (ex. erreur de reconstruction par pixel ou 1-SSIM) sur `test/good` et `test/<defect>`, puis évaluer sa capacité de discrimination (AUC-ROC, AUC-PR) — cf. `etude_desequilibre_classe.md` pour le choix des métriques adaptées au déséquilibre du dataset.
