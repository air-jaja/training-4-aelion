# Designing a Convolutional Autoencoder (Keras/TensorFlow) and Reading its `summary()`

## Context

This guide accompanies `conv_autoencoder.ipynb`, built on the `indusense.vision.dataset` pipeline (`bottle` images, 128×128×3, normalized to `[0,1]`). The autoencoder only learns to reconstruct **normal** images (`train/good`); the reconstruction error on an unseen image becomes the anomaly score.

---

## 1. Importing the `indusense` package from the notebook

As long as the `src/` layout isn't declared in `pyproject.toml` (see `etapes_pipeline_vision.md`), `indusense` isn't installed in the venv — it needs to be added to `sys.path` manually. A relative path (`sys.path.insert(0, "src")` or even `os.path.abspath("src")`) is fragile: it assumes the Jupyter kernel started with its `cwd` at the project root, which isn't guaranteed (depends on where VS Code/Jupyter was launched from, or where the notebook lives).

**Robust solution** — dynamically walk up from the kernel's `cwd` until `pyproject.toml` is found (a reliable marker of the project root):

```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # reduce TensorFlow logs

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

Works regardless of the kernel's `cwd` (project root, `notebooks/` subfolder, etc.), as long as the notebook stays **within the project tree** (under the root containing `pyproject.toml`).

**Definitive solution** (recommended long-term): declare the `src/` layout in `pyproject.toml` (`[tool.setuptools.packages.find] where = ["src"]`) then `uv sync` — `indusense` becomes natively importable, no `sys.path` workaround needed.

---

## 2. Architecture principle

A convolutional autoencoder has two symmetric halves:

- **Encoder**: progressively reduces spatial resolution (`strides=2` or `MaxPooling2D`) while increasing the number of filters — trading spatial information for semantic information.
- **Bottleneck**: the most compact representation, at the middle of the network. This is the constraint that forces the model to learn a structure of "normal" rather than simply copying the image (without a bottleneck, the network could learn the identity function and the reconstruction error would no longer be informative).
- **Decoder**: exact mirror (`Conv2DTranspose` or `UpSampling2D` + `Conv2D`), going back up in resolution until the original image size and channel count are recovered.

---

## 3. Example implementation (Keras, Functional API)

```python
from tensorflow.keras import layers, Model

def build_autoencoder(img_size=128, base_filters=32):
    inputs = layers.Input(shape=(img_size, img_size, 3), name="input_image")

    # Encoder: progressive downsampling
    x = layers.Conv2D(base_filters,     3, strides=2, padding="same", activation="relu", name="enc_conv1")(inputs)  # 128->64
    x = layers.Conv2D(base_filters * 2, 3, strides=2, padding="same", activation="relu", name="enc_conv2")(x)       # 64->32
    x = layers.Conv2D(base_filters * 4, 3, strides=2, padding="same", activation="relu", name="enc_conv3")(x)       # 32->16
    x = layers.Conv2D(base_filters * 8, 3, strides=2, padding="same", activation="relu", name="enc_conv4")(x)       # 16->8

    latent = x  # bottleneck: (8, 8, 256)

    # Decoder: symmetric upsampling
    x = layers.Conv2DTranspose(base_filters * 4, 3, strides=2, padding="same", activation="relu", name="dec_conv1")(latent)  # 8->16
    x = layers.Conv2DTranspose(base_filters * 2, 3, strides=2, padding="same", activation="relu", name="dec_conv2")(x)       # 16->32
    x = layers.Conv2DTranspose(base_filters,     3, strides=2, padding="same", activation="relu", name="dec_conv3")(x)       # 32->64
    outputs = layers.Conv2DTranspose(3, 3, strides=2, padding="same", activation="sigmoid", name="dec_output")(x)            # 64->128

    return Model(inputs, outputs, name="conv_autoencoder")
```

## 4. Design choices to justify

| Choice | Why |
|---|---|
| `strides=2` instead of `MaxPooling2D` | Strided convolution learns how to downsample itself, rather than a fixed max |
| Filters double at each encoder stage (32→64→128→256) | Compensates for the loss of spatial resolution with more representational capacity |
| `padding="same"` everywhere | Simplifies shape computation, guarantees exact encoder/decoder symmetry |
| `sigmoid` output activation | Consistent with the pipeline's `[0,1]` normalization — `tanh`/`linear` would require renormalizing to `[-1,1]` or adjusting the loss |
| `MeanSquaredError` loss (or `binary_crossentropy`) | No label: the input image is also the target (`fit(x, x)`) |

---

## 5. Reading the `summary()`

Example output (128×128×3, `base_filters=32`):

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

### Columns

- **Layer (type)**: the given name (`name=...`) + Keras layer type.
- **Output Shape**: `(None, H, W, C)` — `None` is the batch size (variable). Check here that downsampling/upsampling behaves as expected (`128→64→32→16→8` then `8→16→32→64→128`).
- **Param #**: number of trainable weights. For a `Conv2D`:
  ```
  params = (kernel_h × kernel_w × in_channels + 1) × out_channels
  ```
  The `+1` is the per-filter bias. Check: `enc_conv1` = `(3×3×3 + 1) × 32 = 896`; `enc_conv2` = `(3×3×32 + 1) × 64 = 18,496`.
- **Total params**: sum across all layers, with a memory equivalent (weights in `float32` → `total × 4 bytes`).
- **Trainable / Non-trainable**: everything is trainable here; non-trainable layers would appear with a frozen pretrained encoder (`layer.trainable = False`).

### Points to check systematically

1. **Last layer's Output Shape == input's Output Shape** — necessary condition to compare the reconstruction to the original pixel by pixel.
2. **Symmetry of parameter counts** between encoder and decoder — a marked imbalance often signals a poorly sized bottleneck.
3. **Total params vs training data volume**: worth comparing (e.g. 776k parameters for 177 images is a tight ratio, risk of overfitting — reduce `base_filters` or strengthen regularization if validation loss plateaus/increases).

### Compression ratio

The bottleneck isn't judged by parameter count alone — what matters for anomaly detection is how many **values** are needed to represent an image once it has passed through the encoder, compared to the number of values in the original image:

```
compression ratio = (nb of input values) / (nb of values in the latent)
                   = (H × W × C input) / (H' × W' × C' latent)
```

For this model (`128×128×3` → bottleneck `8×8×256`):

```python
input_values = 128 * 128 * 3    # 49,152
latent_values = 8 * 8 * 256     # 16,384
ratio = input_values / latent_values   # 3.0x
```

**Reading it**: a high ratio (very compact bottleneck) forces the model to learn a more abstract representation of "normal" — useful for anomaly detection, but risks underfitting (degraded reconstruction even on normal images) if the ratio is too aggressive. Too low a ratio (bottleneck almost as large as the input) risks letting the model learn a near-identity mapping instead, which is uninformative for distinguishing normal from defective. A 3x ratio here is moderate — adjust (`base_filters`, number of stages) depending on whether reconstruction underfits or overfits in practice.

---

## 6. Compilation (MSE + SSIM)

No label: the input image is also the target (`fit(x, x)`, target = input = reconstruction).

- **Loss = MSE** (`mean_squared_error`): penalizes pixel-by-pixel reconstruction error, drives the optimization.
- **SSIM** (Structural Similarity Index) tracked as an additional metric, not as the main loss: unlike MSE, it's sensitive to local structure (luminance, contrast, texture) rather than raw pixel-by-pixel difference — closer to how a structural defect would perceptually degrade the reconstruction. Keras has no native SSIM metric, so it's wrapped via `tf.image.ssim`:

```python
def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))  # max_val=1.0 since images are normalized [0,1]

ssim_metric.__name__ = "ssim"  # name used in history.history and logs

autoencoder.compile(optimizer="adam", loss="mse", metrics=[ssim_metric])

train_ds_xy = pipeline.train_ds.map(lambda x: (x, x))
val_ds_xy = pipeline.val_ds.map(lambda x: (x, x))
```

- **Optimizer = Adam**: robust default choice for this kind of model, no fine learning-rate tuning needed to get started.

---

## 7. MLflow tracking

Same convention as the rest of the project (local SQLite, `mlflow/mlflow.db`):

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
mlflow.set_experiment("bottle_autoencoder")
```

- **Params** logged once per run (`img_size`, `batch_size`, `epochs`, `base_filters`, `optimizer`, `loss`).
- **Metrics** logged at every epoch (`train_loss`, `val_loss`, `train_ssim`, `val_ssim`) — allows comparing multiple runs in the MLflow UI (`mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db`).
- **Artifacts**: training curves and reconstruction examples, saved as figures attached to the run (`mlflow.log_figure`).

---

## 8. Training

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

`EPOCHS` deliberately low at first for a quick pipeline sanity check (e.g. 3) — increase once loss and SSIM are confirmed to move in the right direction.

---

## 9. Training curves

Two relevant curves to track, each in train **and** validation (the train/validation gap tells you about overfitting):

- **Loss (MSE)**: should decrease on both curves; a `val_loss` that rises while `train_loss` keeps falling signals overfitting.
- **SSIM**: should increase towards 1.0 (structurally faithful reconstruction); read alongside the loss, not instead of it — the two metrics can diverge slightly (MSE penalizes raw difference, SSIM penalizes structure).

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

## 10. Reconstruction examples after training

Visual comparison on a **validation** batch (never seen with augmentation, never used for the training loss): unlike the section 5 preview in the notebook (random weights, before any training), the reconstruction should now resemble the original if training has converged.

```python
for batch in val_ds_xy.take(1):
    x_val, _ = batch
    reconstruction_trained = autoencoder(x_val)

# ... display original vs reconstruction (see notebook) ...

with mlflow.start_run(run_id=run_id):
    mlflow.log_figure(fig, "reconstructions.png")
```

---

## 11. Next step

Once the MLflow run looks good (decreasing loss, increasing SSIM, visually correct reconstructions on normal images): define an anomaly score (e.g. per-pixel reconstruction error or 1-SSIM) on `test/good` and `test/<defect>`, then evaluate its discriminative power (AUC-ROC, AUC-PR) — see `etude_desequilibre_classe.md` for metric choices suited to the dataset's imbalance.
