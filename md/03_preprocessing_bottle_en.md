# Full `bottle` Pipeline — Preprocessing + Convolutional Autoencoder (Anomaly Detection, MVTec-AD)

Two-part document: **Part 1** (Context through section 6) covers data preprocessing; **Part 2** (sections 7 to 17) covers designing, training, and MLflow-tracking the convolutional autoencoder. Associated notebook: `pipeline_bottle_full.ipynb`.

## Context

Dataset structure (MVTec-AD format):

```
data/bottle/
├── train/good/            # normal images  → training
├── test/good/             # normal images  → evaluation
├── test/<defect>/         # defects (broken_large, broken_small, contamination)
└── ground_truth/<defect>/ # binary defect masks (for pixel-level evaluation)
```

Typical approach: train the model **only on normal images** (`train/good`) — an autoencoder, a one-class network, etc. learns to reconstruct/represent "normal"; evaluation is done on `test/` (normal + defective) and, at pixel level, on `ground_truth/` to segment the defect.

---

## Preprocessing goals

1. Resize all images to a fixed size (256×256 or 128×128).
2. Normalize pixels to `[0, 1]`.
3. Reserve a fraction of the normal images from `train/good` for validation.
4. Build the pipelines with `tensorflow`, `albumentations`, `opencv`, `scikit-image`, `pillow`.

---

## 1. Choosing the target size

| Size | Advantage | Drawback |
|---|---|---|
| **128×128** | Fast training, lightweight in memory | Fine details of small defects (`broken_small`) harder to capture |
| **256×256** | Better resolution for small defects and masks | Heavier (RAM/VRAM), slower |

**Recommendation**: start with **128×128** to iterate quickly on the architecture, then move to **256×256** once the pipeline is validated (`broken_small` defects are sensitive to resolution loss).

Make the size configurable (`IMG_SIZE = 128` or `256`) rather than hardcoding it, to make comparisons easy.

---

## 2. Role of each library

| Library | Role in the pipeline |
|---|---|
| **Pillow (PIL)** | Initial file reading (various formats, EXIF/color mode handling), systematic conversion to RGB |
| **OpenCV (`cv2`)** | Fast resizing (`cv2.resize`, `INTER_AREA` interpolation for downscaling, `INTER_LINEAR` for upscaling), fast batch reading |
| **scikit-image** | Preprocessing of `ground_truth` **masks** (clean binarization, `skimage.transform.resize` which preserves binary content better than a standard image resize), segmentation metrics for evaluation (IoU, etc.) |
| **Albumentations** | Data augmentation (only on normal training images): flips, slight rotations, brightness/contrast variations — with **synchronized image+mask transforms** when needed |
| **TensorFlow (`tf.data`)** | Final performant pipeline: loading, caching, batching, prefetching, integration with the model (`tf.data.Dataset`) |

Principle: Pillow/OpenCV for I/O and low-level resizing, Albumentations for augmentation, scikit-image specifically for masks and evaluation, TensorFlow to orchestrate the training pipeline.

---

## 3. Detailed steps

### 3.1 File inventory and validation

- List files in `train/good`, `test/good`, `test/<defect>/*`, `ground_truth/<defect>/*`.
- Verify the 1-to-1 correspondence between each defective image and its mask (`ground_truth`), usually via a suffix (`000_mask.png` ↔ `000.png` following the MVTec-AD convention).
- Check for corrupted files (`PIL.Image.verify()`).

### 3.2 Loading + resizing images

- Read with Pillow, force conversion to RGB (`convert("RGB")`) to normalize channels (some MVTec images are grayscale depending on the category).
- Resize with OpenCV (`cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)`), faster than PIL in batch and appropriate for downscaling.
- Keep a consistent channel order (RGB) between Pillow and OpenCV (OpenCV reads in BGR — watch for the `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` conversion if reading directly with `cv2.imread`).

### 3.3 Resizing masks (`ground_truth`)

- Masks are binary (0/255): a standard resize (bilinear interpolation) can introduce intermediate values that break binarity.
- Use `skimage.transform.resize(mask, (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True, anti_aliasing=False)` (`order=0` = nearest neighbor) to keep the mask strictly binary after resizing.
- Explicitly re-binarize after resizing (`mask > 127` → `0/1`) as a safety check.

### 3.4 Normalization to `[0, 1]`

- Once the image is resized (`uint8`, values 0–255), divide by `255.0` to get `float32` values in `[0, 1]`.
- Do this normalization **as the last step**, after resizing and geometric augmentation (avoid handling floats during geometric augmentation, which is slower with no benefit here).
- Masks stay in `{0, 1}` (no additional `/255` normalization once already binarized).

### 3.5 Train / validation split (normal images only)

- The split is done **only on `train/good`** (defects are only used for testing, never for training in this paradigm).
- Simple random split with a fixed seed (e.g. 85% train / 15% validation), via `sklearn.model_selection.train_test_split` on the list of file paths (no stratification needed, single class).
- Fix the seed for reproducibility (`random_state=42`).
- Important: the split is done **on file paths before loading**, not on tensors, to stay memory-efficient.

### 3.6 Augmentation pipeline (Albumentations) — normal training images only

A dedicated augmentation pipeline is applied **only** to the `train/good` subset after splitting (never on validation/test, to evaluate on data representative of reality — no associated mask anyway since these are normal images).

Transformations chosen, consistent with the use case (a bottle must still look recognizably "normal"):

| Transformation | Purpose | Suggested parameters |
|---|---|---|
| `HorizontalFlip` | Horizontal symmetry (bottle viewed from a fairly symmetric angle) | `p=0.5` |
| `Rotate` | Slight rotation (variable camera angle) | `limit=15°`, `p=0.5` |
| `ShiftScaleRotate` | Translation + slight scaling (imperfect camera centering/zoom) | `shift_limit=0.05`, `scale_limit=0.1`, `rotate_limit=0` (rotation already handled separately), `p=0.5` |
| `RandomBrightnessContrast` | Lighting variations (industrial conditions are never perfectly stable) | `brightness_limit=0.15`, `contrast_limit=0.15`, `p=0.5` |

**Pitfalls to avoid**: no strong distortions (`ElasticTransform`, `GridDistortion`) or aggressive cropping — these would distort the notion of "normal" in an anomaly detection context, where the model needs to learn a stable geometric shape.

### 3.7 Visual check of augmentation (original vs augmented)

Before wiring augmentation into the training pipeline, visually compare, for several normal images:
- the **original** resized image (no augmentation)
- **several augmented draws** of the same image (the pipeline being stochastic, each call produces a different result)

Goal: confirm by eye that the transformations stay realistic (no unrecognizable bottle, no aberrant colors) and that each draw actually produces a different variation (the pipeline isn't a no-op).

### 3.8 Visual check of the loading pipeline

Before building the `tf.data` pipeline, display a grid of loaded/resized images: a few normal images (`train/good`) and a few images from each defect class (`test/<defect>`), with their `ground_truth` mask overlaid for the defects.

- Goal: catch a loading issue by eye (BGR/RGB channel swap, image/mask misalignment, color corruption after resizing) before spending time on training.
- For each defect, display the resized image and its resized mask side by side (or overlaid), to visually confirm the mask actually matches the defective area.

### 3.9 Building the `tf.data` pipeline (training / validation)

- `tf.data.Dataset.from_tensor_slices(file_paths)` → `.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)` → `.batch(BATCH_SIZE)` → `.prefetch(tf.data.AUTOTUNE)`.
- `load_and_preprocess` function: reading (Pillow/OpenCV) + resize + normalization, wrapped via `tf.py_function` or `tf.numpy_function` if keeping OpenCV/Albumentations (not natively compatible with the TensorFlow graph), or alternatively a 100% TensorFlow variant (`tf.io.decode_image`, `tf.image.resize`) to avoid `py_function` and gain performance.
- Augmentation (step 3.6) applied only to the training pipeline, never to the validation one.
- `.cache()` possible after the first pass if the dataset fits in memory (bottle is a small dataset, a few hundred images) — avoid on the training pipeline if augmentation is enabled (a different draw is wanted at each epoch).

### 3.10 Building the test datasets (evaluation)

- `test/good` + each `test/<defect>`: same loading function as training, but **no augmentation**, just resize + normalization.
- `ground_truth/<defect>` masks resized in parallel (step 3.3) for future pixel-level evaluation (IoU, pixel AUC-ROC), grouped by defect class.
- One `tf.data` dataset per defect class (plus one for `test/good`), to evaluate and report metrics separately per defect type.

### 3.11 Class imbalance study

The imbalance to study here sits at two different levels, both inside `test/` (`train/good` has only one class, nothing to study there):

**Image level** — proportion of normal vs defective in `test/`, and the split across defect types:
- Per-class counts (`test/good`, `test/<defect>`) from `inventory()`.
- Overall normal/defective ratio (on this dataset: ~20/83 normal, ~63/83 defective, roughly 1:3) — moderate, but enough to make raw accuracy misleading. Prefer AUC-ROC, AUC-PR, F1-score.
- Split across defect types (on this dataset: `broken_large`/`broken_small`/`contamination` are almost identical in count) — generally homogeneous, the notable imbalance is "normal vs defective," not between defect types.

**Pixel level** — inside each `ground_truth` mask, proportion of "defect" vs "background" pixels (relevant only if a segmentation model, not just image-level classification, is being considered):
- For each mask, fraction of defective pixels (`mask > 127`) over the total, at original resolution (before resizing).
- Distribution per defect class (histogram or boxplot), not just the mean — intra-class variance is informative (a defect class can be very heterogeneous in defect size).
- This imbalance is typically far more severe than the image-level one (often 85–99% "background" pixels even on a defective image): implies a weighted loss (`weighted BCE`, `Dice`, `Focal loss`) if segmenting, never raw pixel accuracy for evaluation.

**Recommended visual summary**: a two-panel figure — per-class counts (bar chart) + distribution of defective pixel fraction per defect class (boxplot) — to document the imbalance at a glance.

**What doesn't need rebalancing**: no over/under-sampling (SMOTE, etc.) is relevant here — duplicating `test/` images would bias evaluation, and there's no supervised training on defects in this "one-class" paradigm. Rebalancing, if needed, happens via the **loss** and **evaluation metrics**, not via data resampling.

---

## 4. Proposed output structure

```
data/bottle_processed/
├── train/          # normal images, preprocessed (resize + normalization), no augmentation stored on disk
├── val/            # reserved fraction of train/good
├── test/
│   ├── good/
│   └── <defect>/
└── ground_truth/
    └── <defect>/   # resized masks, aligned with test/<defect>/
```

Augmentation (Albumentations) is applied **on the fly** in the `tf.data` pipeline, not stored on disk — only resize and normalization are precomputed/persisted if reproducibility on disk is needed.

---

## 5. Points of attention

- **Image/mask resize consistency**: always use the same target size and verify the resized mask still spatially matches the defect on the resized image.
- **No augmentation on validation/test**: would bias the evaluation.
- **No leakage**: the validation split comes only from `train/good`, never from the `test/` folders.
- **BGR vs RGB**: classic source of error when mixing `cv2.imread` (BGR) and Pillow (RGB) in the same pipeline — standardize on a single channel order from the loading step.
- **Mask binarity after resizing**: always check (`np.unique(mask)` should remain `{0, 1}` after processing).
- **Class imbalance** (step 3.11): don't evaluate with raw accuracy, at either level (image or pixel) — the pixel-level imbalance in particular would push this figure close to 100% without being informative.

---

## 6. Transition to Part 2

The preprocessing above produces `train_ds`, `val_ds`, `test_good_ds`, `test_defect_ds`, and `test_defect_masks`, ready to use. Part 2 (sections 7 to 17) reuses them directly to design, train, and evaluate a convolutional autoencoder.

---

# Part 2 — Convolutional Autoencoder


---

## 7. Designing the autoencoder architecture

A convolutional autoencoder has two symmetric halves:

- **Encoder**: progressively reduces spatial resolution (`strides=2` or `MaxPooling2D`) while increasing the number of filters — trading spatial information for semantic information.
- **Bottleneck**: the most compact representation, at the middle of the network. This is the constraint that forces the model to learn a structure of "normal" rather than simply copying the image (without a bottleneck, the network could learn the identity function and the reconstruction error would no longer be informative).
- **Decoder**: exact mirror (`Conv2DTranspose` or `UpSampling2D` + `Conv2D`), going back up in resolution until the original image size and channel count are recovered.

---

## 8. Example implementation (Keras, Functional API)

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

---

## 9. Design choices to justify

| Choice | Why |
|---|---|
| `strides=2` instead of `MaxPooling2D` | Strided convolution learns how to downsample itself, rather than a fixed max |
| Filters double at each encoder stage (32→64→128→256) | Compensates for the loss of spatial resolution with more representational capacity |
| `padding="same"` everywhere | Simplifies shape computation, guarantees exact encoder/decoder symmetry |
| `sigmoid` output activation | Consistent with the pipeline's `[0,1]` normalization — `tanh`/`linear` would require renormalizing to `[-1,1]` or adjusting the loss |
| `MeanSquaredError` loss (or `binary_crossentropy`) | No label: the input image is also the target (`fit(x, x)`) |

---

## 10. Reading the `summary()`

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

## 11. Checking a real batch from the pipeline

Before compiling and training, run a real batch from `train_ds` through the model (weights still random at this stage) to confirm shapes flow correctly end to end:

```python
for batch in train_ds.take(1):
    reconstruction = autoencoder(batch)
    print("Input batch:", batch.shape)
    print("Reconstruction:", reconstruction.shape)

    mse = tf.keras.losses.MeanSquaredError()
    print("MSE (random weights, before any training):", float(mse(batch, reconstruction)))
```

**What to check**: `reconstruction.shape == batch.shape` (otherwise the loss can't be computed), and a non-zero but not aberrant MSE (random weights → uninformative reconstruction, expected at this stage — the value only becomes meaningful after training, see section 15).

---

## 12. Compilation (MSE + SSIM)

No label: the input image is also the target (`fit(x, x)`, target = input = reconstruction).

- **Loss = MSE** (`mean_squared_error`): penalizes pixel-by-pixel reconstruction error, drives the optimization.
- **SSIM** (Structural Similarity Index) tracked as an additional metric, not as the main loss: unlike MSE, it's sensitive to local structure (luminance, contrast, texture) rather than raw pixel-by-pixel difference — closer to how a structural defect would perceptually degrade the reconstruction. Keras has no native SSIM metric, so it's wrapped via `tf.image.ssim`:

```python
def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))  # max_val=1.0 since images are normalized [0,1]

ssim_metric.__name__ = "ssim"  # name used in history.history and logs

autoencoder.compile(optimizer="adam", loss="mse", metrics=[ssim_metric])

train_ds_xy = train_ds.map(lambda x: (x, x))
val_ds_xy = val_ds.map(lambda x: (x, x))
```

- **Optimizer = Adam**: robust default choice for this kind of model, no fine learning-rate tuning needed to get started.

---

## 13. MLflow tracking

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

## 14. Training

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

## 15. Training curves

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

## 16. Reconstruction examples after training

Visual comparison on a **validation** batch (never seen with augmentation, never used for the training loss): unlike the section 11 preview in the notebook (random weights, before any training), the reconstruction should now resemble the original if training has converged.

```python
for batch in val_ds_xy.take(1):
    x_val, _ = batch
    reconstruction_trained = autoencoder(x_val)

# ... display original vs reconstruction (see notebook) ...

with mlflow.start_run(run_id=run_id):
    mlflow.log_figure(fig, "reconstructions.png")
```

---

## 17. Next step

Once the MLflow run looks good (decreasing loss, increasing SSIM, visually correct reconstructions on normal images): define an anomaly score (e.g. per-pixel reconstruction error or 1-SSIM) on `test/good` and `test/<defect>`, then evaluate its discriminative power (AUC-ROC, AUC-PR) — see `etude_desequilibre_classe.md` for metric choices suited to the dataset's imbalance.
