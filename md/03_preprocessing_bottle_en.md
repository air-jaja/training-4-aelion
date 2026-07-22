# Full `bottle` Pipeline — Preprocessing + Convolutional Autoencoder (Anomaly Detection, MVTec-AD)

Two-part document: **Part 1** (Context through section 6) covers data preprocessing; **Part 2** (sections 7 to 24) covers designing, training, and MLflow-tracking the convolutional autoencoder. Associated notebook: `pipeline_bottle_full.ipynb`.

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

The preprocessing above produces `train_ds`, `val_ds`, `test_good_ds`, `test_defect_ds`, and `test_defect_masks`, ready to use. Part 2 (sections 7 to 24) reuses them directly to design, train, and evaluate a convolutional autoencoder.

---

# Part 2 — Convolutional Autoencoder

Reuses `train_ds` and `val_ds` built at step 3.9 directly (already normalized `[0,1]`, `train_ds` with augmentation, `val_ds` without), along with `IMG_SIZE`/`BATCH_SIZE` defined in configuration. No data reloading here.

```python
from tensorflow.keras import layers, Model
import mlflow
```

---

## 7. Designing the autoencoder architecture

A convolutional autoencoder has two symmetric halves:

- **Encoder**: progressively reduces spatial resolution (`strides=2`) while increasing the number of filters — trading spatial information for semantic information.
- **Bottleneck**: the most compact representation, at the middle of the network — the constraint that forces the model to learn a structure of "normal" rather than simply copying the image.
- **Decoder**: exact mirror (`Conv2DTranspose`), goes back up in resolution to recover the original image size and channel count.

**Design choices made here**:
- `strides=2` instead of `MaxPooling2D`: strided convolution learns how to downsample itself.
- Filters double at each encoder stage (32→64→128→256): compensates for spatial resolution loss with more representational capacity.
- `padding="same"` everywhere: simplifies shape computation, guarantees encoder/decoder symmetry.
- `sigmoid` output activation: consistent with the pipeline's `[0,1]` normalization.

```python
def build_autoencoder(img_size: int = IMG_SIZE, base_filters: int = 32) -> Model:
    inputs = layers.Input(shape=(img_size, img_size, 3), name="input_image")

    # --- Encoder ---
    x = layers.Conv2D(base_filters,     3, strides=2, padding="same", activation="relu", name="enc_conv1")(inputs)
    x = layers.Conv2D(base_filters * 2, 3, strides=2, padding="same", activation="relu", name="enc_conv2")(x)
    x = layers.Conv2D(base_filters * 4, 3, strides=2, padding="same", activation="relu", name="enc_conv3")(x)
    x = layers.Conv2D(base_filters * 8, 3, strides=2, padding="same", activation="relu", name="enc_conv4")(x)

    latent = x  # bottleneck

    # --- Decoder ---
    x = layers.Conv2DTranspose(base_filters * 4, 3, strides=2, padding="same", activation="relu", name="dec_conv1")(latent)
    x = layers.Conv2DTranspose(base_filters * 2, 3, strides=2, padding="same", activation="relu", name="dec_conv2")(x)
    x = layers.Conv2DTranspose(base_filters,     3, strides=2, padding="same", activation="relu", name="dec_conv3")(x)
    outputs = layers.Conv2DTranspose(3, 3, strides=2, padding="same", activation="sigmoid", name="dec_output")(x)

    return Model(inputs, outputs, name="conv_autoencoder")


autoencoder = build_autoencoder()
```

### Key bottleneck information

Before reading the full `summary()`, we isolate the information that characterizes the bottleneck itself.

```python
bottleneck_layer_name = "enc_conv4"
bottleneck_layer = autoencoder.get_layer(bottleneck_layer_name)
bottleneck_shape = bottleneck_layer.output.shape[1:]
input_shape = autoencoder.input.shape[1:]

bottleneck_values = int(np.prod(bottleneck_shape))
input_values = int(np.prod(input_shape))
spatial_reduction = input_shape[0] // bottleneck_shape[0]
compression_ratio = input_values / bottleneck_values

print(f"Bottleneck layer          : {bottleneck_layer_name}")
print(f"Bottleneck shape           : {tuple(bottleneck_shape)}  ({bottleneck_values} values)")
print(f"Spatial reduction         : ÷{spatial_reduction} in height and width")
print(f"Compression ratio         : {compression_ratio:.2f}x")
```

On this model (`128×128×3`, `base_filters=32`): bottleneck `enc_conv4` = `(8, 8, 256)`, spatial reduction ÷16, compression ratio 3.00x.

---

## 8. Reading the `summary()`

**Bottleneck recap** (computed in step 7, shown prominently right before the table): `enc_conv4` carries the smallest `Output Shape` — spot this row first.

```python
print("=" * 60)
print(f"BOTTLENECK ({bottleneck_layer_name}) — spot it in the summary() below")
print(f"  Shape             : {tuple(bottleneck_shape)}")
print(f"  Compression       : {input_values} -> {bottleneck_values} values ({compression_ratio:.2f}x)")
print("=" * 60)

autoencoder.summary()
```

Points to check systematically:
1. **Last layer's Output Shape == input's Output Shape**.
2. **`Conv2D` Param #**: `(kernel_h × kernel_w × in_channels + 1) × out_channels`.
3. **Symmetry of parameter counts** between encoder and decoder.
4. **Total params vs data volume**.

### Compression ratio

What matters is how many **values** are needed to represent an image once encoded, compared to the original — already computed in step 7, reshown here in context. A high ratio forces a more abstract representation (risk of underfitting if too aggressive); too low a ratio risks a near-identity mapping.

---

## 9. Checking a real batch from the pipeline

```python
for batch in train_ds.take(1):
    reconstruction = autoencoder(batch)
    mse = tf.keras.losses.MeanSquaredError()
    print("MSE (random weights, before any training):", float(mse(batch, reconstruction)))
```

Visual preview (input vs reconstruction, random weights — uninformative, just confirms the pipeline works).

---

## 10. Two algorithms to compare: MSE loss vs SSIM loss

So far, SSIM was only a **tracked metric** alongside the MSE loss (a single model). To understand the before/after training gap specific to each optimization criterion, we train **two independent models**, same architecture, each optimized on its own loss:

- **Algorithm A**: `loss = MSE` (SSIM still tracked).
- **Algorithm B**: `loss = 1 - SSIM` (defined explicitly, Keras has no native SSIM loss; MSE still tracked).

**Reproducibility fix (configuration)**: the MSE vs SSIM comparison turned out not to be reproducible run to run, even at identical `VAL_FRACTION` — cause identified: only `np.random.seed(SEED)` was set, never `tf.random.set_seed(SEED)`. `build_autoencoder()`'s weight initialization (Glorot uniform) depends on TensorFlow's random generator, not NumPy's. Fixed in configuration:

```python
np.random.seed(SEED)
tf.random.set_seed(SEED)  # fixes the instability observed across runs
import random
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
```

Effect verified in step 24: with the seed fixed, the three `VAL_FRACTION` values tested give near-identical AUC-ROC — the wild variance observed before the fix is gone.

**Control flag: `RUN_FULL_COMPARISON`** — `True` the first time (or after any dataset/architecture change) to replay the full MSE vs SSIM comparison (sections 10-21). Once the SSIM choice is validated (step 22), switch to `False`: the MSE model is no longer built or trained, all comparison sections adapt automatically (a single panel instead of two on charts), and only the SSIM part (sections 15-17) runs — measured time savings: ~35% on this dataset (see step 25).

```python
RUN_FULL_COMPARISON = True  # False to only retrain SSIM
MODEL_COLORS = {"MSE": "#4C72B0", "SSIM": "#DD8452"}  # consistent colors, used dynamically everywhere

train_ds_xy = train_ds.map(lambda x: (x, x))
val_ds_xy = val_ds.map(lambda x: (x, x))

model_ssim = build_autoencoder(IMG_SIZE)  # retained model: always built

if RUN_FULL_COMPARISON:
    model_mse = build_autoencoder(IMG_SIZE)  # comparison model: only if requested

for batch in val_ds_xy.take(1):
    x_val_fixed, _ = batch

recon_ssim_before = model_ssim(x_val_fixed)
if RUN_FULL_COMPARISON:
    recon_mse_before = model_mse(x_val_fixed)
```

---

## 11. MLflow tracking

Same convention as the rest of the project (local SQLite, `mlflow/mlflow.db`). **One distinct MLflow run per algorithm actually trained.**

```python
MLFLOW_TRACKING_URI = "sqlite:///mlflow/mlflow.db"
EXPERIMENT_NAME = "bottle_autoencoder"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

EPOCHS = 30
PATIENCE = 5
```

---

## 12. Algorithm A — Training (MSE loss)

*(Skipped if `RUN_FULL_COMPARISON=False`.)*

```python
if RUN_FULL_COMPARISON:
    early_stopping_mse = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    model_mse.compile(optimizer="adam", loss="mse", metrics=[ssim_metric])

    with mlflow.start_run(run_name="conv_autoencoder_mse") as run_mse:
        mlflow.log_params({...})
        history_mse = model_mse.fit(train_ds_xy, validation_data=val_ds_xy, epochs=EPOCHS, callbacks=[early_stopping_mse], verbose=2)
        epochs_trained_mse = len(history_mse.history["loss"])
        mlflow.log_param("epochs_trained", epochs_trained_mse)
        for epoch in range(epochs_trained_mse):
            mlflow.log_metrics({...}, step=epoch)
        run_id_mse = run_mse.info.run_id
```

---

## 13. Algorithm A — Training curves (MSE)

*(Skipped if `RUN_FULL_COMPARISON=False`.)* Loss (MSE) train/val + SSIM (cross-metric) train/val, same conventions as before.

---

## 14. Algorithm A — Reconstruction before / after training

*(Skipped if `RUN_FULL_COMPARISON=False`.)* Same fixed batch as step 10, 3 rows: original / before / after (MSE loss).

---

## 15. Algorithm B — Training (SSIM loss)

Always runs (retained model). `loss=ssim_loss` (1 - SSIM), MSE tracked as a cross-metric.

```python
early_stopping_ssim = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
model_ssim.compile(optimizer="adam", loss=ssim_loss, metrics=[mse_metric])

with mlflow.start_run(run_name="conv_autoencoder_ssim") as run_ssim:
    mlflow.log_params({..., "loss": "ssim_loss (1 - SSIM)"})
    history_ssim = model_ssim.fit(train_ds_xy, validation_data=val_ds_xy, epochs=EPOCHS, callbacks=[early_stopping_ssim], verbose=2)
    epochs_trained_ssim = len(history_ssim.history["loss"])
    mlflow.log_param("epochs_trained", epochs_trained_ssim)
    for epoch in range(epochs_trained_ssim):
        mlflow.log_metrics({...}, step=epoch)
    run_id_ssim = run_ssim.info.run_id
```

---

## 16. Algorithm B — Training curves (SSIM)

Always runs. Loss (1 - SSIM) train/val + MSE (cross-metric) train/val.

---

## 17. Algorithm B — Reconstruction before / after training

Always runs. Same layout as step 14, for the SSIM model.

---

## 18. Comparing the two algorithms

*(Skipped if `RUN_FULL_COMPARISON=False` — no MSE model to compare.)*

Both models reconstruct the same validation images; final summary with cross-metrics from both runs (`val_loss`/`val_ssim` for MSE, `val_loss`/`val_mse` for SSIM).

---

## 19. Anomaly score and threshold calibration

Score = per-image MSE reconstruction error, threshold calibrated only on `val_ds`. Two threshold criteria in parallel: **percentile** (p95) and **mean + K·std** (K=3, "3-sigma" rule). The `models` dict adapts to `RUN_FULL_COMPARISON`:

```python
models = {"MSE": (model_mse, run_id_mse), "SSIM": (model_ssim, run_id_ssim)} if RUN_FULL_COMPARISON else {"SSIM": (model_ssim, run_id_ssim)}

for name, (model, rid) in models.items():
    val_scores = reconstruction_errors(val_ds, model)
    test_good_scores = reconstruction_errors(test_good_ds, model)
    test_defect_scores = np.concatenate([reconstruction_errors(ds, model) for ds in test_defect_ds.values()])

    threshold = np.percentile(val_scores, THRESHOLD_PERCENTILE)
    threshold_std = val_scores.mean() + K_STD * val_scores.std()
    # ... recall/FP for both criteria, MLflow logging ...
```

All following render cells (histograms, percentile sweep) iterate over `models.keys()` — a single panel if `RUN_FULL_COMPARISON=False`, two otherwise.

### Checking threshold generalization: `val` vs `test/good`

If the two distributions differ markedly, the threshold calibrated on `val` doesn't generalize well to `test/good` — observed for MSE (35-40% FP instead of ~5%), not for SSIM.

### Visual render — normal vs defect histograms

One panel per algorithm present in `models`, with both thresholds (percentile and mean+K·std) overlaid.

### Lowering the threshold increases recall

Percentile sweep (50 to 99), recall and false positives against threshold, `threshold_sweep` stored for reuse in step 22 (no recomputation).

**Reading it**: recall increases as the threshold is lowered, at the cost of more false positives — no "free" threshold. If one curve dominates the other, that algorithm offers a better intrinsic anomaly score.

---

## 21. Heatmaps & evaluation

Goes down to pixel level: **error map** (heatmap) to localize the defect, then quantitative evaluation at two scales.

### Per-pixel error map (heatmap)

```python
def error_heatmap(original, reconstruction):
    return tf.reduce_mean(tf.square(original - reconstruction), axis=-1)
```

Render `original | reconstruction | heatmap | ground-truth mask`, one example per defect class. The MSE call is conditioned on `RUN_FULL_COMPARISON`; SSIM always shown.

### Threshold-independent aggregate metrics (AUC-ROC, AUC-PR)

- **AUC-ROC**: already seen in steps 19-20.
- **AUC-PR**: more informative than AUC-ROC when the positive class is minority — **relevant especially at pixel level** (rare "defect" pixels, ~85-99% background, see study 3.11). A random classifier gets an AUC-PR close to the positive class prevalence — compare, don't read in absolute terms.

```python
auroc_image[name] = roc_auc_score(y_true_img, y_score_img)
aucpr_image[name] = average_precision_score(y_true_img, y_score_img)
# pixel level: same functions on concatenated heatmaps/masks (all pixels, all test images)
auroc_pixel[name] = roc_auc_score(all_masks.flatten(), all_heatmaps.flatten())
aucpr_pixel[name] = average_precision_score(all_masks.flatten(), all_heatmaps.flatten())
```

**Render — diagrams**: ROC and Precision-Recall curves (image level) side by side, one color per algorithm present in `models`.

### Effect of moving to a pixel-level score: localization via IoU

Image-level metrics answer "is the image defective?", not "where is the defect?". A **pixel threshold** is calibrated (percentile of `val_ds` pixels), the heatmap is binarized to get a **predicted segmentation mask**, compared to the real mask via **IoU**.

```python
def iou_score(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union > 0 else 1.0

PIXEL_THRESHOLD_PERCENTILE = 99
for name, (model, rid) in models.items():
    val_heatmaps, _ = collect_pixel_arrays(model, val_ds)
    pixel_threshold = np.percentile(val_heatmaps.flatten(), PIXEL_THRESHOLD_PERCENTILE)
    # ... IoU per defective image, mean, MLflow logging ...
```

**Render**: predicted segmentation vs real mask, best/worst example, per algorithm.

**Image-level vs pixel-level synthesis**: the two scales are complementary — a high score on one doesn't guarantee a high score on the other. To localize (not just detect), pixel-level IoU is the metric that matters.

### Confusion matrix (at the calibrated threshold)

At the percentile threshold (p95), an image is classified "defective" if its score exceeds the threshold — one matrix per algorithm present in `models`.

### Failure analysis

Missed defects (false negatives) and false alarms (false positives), up to 3 examples of each, per algorithm. **Reading it**: missed defects are typically the smallest/least contrasted (consistent with the pixel-level imbalance study, step 3.11 — `broken_small`). False alarms often come from atypical normal images, underrepresented in training.

---

## 22. Final decision: retained model, adjusted threshold, and per-image IoU

**Retained model: SSIM algorithm.** Justification, from metrics already computed (no new training):
- Step 19 (p95): comparable recall, but FP close to target for SSIM (~5-10%) vs 4 to 8x above target for MSE (35-40%).
- Step 19 (`val` vs `test/good` generalization): SSIM's distribution transfers correctly; MSE's doesn't.
- Step 21 (AUC-ROC/AUC-PR, IoU): consistent with the above.

The rest of this section triggers **no training** and **no reconstruction recomputation** — reuses `scores["SSIM"]`, `pixel_data["SSIM"]` and `threshold_sweep["SSIM"]` already in memory.

### Adjusting the calibration percentile

Rather than the default p95 (arbitrary), the percentile that **maximizes Youden's J** (`recall - false positives`) is selected from the already-computed sweep:

```python
sweep = threshold_sweep["SSIM"]
youden = sweep["recall"] - sweep["fpr"]
best_idx = int(np.argmax(youden))
FINAL_PERCENTILE = int(sweep["percentiles"][best_idx])
final_threshold_image = np.percentile(scores["SSIM"]["val"], FINAL_PERCENTILE)
```

### Pixel segmentation threshold and per-image IoU (adjusted percentile)

Same principle as step 21, with `FINAL_PERCENTILE` (instead of fixed p99), **for each defective image individually**:

```python
val_heatmaps_final, _ = collect_pixel_arrays(model_ssim, val_ds)  # only new forward pass in this section
pixel_threshold_final = np.percentile(val_heatmaps_final.flatten(), FINAL_PERCENTILE)
all_heatmaps_final, all_masks_final = pixel_data["SSIM"]  # reused as-is

iou_rows = []
for i, (label, heatmap, mask) in enumerate(zip(image_labels, all_heatmaps_final, all_masks_final)):
    if mask.sum() == 0:
        continue
    pred_mask = (heatmap > pixel_threshold_final).astype(np.uint8)
    iou_rows.append({"index_global": i, "defect_class": label, "iou": iou_score(pred_mask, mask)})
```

Render: full table (class, index, IoU) for all defective images, per-class + global means, scatter plot by class. **Reading it**: the spread of IoU by defect class reflects the pixel-level imbalance study (step 3.11) — `broken_small` tends to have lower IoU (smaller area, more sensitive to a slight mask misalignment).

---

## 23. Next step

The SSIM model and adjusted threshold (Youden) are a solid starting point, not a final value:
- Validate the Youden percentile on a larger validation set or via k-fold.
- Adjust the percentile based on the actual business cost of FP vs FN.
- Morphological post-processing (erosion/dilation, small-component filtering) on the predicted mask to improve IoU without changing the model.
- In production, only retrain SSIM (`RUN_FULL_COMPARISON=False`) — the full comparison only needs replaying after a dataset/architecture change.

---

## 24. Robustness study: sensitivity to `VAL_FRACTION` and calibration threshold

Direct motivation: the MSE vs SSIM comparison from an isolated run turned out not to be reproducible run to run (see the seed fix, step 10). Rather than trusting a single run, this step formalizes a systematic sensitivity study:

- **3 `VAL_FRACTION` values**: 0.15, 0.25, 0.30 — one SSIM model retrained for each (seed now fixed, so each training run is individually reproducible).
- **3 threshold percentiles**: 92, 95, 99 — applied to each trained model, no retraining needed (recomputed instantly from scores already available for that model).

Result: a 3×3 = 9-combination grid (recall, false positives), plus threshold-independent metrics (AUC-ROC, AUC-PR) per `VAL_FRACTION` value — 3 trainings total, not 9.

```python
VAL_FRACTIONS_TO_TEST = [0.15, 0.25, 0.30]
PERCENTILES_TO_TEST = [92, 95, 99]
ROBUSTNESS_EPOCHS = 20
ROBUSTNESS_PATIENCE = 5

robustness_results = []        # one row per (VAL_FRACTION, percentile) combination
robustness_by_fraction = {}    # one entry per VAL_FRACTION (model, scores, AUC)

for vf in VAL_FRACTIONS_TO_TEST:
    tf.random.set_seed(SEED)   # reset random state before each training: only VAL_FRACTION varies

    train_paths_r, val_paths_r = train_test_split(train_good, test_size=vf, random_state=SEED)
    train_ds_r = build_dataset(train_paths_r, augment=True, shuffle=True)
    val_ds_r = build_dataset(val_paths_r, augment=False, shuffle=False)

    model_r = build_autoencoder(IMG_SIZE)
    model_r.compile(optimizer="adam", loss=ssim_loss, metrics=[mse_metric])
    early_stopping_r = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=ROBUSTNESS_PATIENCE, restore_best_weights=True)

    with mlflow.start_run(run_name=f"robustness_ssim_valfrac_{vf}") as run_r:
        mlflow.log_params({"val_fraction": vf, "epochs": ROBUSTNESS_EPOCHS, "n_train": len(train_paths_r), "n_val": len(val_paths_r)})
        history_r = model_r.fit(train_ds_r.map(lambda x: (x, x)), validation_data=val_ds_r.map(lambda x: (x, x)),
                                 epochs=ROBUSTNESS_EPOCHS, callbacks=[early_stopping_r], verbose=0)
        run_id_r = run_r.info.run_id

    # ... scores, image-level AUC-ROC/AUC-PR, stored in robustness_by_fraction[vf] ...

    for pct in PERCENTILES_TO_TEST:
        threshold_r = np.percentile(val_scores_r, pct)
        recall_r = (test_defect_scores_r > threshold_r).mean()
        fpr_r = (test_good_scores_r > threshold_r).mean()
        robustness_results.append({"val_fraction": vf, "percentile": pct, "threshold": threshold_r, "recall": recall_r, "fpr": fpr_r})
```

### Comparative results and charts

- **Full table** of the 9 combinations (threshold, recall, FP), plus AUC-ROC/AUC-PR per `VAL_FRACTION`.
- **Recall / false-positive grids** (VAL_FRACTION × percentile): annotated heatmap, one cell per combination actually tested.
- **AUC-ROC/AUC-PR vs `VAL_FRACTION` curve**: visualizes whether the SSIM model's performance depends on the validation split size.

Each run (`robustness_ssim_valfrac_0.15/0.25/0.30`) is logged to MLflow with its metrics and figures.

**Seed fix validation**: with the seed now fixed, the three `VAL_FRACTION` values give near-identical AUC-ROC on a low-epoch-budget test — confirming the wild variance observed before the fix came from TensorFlow's random initialization, not `VAL_FRACTION` itself.

### Reading it

- **Stability across `VAL_FRACTION`**: with the seed fixed, observed differences now reflect the real effect of split size, plus the initialization randomness (eliminated).
- **Percentile effect**: at fixed `VAL_FRACTION`, increasing the percentile (92→95→99) mechanically reduces recall and false positives (stricter threshold).
- **`VAL_FRACTION` effect**: a larger validation split (0.30) gives a less noisy threshold estimate but leaves fewer images for training — the classic bias/variance trade-off of split size, now visible without being confounded with initialization randomness.
- If, with the seed fixed, results still differ meaningfully across the three `VAL_FRACTION` values, that's a robust signal to factor into the production validation split choice — not an artifact.

---

## 25. Overall processing time

Timer set in the very first cell of the notebook (`_notebook_start_time = time.time()`), displayed here:

```python
_total_elapsed = time.time() - _notebook_start_time
print(f"Total notebook execution time: {_total_elapsed:.1f} s  ({_total_elapsed/60:.1f} min)")
```

A local timer is also set around step 22 (calibration + per-image IoU), displayed separately — on this dataset, this section typically takes under a second (no significant new forward pass, everything reused). The main gain from `RUN_FULL_COMPARISON=False` comes from training (one model instead of two): ~35% less total time observed on this dataset.

The robustness study (step 24) adds its own cost — 3 full trainings — timed separately (`_robustness_elapsed`): on the order of a minute on this dataset for a reduced epoch budget, to be scaled with `ROBUSTNESS_EPOCHS`.
