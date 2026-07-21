# Preprocessing Pipeline — `bottle` Dataset (Anomaly Detection)

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

---

## 6. Suggested next step

Once this preprocessing is validated: define the architecture (convolutional autoencoder or variational autoencoder for reconstruction, possibly a pretrained model as a feature extractor for a distance-based/PaDiM-like approach) and the evaluation protocol (image-level AUC-ROC on `test/`, pixel-level IoU/AUC-ROC via `ground_truth/`).