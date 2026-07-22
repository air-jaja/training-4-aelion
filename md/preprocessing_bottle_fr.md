# Pipeline complet `bottle` — Prétraitement + Autoencodeur convolutionnel (détection d'anomalies, MVTec-AD)

Document en deux parties : **Partie 1** (sections Contexte à 6) couvre le prétraitement des données ; **Partie 2** (sections 7 à 17) couvre la conception, l'entraînement et le suivi MLflow de l'autoencodeur convolutionnel. Notebook associé : `pipeline_bottle_full.ipynb`.

## Contexte

Structure du dataset (format MVTec-AD) :

```
data/bottle/
├── train/good/            # images saines  → entraînement
├── test/good/             # images saines  → évaluation
├── test/<defect>/         # défauts (broken_large, broken_small, contamination)
└── ground_truth/<defect>/ # masques binaires des défauts (pour l'évaluation pixel)
```

Approche typique : entraînement **uniquement sur les images saines** (`train/good`), le modèle (autoencodeur, réseau à une classe, etc.) apprend à reconstruire/représenter le "normal" ; l'évaluation se fait sur `test/` (saines + défauts) et, au niveau pixel, sur `ground_truth/` pour segmenter le défaut.

---

## Objectifs du prétraitement

1. Redimensionner toutes les images à une taille fixe (256×256 ou 128×128).
2. Normaliser les pixels dans `[0, 1]`.
3. Réserver une fraction des images saines de `train/good` pour la validation.
4. Construire les pipelines avec `tensorflow`, `albumentations`, `opencv`, `scikit-image`, `pillow`.

---

## 1. Choix de la taille cible

| Taille | Avantage | Inconvénient |
|---|---|---|
| **128×128** | Entraînement rapide, léger en mémoire | Détails fins de petits défauts (`broken_small`) plus difficiles à capter |
| **256×256** | Meilleure résolution pour les petits défauts et les masques | Plus lourd (RAM/VRAM), plus lent |

**Recommandation** : démarrer en **128×128** pour itérer vite sur l'architecture, repasser en **256×256** une fois le pipeline validé (les défauts `broken_small` sont sensibles à la perte de résolution).

Rendre la taille configurable (`IMG_SIZE = 128` ou `256`) plutôt que la coder en dur, pour pouvoir comparer facilement.

---

## 2. Rôle de chaque bibliothèque

| Bibliothèque | Rôle dans le pipeline |
|---|---|
| **Pillow (PIL)** | Lecture initiale des fichiers image (formats variés, gestion EXIF/mode couleur), conversion en RGB systématique |
| **OpenCV (`cv2`)** | Redimensionnement performant (`cv2.resize`, interpolation `INTER_AREA` pour réduire, `INTER_LINEAR` pour agrandir), lecture rapide en batch |
| **scikit-image** | Prétraitement des **masques de `ground_truth`** (binarisation propre, `skimage.transform.resize` qui préserve le caractère binaire mieux qu'un resize "image" classique), métriques de segmentation pour l'évaluation (IoU, etc.) |
| **Albumentations** | Augmentation de données (uniquement sur les images saines d'entraînement) : flips, légères rotations, variations de luminosité/contraste — **avec transformation synchronisée image+masque** quand nécessaire |
| **TensorFlow (`tf.data`)** | Pipeline final performant : chargement, cache, batching, prefetch, intégration au modèle (`tf.data.Dataset`) |

Principe : Pillow/OpenCV pour l'I/O et le resize bas niveau, Albumentations pour l'augmentation, scikit-image spécifiquement pour les masques et l'évaluation, TensorFlow pour orchestrer le pipeline d'entraînement.

---

## 3. Étapes détaillées

### 3.1 Inventaire et validation des fichiers

- Lister les fichiers de `train/good`, `test/good`, `test/<defect>/*`, `ground_truth/<defect>/*`.
- Vérifier la correspondance 1-à-1 entre chaque image défectueuse et son masque (`ground_truth`), généralement via un suffixe (`000_mask.png` ↔ `000.png` selon la convention MVTec-AD).
- Vérifier qu'il n'y a pas de fichiers corrompus (`PIL.Image.verify()`).

### 3.2 Chargement + redimensionnement des images

- Lecture avec Pillow, conversion forcée en RGB (`convert("RGB")`) pour homogénéiser (certaines images MVTec sont en niveaux de gris selon la catégorie).
- Redimensionnement avec OpenCV (`cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)`), plus rapide que PIL en batch et pertinent pour la réduction de taille.
- Conserver systématiquement le même ordre de canaux (RGB) entre Pillow et OpenCV (OpenCV lit en BGR — attention à la conversion `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` si l'image passe par `cv2.imread` directement).

### 3.3 Redimensionnement des masques (`ground_truth`)

- Les masques sont binaires (0/255) : un resize classique (interpolation bilinéaire) peut créer des valeurs intermédiaires qui cassent la binarité.
- Utiliser `skimage.transform.resize(mask, (IMG_SIZE, IMG_SIZE), order=0, preserve_range=True, anti_aliasing=False)` (`order=0` = plus proche voisin) pour garder un masque strictement binaire après redimensionnement.
- Re-binariser explicitement après resize (`mask > 127` → `0/1`) par sécurité.

### 3.4 Normalisation `[0, 1]`

- Une fois l'image redimensionnée (`uint8`, valeurs 0–255), diviser par `255.0` pour obtenir des `float32` dans `[0, 1]`.
- Faire cette normalisation **en dernière étape**, après resize et augmentation géométrique (éviter de manipuler des floats dans les étapes d'augmentation géométrique, plus lentes et sans bénéfice ici).
- Les masques restent en `{0, 1}` (pas de normalisation `/255` supplémentaire une fois déjà binarisés).

### 3.5 Split train / validation (images saines uniquement)

- Le split se fait **uniquement sur `train/good`** (les défauts servent uniquement au test, jamais à l'entraînement dans ce paradigme).
- Split simple aléatoire avec graine fixée (ex. 85 % train / 15 % validation), via `sklearn.model_selection.train_test_split` sur la liste des chemins de fichiers (pas besoin de stratification, une seule classe).
- Fixer la seed pour reproductibilité (`random_state=42`).
- Important : le split se fait **sur les chemins de fichiers avant chargement**, pas sur les tenseurs, pour rester mémoire-efficace.

### 3.6 Pipeline d'augmentation (Albumentations) — images saines d'entraînement uniquement

Un pipeline d'augmentation dédié est appliqué **uniquement** au sous-ensemble `train/good` après split (jamais sur validation/test, pour évaluer sur des données représentatives de la réalité — pas de masque associé de toute façon puisqu'il s'agit d'images saines).

Transformations retenues, cohérentes avec le cas d'usage (une bouteille doit rester reconnaissable comme "normale") :

| Transformation | Rôle | Paramètres suggérés |
|---|---|---|
| `HorizontalFlip` | Symétrie horizontale (bouteille vue de dessus/dessous, souvent invariante par rotation/flip) | `p=0.5` |
| `Rotate` | Rotation légère (angle de prise de vue variable) | `limit=15°`, `p=0.5` |
| `ShiftScaleRotate` | Translation + mise à l'échelle légère (centrage/zoom imparfait de la caméra) | `shift_limit=0.05`, `scale_limit=0.1`, `rotate_limit=0` (rotation déjà gérée séparément), `p=0.5` |
| `RandomBrightnessContrast` | Variations d'éclairage (conditions industrielles jamais parfaitement stables) | `brightness_limit=0.15`, `contrast_limit=0.15`, `p=0.5` |

**Pièges à éviter** : pas de distorsions fortes (`ElasticTransform`, `GridDistortion`) ni de crop agressif — dénatureraient la notion de "normal" dans un contexte de détection d'anomalie, où le modèle doit apprendre une forme géométrique stable.

### 3.7 Vérification visuelle de l'augmentation (original vs augmenté)

Avant d'intégrer l'augmentation dans le pipeline d'entraînement, comparer visuellement, pour plusieurs images saines :
- l'image **originale** redimensionnée (sans augmentation)
- **plusieurs tirages augmentés** de la même image (le pipeline étant stochastique, chaque appel donne un résultat différent)

Objectif : confirmer à l'œil que les transformations restent réalistes (pas de bouteille méconnaissable, pas de couleurs aberrantes) et que chaque tirage produit bien une variation différente (le pipeline n'est pas figé/inopérant).

### 3.8 Vérification visuelle du chargement

Avant de construire le pipeline `tf.data`, afficher une grille d'images chargées/redimensionnées : quelques images saines (`train/good`) et quelques images de chaque classe de défaut (`test/<defect>`), avec leur masque (`ground_truth`) superposé pour les défauts.

- Objectif : détecter à l'œil un problème de chargement (inversion de canaux BGR/RGB, mauvais alignement image/masque, écrasement de couleurs après resize) avant d'investir du temps dans l'entraînement.
- Afficher, pour chaque défaut : l'image redimensionnée et son masque redimensionné côte à côte (ou en surimpression), pour confirmer visuellement que le masque correspond bien à la zone défectueuse.

### 3.9 Construction du pipeline `tf.data` (entraînement / validation)

- `tf.data.Dataset.from_tensor_slices(file_paths)` → `.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)` → `.batch(BATCH_SIZE)` → `.prefetch(tf.data.AUTOTUNE)`.
- Fonction `load_and_preprocess` : lecture (Pillow/OpenCV) + resize + normalisation, encapsulée via `tf.py_function` ou `tf.numpy_function` si on garde OpenCV/Albumentations (pas nativement compatibles avec le graphe TensorFlow), sinon variante 100 % TensorFlow (`tf.io.decode_image`, `tf.image.resize`) si on veut éviter le `py_function` et gagner en performance.
- Augmentation (étape 3.6) appliquée uniquement au pipeline d'entraînement, jamais à celui de validation.
- `.cache()` possible après le premier passage si le dataset tient en mémoire (bottle = petit dataset, quelques centaines d'images) — à ne pas utiliser sur le pipeline d'entraînement si l'augmentation est activée (on veut un tirage différent à chaque epoch).

### 3.10 Construction des datasets de test (évaluation)

- `test/good` + chaque `test/<defect>` : même fonction de chargement que l'entraînement, mais **sans augmentation**, juste resize + normalisation.
- Masques `ground_truth/<defect>` redimensionnés en parallèle (étape 3.3) pour l'évaluation pixel-level future (IoU, AUC-ROC pixel), regroupés par classe de défaut.
- Un dataset `tf.data` par classe de défaut (plus un pour `test/good`), pour pouvoir évaluer et rapporter les métriques séparément par type de défaut.

### 3.11 Étude de déséquilibre de classe

Le déséquilibre à étudier ici se situe à deux niveaux différents, tous les deux dans `test/` (`train/good` ne contient qu'une seule classe, rien à étudier de ce côté) :

**Niveau image** — proportion saines vs défectueuses dans `test/`, et répartition entre les différents types de défauts :
- Comptage par classe (`test/good`, `test/<defect>`) à partir de `inventory()`.
- Ratio saines/défectueuses global (sur ce dataset : ~20/83 saines, ~63/83 défectueuses, soit environ 1:3) — modéré, mais suffisant pour rendre l'accuracy brute trompeuse. Préférer AUC-ROC, AUC-PR, F1-score.
- Répartition entre types de défauts (sur ce dataset : `broken_large`/`broken_small`/`contamination` quasi identiques en nombre) — généralement homogène, le déséquilibre notable est "sain vs défectueux", pas entre types de défauts.

**Niveau pixel** — à l'intérieur de chaque masque `ground_truth`, proportion de pixels "défaut" vs "fond" (pertinent seulement si un modèle de segmentation, pas seulement de classification image-level, est envisagé) :
- Pour chaque masque, fraction de pixels défectueux (`mask > 127`) sur le total, à la résolution originale (avant resize).
- Distribution par classe de défaut (histogramme ou boxplot), pas seulement la moyenne — la variance intra-classe est informative (une classe de défaut peut être très hétérogène en taille de défaut).
- Ce déséquilibre est typiquement beaucoup plus sévère que celui du niveau image (souvent 85–99 % de pixels "fond" même sur une image défectueuse) : implique une loss pondérée (`weighted BCE`, `Dice`, `Focal loss`) si segmentation, jamais l'accuracy pixel brute pour évaluer.

**Synthèse visuelle recommandée** : un graphique à deux panneaux — comptage par classe (bar chart) + distribution de la fraction de pixels défectueux par classe de défaut (boxplot) — pour documenter le déséquilibre en un coup d'œil.

**Ce qui n'a pas besoin d'être rééquilibré** : pas de sur-échantillonnage/sous-échantillonnage (SMOTE, etc.) pertinent ici — on ne va pas dupliquer des images de `test/` (biaiserait l'évaluation), et il n'y a pas d'entraînement supervisé sur les défauts dans ce paradigme "one-class". Le rééquilibrage, si nécessaire, se fait via la **loss** et les **métriques d'évaluation**, pas via le rééchantillonnage des données.

---

## 4. Arborescence de sortie proposée

```
data/bottle_processed/
├── train/          # images saines, prétraitées (resize + normalisation), sans augmentation stockée sur disque
├── val/            # fraction réservée de train/good
├── test/
│   ├── good/
│   └── <defect>/
└── ground_truth/
    └── <defect>/   # masques redimensionnés, réalignés avec test/<defect>/
```

L'augmentation (Albumentations) se fait **à la volée** dans le pipeline `tf.data`, pas stockée sur disque — seuls le resize et la normalisation sont pré-calculés/persistés si besoin de rejouabilité.

---

## 5. Points de vigilance

- **Cohérence resize image/masque** : toujours utiliser la même taille cible et vérifier que le masque redimensionné correspond toujours spatialement au défaut sur l'image redimensionnée.
- **Pas d'augmentation sur validation/test** : biaiserait l'évaluation.
- **Pas de fuite** : le split validation vient uniquement de `train/good`, jamais des dossiers `test/`.
- **BGR vs RGB** : source d'erreur classique si on mélange `cv2.imread` (BGR) et Pillow (RGB) dans le même pipeline — normaliser un seul ordre de canaux dès le chargement.
- **Binarité des masques après resize** : à vérifier systématiquement (`np.unique(mask)` doit rester `{0, 1}` après traitement).
- **Déséquilibre de classe** (étape 3.11) : ne pas évaluer avec l'accuracy brute, à aucun des deux niveaux (image ou pixel) — le déséquilibre pixel notamment rendrait ce chiffre proche de 100 % sans être informatif.

---

## 6. Transition vers la partie 2

Le prétraitement ci-dessus produit `train_ds`, `val_ds`, `test_good_ds`, `test_defect_ds` et `test_defect_masks`, prêts à l'emploi. La partie 2 (sections 7 à 17) les réutilise directement pour concevoir, entraîner et évaluer un autoencodeur convolutionnel.

---

# Partie 2 — Autoencodeur convolutionnel


---

## 7. Conception de l'architecture de l'autoencodeur

Un autoencodeur convolutionnel a deux moitiés symétriques :

- **Encodeur** : réduit progressivement la résolution spatiale (`strides=2` ou `MaxPooling2D`) tout en augmentant le nombre de filtres — échange l'information spatiale contre de l'information sémantique.
- **Bottleneck** : la représentation la plus compacte, au milieu du réseau. C'est la contrainte qui force le modèle à apprendre une structure du "normal" plutôt que de simplement copier l'image (sans bottleneck, le réseau pourrait apprendre l'identité et l'erreur de reconstruction ne serait plus informative).
- **Décodeur** : miroir inverse (`Conv2DTranspose` ou `UpSampling2D` + `Conv2D`), remonte en résolution jusqu'à retrouver la taille et le nombre de canaux de l'image d'origine.

---

## 8. Exemple d'implémentation (Keras, API fonctionnelle)

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

---

## 9. Choix de conception à justifier

| Choix | Pourquoi |
|---|---|
| `strides=2` plutôt que `MaxPooling2D` | La convolution stridée apprend elle-même comment sous-échantillonner, plutôt qu'un max fixe |
| Filtres qui doublent à chaque étage (32→64→128→256) | Compense la perte de résolution spatiale par plus de capacité de représentation |
| `padding="same"` partout | Simplifie le calcul des tailles, garantit la symétrie exacte encodeur/décodeur |
| Activation `sigmoid` en sortie | Cohérente avec la normalisation `[0,1]` du pipeline — un `tanh`/`linear` demanderait de renormaliser en `[-1,1]` ou d'ajuster la loss |
| Loss `MeanSquaredError` (ou `binary_crossentropy`) | Pas de label : l'image d'entrée est aussi la cible (`fit(x, x)`) |

---

## 10. Lire le `summary()`

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

## 11. Vérification sur un vrai batch du pipeline

Avant de compiler et d'entraîner, faire passer un vrai batch de `train_ds` dans le modèle (poids encore aléatoires à ce stade) pour vérifier que les formes s'enchaînent correctement de bout en bout :

```python
for batch in train_ds.take(1):
    reconstruction = autoencoder(batch)
    print("Batch d'entrée :", batch.shape)
    print("Reconstruction :", reconstruction.shape)

    mse = tf.keras.losses.MeanSquaredError()
    print("MSE (poids aléatoires, avant tout entraînement) :", float(mse(batch, reconstruction)))
```

**Ce qu'il faut vérifier** : `reconstruction.shape == batch.shape` (sinon la loss ne pourra pas se calculer), et un MSE non nul mais pas aberrant (poids aléatoires → reconstruction non informative, c'est normal et attendu à ce stade — la valeur ne devient intéressante qu'après entraînement, cf. section 15).

---

## 12. Compilation (MSE + SSIM)

Pas de label : l'image d'entrée est aussi la cible (`fit(x, x)`, target = entrée = reconstruction).

- **Loss = MSE** (`mean_squared_error`) : pénalise l'erreur de reconstruction pixel à pixel, c'est elle qui pilote l'optimisation.
- **SSIM** (Structural Similarity Index) suivi en métrique additionnelle, pas comme loss principale : contrairement au MSE, elle est sensible à la structure locale (luminance, contraste, texture) plutôt qu'à l'écart pixel à pixel brut — plus proche de la façon dont un défaut structurel dégraderait perceptuellement la reconstruction. Keras n'a pas de métrique SSIM native, on l'encapsule via `tf.image.ssim` :

```python
def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))  # max_val=1.0 car images normalisées [0,1]

ssim_metric.__name__ = "ssim"  # nom utilisé dans history.history et les logs

autoencoder.compile(optimizer="adam", loss="mse", metrics=[ssim_metric])

train_ds_xy = train_ds.map(lambda x: (x, x))
val_ds_xy = val_ds.map(lambda x: (x, x))
```

- **Optimiseur = Adam** : choix par défaut robuste pour ce type de modèle, pas de réglage fin du learning rate nécessaire pour démarrer.

---

## 13. Suivi MLflow

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

## 14. Entraînement

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

## 15. Courbes d'apprentissage

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

## 16. Exemples de reconstruction après entraînement

Comparaison visuelle sur un batch de **validation** (jamais vu en augmentation, jamais utilisé pour la loss d'entraînement) : contrairement à l'aperçu de la section 11 du notebook (poids aléatoires, avant tout entraînement), la reconstruction doit maintenant ressembler à l'original si l'entraînement a convergé.

```python
for batch in val_ds_xy.take(1):
    x_val, _ = batch
    reconstruction_trained = autoencoder(x_val)

# ... affichage original vs reconstruction (voir notebook) ...

with mlflow.start_run(run_id=run_id):
    mlflow.log_figure(fig, "reconstructions.png")
```

---

## 17. Prochaine étape

Une fois le run MLflow validé (loss qui décroît, SSIM qui croît, reconstructions visuellement correctes sur les images saines) : établir un score d'anomalie (ex. erreur de reconstruction par pixel ou 1-SSIM) sur `test/good` et `test/<defect>`, puis évaluer sa capacité de discrimination (AUC-ROC, AUC-PR) — cf. `etude_desequilibre_classe.md` pour le choix des métriques adaptées au déséquilibre du dataset.
