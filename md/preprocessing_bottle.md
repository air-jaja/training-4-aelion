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

Le prétraitement ci-dessus produit `train_ds`, `val_ds`, `test_good_ds`, `test_defect_ds` et `test_defect_masks`, prêts à l'emploi. La partie 2 (sections 7 à 20) les réutilise directement pour concevoir, entraîner et évaluer un autoencodeur convolutionnel.

---

# Partie 2 — Autoencodeur convolutionnel

Réutilise directement `train_ds` et `val_ds` construits à l'étape 3.9 (déjà normalisés `[0,1]`, `train_ds` avec augmentation, `val_ds` sans), ainsi que `IMG_SIZE`/`BATCH_SIZE` définis en configuration. Pas de rechargement de données ici.

```python
from tensorflow.keras import layers, Model
import mlflow
```

---

## 7. Conception de l'architecture de l'autoencodeur

Un autoencodeur convolutionnel a deux moitiés symétriques :

- **Encodeur** : réduit progressivement la résolution spatiale (`strides=2`) tout en augmentant le nombre de filtres — échange l'information spatiale contre de l'information sémantique.
- **Bottleneck** : la représentation la plus compacte, au milieu du réseau — c'est la contrainte qui force le modèle à apprendre une structure du "normal" plutôt que de simplement copier l'image.
- **Décodeur** : miroir inverse (`Conv2DTranspose`), remonte en résolution jusqu'à retrouver la taille et le nombre de canaux de l'image d'origine.

**Choix de conception faits ici** :
- `strides=2` plutôt que `MaxPooling2D` : la convolution stridée apprend elle-même comment sous-échantillonner.
- Filtres qui doublent à chaque étage de l'encodeur (32→64→128→256) : compense la perte de résolution spatiale par plus de capacité de représentation.
- `padding="same"` partout : simplifie le calcul des tailles, garantit la symétrie encodeur/décodeur.
- Activation `sigmoid` en sortie : cohérente avec la normalisation `[0,1]` du pipeline.

```python
def build_autoencoder(img_size: int = IMG_SIZE, base_filters: int = 32) -> Model:
    inputs = layers.Input(shape=(img_size, img_size, 3), name="input_image")

    # --- Encodeur ---
    x = layers.Conv2D(base_filters,     3, strides=2, padding="same", activation="relu", name="enc_conv1")(inputs)
    x = layers.Conv2D(base_filters * 2, 3, strides=2, padding="same", activation="relu", name="enc_conv2")(x)
    x = layers.Conv2D(base_filters * 4, 3, strides=2, padding="same", activation="relu", name="enc_conv3")(x)
    x = layers.Conv2D(base_filters * 8, 3, strides=2, padding="same", activation="relu", name="enc_conv4")(x)

    latent = x  # bottleneck

    # --- Décodeur ---
    x = layers.Conv2DTranspose(base_filters * 4, 3, strides=2, padding="same", activation="relu", name="dec_conv1")(latent)
    x = layers.Conv2DTranspose(base_filters * 2, 3, strides=2, padding="same", activation="relu", name="dec_conv2")(x)
    x = layers.Conv2DTranspose(base_filters,     3, strides=2, padding="same", activation="relu", name="dec_conv3")(x)
    outputs = layers.Conv2DTranspose(3, 3, strides=2, padding="same", activation="sigmoid", name="dec_output")(x)

    return Model(inputs, outputs, name="conv_autoencoder")


autoencoder = build_autoencoder()
```

### Informations clés du bottleneck

Avant de lire le `summary()` complet, on isole les informations qui caractérisent le bottleneck lui-même — c'est la partie de l'architecture qui détermine la capacité de compression du modèle, donc sa capacité à distinguer le "normal" de l'anormal.

```python
bottleneck_layer_name = "enc_conv4"
bottleneck_layer = autoencoder.get_layer(bottleneck_layer_name)
bottleneck_shape = bottleneck_layer.output.shape[1:]   # (H', W', C'), sans la dimension de batch
input_shape = autoencoder.input.shape[1:]              # (H, W, C)

bottleneck_values = int(np.prod(bottleneck_shape))
input_values = int(np.prod(input_shape))
spatial_reduction = input_shape[0] // bottleneck_shape[0]   # ex. 128 // 8 = 16
compression_ratio = input_values / bottleneck_values

print(f"Couche bottleneck         : {bottleneck_layer_name}")
print(f"Forme entrée              : {tuple(input_shape)}  ({input_values} valeurs)")
print(f"Forme bottleneck          : {tuple(bottleneck_shape)}  ({bottleneck_values} valeurs)")
print(f"Réduction spatiale        : ÷{spatial_reduction} en hauteur et en largeur")
print(f"Canaux bottleneck         : {bottleneck_shape[-1]}")
print(f"Ratio de compression      : {compression_ratio:.2f}x")
```

Sur ce modèle (`128×128×3`, `base_filters=32`) : bottleneck `enc_conv4` = `(8, 8, 256)`, soit 16 384 valeurs contre 49 152 en entrée — réduction spatiale ÷16, ratio de compression 3.00x.

---

## 8. Lecture du `summary()`

**Rappel bottleneck** (calculé en étape 7, affiché en évidence juste avant le tableau) : c'est `enc_conv4` qui porte le `Output Shape` le plus petit — repérer cette ligne en premier, c'est elle qui contraint toute la capacité du modèle.

```python
print("=" * 60)
print(f"BOTTLENECK ({bottleneck_layer_name}) — à repérer dans le summary() ci-dessous")
print(f"  Forme            : {tuple(bottleneck_shape)}")
print(f"  Compression       : {input_values} -> {bottleneck_values} valeurs ({compression_ratio:.2f}x)")
print("=" * 60)

autoencoder.summary()
```

Points à vérifier systématiquement dans le tableau affiché :

1. **Output Shape de la dernière couche == Output Shape de l'entrée** (`(None, 128, 128, 3)` des deux côtés) — condition nécessaire pour comparer la reconstruction à l'original pixel à pixel.
2. **Param # d'une `Conv2D`** : `(kernel_h × kernel_w × canaux_entrée + 1) × canaux_sortie` (le `+1` = biais par filtre).
3. **Symétrie du nombre de paramètres** entre encodeur et décodeur — un déséquilibre marqué signale souvent un bottleneck mal dimensionné.
4. **Total params vs volume de données** : à mettre en regard du nombre d'images d'entraînement pour anticiper un risque de sur-apprentissage.

```python
def conv2d_params(kernel, in_channels, out_channels):
    return (kernel * kernel * in_channels + 1) * out_channels

print("enc_conv1 attendu :", conv2d_params(3, 3, 32), "  (résultat summary() : 896)")
print("enc_conv2 attendu :", conv2d_params(3, 32, 64), " (résultat summary() : 18496)")
```

### Ratio de compression

Le bottleneck ne se juge pas qu'au nombre de paramètres — ce qui compte, c'est combien de **valeurs** il faut pour représenter une image une fois passée dans l'encodeur, comparé au nombre de valeurs de l'image d'origine. Ces valeurs ont déjà été calculées en étape 7 — on les réaffiche ici en contexte, juste après le `summary()` :

```
ratio de compression = (nb de valeurs en entrée) / (nb de valeurs dans le latent)
                      = (H × W × C entrée) / (H' × W' × C' latent)
```

```python
print(f"Entrée   : {tuple(input_shape)} -> {input_values} valeurs")
print(f"Latent   : {tuple(bottleneck_shape)} -> {bottleneck_values} valeurs")
print(f"Ratio de compression : {compression_ratio:.2f}x")
```

**Lecture** : un ratio élevé (bottleneck très compact) force le modèle à apprendre une représentation plus abstraite du "normal" — utile pour la détection d'anomalie, mais risque de sous-apprentissage si le ratio est trop agressif. Un ratio trop faible risque à l'inverse de laisser le modèle apprendre une quasi-identité, peu informative pour distinguer sain/défectueux.

---

## 9. Vérification sur un vrai batch du pipeline

Avant d'aller plus loin, faire passer un vrai batch de `train_ds` dans le modèle de référence (poids encore aléatoires) pour vérifier que les formes s'enchaînent correctement de bout en bout :

```python
for batch in train_ds.take(1):
    reconstruction = autoencoder(batch)
    print("Batch d'entrée :", batch.shape)
    print("Reconstruction :", reconstruction.shape)

    mse = tf.keras.losses.MeanSquaredError()
    print("MSE (poids aléatoires, avant tout entraînement) :", float(mse(batch, reconstruction)))
```

Aperçu visuel (entrée vs reconstruction, poids aléatoires — non informatif à ce stade, sert juste à confirmer que le pipeline image → modèle → image fonctionne).

---

## 10. Deux algorithmes à comparer : perte MSE vs perte SSIM

Jusqu'ici, SSIM n'était qu'une **métrique suivie** à côté de la loss MSE (un seul modèle). Pour comprendre l'écart avant/après entraînement propre à chaque critère d'optimisation, on entraîne maintenant **deux modèles indépendants**, de même architecture (`build_autoencoder`), mais chacun optimisé sur sa propre perte :

- **Algorithme A** : `loss = MSE` (la métrique SSIM reste suivie, pour comparaison).
- **Algorithme B** : `loss = 1 - SSIM` (Keras n'a pas de loss SSIM native, on la définit explicitement ; la métrique MSE reste suivie, pour comparaison).

Les deux modèles partent de poids aléatoires indépendants, et sont évalués sur le **même batch de validation fixe** (`x_val_fixed`) avant et après entraînement — condition nécessaire pour que la comparaison "avant/après" soit valide pour chaque algorithme.

```python
def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
ssim_metric.__name__ = "ssim"

def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def mse_metric(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))
mse_metric.__name__ = "mse"

train_ds_xy = train_ds.map(lambda x: (x, x))
val_ds_xy = val_ds.map(lambda x: (x, x))

# Deux instances indépendantes, mêmes hyperparamètres d'architecture
model_mse = build_autoencoder(IMG_SIZE)
model_ssim = build_autoencoder(IMG_SIZE)

# Même batch de validation fixe pour une comparaison avant/après valide sur les deux algorithmes
for batch in val_ds_xy.take(1):
    x_val_fixed, _ = batch

recon_mse_before = model_mse(x_val_fixed)
recon_ssim_before = model_ssim(x_val_fixed)
```

---

## 11. Suivi MLflow

Même convention que le reste du projet (SQLite locale, `mlflow/mlflow.db`). **Un run MLflow distinct par algorithme** (`conv_autoencoder_mse` et `conv_autoencoder_ssim`), pour comparer les deux dans l'UI MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db`) :
- **Params** : `img_size`, `batch_size`, `epochs`, `early_stopping_patience`, `base_filters`, `optimizer`, `loss` (différent selon l'algorithme).
- **Metrics** par epoch : la loss native de l'algorithme (train/val) + la métrique croisée de l'autre critère (train/val), pour comparaison directe.
- **Artifacts** : courbes d'apprentissage et reconstructions avant/après, par algorithme — nommés avec le `run_id` pour éviter toute collision (`curves_{run_id}.png`, etc.).

```python
MLFLOW_TRACKING_URI = "sqlite:///mlflow/mlflow.db"
EXPERIMENT_NAME = "bottle_autoencoder"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

EPOCHS = 30          # nombre maximum d'epochs
PATIENCE = 5         # arrêt anticipé si val_loss ne s'améliore plus pendant PATIENCE epochs
```

---

## 12. Algorithme A — Entraînement (perte MSE)

`loss="mse"`, SSIM suivie comme métrique croisée.

```python
early_stopping_mse = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=PATIENCE, restore_best_weights=True,
)

model_mse.compile(optimizer="adam", loss="mse", metrics=[ssim_metric])

with mlflow.start_run(run_name="conv_autoencoder_mse") as run_mse:
    mlflow.log_params({
        "img_size": IMG_SIZE, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "early_stopping_patience": PATIENCE, "base_filters": 32,
        "optimizer": "adam", "loss": "mse",
    })

    history_mse = model_mse.fit(
        train_ds_xy, validation_data=val_ds_xy,
        epochs=EPOCHS, callbacks=[early_stopping_mse], verbose=2,
    )

    epochs_trained_mse = len(history_mse.history["loss"])
    mlflow.log_param("epochs_trained", epochs_trained_mse)

    for epoch in range(epochs_trained_mse):
        mlflow.log_metrics({
            "train_loss": history_mse.history["loss"][epoch],
            "val_loss": history_mse.history["val_loss"][epoch],
            "train_ssim": history_mse.history["ssim"][epoch],
            "val_ssim": history_mse.history["val_ssim"][epoch],
        }, step=epoch)

    run_id_mse = run_mse.info.run_id
```

---

## 13. Algorithme A — Courbes d'apprentissage (MSE)

- **Loss (MSE)** : doit décroître sur train et validation.
- **SSIM (métrique croisée)** : doit croître vers 1.0 — permet de voir si optimiser directement le MSE améliore aussi la similarité structurelle, ou si les deux critères divergent.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(history_mse.history["loss"], label="train")
axes[0].plot(history_mse.history["val_loss"], label="validation")
axes[0].set_title("Loss (MSE) — Algorithme MSE")
axes[1].plot(history_mse.history["ssim"], label="train")
axes[1].plot(history_mse.history["val_ssim"], label="validation")
axes[1].set_title("SSIM (métrique croisée) — Algorithme MSE")

with mlflow.start_run(run_id=run_id_mse):
    mlflow.log_figure(fig, f"curves_{run_id_mse}.png")
```

---

## 14. Algorithme A — Reconstruction avant / après entraînement

Même batch de validation (`x_val_fixed`) que la capture "avant entraînement" de la section 10 — comparaison directe de l'écart apporté par l'entraînement sur la perte MSE. Rendu en 3 lignes : original / avant entraînement / après entraînement.

```python
recon_mse_after = model_mse(x_val_fixed)
# ... affichage 3 lignes (original, avant, après) — voir notebook ...

with mlflow.start_run(run_id=run_id_mse):
    mlflow.log_figure(fig, f"reconstructions_before_after_{run_id_mse}.png")
```

---

## 15. Algorithme B — Entraînement (perte SSIM)

`loss=ssim_loss` (1 - SSIM), MSE suivie comme métrique croisée. Même `EPOCHS`/`PATIENCE` que l'algorithme A, pour une comparaison équitable.

```python
early_stopping_ssim = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=PATIENCE, restore_best_weights=True,
)

model_ssim.compile(optimizer="adam", loss=ssim_loss, metrics=[mse_metric])

with mlflow.start_run(run_name="conv_autoencoder_ssim") as run_ssim:
    mlflow.log_params({
        "img_size": IMG_SIZE, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "early_stopping_patience": PATIENCE, "base_filters": 32,
        "optimizer": "adam", "loss": "ssim_loss (1 - SSIM)",
    })

    history_ssim = model_ssim.fit(
        train_ds_xy, validation_data=val_ds_xy,
        epochs=EPOCHS, callbacks=[early_stopping_ssim], verbose=2,
    )

    epochs_trained_ssim = len(history_ssim.history["loss"])
    mlflow.log_param("epochs_trained", epochs_trained_ssim)

    for epoch in range(epochs_trained_ssim):
        mlflow.log_metrics({
            "train_loss": history_ssim.history["loss"][epoch],
            "val_loss": history_ssim.history["val_loss"][epoch],
            "train_mse": history_ssim.history["mse"][epoch],
            "val_mse": history_ssim.history["val_mse"][epoch],
        }, step=epoch)

    run_id_ssim = run_ssim.info.run_id
```

---

## 16. Algorithme B — Courbes d'apprentissage (SSIM)

- **Loss (1 - SSIM)** : doit décroître (donc SSIM croît vers 1.0) sur train et validation.
- **MSE (métrique croisée)** : à surveiller — optimiser SSIM ne garantit pas de minimiser le MSE, les deux courbes peuvent diverger, c'est justement ce que cette comparaison doit révéler.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(history_ssim.history["loss"], label="train")
axes[0].plot(history_ssim.history["val_loss"], label="validation")
axes[0].set_title("Loss (1 - SSIM) — Algorithme SSIM")
axes[1].plot(history_ssim.history["mse"], label="train")
axes[1].plot(history_ssim.history["val_mse"], label="validation")
axes[1].set_title("MSE (métrique croisée) — Algorithme SSIM")

with mlflow.start_run(run_id=run_id_ssim):
    mlflow.log_figure(fig, f"curves_{run_id_ssim}.png")
```

---

## 17. Algorithme B — Reconstruction avant / après entraînement

Même batch de validation fixe, même mise en page que pour l'algorithme A (section 14) — comparaison directe possible entre les deux algorithmes (section 18).

```python
recon_ssim_after = model_ssim(x_val_fixed)
# ... affichage 3 lignes (original, avant, après) — voir notebook ...

with mlflow.start_run(run_id=run_id_ssim):
    mlflow.log_figure(fig, f"reconstructions_before_after_{run_id_ssim}.png")
```

---

## 18. Comparaison des deux algorithmes

Les deux modèles reconstruisent les **mêmes images de validation** — comparaison directe de l'effet du choix de la loss sur la qualité perçue de la reconstruction, et sur les métriques finales.

```python
# ... affichage 3 lignes : original / après MSE / après SSIM — voir notebook ...

print("Résumé final :")
print(f"  MSE  — {epochs_trained_mse} epochs — val_loss(MSE)={history_mse.history['val_loss'][-1]:.5f} — val_ssim={history_mse.history['val_ssim'][-1]:.4f}")
print(f"  SSIM — {epochs_trained_ssim} epochs — val_loss(1-SSIM)={history_ssim.history['val_loss'][-1]:.5f} — val_mse={history_ssim.history['val_mse'][-1]:.5f}")
```

**Lecture** : le "Résumé final" permet de comparer les deux algorithmes sur des critères croisés — ex. le modèle SSIM peut afficher un `val_mse` plus bas que le modèle MSE lui-même, ce qui n'est pas garanti a priori (optimiser une loss n'optimise pas forcément l'autre) mais peut arriver si les deux critères sont globalement alignés sur ce dataset.

---

## 19. Score d'anomalie et calibration du seuil (comparaison MSE vs SSIM)

Même principe que précédemment (score = erreur de reconstruction MSE par image, seuil calibré uniquement sur `val_ds`), appliqué **séparément aux deux modèles entraînés**, pour voir si le choix de la loss d'entraînement change la capacité de discrimination saines/défauts du score d'anomalie.

```python
def reconstruction_errors(dataset, model):
    errors = []
    for batch in dataset:
        recon = model(batch)
        err = tf.reduce_mean(tf.square(batch - recon), axis=[1, 2, 3])
        errors.append(err.numpy())
    return np.concatenate(errors)

THRESHOLD_PERCENTILE = 95
models = {"MSE": (model_mse, run_id_mse), "SSIM": (model_ssim, run_id_ssim)}
scores, thresholds, recalls, fprs = {}, {}, {}, {}

for name, (model, rid) in models.items():
    val_scores = reconstruction_errors(val_ds, model)
    test_good_scores = reconstruction_errors(test_good_ds, model)
    test_defect_scores = np.concatenate([reconstruction_errors(ds, model) for ds in test_defect_ds.values()])

    threshold = np.percentile(val_scores, THRESHOLD_PERCENTILE)
    recall = (test_defect_scores > threshold).mean()
    fpr = (test_good_scores > threshold).mean()

    scores[name] = {"val": val_scores, "good": test_good_scores, "defect": test_defect_scores}
    thresholds[name], recalls[name], fprs[name] = threshold, recall, fpr

    with mlflow.start_run(run_id=rid):
        mlflow.log_metrics({"threshold_p95": threshold, "recall_p95": recall, "fpr_p95": fpr})
```

### Vérification de la généralisation du seuil : `val` vs `test/good`

Le seuil est calibré sur `val_scores` (saines de validation) puis appliqué à `test_good_scores` (saines de test) et `test_defect_scores`. Pour que ça marche, il faut que **`val_scores` et `test_good_scores` suivent une distribution similaire** — ce sont deux échantillons de la même population ("images saines"), juste des images différentes.

Si les deux distributions diffèrent nettement (décalage, forme différente), le seuil calibré sur `val` ne se généralise pas bien à `test/good` — signe d'une instabilité du score d'anomalie pour l'algorithme concerné, indépendante de la taille du split de validation (à vérifier en premier avant d'incriminer un échantillon trop petit).

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, name in zip(axes, ["MSE", "SSIM"]):
    val_s, good_s, threshold = scores[name]["val"], scores[name]["good"], thresholds[name]
    ax.hist(val_s, alpha=0.6, label="val (calibration)")
    ax.hist(good_s, alpha=0.6, label="test/good (application)")
    ax.axvline(threshold, color="black", linestyle="--", label=f"seuil (p{THRESHOLD_PERCENTILE})")
    ax.set_title(f"Algorithme {name} — val vs test/good")

with mlflow.start_run(run_id=run_id_mse):
    mlflow.log_figure(fig, f"val_vs_test_good_{run_id_mse}.png")

# Écart quantifié entre les deux distributions (moyenne et p95), par algorithme
for name in ["MSE", "SSIM"]:
    val_s, good_s = scores[name]["val"], scores[name]["good"]
    print(f"[{name}] mean(val)={val_s.mean():.5f}  mean(test/good)={good_s.mean():.5f}  "
          f"écart={100*(good_s.mean()-val_s.mean())/val_s.mean():+.1f}%")
```

### Rendu visuel — histogrammes saines vs défauts, un panneau par algorithme

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, name in zip(axes, ["MSE", "SSIM"]):
    good, defect, threshold = scores[name]["good"], scores[name]["defect"], thresholds[name]
    ax.hist(good, alpha=0.6, label="test/good (saines)")
    ax.hist(defect, alpha=0.6, label="test/<defect> (défauts)")
    ax.axvline(threshold, color="black", linestyle="--")
    ax.set_title(f"Algorithme {name} — rappel={recalls[name]:.0%}, FP={fprs[name]:.0%}")
```

### Abaisser le seuil augmente le rappel — comparaison des deux algorithmes

Rendu distinct : rappel et taux de faux positifs en fonction du seuil (balayage de percentiles), **une courbe par algorithme** — permet de voir directement si un algorithme offre un meilleur compromis rappel/faux-positifs que l'autre, à n'importe quel seuil.

```python
percentiles = np.arange(50, 100, 2)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

for name, color in [("MSE", "#4C72B0"), ("SSIM", "#DD8452")]:
    val_scores, good, defect = scores[name]["val"], scores[name]["good"], scores[name]["defect"]
    recall_curve = [(defect > np.percentile(val_scores, p)).mean() for p in percentiles]
    fpr_curve = [(good > np.percentile(val_scores, p)).mean() for p in percentiles]
    axes[0].plot(percentiles, recall_curve, marker="o", label=name, color=color)
    axes[1].plot(percentiles, fpr_curve, marker="o", label=name, color=color)

with mlflow.start_run(run_id=run_id_mse):
    mlflow.log_figure(fig, f"recall_vs_threshold_comparison_{run_id_mse}.png")
```

**Lecture** : pour les deux algorithmes, le rappel augmente quand on abaisse le seuil, au prix de plus de faux positifs — pas de seuil "gratuit". Si une courbe domine l'autre (meilleur rappel pour un même taux de faux positifs), l'algorithme correspondant offre un meilleur score d'anomalie intrinsèque. Le choix final du seuil dépend ensuite du coût métier relatif d'un défaut manqué vs d'une fausse alerte (cf. `etude_desequilibre_classe.md`).

---

## 20. Prochaine étape

Le seuil calibré ici (p95 des scores de validation) est un point de départ raisonnable, pas une valeur définitive, pour l'un ou l'autre algorithme. Pistes de suite :
- Choisir l'algorithme (MSE ou SSIM) qui domine sur la courbe rappel/faux-positifs (section 19), plutôt que d'en présupposer un a priori.
- Ajuster le percentile de calibration selon le coût métier réel d'un faux positif vs d'un faux négatif.
- Si le taux de faux positifs observé s'écarte fortement de l'objectif visé par le percentile de calibration (ex. 35-40 % au lieu de ~5 %), vérifier d'abord la généralisation `val` vs `test/good` (section 19) et envisager d'agrandir `VAL_FRACTION` ou de passer en k-fold avant d'ajuster le seuil lui-même.
- Évaluer avec des métriques agrégées indépendantes du seuil (AUC-ROC, AUC-PR) plutôt qu'à un seul point de fonctionnement.
- Passer à un score pixel-level (carte d'erreur de reconstruction par pixel) pour localiser le défaut, en s'appuyant sur `ground_truth/` (IoU, AUC-ROC pixel) — cf. étude de déséquilibre pixel-level en étape 3.11.
