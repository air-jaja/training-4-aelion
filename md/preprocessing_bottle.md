# Pipeline complet `bottle` — Prétraitement + Autoencodeur convolutionnel (détection d'anomalies, MVTec-AD)

Document en deux parties : **Partie 1** (sections Contexte à 6) couvre le prétraitement des données ; **Partie 2** (sections 7 à 24) couvre la conception, l'entraînement et le suivi MLflow de l'autoencodeur convolutionnel. Notebook associé : `pipeline_bottle_full.ipynb`.

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

Le prétraitement ci-dessus produit `train_ds`, `val_ds`, `test_good_ds`, `test_defect_ds` et `test_defect_masks`, prêts à l'emploi. La partie 2 (sections 7 à 24) les réutilise directement pour concevoir, entraîner et évaluer un autoencodeur convolutionnel.

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

Avant de lire le `summary()` complet, on isole les informations qui caractérisent le bottleneck lui-même.

```python
bottleneck_layer_name = "enc_conv4"
bottleneck_layer = autoencoder.get_layer(bottleneck_layer_name)
bottleneck_shape = bottleneck_layer.output.shape[1:]
input_shape = autoencoder.input.shape[1:]

bottleneck_values = int(np.prod(bottleneck_shape))
input_values = int(np.prod(input_shape))
spatial_reduction = input_shape[0] // bottleneck_shape[0]
compression_ratio = input_values / bottleneck_values

print(f"Couche bottleneck         : {bottleneck_layer_name}")
print(f"Forme bottleneck          : {tuple(bottleneck_shape)}  ({bottleneck_values} valeurs)")
print(f"Réduction spatiale        : ÷{spatial_reduction} en hauteur et en largeur")
print(f"Ratio de compression      : {compression_ratio:.2f}x")
```

Sur ce modèle (`128×128×3`, `base_filters=32`) : bottleneck `enc_conv4` = `(8, 8, 256)`, réduction spatiale ÷16, ratio de compression 3.00x.

---

## 8. Lecture du `summary()`

**Rappel bottleneck** (calculé en étape 7, affiché en évidence juste avant le tableau) : c'est `enc_conv4` qui porte le `Output Shape` le plus petit — repérer cette ligne en premier.

```python
print("=" * 60)
print(f"BOTTLENECK ({bottleneck_layer_name}) — à repérer dans le summary() ci-dessous")
print(f"  Forme            : {tuple(bottleneck_shape)}")
print(f"  Compression       : {input_values} -> {bottleneck_values} valeurs ({compression_ratio:.2f}x)")
print("=" * 60)

autoencoder.summary()
```

Points à vérifier systématiquement :
1. **Output Shape de la dernière couche == Output Shape de l'entrée**.
2. **Param # d'une `Conv2D`** : `(kernel_h × kernel_w × canaux_entrée + 1) × canaux_sortie`.
3. **Symétrie du nombre de paramètres** entre encodeur et décodeur.
4. **Total params vs volume de données**.

### Ratio de compression

Ce qui compte, c'est combien de **valeurs** il faut pour représenter une image une fois passée dans l'encodeur, comparé au nombre de valeurs de l'image d'origine — déjà calculé en étape 7, réaffiché ici en contexte. Un ratio élevé force une représentation plus abstraite (risque de sous-apprentissage si trop agressif) ; un ratio trop faible risque une quasi-identité peu informative.

---

## 9. Vérification sur un vrai batch du pipeline

```python
for batch in train_ds.take(1):
    reconstruction = autoencoder(batch)
    mse = tf.keras.losses.MeanSquaredError()
    print("MSE (poids aléatoires, avant tout entraînement) :", float(mse(batch, reconstruction)))
```

Aperçu visuel (entrée vs reconstruction, poids aléatoires — non informatif, sert juste à confirmer le pipeline).

---

## 10. Deux algorithmes à comparer : perte MSE vs perte SSIM

Jusqu'ici, SSIM n'était qu'une **métrique suivie** à côté de la loss MSE (un seul modèle). Pour comprendre l'écart avant/après entraînement propre à chaque critère d'optimisation, on entraîne **deux modèles indépendants**, de même architecture, chacun optimisé sur sa propre perte :

- **Algorithme A** : `loss = MSE` (SSIM restant suivie).
- **Algorithme B** : `loss = 1 - SSIM` (définie explicitement, Keras n'a pas de loss SSIM native ; MSE restant suivie).

**Correction de reproductibilité (configuration)** : la comparaison MSE vs SSIM s'est révélée non reproductible d'un run à l'autre y compris à `VAL_FRACTION` identique — cause identifiée : seul `np.random.seed(SEED)` était fixé, jamais `tf.random.set_seed(SEED)`. Or l'initialisation des poids de `build_autoencoder()` (Glorot uniform) dépend du générateur aléatoire de TensorFlow, pas de NumPy. Corrigé en configuration :

```python
np.random.seed(SEED)
tf.random.set_seed(SEED)  # corrige l'instabilité observée entre exécutions
import random
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
```

Effet vérifié à l'étape 24 : à seed fixée, les trois valeurs de `VAL_FRACTION` testées donnent des AUC-ROC quasi identiques — la variance sauvage observée avant correction a disparu.

**Flag de contrôle : `RUN_FULL_COMPARISON`** — à `True` la première fois (ou après tout changement de dataset/architecture) pour rejouer la comparaison complète MSE vs SSIM (sections 10-21). Une fois le choix de SSIM validé (étape 22), repasser à `False` : le modèle MSE n'est plus construit ni entraîné, toutes les sections de comparaison s'adaptent automatiquement (un seul panneau au lieu de deux sur les graphiques), et seule la partie SSIM (sections 15-17) s'exécute — gain de temps mesuré : ~35 % sur ce dataset (voir étape 25).

```python
RUN_FULL_COMPARISON = True  # False pour ne réentraîner que SSIM
MODEL_COLORS = {"MSE": "#4C72B0", "SSIM": "#DD8452"}  # couleurs cohérentes, utilisées dynamiquement partout

train_ds_xy = train_ds.map(lambda x: (x, x))
val_ds_xy = val_ds.map(lambda x: (x, x))

model_ssim = build_autoencoder(IMG_SIZE)  # modèle retenu : toujours construit

if RUN_FULL_COMPARISON:
    model_mse = build_autoencoder(IMG_SIZE)  # modèle de comparaison : uniquement si demandé

for batch in val_ds_xy.take(1):
    x_val_fixed, _ = batch

recon_ssim_before = model_ssim(x_val_fixed)
if RUN_FULL_COMPARISON:
    recon_mse_before = model_mse(x_val_fixed)
```

---

## 11. Suivi MLflow

Même convention que le reste du projet (SQLite locale, `mlflow/mlflow.db`). **Un run MLflow distinct par algorithme réellement entraîné.**

```python
MLFLOW_TRACKING_URI = "sqlite:///mlflow/mlflow.db"
EXPERIMENT_NAME = "bottle_autoencoder"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

EPOCHS = 30
PATIENCE = 5
```

---

## 12. Algorithme A — Entraînement (perte MSE)

*(Ignoré si `RUN_FULL_COMPARISON=False`.)*

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

## 13. Algorithme A — Courbes d'apprentissage (MSE)

*(Ignoré si `RUN_FULL_COMPARISON=False`.)* Loss (MSE) train/val + SSIM (métrique croisée) train/val, mêmes conventions que précédemment.

---

## 14. Algorithme A — Reconstruction avant / après entraînement

*(Ignoré si `RUN_FULL_COMPARISON=False`.)* Même batch fixe qu'en étape 10, 3 lignes : original / avant / après (perte MSE).

---

## 15. Algorithme B — Entraînement (perte SSIM)

Toujours exécuté (modèle retenu). `loss=ssim_loss` (1 - SSIM), MSE suivie comme métrique croisée.

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

## 16. Algorithme B — Courbes d'apprentissage (SSIM)

Toujours exécuté. Loss (1 - SSIM) train/val + MSE (métrique croisée) train/val.

---

## 17. Algorithme B — Reconstruction avant / après entraînement

Toujours exécuté. Même mise en page que l'étape 14, pour le modèle SSIM.

---

## 18. Comparaison des deux algorithmes

*(Ignoré si `RUN_FULL_COMPARISON=False` — pas de modèle MSE à comparer.)*

Les deux modèles reconstruisent les mêmes images de validation ; résumé final avec les métriques croisées des deux runs (`val_loss`/`val_ssim` pour MSE, `val_loss`/`val_mse` pour SSIM).

---

## 19. Score d'anomalie et calibration du seuil

Score = erreur de reconstruction MSE par image, seuil calibré uniquement sur `val_ds`. Deux critères de seuil en parallèle : **percentile** (p95) et **moyenne + K·écart-type** (K=3, règle "3-sigma"). Le dict `models` s'adapte à `RUN_FULL_COMPARISON` :

```python
models = {"MSE": (model_mse, run_id_mse), "SSIM": (model_ssim, run_id_ssim)} if RUN_FULL_COMPARISON else {"SSIM": (model_ssim, run_id_ssim)}

for name, (model, rid) in models.items():
    val_scores = reconstruction_errors(val_ds, model)
    test_good_scores = reconstruction_errors(test_good_ds, model)
    test_defect_scores = np.concatenate([reconstruction_errors(ds, model) for ds in test_defect_ds.values()])

    threshold = np.percentile(val_scores, THRESHOLD_PERCENTILE)
    threshold_std = val_scores.mean() + K_STD * val_scores.std()
    # ... rappel/FP pour les deux critères, log MLflow ...
```

Toutes les cellules de rendu qui suivent (histogrammes, sweep de percentiles) itèrent sur `models.keys()` — un seul panneau si `RUN_FULL_COMPARISON=False`, deux sinon.

### Vérification de la généralisation du seuil : `val` vs `test/good`

Si les deux distributions diffèrent nettement, le seuil calibré sur `val` ne se généralise pas bien à `test/good` — observé pour MSE (35-40 % de FP au lieu de ~5 %), pas pour SSIM.

### Rendu visuel — histogrammes saines vs défauts

Un panneau par algorithme présent dans `models`, avec les deux seuils (percentile et moyenne+K·std) superposés.

### Abaisser le seuil augmente le rappel

Balayage de percentiles (50 à 99), rappel et faux positifs en fonction du seuil, `threshold_sweep` stocké pour réutilisation en étape 22 (pas de recalcul).

**Lecture** : le rappel augmente quand on abaisse le seuil, au prix de plus de faux positifs — pas de seuil "gratuit". Si une courbe domine l'autre, l'algorithme correspondant offre un meilleur score d'anomalie intrinsèque.

---

## 21. Heatmaps & évaluation

Descend au niveau pixel : **carte d'erreur** (heatmap) pour localiser le défaut, puis évaluation quantitative à deux échelles.

### Carte d'erreur par pixel (heatmap)

```python
def error_heatmap(original, reconstruction):
    return tf.reduce_mean(tf.square(original - reconstruction), axis=-1)
```

Rendu `original | reconstruction | heatmap | masque ground-truth`, un exemple par classe de défaut. L'appel pour MSE est conditionné par `RUN_FULL_COMPARISON` ; SSIM toujours affiché.

### Métriques agrégées indépendantes du seuil (AUC-ROC, AUC-PR)

- **AUC-ROC** : déjà vue en étape 19-20.
- **AUC-PR** : plus informative que l'AUC-ROC quand la classe positive est minoritaire — **pertinente surtout au niveau pixel** (pixels "défaut" rares, ~85-99 % de fond, cf. étude 3.11). Un classifieur aléatoire obtient un AUC-PR proche de la prévalence de la classe positive — à comparer, pas à lire en absolu.

```python
auroc_image[name] = roc_auc_score(y_true_img, y_score_img)
aucpr_image[name] = average_precision_score(y_true_img, y_score_img)
# niveau pixel : mêmes fonctions sur les heatmaps/masques concaténés (tous les pixels, toutes les images de test)
auroc_pixel[name] = roc_auc_score(all_masks.flatten(), all_heatmaps.flatten())
aucpr_pixel[name] = average_precision_score(all_masks.flatten(), all_heatmaps.flatten())
```

**Rendu — diagrammes** : courbes ROC et Precision-Recall (niveau image) côte à côte, une couleur par algorithme présent dans `models`.

### Effet du passage à un score pixel-level : localisation via IoU

Les métriques image-level répondent à "l'image est-elle défectueuse ?", pas à "où est le défaut ?". On calibre un **seuil pixel** (percentile des pixels de `val_ds`), on binarise la heatmap pour obtenir un **masque de segmentation prédit**, comparé au masque réel via l'**IoU**.

```python
def iou_score(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union > 0 else 1.0

PIXEL_THRESHOLD_PERCENTILE = 99
for name, (model, rid) in models.items():
    val_heatmaps, _ = collect_pixel_arrays(model, val_ds)
    pixel_threshold = np.percentile(val_heatmaps.flatten(), PIXEL_THRESHOLD_PERCENTILE)
    # ... IoU par image défectueuse, moyenne, log MLflow ...
```

**Rendu** : segmentation prédite vs masque réel, meilleur/pire exemple, par algorithme.

**Synthèse image-level vs pixel-level** : les deux échelles se complètent — un score élevé à l'un ne garantit pas un score élevé à l'autre. Pour localiser (pas seulement détecter), l'IoU pixel-level est la métrique qui compte.

### Matrice de confusion (au seuil calibré)

Au seuil percentile (p95), une image est classée "défectueuse" si son score dépasse le seuil — une matrice par algorithme présent dans `models`.

### Analyse des ratés

Défauts manqués (faux négatifs) et fausses alertes (faux positifs), jusqu'à 3 exemples de chaque, par algorithme. **Lecture** : les défauts manqués sont typiquement les plus petits/peu contrastés (cohérent avec l'étude de déséquilibre pixel-level, étape 3.11 — `broken_small`). Les fausses alertes viennent souvent d'images saines atypiques, peu vues à l'entraînement.

---

## 22. Décision finale : modèle retenu, seuil ajusté et IoU par image

**Modèle retenu : algorithme SSIM.** Justification, à partir des métriques déjà calculées (pas de nouvel entraînement) :
- Étape 19 (p95) : rappel comparable, mais FP proches de la cible pour SSIM (~5-10 %) contre 4 à 8x supérieurs pour MSE (35-40 %).
- Étape 19 (généralisation `val` vs `test/good`) : la distribution SSIM se transporte correctement ; celle de MSE non.
- Étape 21 (AUC-ROC/AUC-PR, IoU) : cohérent avec ce qui précède.

Le reste de cette section ne relance **aucun entraînement** ni recalcul de reconstruction — réutilise `scores["SSIM"]`, `pixel_data["SSIM"]` et `threshold_sweep["SSIM"]` déjà en mémoire.

### Ajustement du percentile de calibration

Plutôt que le p95 par défaut (arbitraire), on sélectionne le percentile qui **maximise l'indice de Youden** (`rappel - faux positifs`) sur le balayage déjà calculé :

```python
sweep = threshold_sweep["SSIM"]
youden = sweep["recall"] - sweep["fpr"]
best_idx = int(np.argmax(youden))
FINAL_PERCENTILE = int(sweep["percentiles"][best_idx])
final_threshold_image = np.percentile(scores["SSIM"]["val"], FINAL_PERCENTILE)
```

### Seuil de segmentation pixel et IoU par image (percentile ajusté)

Même principe que l'étape 21, avec `FINAL_PERCENTILE` (au lieu de p99 fixe), **pour chaque image défectueuse individuellement** :

```python
val_heatmaps_final, _ = collect_pixel_arrays(model_ssim, val_ds)  # seul nouveau passage forward de la section
pixel_threshold_final = np.percentile(val_heatmaps_final.flatten(), FINAL_PERCENTILE)
all_heatmaps_final, all_masks_final = pixel_data["SSIM"]  # réutilisé tel quel

iou_rows = []
for i, (label, heatmap, mask) in enumerate(zip(image_labels, all_heatmaps_final, all_masks_final)):
    if mask.sum() == 0:
        continue
    pred_mask = (heatmap > pixel_threshold_final).astype(np.uint8)
    iou_rows.append({"index_global": i, "classe_defaut": label, "iou": iou_score(pred_mask, mask)})
```

Rendu : tableau complet (classe, index, IoU) pour toutes les images défectueuses, moyennes par classe + globale, graphique de dispersion par classe. **Lecture** : la dispersion de l'IoU par classe reflète l'étude de déséquilibre pixel-level (étape 3.11) — `broken_small` tend à avoir un IoU plus bas (zone plus petite, plus sensible à un décalage du masque prédit).

---

## 23. Prochaine étape

Le modèle SSIM et le seuil ajusté (Youden) constituent un point de départ solide, pas une valeur définitive :
- Valider le percentile de Youden sur un validation set plus grand ou en k-fold.
- Ajuster le percentile selon le coût métier réel FP vs FN.
- Post-traitement morphologique (érosion/dilatation, filtrage de petites composantes) sur le masque prédit pour améliorer l'IoU sans changer le modèle.
- En production, ne réentraîner que SSIM (`RUN_FULL_COMPARISON=False`) — la comparaison complète n'a besoin d'être rejouée qu'en cas de changement de dataset/architecture.

---

## 24. Étude de robustesse : sensibilité à `VAL_FRACTION` et au seuil de calibration

Motivation directe : la comparaison MSE vs SSIM d'un run isolé s'est révélée non reproductible d'une exécution à l'autre (cf. correction de seed, étape 10). Plutôt que de se fier à un seul run, cette étape formalise une étude de sensibilité systématique :

- **3 valeurs de `VAL_FRACTION`** : 0.15, 0.25, 0.30 — un modèle SSIM réentraîné pour chacune (seed fixée, donc chaque entraînement est désormais individuellement reproductible).
- **3 percentiles de seuil** : 92, 95, 99 — appliqués à chaque modèle entraîné, sans réentraînement (recalcul immédiat à partir des scores déjà calculés pour ce modèle).

Résultat : une grille de 3×3 = 9 combinaisons (rappel, faux positifs), plus les métriques indépendantes du seuil (AUC-ROC, AUC-PR) par valeur de `VAL_FRACTION` — 3 entraînements au total, pas 9.

```python
VAL_FRACTIONS_TO_TEST = [0.15, 0.25, 0.30]
PERCENTILES_TO_TEST = [92, 95, 99]
ROBUSTNESS_EPOCHS = 20
ROBUSTNESS_PATIENCE = 5

robustness_results = []        # une ligne par combinaison (VAL_FRACTION, percentile)
robustness_by_fraction = {}    # une entrée par VAL_FRACTION (modèle, scores, AUC)

for vf in VAL_FRACTIONS_TO_TEST:
    tf.random.set_seed(SEED)   # réinitialise l'état aléatoire avant chaque entraînement : seul VAL_FRACTION varie

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

    # ... scores, AUC-ROC/AUC-PR image, stockage dans robustness_by_fraction[vf] ...

    for pct in PERCENTILES_TO_TEST:
        threshold_r = np.percentile(val_scores_r, pct)
        recall_r = (test_defect_scores_r > threshold_r).mean()
        fpr_r = (test_good_scores_r > threshold_r).mean()
        robustness_results.append({"val_fraction": vf, "percentile": pct, "threshold": threshold_r, "recall": recall_r, "fpr": fpr_r})
```

### Résultats comparatifs et graphes

- **Tableau complet** des 9 combinaisons (seuil, rappel, FP), plus AUC-ROC/AUC-PR par `VAL_FRACTION`.
- **Grilles rappel / faux positifs** (VAL_FRACTION × percentile) : heatmap annotée, une case par combinaison réellement testée.
- **Courbe AUC-ROC/AUC-PR vs `VAL_FRACTION`** : visualise si la performance du modèle SSIM dépend de la taille du split de validation.

Chaque run (`robustness_ssim_valfrac_0.15/0.25/0.30`) est loggé dans MLflow avec ses métriques et figures.

**Validation de la correction de seed** : à seed désormais fixée, les trois valeurs de `VAL_FRACTION` donnent des AUC-ROC quasi identiques sur un test à faible budget d'epochs — confirmant que la variance sauvage observée avant correction provenait bien de l'initialisation aléatoire de TensorFlow, pas de `VAL_FRACTION` lui-même.

### Lecture

- **Stabilité entre `VAL_FRACTION`** : la graine étant fixée, les écarts observés reflètent désormais l'effet réel de la taille du split, plus l'aléa d'initialisation (éliminé).
- **Effet du percentile** : à `VAL_FRACTION` fixé, augmenter le percentile (92→95→99) réduit mécaniquement le rappel et les faux positifs (seuil plus strict).
- **Effet de `VAL_FRACTION`** : un split de validation plus grand (0.30) donne une estimation du seuil moins bruitée mais laisse moins d'images pour l'entraînement — compromis classique biais/variance, désormais visible sans être confondu avec l'aléa d'initialisation.
- Si, seed fixée, les résultats restent sensiblement différents entre les trois `VAL_FRACTION`, c'est un signal robuste à prendre en compte dans le choix du split de validation en production — pas un artefact.

---

## 25. Temps de traitement global

Chronométrage posé en toute première cellule du notebook (`_notebook_start_time = time.time()`), affiché ici :

```python
_total_elapsed = time.time() - _notebook_start_time
print(f"Temps total d'exécution du notebook : {_total_elapsed:.1f} s  ({_total_elapsed/60:.1f} min)")
```

Un chronométrage local est également posé autour de la section 22 (calibration + IoU par image), affiché séparément — sur ce dataset, cette section prend typiquement moins d'une seconde (aucun nouveau passage forward significatif, tout est réutilisé). Le gain principal de `RUN_FULL_COMPARISON=False` vient de l'entraînement (un seul modèle au lieu de deux) : ~35 % de temps total en moins observé sur ce dataset.

L'étude de robustesse (étape 24) ajoute son propre coût — 3 entraînements complets — chronométré séparément (`_robustness_elapsed`) : de l'ordre de la minute sur ce dataset pour un budget d'epochs réduit, à mettre à l'échelle selon `ROBUSTNESS_EPOCHS`.
