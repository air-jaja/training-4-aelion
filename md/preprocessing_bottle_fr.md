# Pipeline de prétraitement — Dataset `bottle` (détection d'anomalies)

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

---

## 6. Prochaine étape suggérée

Une fois ce prétraitement validé : définir l'architecture (autoencodeur convolutif ou variational autoencoder pour la reconstruction, éventuellement un modèle pré-entraîné en feature extractor pour une approche par distance/PaDiM-like) et le protocole d'évaluation (AUC-ROC image-level sur `test/`, IoU/AUC-ROC pixel-level via `ground_truth/`).
