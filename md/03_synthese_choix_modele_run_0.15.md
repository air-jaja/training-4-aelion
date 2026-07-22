# Synthèse — Choix du modèle d'autoencodeur (MSE vs SSIM)

Ce document explique **pourquoi** le modèle entraîné avec la perte **SSIM** a été retenu plutôt que celui entraîné avec la perte **MSE**, et reprend, avec les valeurs chiffrées et les graphiques associés, chaque étape du notebook (`pipeline_bottle_full.ipynb`) qui permet de vérifier ce choix. Les chiffres cités proviennent d'une exécution réelle sur le dataset `bottle` (8 epochs, `EarlyStopping` patience=5 — un entraînement plus long accentuerait probablement ces écarts plutôt que de les inverser).

---

## 1. Point de détail important : l'architecture et son `summary()`

Avant de comparer deux entraînements, il faut savoir *ce qu'on compare* : les deux modèles (MSE et SSIM) partagent exactement la même architecture, seule la loss d'optimisation change. Le `summary()` Keras sert à vérifier cette architecture avant tout entraînement.

```
Model: "conv_autoencoder"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_image (InputLayer)        │ (None, 128, 128, 3)    │             0 │
│ enc_conv1 (Conv2D)              │ (None, 64, 64, 32)     │           896 │
│ enc_conv2 (Conv2D)              │ (None, 32, 32, 64)     │        18,496 │
│ enc_conv3 (Conv2D)              │ (None, 16, 16, 128)    │        73,856 │
│ enc_conv4 (Conv2D)              │ (None, 8, 8, 256)      │       295,168 │  ← bottleneck
│ dec_conv1 (Conv2DTranspose)     │ (None, 16, 16, 128)    │       295,040 │
│ dec_conv2 (Conv2DTranspose)     │ (None, 32, 32, 64)     │        73,792 │
│ dec_conv3 (Conv2DTranspose)     │ (None, 64, 64, 32)     │        18,464 │
│ dec_output (Conv2DTranspose)    │ (None, 128, 128, 3)    │           867 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 776,579 (2.96 MB)
```

**Comment le lire :**

- **Output Shape de la dernière ligne == Output Shape de la première** (`(None, 128, 128, 3)` des deux côtés) : condition nécessaire pour que le modèle puisse reconstruire une image de même taille que l'originale, et donc comparer pixel à pixel.
- **`enc_conv4` porte le `Output Shape` le plus petit** (`8, 8, 256`) : c'est le **bottleneck**, le goulot d'étranglement qui force le modèle à compresser l'information plutôt qu'à recopier l'image. Sans cette contrainte, le modèle pourrait apprendre l'identité et l'erreur de reconstruction ne serait plus informative pour détecter une anomalie.
  - Réduction spatiale : 128 → 8, soit **÷16** en hauteur et en largeur.
  - **Ratio de compression** : `(128×128×3) / (8×8×256) = 49 152 / 16 384 = 3.00x` — 3 fois moins de valeurs à la sortie de l'encodeur qu'à l'entrée.
- **Param #** se vérifie avec la formule `(kernel_h × kernel_w × canaux_entrée + 1) × canaux_sortie` : `enc_conv1` = `(3×3×3+1)×32 = 896` ✓, `enc_conv2` = `(3×3×32+1)×64 = 18 496` ✓.
- **Symétrie encodeur/décodeur** : le nombre de paramètres croît puis décroît de façon quasi symétrique (896 → 295 168 → 867) — signe d'une architecture cohérente, pas de déséquilibre suspect entre les deux moitiés.
- **776 579 paramètres entraînables**, à mettre en regard des 177 images d'entraînement disponibles : ratio paramètres/données volontairement surveillé (risque de sur-apprentissage), d'où l'usage de l'augmentation de données et de l'`EarlyStopping`.

---

## 2. Étape 19 — Rappel et faux positifs au seuil par défaut (p95)

Le seuil est calibré **uniquement sur les images saines de validation** (`val_ds`, jamais sur le test — pas de fuite), au 95ᵉ percentile des scores d'erreur de reconstruction.

| Algorithme | Seuil (p95) | Rappel (défauts détectés) | Faux positifs (saines flaguées) |
|---|---|---|---|
| **MSE**  | 0.00266 | 57.14% | **35.00 %** |
| **SSIM** | 0.00105 | 52.38% | **0.00 %** |

À ce seuil précis, MSE et SSIM ont un rappel comparable, mais SSIM ne produit aucun faux positif contre 35 % pour MSE — très loin des ~5 % attendus par construction d'un seuil p95. Ce chiffre est le premier signal d'alerte sur MSE, creusé à l'étape suivante.

---

## 3. Étape 19 — Vérification de la généralisation du seuil (`val` vs `test/good`)

Le seuil n'a de sens que si les scores des images saines de validation et de test suivent la **même distribution**. C'est le test décisif qui a fait pencher la balance vers SSIM :

![Val vs test/good](synthesis_val_vs_test.png)

Un seuil calibré sur `val` avec MSE produit un taux de FP très supérieur à l'objectif visé sur `test/good` — le seuil ne se généralise pas correctement. Avec SSIM, le seuil se généralise correctement (0 % de FP, cohérent avec l'objectif). Ce test suffit à lui seul à mettre en doute la fiabilité de MSE comme score de production.

---

## 4. Étape 21 — Métriques agrégées indépendantes du seuil (AUC-ROC, AUC-PR)

Pour ne pas dépendre d'un seul point de fonctionnement (le seuil p95), on évalue le score sur toute la plage de seuils possibles, à deux échelles :

| Algorithme | AUC-ROC image | AUC-PR image | AUC-ROC pixel | AUC-PR pixel |
|---|---|---|---|---|
| **MSE**  | 0.441 | 0.786 | 0.692 | 0.088 |
| **SSIM** | **0.744** | **0.922** | **0.779** | **0.211** |

![Courbes ROC et Precision-Recall](synthesis_roc_pr.png)

SSIM domine sur les **quatre** métriques simultanément, à l'image comme au pixel. L'AUC-PR pixel de MSE est proche de ce qu'obtiendrait un classifieur aléatoire compte tenu du déséquilibre pixel-level (cf. étude de déséquilibre, étape 3.11) ; celui de SSIM, bien que modeste en absolu, double ce niveau de référence.

---

## 5. Étape 21 — Rappel vs faux positifs en fonction du seuil (balayage complet)

Plutôt qu'un seul point (p95), on trace le compromis rappel/FP sur toute la plage de seuils :

![Rappel et faux positifs vs seuil](synthesis_recall_vs_threshold.png)

À n'importe quel seuil de ce balayage, on peut comparer les deux courbes point par point — c'est cette figure qui sert de base à l'ajustement du seuil final (section 7 ci-dessous, indice de Youden).

---

## 6. Étape 21 — Effet du score pixel-level : IoU (localisation du défaut)

Au-delà de "l'image est-elle défectueuse ?", peut-on **localiser** le défaut ? Seuil pixel calibré (p99 des pixels de `val_ds`), masque de segmentation prédit comparé au masque réel via l'IoU :

| Algorithme | Seuil pixel (p99) | IoU moyen (63 images défectueuses) |
|---|---|---|
| **MSE**  | 0.05519 | 0.047 |
| **SSIM** | 0.08403 | **0.138** |

![Segmentation prédite vs masque réel](synthesis_segmentation.png)

L'IoU de SSIM est quasiment **3 fois supérieur** à celui de MSE — SSIM ne détecte pas seulement mieux la présence d'un défaut, il le localise aussi mieux spatialement. Cohérent avec le fait que la loss SSIM est elle-même construite pour être sensible à la structure locale (luminance, contraste, texture), pas seulement à l'écart pixel brut que pénalise MSE.

---

## 7. Étape 22 — Décision finale et ajustement du seuil (indice de Youden)

**Modèle retenu : SSIM.** Les étapes précédentes convergent toutes dans le même sens (rappel/FP, généralisation, AUC-ROC/AUC-PR, IoU) — aucune ne favorise MSE.

Plutôt que de garder le p95 par défaut (arbitraire), le seuil final est celui qui **maximise l'indice de Youden** (`rappel − faux positifs`) sur le balayage de la section 5 :

```
Percentile retenu : p60  (indice de Youden = 0.521, rappel = 57.14%, FP = 5.00%)
Pour comparaison, p95 par défaut : rappel = 19.05%, FP = 0.00%
```

En acceptant de passer de 0 % à 5 % de faux positifs (toujours un niveau raisonnable), le rappel augmente substantiellement — un gain net pour un coût de FP resté modéré. C'est ce compromis, justifié par les données plutôt que choisi arbitrairement, qui est retenu.

### IoU par image, seuil ajusté

Avec ce nouveau seuil pixel (calibré au même percentile ajusté), l'IoU par classe de défaut :

| Classe | IoU moyen | n images |
|---|---|---|
| `broken_large` | 0.221 | 20 |
| `broken_small` | **0.064** | 22 |
| `contamination` | 0.206 | 21 |
| **Global** | **0.161** (± 0.119) | 63 |

![IoU par image défectueuse](synthesis_iou_per_image.png)

`broken_small` a l'IoU le plus faible — cohérent avec l'étude de déséquilibre pixel-level (étape 3.11), où cette classe a la plus petite fraction de pixels défectueux : une zone de défaut plus petite est mécaniquement plus sensible à un léger décalage du masque prédit.

### Matrice de confusion (seuil calibré)

![Matrice de confusion](synthesis_confusion_matrix.png)

---

## 8. Résumé décisionnel

| Critère | MSE | SSIM | Gagnant |
|---|---|---|---|
| Rappel @ p95 | 57.1 % | 52.4 % | ≈ égal |
| FP @ p95 | **35.0 %** | 0.0 % | **SSIM** |
| Généralisation val→test | Seuil non fiable | Seuil fiable | **SSIM** |
| AUC-ROC image | 0.441 | **0.744** | **SSIM** |
| AUC-PR image | 0.786 | **0.922** | **SSIM** |
| AUC-ROC pixel | 0.692 | **0.779** | **SSIM** |
| AUC-PR pixel | 0.088 | **0.211** | **SSIM** |
| IoU moyen | 0.047 | **0.138** | **SSIM** |
| Rappel @ seuil ajusté (Youden) | — | 57.1 % (FP 5 %) | **SSIM** |

Sur la quasi-totalité des critères comparés, SSIM domine — le seul point de parité (rappel brut au p95) s'accompagne d'un taux de faux positifs bien supérieur pour MSE, ce qui invalide la comparaison à seuil égal. **Le choix de SSIM comme perte d'entraînement est donc soutenu par l'ensemble des métriques, pas par un seul chiffre isolé.**

> ⚠️ **Mise à jour ultérieure** : des réexécutions du même notebook (avec `VAL_FRACTION` différent, puis rejoué à l'identique) ont montré que ces résultats ne sont **pas stables d'une exécution à l'autre** — voir le fichier de synthèse le plus récent pour l'analyse de cette variance et sa cause probable (absence de `tf.random.set_seed`). Ce document reste la trace du tout premier run, mais ne doit plus être lu comme une conclusion définitive isolément.

*Notebook de référence : `pipeline_bottle_full.ipynb`, sections 7-8 (architecture), 19 (seuil et généralisation), 21 (AUC-ROC/PR, IoU), 22 (décision finale et ajustement).*
