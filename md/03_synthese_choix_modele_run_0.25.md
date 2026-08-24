# Synthèse — Choix du modèle d'autoencodeur (MSE vs SSIM)

**Mise à jour : run réexécuté avec `VAL_FRACTION = 0.25`** (156 images train / 53 validation, contre 177/32 pour la précédente version à 0.15). Mêmes hyperparamètres sinon (8 epochs, `EarlyStopping` patience=5).

> ⚠️ **Ce nouveau run inverse une partie des conclusions précédentes.** Les deux runs sont présentés côte à côte ci-dessous, avec une lecture honnête de ce que cette instabilité signifie (section 8).

---

## 1. Point de détail important : l'architecture et son `summary()`

Inchangé par rapport à la version précédente — `VAL_FRACTION` ne change pas l'architecture, seulement le split des données. Rappel :

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

- **Bottleneck `enc_conv4`** : `(8, 8, 256)`, ratio de compression `49 152 / 16 384 = 3.00x`, réduction spatiale ÷16.
- **776 579 paramètres entraînables** — avec `VAL_FRACTION=0.25`, seulement **156 images d'entraînement** (contre 177 à 0.15) : le ratio paramètres/données se tend encore un peu plus, à surveiller.

---

## 2. Étape 19 — Rappel et faux positifs au seuil par défaut (p95)

| Algorithme | Seuil (p95) | Rappel | Faux positifs |
|---|---|---|---|
| **MSE**  | 0.00746 | **47.6 %** | 40.0 % |
| **SSIM** | 0.01519 | 11.1 % | **5.0 %** |

Comme sur le run précédent, MSE détecte plus de défauts au prix de plus de faux positifs. Le taux de FP de MSE (40 %) reste très supérieur à la cible de ~5 %.

---

## 3. Étape 19 — Vérification de la généralisation du seuil (`val` vs `test/good`)

| Algorithme | mean(val) | mean(test/good) | Écart |
|---|---|---|---|
| **MSE**  | 0.00708 | 0.00736 | +3.8 % |
| **SSIM** | 0.01374 | 0.01394 | +1.4 % |

![Val vs test/good](figures/synthesis2_val_vs_test.png)

Ici, **les deux algorithmes généralisent correctement** cette fois (écarts faibles, du même ordre de grandeur) — contrairement au run à 0.15 où SSIM se distinguait nettement de MSE sur ce critère. Ce test ne permet donc plus, à lui seul, de départager les deux algorithmes sur ce run.

---

## 4. Étape 21 — Métriques agrégées indépendantes du seuil (AUC-ROC, AUC-PR)

| Algorithme | AUC-ROC image | AUC-PR image | AUC-ROC pixel | AUC-PR pixel |
|---|---|---|---|---|
| **MSE**  | **0.556** | **0.829** | **0.648** | **0.098** |
| **SSIM** | 0.544 | 0.808 | 0.333 | 0.042 |

![Courbes ROC et Precision-Recall](figures/synthesis2_roc_pr.png)

**Retournement complet** par rapport au run précédent : MSE devance désormais SSIM sur les quatre métriques, et l'AUC-ROC pixel de SSIM (0.333) est même **inférieur au niveau du hasard** (0.5) — signe que sur ce run, le score pixel de SSIM anti-corrèle localement avec le masque réel plutôt que de le suivre.

---

## 5. Étape 21 — Rappel vs faux positifs en fonction du seuil

![Rappel et faux positifs vs seuil](figures/synthesis2_recall_vs_threshold.png)

---

## 6. Étape 21 — Effet du score pixel-level : IoU

| Algorithme | Seuil pixel (p99) | IoU moyen (63 images défectueuses) |
|---|---|---|
| **MSE**  | 0.05738 | **0.047** |
| **SSIM** | 0.13017 | 0.010 |

![Segmentation prédite vs masque réel](figures/synthesis2_segmentation.png)

Là aussi, l'écart s'inverse : MSE localise près de 5x mieux le défaut que SSIM sur ce run (contre l'inverse sur le run à 0.15).

---

## 7. Étape 22 — Ajustement du seuil (indice de Youden), sur le modèle SSIM

Le notebook reste configuré pour retenir SSIM par défaut (choix figé en amont) ; l'ajustement de percentile est donc recalculé pour SSIM, mais **à la lumière des chiffres ci-dessus, ce choix de modèle n'est plus soutenu par ce run** — voir section 8.

```
Percentile retenu : p70  (indice de Youden = 0.160, rappel = 46.03%, FP = 30.00%)
Pour comparaison, p95 par défaut : rappel = 11.11%, FP = 5.00%
```

L'indice de Youden au point optimal (0.160) est nettement plus faible que sur le run précédent (0.521) — signe d'un compromis rappel/FP globalement moins favorable pour SSIM sur ce run.

### IoU par image, seuil ajusté (p70)

| Classe | IoU moyen | n images |
|---|---|---|
| `broken_large` | 0.036 | 20 |
| `broken_small` | 0.025 | 22 |
| `contamination` | 0.031 | 21 |
| **Global** | **0.030** (± 0.026) | 63 |

![IoU par image défectueuse](figures/synthesis2_iou_per_image.png)

IoU global divisé par ~5 par rapport au run précédent (0.161 → 0.030) pour ce même modèle SSIM — confirme que la performance de segmentation de SSIM sur ce run est nettement dégradée.

### Matrice de confusion

![Matrice de confusion](figures/synthesis2_confusion_matrix.png)

---

## 8. Comparaison des deux runs — lecture honnête

| Critère | Run `VAL_FRACTION=0.15` | Run `VAL_FRACTION=0.25` |
|---|---|---|
| Split train/val | 177 / 32 | 156 / 53 |
| FP @ p95 (MSE) | 30.0 % | 40.0 % |
| FP @ p95 (SSIM) | 0.0 % | 5.0 % |
| Généralisation val→test | SSIM nettement meilleur | Les deux comparables |
| AUC-ROC image | SSIM (0.744) > MSE (0.551) | **MSE (0.556) > SSIM (0.544)** |
| AUC-PR image | SSIM (0.922) > MSE (0.836) | **MSE (0.829) > SSIM (0.808)** |
| AUC-ROC pixel | SSIM (0.779) > MSE (0.639) | **MSE (0.648) > SSIM (0.333)** |
| IoU moyen | SSIM (0.138) > MSE (0.047) | **MSE (0.047) > SSIM (0.010)** |
| Youden (SSIM) | 0.521 | 0.160 |

**Ce que ça signifie :**

1. **Le résultat n'est pas stable d'un split à l'autre**, à 8 epochs. Avec seulement 8 epochs, les deux modèles sont probablement encore loin de leur convergence — l'ordre de mérite entre MSE et SSIM peut dépendre autant du split validation que du choix de la loss elle-même. **La conclusion précédente ("SSIM domine sur 8/9 critères") ne doit plus être considérée comme établie** — elle décrivait un run particulier, pas une propriété robuste de la loss SSIM sur ce dataset.
2. Ce constat corrobore une remarque déjà faite plus tôt dans ce travail : *le rappel/FP à un seuil donné dépend de la taille du split validation* — c'est exactement ce qu'on observe ici, mais à une échelle plus large que prévu (inversion de classement, pas seulement un décalage de valeurs).
3. **Recommandation** : avant de trancher définitivement entre MSE et SSIM, il faut soit (a) entraîner avec un budget d'epochs nettement supérieur (30+ avec `EarlyStopping`, pas 8) pour réduire la variance liée à un entraînement incomplet, soit (b) répéter la comparaison sur plusieurs splits (k-fold ou plusieurs seeds) et comparer les moyennes ± écart-type plutôt qu'un seul run.
4. **Le mécanisme du notebook (comparaison, ajustement Youden, IoU par image) reste valide et correctement outillé** — c'est la conclusion tirée d'un seul run à 8 epochs qui n'est pas fiable, pas la méthode.

**Décision révisée à ce stade** : ne pas figer le choix de SSIM comme définitif tant qu'une comparaison plus robuste (plus d'epochs et/ou plusieurs splits) n'a pas été menée. Le notebook et la section 22 restent utiles pour documenter *comment* comparer et choisir, mais le résultat numérique d'un run isolé à 8 epochs ne doit pas être interprété comme la réponse finale.

*Notebook de référence : `pipeline_bottle_full.ipynb`, run avec `VAL_FRACTION=0.25`, `EPOCHS=8`.*
