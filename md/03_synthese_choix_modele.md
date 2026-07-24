# Synthèse — Choix du modèle d'autoencodeur (MSE vs SSIM)

**Mise à jour : `VAL_FRACTION` rejoué à sa valeur initiale 0.15** (177 images train / 32 validation), pour vérifier si les résultats à 0.25 (run précédent) tenaient au split ou à autre chose.

> 🔴 **Résultat clé de cette mise à jour : même avec `VAL_FRACTION=0.15` identique au tout premier run, les chiffres changent encore d'une exécution à l'autre.** Ce n'est donc pas (uniquement) `VAL_FRACTION` qui explique l'instabilité observée précédemment — la section 8 identifie la cause probable et une correction concrète.

Trois runs sont maintenant disponibles et comparés :
- **Run A** (premier run, 0.15) : SSIM dominait largement.
- **Run B** (0.25) : MSE dominait sur presque tous les critères.
- **Run C** (ce run, 0.15 rejoué à l'identique) : résultat mixte, ni Run A ni Run B.

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
| **MSE**  | 0.00426 | **47.6 %** | 40.0 % |
| **SSIM** | 0.00691 | 41.3 % | **0.0 %** |

Cette fois, MSE et SSIM détectent un rappel comparable (47.6 % vs 41.3 %), mais SSIM reste sans faux positif contre 40 % pour MSE — cohérent avec le run A, différent du run B (où SSIM avait 5 % de FP).

---

## 3. Étape 19 — Vérification de la généralisation du seuil (`val` vs `test/good`)

| Algorithme | mean(val) | mean(test/good) | Écart |
|---|---|---|---|
| **MSE**  | 0.00380 | 0.00410 | +7.8 % |
| **SSIM** | 0.00610 | 0.00616 | +1.0 % |

![Val vs test/good](figures/synthesis3_val_vs_test.png)

SSIM généralise mieux que MSE (écart +1.0 % contre +7.8 %) — comme sur le run A, à un degré moindre.

---

## 4. Étape 21 — Métriques agrégées indépendantes du seuil (AUC-ROC, AUC-PR)

| Algorithme | AUC-ROC image | AUC-PR image | AUC-ROC pixel | AUC-PR pixel |
|---|---|---|---|---|
| **MSE**  | 0.570 | 0.840 | **0.645** | **0.101** |
| **SSIM** | **0.766** | **0.924** | 0.485 | 0.071 |

![Courbes ROC et Precision-Recall](figures/synthesis3_roc_pr.png)

**Résultat mixte, différent des deux runs précédents** : SSIM domine au niveau image (comme le run A), mais MSE domine au niveau pixel (comme le run B), et l'AUC-ROC pixel de SSIM (0.485) est proche du hasard.

---

## 5. Étape 21 — Effet du score pixel-level : IoU

| Algorithme | Seuil pixel (p99) | IoU moyen (63 images défectueuses) |
|---|---|---|
| **MSE**  | 0.03867 | **0.053** |
| **SSIM** | 0.11268 | 0.018 |

MSE localise mieux le défaut que SSIM sur ce run (comme le run B), malgré une meilleure détection image-level de SSIM (section 4). Ces deux mesures ne racontent donc pas la même histoire, sur ce run comme sur les précédents.

---

## 6. Étape 22 — Ajustement du seuil (indice de Youden), sur le modèle SSIM

```
Percentile retenu : p86  (indice de Youden = 0.456, rappel = 55.56%, FP = 10.00%)
Pour comparaison, p95 par défaut : rappel = 41.27%, FP = 0.00%
```

IoU par image, seuil ajusté (p86) :

| Classe | IoU moyen | n images |
|---|---|---|
| `broken_large` | 0.079 | 20 |
| `broken_small` | 0.066 | 22 |
| `contamination` | 0.083 | 21 |
| **Global** | **0.076** (± 0.071) | 63 |

![IoU par image défectueuse](figures/synthesis3_iou_per_image.png)

---

## 7. Comparaison des trois runs — lecture honnête

| Critère | Run A (0.15, 1ᵉʳ run) | Run B (0.25) | Run C (0.15, rejoué) |
|---|---|---|---|
| Split train/val | 177 / 32 | 156 / 53 | 177 / 32 (identique à A) |
| FP @ p95 (MSE / SSIM) | 30 % / 0 % | 40 % / 5 % | 40 % / 0 % |
| AUC-ROC image (MSE / SSIM) | 0.551 / **0.744** | **0.556** / 0.544 | 0.570 / **0.766** |
| AUC-ROC pixel (MSE / SSIM) | 0.639 / **0.779** | **0.648** / 0.333 | **0.645** / 0.485 |
| IoU moyen (MSE / SSIM) | 0.047 / **0.138** | **0.047** / 0.010 | **0.053** / 0.018 |
| Youden (SSIM) | 0.521 | 0.160 | 0.456 |

**Point le plus important de cette mise à jour** : les runs A et C utilisent **exactement le même `VAL_FRACTION=0.15`**, et pourtant leurs résultats diffèrent (A : SSIM gagne partout ; C : SSIM gagne au niveau image mais perd au niveau pixel). `VAL_FRACTION` n'est donc **pas la seule source de variance** — un autre facteur, non contrôlé, change d'une exécution à l'autre même à split identique.

### Cause probable identifiée

```
grep "tf.random\|set_seed" pipeline_bottle_full.ipynb
→ seul np.random.seed(SEED) est présent, aucun tf.random.set_seed(SEED)
```

Le notebook fixe `SEED` pour `numpy` (utilisé par `train_test_split` et le `shuffle` de `tf.data`), **mais jamais pour TensorFlow/Keras**. Or l'initialisation des poids de `build_autoencoder()` (Glorot uniform par défaut) dépend du générateur aléatoire de TensorFlow, pas de celui de NumPy. **Chaque exécution repart donc d'une initialisation de poids différente**, même à split de données identique — c'est très probablement la source dominante de variance observée, plus que `VAL_FRACTION` lui-même.

### Recommandation concrète

Ajouter, juste après la fixation de `SEED` en configuration :

```python
tf.random.set_seed(SEED)
```

Et idéalement, pour une reproductibilité complète sous CPU :

```python
os.environ["PYTHONHASHSEED"] = str(SEED)
import random; random.seed(SEED)
```

Sans cette correction, comparer MSE et SSIM sur un seul run (quel que soit `VAL_FRACTION`) revient à comparer deux tirages aléatoires différents plutôt que deux algorithmes — d'où l'absence de conclusion stable sur les trois runs déjà réalisés.

### Ce que ça signifie pour la décision

1. **Aucun des trois runs ne peut être retenu comme "la" comparaison définitive.** Chacun décrit une combinaison (split, initialisation aléatoire, 8 epochs) particulière, pas une propriété robuste de MSE ou SSIM sur ce dataset.
2. Un point cependant traverse les trois runs : **SSIM a systématiquement un taux de faux positifs inférieur ou égal à MSE au seuil p95** (0 %, 5 %, 0 % contre 30-40 % pour MSE) — c'est le seul signal reproductible dans les trois exécutions.
3. **Avant de choisir définitivement**, il faut : (a) fixer `tf.random.set_seed(SEED)` pour éliminer cette source de variance, (b) entraîner avec un budget d'epochs plus réaliste (30+), (c) répéter la comparaison sur plusieurs seeds (au moins 3-5) et comparer moyenne ± écart-type plutôt qu'un run isolé.
4. **Le mécanisme du notebook reste valide** (comparaison, ajustement Youden, IoU par image, MLflow) — c'est la robustesse statistique de la conclusion tirée d'un run isolé à 8 epochs qui ne l'est pas.

**Décision révisée** : ne pas figer de choix final entre MSE et SSIM tant que le seed TensorFlow n'est pas fixé et qu'une moyenne sur plusieurs runs n'a pas été établie. Le seul constat robuste à ce stade est la maîtrise des faux positifs par SSIM, observée de façon cohérente sur les trois runs.

*Notebook de référence : `pipeline_bottle_full.ipynb`. Runs comparés : A (`VAL_FRACTION=0.15`, run initial), B (`VAL_FRACTION=0.25`), C (`VAL_FRACTION=0.15`, rejoué). `EPOCHS=8` pour les trois.*
