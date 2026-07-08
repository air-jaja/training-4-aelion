# Préparation des features — Modèle de maintenance prédictive (variante : CV calendaire)

**Contexte technique** : Python, pandas, Jupyter, pyarrow, scikit-learn. Installation des packages via `uv`.

**Fichier source** : `artifacts/ingestions/datas/gold_dataset.parquet` (134 280 lignes × 100 colonnes)

---

## 0) Installation de l'environnement

```bash
uv add ipykernel pandas jupyter
uv add matplotlib seaborn
uv add pyarrow scikit-learn 
uv add xgboost
```

---

## 1) Charger le parquet, trié par machine puis par temps

```python
import pandas as pd

df = pd.read_parquet("artifacts/ingestions/datas/gold_dataset.parquet")

df = df.sort_values(["machine_id_std", "window_start"]).reset_index(drop=True)
```

- `machine_id_std` : identifiant machine (catégorie)
- `window_start` : horodatage de la fenêtre horaire (clé temporelle)

Le tri garantit que toute opération ultérieure basée sur l'ordre (lag, vérification de fenêtres glissantes, etc.) reste cohérente par machine.

---

## 2) Exclure les colonnes de fuite, les identifiants et les labels des features

Le dataset contient trois familles de colonnes à ne **jamais** mettre dans `X` :

| Catégorie | Colonnes | Raison |
|---|---|---|
| **Identifiants / temporel** | `machine_id_std`, `window_start`, `window_end` | Ne sont pas des features prédictives (l'identifiant machine peut être conservé séparément pour du group-by, mais pas encodé brut comme feature) |
| **Fuite de données (leakage)** | `future_incident_count_6h`, `future_incident_count_12h`, `future_incident_count_24h`, `future_incident_count_48h` | Ces colonnes comptent des incidents **futurs** — elles ont servi à construire les labels et donneraient au modèle une information qu'il n'aura jamais en production |
| **Labels** | `label_failure_next_6h`, `label_failure_next_12h`, `label_failure_next_24h`, `label_failure_next_48h` | Ce sont les cibles à prédire, pas des features |
| **Split** | `split_set` | Colonne de contrôle du split, pas une feature |

```python
id_cols = ["machine_id_std", "window_start", "window_end"]

leakage_cols = [
    "future_incident_count_6h",
    "future_incident_count_12h",
    "future_incident_count_24h",
    "future_incident_count_48h",
]

label_cols = [
    "label_failure_next_6h",
    "label_failure_next_12h",
    "label_failure_next_24h",
    "label_failure_next_48h",
]

control_cols = ["split_set"]

exclude_cols = id_cols + leakage_cols + label_cols + control_cols

feature_cols = [c for c in df.columns if c not in exclude_cols]

X_all = df[feature_cols]
```

⚠️ Vérifier qu'aucune autre colonne dérivée des incidents futurs ne s'est glissée dans les features (toute colonne contenant `future_` doit être exclue).

---

## 3) Choisir un horizon et construire la cible y (0/1)

Quatre horizons sont disponibles : `label_failure_next_6h`, `label_failure_next_12h`, `label_failure_next_24h`, `label_failure_next_48h`.

Taux de positifs observés (déséquilibre croissant avec l'horizon) :

| Horizon | % positifs |
|---|---|
| 6h | ~4.5 % |
| 12h | ~8.8 % |
| 24h | ~16.8 % |
| 48h | ~25.5 % |

```python
horizon = "label_failure_next_24h"  # à changer selon l'horizon étudié

y = df[horizon].astype(int)  # bool -> 0/1
```

Pour comparer plusieurs horizons, répéter l'entraînement en bouclant sur la liste `label_cols`, avec le **même** `X` et le **même** split.

⚠️ Le déséquilibre de classes (surtout à 6h) justifie de surveiller autre chose que l'accuracy (ex. : PR-AUC, F1, recall) lors de l'évaluation du modèle.

---

## 4) Utiliser le split temporel fourni (pas de split aléatoire)

Le dataset contient déjà une colonne `split_set` avec un découpage chronologique respectant `train < validation < test`.

```python
print(df["split_set"].value_counts())
# train        93990
# validation   20145
# test         20145

train_mask = df["split_set"] == "train"
val_mask   = df["split_set"] == "validation"
test_mask  = df["split_set"] == "test"

X_train, y_train = X_all[train_mask], y[train_mask]
X_val,   y_val   = X_all[val_mask],   y[val_mask]
X_test,  y_test  = X_all[test_mask],  y[test_mask]
```

⚠️ Ne jamais utiliser `train_test_split` avec mélange aléatoire (`shuffle=True`) : cela romprait la logique temporelle et provoquerait une fuite d'information (le modèle verrait des données futures pendant l'entraînement).

Vérification recommandée : confirmer que les plages de `window_start` de chaque split ne se chevauchent pas.

```python
for name, mask in [("train", train_mask), ("validation", val_mask), ("test", test_mask)]:
    print(name, df.loc[mask, "window_start"].min(), "->", df.loc[mask, "window_start"].max())
```

---

## 5) Imputer les NaN (médiane) et standardiser pour les modèles linéaires

Plusieurs colonnes de features contiennent des NaN (agrégats glissants en début de série, z-scores, colonnes liées aux incidents/maintenance jamais survenus, etc.).

**Règle impérative** : fitter l'imputer et le scaler **uniquement sur le train**, puis appliquer (`transform`) sur validation et test — jamais de fit sur validation/test, pour éviter toute fuite d'information.

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Imputation (médiane) — pour tous les modèles
imputer = SimpleImputer(strategy="median")
imputer.fit(X_train)

X_train_imp = imputer.transform(X_train)
X_val_imp   = imputer.transform(X_val)
X_test_imp  = imputer.transform(X_test)

# Standardisation — nécessaire seulement pour les modèles linéaires
# (régression logistique, SVM linéaire, etc.), pas pour les modèles à arbres
scaler = StandardScaler()
scaler.fit(X_train_imp)

X_train_scaled = scaler.transform(X_train_imp)
X_val_scaled   = scaler.transform(X_val_imp)
X_test_scaled  = scaler.transform(X_test_imp)
```

Alternative propre avec `Pipeline` (évite les fuites par construction) :

```python
linear_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

X_train_scaled = linear_pipeline.fit_transform(X_train)
X_val_scaled   = linear_pipeline.transform(X_val)
X_test_scaled  = linear_pipeline.transform(X_test)
```

Pour un modèle à base d'arbres (RandomForest, LightGBM, XGBoost), la standardisation n'est pas nécessaire — seule l'imputation (ou la gestion native des NaN par le modèle) est requise.

---

## 6) Vérifier le taux de panne (rare) → ne pas utiliser l'accuracy

Les labels sont fortement déséquilibrés (voir étape 3 : de ~4.5 % à ~25.5 % de positifs selon l'horizon). Avec un tel déséquilibre, un modèle qui prédit toujours "pas de panne" obtient déjà une accuracy élevée sans aucune valeur prédictive.

```python
print(f"Taux de panne (train) : {y_train.mean():.2%}")
print(f"Taux de panne (val)   : {y_val.mean():.2%}")
print(f"Taux de panne (test)  : {y_test.mean():.2%}")
```

⚠️ Ne pas piloter l'évaluation du modèle avec l'accuracy. Utiliser à la place :
- **PR-AUC** (aire sous la courbe précision-rappel) — la plus informative sur classe rare
- **Recall** — capacité à détecter les vraies pannes (souvent le critère métier prioritaire)
- **F1-score** — compromis précision/rappel
- **ROC-AUC** — utile mais moins discriminant que PR-AUC quand la classe positive est rare

```python
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score

# à calculer sur les prédictions du modèle (y_pred / y_proba), une fois entraîné
# average_precision_score(y_test, y_proba)
# recall_score(y_test, y_pred)
# f1_score(y_test, y_pred)
# roc_auc_score(y_test, y_proba)
```

---

## 7) Compenser le déséquilibre des classes

Deux approches selon la bibliothèque utilisée, toutes deux calculées/appliquées **sur le train uniquement**.

### scikit-learn : `class_weight="balanced"`

Rééquilibre automatiquement le poids des classes en fonction de leur fréquence dans `y_train` (pas besoin de calculer le ratio à la main).

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

log_reg = LogisticRegression(class_weight="balanced", max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(class_weight="balanced", random_state=42)
rf.fit(X_train_imp, y_train)
```

### XGBoost : `scale_pos_weight`

XGBoost n'a pas de `class_weight` : le ratio doit être calculé manuellement sur le train, comme `nb_négatifs / nb_positifs`.

```python
from xgboost import XGBClassifier

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

print(f"scale_pos_weight = {scale_pos_weight:.2f}")

xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
xgb.fit(X_train_imp, y_train)
```

⚠️ `scale_pos_weight` doit être recalculé à chaque changement d'horizon (le taux de positifs varie de ~4.5 % à ~25.5 % selon `label_failure_next_*`), et toujours à partir de `y_train` uniquement — jamais de `y_val`/`y_test`.

---

## 8) Prédire sur le jeu de test et afficher les informations pertinentes

Pour chaque modèle entraîné (`log_reg`, `rf`, `xgb`), prédire sur `X_test` (ou `X_test_scaled` pour le modèle linéaire) et centraliser les métriques adaptées au déséquilibre (voir étape 6) plutôt que l'accuracy.

### 8.1 Prédictions (classe + probabilité)

```python
# Régression logistique (features standardisées)
y_pred_log  = log_reg.predict(X_test_scaled)
y_proba_log = log_reg.predict_proba(X_test_scaled)[:, 1]

# Random Forest (features imputées, pas de standardisation nécessaire)
y_pred_rf  = rf.predict(X_test_imp)
y_proba_rf = rf.predict_proba(X_test_imp)[:, 1]

# XGBoost (features imputées)
y_pred_xgb  = xgb.predict(X_test_imp)
y_proba_xgb = xgb.predict_proba(X_test_imp)[:, 1]
```

### 8.2 Tableau récapitulatif des métriques (PR-AUC, ROC-AUC, F1, recall, precision)

```python
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, recall_score, precision_score,
)
import pandas as pd

models_preds = {
    "LogisticRegression":  (y_pred_log, y_proba_log),
    "RandomForest":        (y_pred_rf,  y_proba_rf),
    "XGBoost":             (y_pred_xgb, y_proba_xgb),
}

rows = []
for name, (y_pred, y_proba) in models_preds.items():
    rows.append({
        "modele":     name,
        "pr_auc":     average_precision_score(y_test, y_proba),
        "roc_auc":    roc_auc_score(y_test, y_proba),
        "precision":  precision_score(y_test, y_pred),
        "recall":     recall_score(y_test, y_pred),
        "f1":         f1_score(y_test, y_pred),
    })

results_df = pd.DataFrame(rows).set_index("modele").round(3)
results_df
```

### 8.3 Matrice de confusion par modèle

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, (y_pred, _)) in zip(axes, models_preds.items()):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["pas de panne", "panne"]).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.show()

# Somme des éléments de chaque matrice de confusion (doit être égale à len(y_test))
for name, (y_pred, _) in models_preds.items():
    cm = confusion_matrix(y_test, y_pred)
    print(f"{name}: somme de la matrice de confusion = {cm.sum()} (taille de y_test = {len(y_test)})")
```

### 8.4 Courbe précision-rappel (comparaison des modèles)

Plus informative que la ROC quand la classe positive est rare.

```python
from sklearn.metrics import PrecisionRecallDisplay

fig, ax = plt.subplots(figsize=(6, 5))
for name, (_, y_proba) in models_preds.items():
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, name=name, ax=ax)
ax.set_title(f"Précision-Rappel — {HORIZON}")
plt.show()
```

### 8.5 Rapport de classification détaillé

```python
from sklearn.metrics import classification_report

for name, (y_pred, _) in models_preds.items():
    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["pas de panne", "panne"]))
```

### 8.6 Importance des features (modèles à arbres)

Utile pour interpréter ce qui pilote la prédiction (Random Forest / XGBoost uniquement).

```python
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
importances.head(15).plot.barh(figsize=(6, 6), title="Top 15 features — Random Forest")
plt.gca().invert_yaxis()
plt.show()
```

⚠️ Toutes les métriques et graphiques de cette étape doivent être calculés sur `X_test` / `y_test`, jamais sur le train (qui a servi à l'entraînement) ni sur la validation (réservée au tuning des hyperparamètres/du seuil).

---

## 9) Lister et choisir les métriques associées aux données

Avant de figer les métriques utilisées aux étapes 6 et 8, il est utile de lister systématiquement les métriques de classification disponibles dans scikit-learn et de choisir celles pertinentes en fonction des caractéristiques des données (classes déséquilibrées, coût métier d'un faux négatif vs faux positif).

### 9.1 Lister les métriques de classification disponibles dans scikit-learn

```python
import sklearn.metrics

# Liste des scorers nommés disponibles (utilisables via scoring="..." dans cross_val_score, GridSearchCV, etc.)
sorted(sklearn.metrics.get_scorer_names())
```

### 9.2 Grille de choix selon les caractéristiques des données

| Caractéristique des données | Métrique(s) recommandée(s) | À éviter |
|---|---|---|
| Classes déséquilibrées (ici : 4.5 % à 25.5 % de positifs) | PR-AUC (`average_precision`), F1, recall | Accuracy |
| Faux négatif coûteux (panne manquée) | Recall, F2-score (pondère le recall) | Precision seule |
| Faux positif coûteux (fausse alerte, intervention inutile) | Precision, F0.5-score | Recall seul |
| Besoin d'un score continu pour prioriser (ranking des machines à risque) | PR-AUC, ROC-AUC | Métriques à seuil fixe (F1, recall à 0.5) |
| Comparaison de plusieurs modèles/horizons sur un même graphique | Courbe précision-rappel, ROC | — |
| Communication à un public non technique | Matrice de confusion, rapport de classification | AUC brute (moins intuitive) |

### 9.3 Sélectionner et calculer les métriques choisies pour ce projet

Sur la base de la grille ci-dessus (classes rares + coût métier orienté détection des pannes) :

```python
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, fbeta_score,
)

selected_metrics = {
    "pr_auc":    lambda y_true, y_pred, y_proba: average_precision_score(y_true, y_proba),
    "roc_auc":   lambda y_true, y_pred, y_proba: roc_auc_score(y_true, y_proba),
    "precision": lambda y_true, y_pred, y_proba: precision_score(y_true, y_pred),
    "recall":    lambda y_true, y_pred, y_proba: recall_score(y_true, y_pred),
    "f1":        lambda y_true, y_pred, y_proba: f1_score(y_true, y_pred),
    "f2":        lambda y_true, y_pred, y_proba: fbeta_score(y_true, y_pred, beta=2),  # priorise le recall
}

rows = []
for name, (y_pred, y_proba) in models_preds.items():
    row = {"modele": name}
    for metric_name, metric_fn in selected_metrics.items():
        row[metric_name] = metric_fn(y_test, y_pred, y_proba)
    rows.append(row)

pd.DataFrame(rows).set_index("modele").round(3)
```

⚠️ Le F2-score (recall pondéré 2x plus que la precision) est pertinent ici si le coût métier d'une panne manquée est plus élevé que celui d'une fausse alerte — à ajuster (`beta`) selon le contexte métier réel.

---

---

## 10) Validation croisée temporelle par découpage calendaire (indépendant de l'ordre des lignes)

Contrairement à `TimeSeriesSplit` (qui s'appuie sur la **position** des lignes dans le tableau), cette approche découpe l'intervalle de dates du train en segments **calendaires** fixes, via des comparaisons directes sur `window_start`. Chaque fold est défini par une date de coupure — train/validation restent strictement séparés dans le temps, quel que soit l'ordre des machines dans le DataFrame.

### 10.1 Définir les bornes calendaires

```python
import pandas as pd

N_SPLITS = 5
n_chunks = N_SPLITS + 1  # nombre de segments calendaires nécessaires pour N_SPLITS folds expansifs

train_dates = df.loc[train_mask, "window_start"]
date_min, date_max = train_dates.min(), train_dates.max()

# n_chunks+1 bornes -> n_chunks segments calendaires égaux
boundaries = pd.date_range(date_min, date_max, periods=n_chunks + 1)
print(boundaries)
```

### 10.2 Construire et évaluer les folds (fenêtre expansive, imputer refitté par fold)

Refitter l'imputer à l'intérieur de chaque fold (sur son train uniquement) évite toute fuite d'information entre folds — plus rigoureux que réutiliser l'imputer global du train.

```python
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, f1_score

cv_rows = []
for fold in range(N_SPLITS):
    cutoff_train_end = boundaries[fold + 1]
    cutoff_val_end   = boundaries[fold + 2]

    fold_train_mask = train_mask & (df["window_start"] < cutoff_train_end)
    fold_val_mask   = train_mask & (df["window_start"] >= cutoff_train_end) & (df["window_start"] < cutoff_val_end)

    X_tr_raw, y_tr = X_all[fold_train_mask], y[fold_train_mask]
    X_va_raw, y_va = X_all[fold_val_mask],   y[fold_val_mask]

    fold_imputer = SimpleImputer(strategy="median")
    X_tr = fold_imputer.fit_transform(X_tr_raw)
    X_va = fold_imputer.transform(X_va_raw)

    model = clone(rf)
    model.fit(X_tr, y_tr)

    y_va_pred  = model.predict(X_va)
    y_va_proba = model.predict_proba(X_va)[:, 1]

    cv_rows.append({
        "fold": fold + 1,
        "train_end": cutoff_train_end,
        "val_end": cutoff_val_end,
        "n_train": int(fold_train_mask.sum()),
        "n_val": int(fold_val_mask.sum()),
        "n_machines_train": df.loc[fold_train_mask, "machine_id_std"].nunique(),
        "n_machines_val": df.loc[fold_val_mask, "machine_id_std"].nunique(),
        "taux_panne_train": y_tr.mean(),
        "taux_panne_val": y_va.mean(),
        "pr_auc": average_precision_score(y_va, y_va_proba),
        "roc_auc": roc_auc_score(y_va, y_va_proba),
        "recall": recall_score(y_va, y_va_pred),
        "f1": f1_score(y_va, y_va_pred),
    })

cv_df = pd.DataFrame(cv_rows).set_index("fold")
cv_df.round(3)
```

### 10.3 Garde-fou : aucun fold vide, aucun chevauchement

```python
for fold in range(N_SPLITS):
    row = cv_df.loc[fold + 1]
    assert row["n_train"] > 0 and row["n_val"] > 0, f"Fold {fold + 1} vide — ajuster N_SPLITS ou les bornes calendaires"

print("Tous les folds sont non-vides ; train/validation strictement séparés par date de coupure.")
```

### 10.4 Résumé (moyenne ± écart-type)

```python
summary = cv_df[["pr_auc", "roc_auc", "recall", "f1"]].agg(["mean", "std"]).round(3)
summary
```

### 10.5 Couverture des machines par fold

Un découpage calendaire n'assure pas une répartition égale des machines entre folds (si certaines machines n'ont des données que sur une partie de la période). Vérifier cette couverture.

```python
cv_df[["n_train", "n_val", "n_machines_train", "n_machines_val"]]
```

### 10.6 Frise temporelle des folds

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
for fold in range(N_SPLITS):
    row = cv_df.loc[fold + 1]
    ax.plot([date_min, row["train_end"]], [fold, fold], color="tab:blue", linewidth=6, solid_capstyle="butt")
    ax.plot([row["train_end"], row["val_end"]], [fold, fold], color="tab:orange", linewidth=6, solid_capstyle="butt")
ax.set_yticks(range(N_SPLITS))
ax.set_yticklabels([f"Fold {i + 1}" for i in range(N_SPLITS)])
ax.set_xlabel("Date calendaire")
ax.set_title("Découpage calendaire des folds (bleu = train, orange = validation)")
plt.tight_layout()
plt.show()
```

### 10.7 Évolution des métriques dans le temps

Axe des abscisses en dates réelles (et non en numéro de fold) pour repérer une dérive calendaire (ex. : saisonnalité, dégradation progressive).

```python
fig, ax = plt.subplots(figsize=(8, 4))
cv_df.set_index("train_end")[["pr_auc", "roc_auc", "recall", "f1"]].plot(marker="o", ax=ax)
ax.set_xlabel("Date de fin du train (cutoff)")
ax.set_ylabel("Score")
ax.set_title("Stabilité des métriques par fold — découpage calendaire")
plt.tight_layout()
plt.show()
```

⚠️ Différences clés avec l'approche `TimeSeriesSplit` positionnelle :
- Les bornes sont des **dates**, pas des index de lignes → chaque fold couvre une période calendaire identique pour toutes les machines, sans effet d'entrelacement lié au tri par `machine_id_std`.
- L'imputer est refitté à chaque fold (pas de réutilisation de l'imputer global du train) → estimation moins optimiste.
- Si une machine n'a des données que sur une sous-période, elle peut être absente de certains folds (voir 10.5) — à surveiller si l'objectif est une performance par machine.

---

## Résumé du pipeline

1. `read_parquet` → `sort_values(["machine_id_std", "window_start"])`
2. Exclusion : identifiants, colonnes `future_*` (leakage), 4 labels, `split_set`
3. Choix de l'horizon → `y = df[horizon].astype(int)`
4. Split via `split_set` (`train` / `validation` / `test`), aucun `shuffle`
5. `SimpleImputer(strategy="median")` fit sur train uniquement, puis `StandardScaler` (modèles linéaires seulement) fit sur train uniquement
6. Vérifier le taux de panne sur train/val/test → écarter l'accuracy, privilégier PR-AUC / recall / F1
7. Compenser le déséquilibre : `class_weight="balanced"` (scikit-learn) ou `scale_pos_weight = neg/pos` calculé sur le train (XGBoost)
8. Prédire sur le test pour chaque modèle → tableau récapitulatif (PR-AUC/ROC-AUC/F1/recall), matrices de confusion, courbe précision-rappel, rapport de classification, importance des features
9. Lister les métriques scikit-learn disponibles, choisir celles adaptées au déséquilibre et au coût métier (PR-AUC, recall, F1/F2), puis les calculer pour chaque modèle
10. Validation croisée **calendaire** (bornes sur `window_start`, indépendantes de l'ordre des lignes, imputer refitté par fold) → tableau par fold, garde-fou anti-fold-vide, moyenne ± écart-type, couverture des machines, frise temporelle, dérive des métriques
