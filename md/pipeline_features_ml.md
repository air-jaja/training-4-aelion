# Préparation des features — Modèle de maintenance prédictive

**Contexte technique** : Python, pandas, Jupyter, pyarrow, scikit-learn. Installation des packages via `uv`.

**Fichier source** : `artifacts/ingestions/datas/gold_dataset.parquet` (134 280 lignes × 100 colonnes)

---

## 0) Installation de l'environnement

```bash
uv add ipykernel setuptools 
uv add pandas jupyter
uv add matplotlib seaborn sqlalchemy
uv add pyarrow scikit-learn 
uv add xgboost
# Environnement (si retour à 3.13 nécessaire à cause de l'erreur protobuf/Python 3.14)
uv python pin 3.13.5
# Dépendances avec versions compatibles épinglées
uv add mlflow
uv add "setuptools<82"
uv add "protobuf<5"
uv add "sqlalchemy<2"
# Vérification
uv run mlflow --version
```

### Tracking MLflow avec un backend SQLite

MLflow a besoin d'un backend store pour persister les runs, paramètres et métriques. SQLite est le choix le plus simple pour un usage local/mono-poste (pas de serveur à administrer, un simple fichier `.db`).

```bash
# Lancer le serveur de tracking MLflow avec SQLite comme backend store
mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

Dans le notebook, pointer le tracking URI vers ce serveur (ou directement vers le fichier SQLite si le serveur n'est pas lancé) :

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")  # ou "http://127.0.0.1:5000" si le serveur tourne
mlflow.set_experiment("maintenance_predictive")
```

L'interface web (si le serveur tourne) est accessible sur `http://127.0.0.1:5000`.

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

## 10) Validation croisée temporelle (`TimeSeriesSplit`) pour une estimation robuste

Un seul split train/validation/test donne une estimation ponctuelle de la performance, sensible à la période choisie. `TimeSeriesSplit` répète l'entraînement/évaluation sur plusieurs fenêtres temporelles successives **à l'intérieur du train uniquement** (le test reste intouché, réservé à l'évaluation finale).

⚠️ `TimeSeriesSplit` suppose une série temporelle unique et ordonnée. Le dataset étant trié par `machine_id_std` puis `window_start` (étape 1), les folds mélangent en réalité l'ordre temporel de plusieurs machines entrelacées par leur position dans le tableau — c'est une approximation raisonnable ici, mais à garder en tête (une alternative plus rigoureuse serait un découpage par date calendaire indépendant de l'ordre des lignes).

### 10.1 Définir les folds et lancer la validation croisée

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, f1_score

N_SPLITS = 5
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# Modèle à valider (exemple : Random Forest, déjà entraîné à l'étape 7)
base_model = clone(rf)

cv_rows = []
for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_imp), start=1):
    X_tr, X_va = X_train_imp.iloc[tr_idx], X_train_imp.iloc[val_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    model = clone(base_model)
    model.fit(X_tr, y_tr)

    y_va_pred  = model.predict(X_va)
    y_va_proba = model.predict_proba(X_va)[:, 1]

    cv_rows.append({
        "fold": fold,
        "n_train": len(tr_idx),
        "n_val": len(val_idx),
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

### 10.2 Résumé (moyenne ± écart-type sur les folds)

L'écart-type mesure la stabilité du modèle dans le temps — un écart-type élevé signale une performance instable selon la période.

```python
summary = cv_df[["pr_auc", "roc_auc", "recall", "f1"]].agg(["mean", "std"]).round(3)
summary
```

### 10.3 Bornes temporelles de chaque fold

Vérifier concrètement quelles périodes sont utilisées en train/validation à chaque fold.

```python
window_start_train = df.loc[train_mask, "window_start"].reset_index(drop=True)

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_imp), start=1):
    print(
        f"Fold {fold}: "
        f"train [{window_start_train.iloc[tr_idx].min()} -> {window_start_train.iloc[tr_idx].max()}] ({len(tr_idx)} lignes) | "
        f"val [{window_start_train.iloc[val_idx].min()} -> {window_start_train.iloc[val_idx].max()}] ({len(val_idx)} lignes)"
    )
```

### 10.4 Visualiser le découpage des folds

```python
fig, ax = plt.subplots(figsize=(10, 4))
for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_imp)):
    ax.scatter(tr_idx, [fold] * len(tr_idx), color="tab:blue", s=2, label="train" if fold == 0 else "")
    ax.scatter(val_idx, [fold] * len(val_idx), color="tab:orange", s=2, label="validation" if fold == 0 else "")
ax.set_yticks(range(N_SPLITS))
ax.set_yticklabels([f"Fold {i + 1}" for i in range(N_SPLITS)])
ax.set_xlabel("Index (ordre temporel dans le train)")
ax.set_title("TimeSeriesSplit — découpage des folds")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
```

### 10.5 Évolution de la performance fold par fold

Permet de repérer une dérive dans le temps (ex. : performance qui se dégrade sur les folds les plus récents).

```python
fig, ax = plt.subplots(figsize=(8, 4))
cv_df[["pr_auc", "roc_auc", "recall", "f1"]].plot(marker="o", ax=ax)
ax.set_xlabel("Fold")
ax.set_ylabel("Score")
ax.set_title("Stabilité des métriques par fold (TimeSeriesSplit)")
plt.tight_layout()
plt.show()
```

⚠️ Comme aux étapes précédentes, ne jamais fitter l'imputer/scaler sur l'ensemble `X_train_imp` avant la validation croisée si l'on veut une estimation totalement propre : idéalement, refitter aussi `SimpleImputer`/`StandardScaler` à l'intérieur de chaque fold (sur `X_tr` uniquement). La version ci-dessus, plus simple, réutilise l'imputer déjà fitté sur tout le train — acceptable comme première estimation, mais légèrement optimiste.

---

## 11) Choisir un seuil de décision

`predict()` applique par défaut un seuil de 0.5 sur la probabilité (`predict_proba`) pour décider "panne" / "pas de panne". Ce seuil est arbitraire : rien ne garantit qu'il soit optimal, surtout avec des classes déséquilibrées (étape 6). Le seuil doit être choisi **explicitement**, en fonction de la métrique ou de la contrainte métier prioritaire.

**Règle impérative** : choisir le seuil sur la **validation**, jamais sur le test — le test ne sert qu'à l'évaluation finale, une fois le seuil figé. Une fois choisi sur `X_val`, on l'applique tel quel sur `X_test`.

### 11.1 Générer les probabilités sur la validation

```python
y_val_proba_log = log_reg.predict_proba(X_val_scaled)[:, 1]
y_val_proba_rf  = rf.predict_proba(X_val_imp)[:, 1]
y_val_proba_xgb = xgb.predict_proba(X_val_imp)[:, 1]
```

### 11.2 Visualiser precision / recall / F1 en fonction du seuil

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba_rf)

# f1 aligné sur thresholds (precision_recall_curve retourne un point de plus que thresholds)
f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, precisions[:-1], label="precision")
ax.plot(thresholds, recalls[:-1], label="recall")
ax.plot(thresholds, f1_scores, label="f1", linestyle="--")
ax.set_xlabel("Seuil de décision")
ax.set_ylabel("Score")
ax.set_title("Precision / Recall / F1 en fonction du seuil — Random Forest (validation)")
ax.legend()
plt.tight_layout()
plt.show()
```

### 11.3 Option A — Seuil qui maximise le F1

Compromis équilibré entre precision et recall, sans préférence métier particulière.

```python
best_idx = np.argmax(f1_scores)
best_threshold_f1 = thresholds[best_idx]

print(f"Seuil optimal (F1) : {best_threshold_f1:.3f}")
print(f"  -> precision={precisions[best_idx]:.3f}, recall={recalls[best_idx]:.3f}, f1={f1_scores[best_idx]:.3f}")
```

### 11.4 Option B — Seuil qui respecte une contrainte métier (ex. : recall minimum garanti)

Pour de la maintenance prédictive, manquer une panne est souvent plus coûteux qu'une fausse alerte : on peut vouloir garantir un recall minimum (ex. : détecter au moins 80 % des pannes), quitte à sacrifier de la precision.

```python
TARGET_RECALL = 0.80

# Parmi les seuils qui atteignent le recall cible, prendre celui qui maximise la precision
eligible = recalls[:-1] >= TARGET_RECALL
if eligible.any():
    best_idx_recall = np.argmax(np.where(eligible, precisions[:-1], -1))
    best_threshold_recall = thresholds[best_idx_recall]
    print(f"Seuil pour recall >= {TARGET_RECALL:.0%} : {best_threshold_recall:.3f}")
    print(f"  -> precision={precisions[best_idx_recall]:.3f}, recall={recalls[best_idx_recall]:.3f}")
else:
    print(f"Aucun seuil n'atteint un recall de {TARGET_RECALL:.0%} sur la validation.")
```

### 11.5 Appliquer le seuil choisi sur le test (évaluation finale)

```python
chosen_threshold = best_threshold_f1  # ou best_threshold_recall selon le choix métier retenu

y_pred_rf_thresh = (y_proba_rf >= chosen_threshold).astype(int)

print(f"Seuil appliqué : {chosen_threshold:.3f}")
print(classification_report(y_test, y_pred_rf_thresh, target_names=["pas de panne", "panne"]))
```

### 11.6 Comparer seuil par défaut (0.5) vs seuil choisi

```python
comparison_rows = []
for label, thresh in [("seuil par défaut (0.5)", 0.5), ("seuil choisi", chosen_threshold)]:
    y_pred_t = (y_proba_rf >= thresh).astype(int)
    comparison_rows.append({
        "seuil": label,
        "valeur_seuil": thresh,
        "precision": precision_score(y_test, y_pred_t),
        "recall": recall_score(y_test, y_pred_t),
        "f1": f1_score(y_test, y_pred_t),
    })

pd.DataFrame(comparison_rows).set_index("seuil").round(3)
```

⚠️ Points de vigilance :
- Le seuil optimal dépend de l'horizon choisi (étape 3) et peut varier d'un modèle à l'autre — le refaire pour chaque combinaison horizon/modèle.
- Le PR-AUC et le ROC-AUC (étapes 8-9) sont indépendants du seuil (ils évaluent le classement des scores) ; le choix du seuil n'intervient que pour transformer une probabilité en décision binaire concrète.
- Un seuil recalibré sur la validation peut se dégrader légèrement sur le test (léger optimisme de sélection) — c'est attendu et acceptable tant que l'écart reste faible.

---

## 12) Journaliser les paramètres et métriques dans MLflow

Pour comparer et reproduire les expériences, chaque modèle (régression logistique, Random Forest, XGBoost) est journalisé dans MLflow sous forme d'un **run** distinct : hyperparamètres, métriques de l'étape 8/9, seuil choisi (étape 11), et le modèle lui-même.

Prérequis : MLflow installé et backend SQLite configuré (voir étape 0). En cas d'erreur d'installation ou de compatibilité, voir le fichier de dépannage dédié (`depannage_mlflow.md`).

### 12.1 Configurer le tracking

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")  # ou "http://127.0.0.1:5000" si le serveur tourne
mlflow.set_experiment("maintenance_predictive")
```

### 12.2 Fonction utilitaire de journalisation

Centralise la journalisation pour éviter de dupliquer le code par modèle.

```python
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score,
)

def log_model_run(run_name, model, params, y_true, y_pred, y_proba, model_flavor_log_fn):
    with mlflow.start_run(run_name=run_name):
        # Paramètres
        mlflow.log_param("horizon", HORIZON)
        for name, value in params.items():
            mlflow.log_param(name, value)

        # Métriques (étapes 8-9)
        mlflow.log_metric("pr_auc", average_precision_score(y_true, y_proba))
        mlflow.log_metric("roc_auc", roc_auc_score(y_true, y_proba))
        mlflow.log_metric("precision", precision_score(y_true, y_pred))
        mlflow.log_metric("recall", recall_score(y_true, y_pred))
        mlflow.log_metric("f1", f1_score(y_true, y_pred))

        # Modèle (flavor scikit-learn ou xgboost selon le cas)
        model_flavor_log_fn(model, "model")

        print(f"Run '{run_name}' journalisé (id={mlflow.active_run().info.run_id})")
```

### 12.3 Journaliser la régression logistique

```python
import mlflow.sklearn

log_model_run(
    run_name="logistic_regression",
    model=log_reg,
    params={
        "class_weight": "balanced",
        "max_iter": log_reg.max_iter,
        "features": "standardisées (imputer + scaler)",
    },
    y_true=y_test,
    y_pred=y_pred_log,
    y_proba=y_proba_log,
    model_flavor_log_fn=mlflow.sklearn.log_model,
)
```

### 12.4 Journaliser le Random Forest

```python
log_model_run(
    run_name="random_forest",
    model=rf,
    params={
        "class_weight": "balanced",
        "n_estimators": rf.n_estimators,
        "random_state": rf.random_state,
        "features": "imputées (médiane)",
    },
    y_true=y_test,
    y_pred=y_pred_rf,
    y_proba=y_proba_rf,
    model_flavor_log_fn=mlflow.sklearn.log_model,
)
```

### 12.5 Journaliser XGBoost

```python
import mlflow.xgboost

log_model_run(
    run_name="xgboost",
    model=xgb,
    params={
        "scale_pos_weight": round(scale_pos_weight, 3),
        "random_state": xgb.random_state,
        "features": "imputées (médiane)",
    },
    y_true=y_test,
    y_pred=y_pred_xgb,
    y_proba=y_proba_xgb,
    model_flavor_log_fn=mlflow.xgboost.log_model,
)
```

### 12.6 Comparer les runs

```python
runs_df = mlflow.search_runs(experiment_names=["maintenance_predictive"], order_by=["metrics.pr_auc DESC"])
runs_df[["run_id", "tags.mlflow.runName", "metrics.pr_auc", "metrics.roc_auc", "metrics.recall", "metrics.f1"]]
```

### 12.7 Visualiser dans l'interface MLflow

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

Puis ouvrir `http://127.0.0.1:5000` : chaque run apparaît avec ses paramètres, métriques et le modèle téléchargeable, groupés sous l'expérience `maintenance_predictive`.

⚠️ Points de vigilance :
- Journaliser **un run par combinaison** horizon/modèle si plusieurs horizons sont testés (étape 3) — ajouter `mlflow.log_param("horizon", horizon)` distinctement à chaque run, ou nommer le run en conséquence (ex. `"random_forest_24h"`).
- Le seuil choisi à l'étape 11 (`chosen_threshold`) peut aussi être journalisé comme paramètre (`mlflow.log_param("threshold", chosen_threshold)`) pour tracer la décision associée à chaque run.
- `mlflow.sklearn.log_model` et `mlflow.xgboost.log_model` sauvegardent le modèle comme artefact MLflow (rechargeable ensuite via `mlflow.pyfunc.load_model`), ce qui facilite le déploiement ou la comparaison a posteriori.

---

## 13) Tableau comparatif, graphes et choix du meilleur modèle

Synthèse des résultats obtenus sur le jeu de test pour l'horizon `label_failure_next_24h` (étapes 8 à 11), pour trancher entre les 3 modèles.

### 13.1 Tableau comparatif (seuil par défaut 0.5)

| Modèle | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.577 | 0.762 | 0.554 | 0.544 | 0.549 |
| **Random Forest** | **0.613** | **0.767** | 0.865 | 0.375 | 0.523 |
| XGBoost | 0.579 | 0.745 | 0.614 | 0.478 | 0.537 |

*(PR-AUC et ROC-AUC sont indépendants du seuil — ils mesurent la qualité du classement des scores, pas d'une décision binaire précise. Voir étape 9.)*

### 13.2 Effet du seuil optimisé (étape 11) — exemple Random Forest

| Seuil | Precision | Recall | F1 |
|---|---|---|---|
| 0.5 (défaut) | 0.855 | 0.381 | 0.527 |
| **0.4 (optimisé F1 sur validation)** | 0.787 | 0.440 | **0.564** |

Le Random Forest a la plus forte precision à 0.5 mais un recall faible (beaucoup de pannes manquées). Abaisser le seuil à 0.4 rééquilibre precision/recall et fait passer son F1 de 0.523 à **0.564** — meilleur score F1 toutes méthodes confondues (à comparer aux 0.549 de la régression logistique et 0.537 de XGBoost, eux-mêmes non encore réoptimisés sur seuil).

### 13.3 Générer les graphiques comparatifs

```python
import matplotlib.pyplot as plt
import numpy as np

metrics_to_plot = ["pr_auc", "roc_auc", "precision", "recall", "f1"]
x = np.arange(len(metrics_to_plot))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
for i, model_name in enumerate(results_df.index):
    ax.bar(x + i * width, results_df.loc[model_name, metrics_to_plot], width, label=model_name)

ax.set_xticks(x + width)
ax.set_xticklabels(metrics_to_plot)
ax.set_ylabel("Score")
ax.set_title(f"Comparaison des modèles — {HORIZON} (seuil par défaut 0.5)")
ax.legend()
plt.tight_layout()
plt.show()
```

```python
# Courbe précision-rappel comparative (déjà produite à l'étape 8.4)
# -> c'est le graphique le plus pertinent pour choisir un modèle indépendamment du seuil :
# la courbe la "plus haute à droite" domine les autres sur toute la plage de seuils possibles.
```

```python
# Effet du seuil sur le F1 par modèle (reprendre la logique de l'étape 11.2 pour chaque modèle)
fig, ax = plt.subplots(figsize=(8, 5))
for name, (_, y_proba) in models_preds.items():
    precisions_m, recalls_m, thresholds_m = precision_recall_curve(y_test, y_proba)
    f1_m = 2 * precisions_m[:-1] * recalls_m[:-1] / (precisions_m[:-1] + recalls_m[:-1] + 1e-12)
    ax.plot(thresholds_m, f1_m, label=name)
ax.set_xlabel("Seuil de décision")
ax.set_ylabel("F1")
ax.set_title(f"F1 en fonction du seuil, par modèle — {HORIZON}")
ax.legend()
plt.tight_layout()
plt.show()
```

Ce dernier graphique est le plus utile pour la décision : il montre, pour **chaque** modèle, son meilleur F1 atteignable (pas seulement au seuil 0.5) — condition nécessaire pour une comparaison équitable.

### 13.4 Choix du modèle et argumentation

**Modèle retenu : Random Forest**, avec seuil de décision ajusté à 0.4 (au lieu de 0.5 par défaut).

Arguments :

1. **Meilleure capacité de classement (métriques indépendantes du seuil)** : Random Forest a le PR-AUC le plus élevé (0.613 vs 0.577 et 0.579) et le ROC-AUC le plus élevé (0.767 vs 0.762 et 0.745). C'est le critère le plus fiable pour comparer des modèles sur classes déséquilibrées (étape 9), car il ne dépend pas d'un choix de seuil arbitraire.
2. **Son faible recall par défaut est un artefact de seuil, pas un défaut intrinsèque** : à 0.5, le Random Forest est très précis (0.865) mais rate beaucoup de pannes (recall 0.375). Comme il classe mieux les cas positifs (PR-AUC le plus haut), il suffit d'abaisser le seuil pour révéler cette capacité — ce qui donne le meilleur F1 global (0.564) une fois optimisé.
3. **Interprétabilité opérationnelle** : contrairement à XGBoost (performance intermédiaire ici) et à la régression logistique (limitée aux relations linéaires après standardisation), le Random Forest fournit une importance des features directement exploitable (étape 8.6) pour orienter la maintenance (quelles variables regarder en priorité).
4. **Marge d'ajustement métier** : le seuil peut encore être déplacé vers un recall plus élevé (option B de l'étape 11.4, ex. recall ≥ 80 %) si le coût d'une panne manquée s'avère plus élevé que prévu — le Random Forest conserve alors la meilleure precision à recall égal, grâce à son PR-AUC supérieur.

⚠️ Nuances à garder à l'esprit :
- Cette comparaison porte sur un seul horizon (`24h`) et un split unique ; la refaire pour les autres horizons (`6h`, `12h`, `48h` — étape 3) avant de généraliser le choix.
- Les écarts de PR-AUC entre les 3 modèles restent modestes (0.577 à 0.613) — sur un autre tirage de données ou avec un tuning d'hyperparamètres plus poussé (non couvert ici), XGBoost pourrait rattraper ou dépasser le Random Forest.
- Si la contrainte métier prioritaire est un recall très élevé (détecter un maximum de pannes, quitte à multiplier les fausses alertes), la régression logistique — plus stable en recall à seuil par défaut — mérite d'être réévaluée après son propre tuning de seuil (étape 11, non encore appliqué ici qu'au Random Forest).

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
10. Validation croisée temporelle (`TimeSeriesSplit`, 5 folds sur le train) → tableau par fold, moyenne ± écart-type, bornes temporelles, visualisation du découpage et de la stabilité des métriques
11. Choisir un seuil de décision sur la validation (maximiser F1 ou garantir un recall minimum), l'appliquer sur le test, comparer au seuil par défaut (0.5)
12. Journaliser paramètres, métriques et modèle dans MLflow pour chacun des 3 modèles (régression logistique, Random Forest, XGBoost), comparer les runs et visualiser dans l'UI MLflow
13. Tableau comparatif + graphiques (barres par métrique, F1 vs seuil par modèle) → choix argumenté du meilleur modèle (Random Forest, seuil 0.4)