# Préparation des features — Modèle de maintenance prédictive

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

## Résumé du pipeline

1. `read_parquet` → `sort_values(["machine_id_std", "window_start"])`
2. Exclusion : identifiants, colonnes `future_*` (leakage), 4 labels, `split_set`
3. Choix de l'horizon → `y = df[horizon].astype(int)`
4. Split via `split_set` (`train` / `validation` / `test`), aucun `shuffle`
5. `SimpleImputer(strategy="median")` fit sur train uniquement, puis `StandardScaler` (modèles linéaires seulement) fit sur train uniquement
6. Vérifier le taux de panne sur train/val/test → écarter l'accuracy, privilégier PR-AUC / recall / F1
7. Compenser le déséquilibre : `class_weight="balanced"` (scikit-learn) ou `scale_pos_weight = neg/pos` calculé sur le train (XGBoost)