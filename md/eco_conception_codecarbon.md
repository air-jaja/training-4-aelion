# Éco-conception avec CodeCarbon — ML & DL

Ce document accompagne `eco_conception_codecarbon.ipynb`. **Partie 2** de la démarche d'optimisation raisonnée (fait suite au notebook Optuna/XGBoost). Objectif : mesurer, pas supposer, le coût énergétique et carbone de l'entraînement, et l'utiliser comme critère de décision au même titre que la performance.

1. **Instrumentation** : `EmissionsTracker` autour de l'entraînement ML (baseline XGBoost) et DL (autoencodeur SSIM) — mêmes wrapper et conventions pour les deux.
2. **Étude lourde vs étude frugale** : deux stratégies Optuna comparées sur performance **et** empreinte carbone, coût par point de PR-AUC gagné.
3. Réponses aux questions : gain proportionné ? quand la parcimonie est-elle la bonne décision Green AI ?

Tous les fichiers de sortie (`emissions.csv`, tableaux, graphiques) sont écrits dans `artifacts/ingestions/output/`.

---

## Configuration commune

```python
OUTPUT_DIR = "artifacts/ingestions/output"   # chemin relatif à la racine du projet
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Intensité carbone forcée plutôt que géolocalisation automatique (observée peu fiable/reproductible
# d'une exécution à l'autre) — ordre de grandeur du mix électrique français.
FRANCE_GRID_INTENSITY = 56.0  # gCO2eq/kWh (source ADEME/RTE, mix français très largement décarboné)

def make_tracker(project_name):
    return EmissionsTracker(
        project_name=project_name,
        output_dir=OUTPUT_DIR,
        output_file="emissions.csv",
        force_carbon_intensity_g_co2e_kwh=FRANCE_GRID_INTENSITY,
        log_level="error",
        measure_power_secs=1,
        allow_multiple_runs=True,
    )

steps_log = []  # accumule : étape, durée (s), kWh, gCO2eq, perf obtenue

def get_last_energy_kwh(project_name):
    """Relit emissions.csv pour récupérer le kWh authentique de la dernière ligne d'un projet donné."""
    emissions_df = pd.read_csv(os.path.join(OUTPUT_DIR, "emissions.csv"))
    return emissions_df[emissions_df.project_name == project_name].iloc[-1]["energy_consumed"]
```

**Point technique** : `EmissionsTracker` n'a pas de paramètre `country_iso_code` direct dans la version installée. La géolocalisation automatique s'est montrée incohérente d'un run à l'autre (pays différent détecté à quelques minutes d'intervalle dans le même environnement) — `force_carbon_intensity_g_co2e_kwh` est utilisé à la place pour un résultat reproductible, avec la valeur du mix électrique français comme référence.

---

## 1. Instrumenter l'entraînement

### 1.1 Chargement des données ML (`gold_dataset`)

Mêmes exclusions de colonnes et même horizon (`label_failure_next_24h`) que le notebook Optuna précédent.

### 1.2 Entraînement ML instrumenté (baseline XGBoost)

```python
BASELINE_PARAMS = dict(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    subsample=0.9, colsample_bytree=0.9, min_child_weight=1,
    reg_lambda=1.0, reg_alpha=0.0,
)

tracker = make_tracker("ml_baseline_xgboost")
tracker.start()
_t0 = time.time()

baseline_model = xgb.XGBClassifier(**BASELINE_PARAMS, tree_method="hist", n_jobs=1,
                                     eval_metric="aucpr", random_state=SEED)
baseline_model.fit(X_train, y_train)

_duration_ml = time.time() - _t0
_emissions_kg_ml = tracker.stop()

baseline_prauc = average_precision_score(y_val, baseline_model.predict_proba(X_val)[:, 1])

_kwh_ml = get_last_energy_kwh("ml_baseline_xgboost")
steps_log.append({
    "étape": "ML — baseline XGBoost", "durée_s": _duration_ml,
    "kWh": _kwh_ml, "gCO2eq": _emissions_kg_ml * 1000,
    "perf_obtenue": baseline_prauc, "perf_metric": "PR-AUC (val)",
})
```

### 1.3 Entraînement DL instrumenté (autoencodeur convolutionnel, perte SSIM)

Même wrapper `EmissionsTracker`, même fichier `emissions.csv` — pour comparer directement le coût ML vs DL dans le même tableau. Dataset `bottle` (MVTec-AD), architecture et perte SSIM déjà validées dans les travaux précédents (autoencodeur convolutionnel symétrique, augmentation Albumentations sur les images saines uniquement).

```python
dl_model = build_autoencoder()
dl_model.compile(optimizer="adam", loss=ssim_loss, metrics=[mse_metric])
early_stopping_dl = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

tracker = make_tracker("dl_autoencoder_ssim")
tracker.start()
_t0 = time.time()

dl_history = dl_model.fit(dl_train_ds_xy, validation_data=dl_val_ds_xy,
                           epochs=DL_EPOCHS, callbacks=[early_stopping_dl], verbose=0)

_duration_dl = time.time() - _t0
_emissions_kg_dl = tracker.stop()
dl_val_ssim_final = 1.0 - dl_history.history["val_loss"][-1]

_kwh_dl = get_last_energy_kwh("dl_autoencoder_ssim")
steps_log.append({
    "étape": "DL — autoencodeur SSIM", "durée_s": _duration_dl,
    "kWh": _kwh_dl, "gCO2eq": _emissions_kg_dl * 1000,
    "perf_obtenue": dl_val_ssim_final, "perf_metric": "SSIM (val)",
})
```

### 1.4 Rendu — tableau `étape → durée, kWh, gCO2eq, perf obtenue`

```python
steps_table = pd.DataFrame(steps_log)
steps_table.to_csv(os.path.join(OUTPUT_DIR, "steps_summary.csv"), index=False)
```

### 1.5 Graphiques pertinents

Deux lectures complémentaires : (1) répartition kWh/gCO2eq par étape (échelles très différentes entre ML tabulaire et DL image — barre logarithmique), (2) durée par étape.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(steps_table["étape"], steps_table["gCO2eq"], color=["#4C72B0", "#DD8452"])
axes[0].set_yscale("log")
axes[0].set_title("Émissions par étape")
axes[1].bar(steps_table["étape"], steps_table["durée_s"], color=["#4C72B0", "#DD8452"])
axes[1].set_title("Durée par étape")
```

---

## 2. Étude lourde vs étude frugale

Deux stratégies Optuna, sur le même problème ML (XGBoost, horizon 24h), instrumentées de la même façon :

| | Étude **lourde** | Étude **frugale** |
|---|---|---|
| Espace de recherche | Large (bornes ~2-3x plus larges) | Resserré (bornes "raisonnées") |
| Essais | 40 | 15 |
| Pruning | Aucun (`NopPruner`) | Agressif (`MedianPruner`, coupe dès le 1ᵉʳ fold) |
| Folds CV | 3, données complètes | 2, sous-échantillon 50% |

Optimisation temps de traitement appliquée aux deux : `tree_method="hist"`, `early_stopping_rounds`, `n_estimators` borné par l'early stopping plutôt que fixé arbitrairement haut.

### 2.1 Étude lourde — espace large, 40 essais, sans pruning

```python
def objective_heavy(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 800),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 100.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 100.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, RATIO_FULL * 2),
    }
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for tr_idx, va_idx in tscv.split(X_train_full):
        model = xgb.XGBClassifier(**params, tree_method="hist", n_jobs=1, eval_metric="aucpr",
                                    early_stopping_rounds=20, random_state=SEED)
        model.fit(X_train_full[tr_idx], y_train_full[tr_idx],
                  eval_set=[(X_train_full[va_idx], y_train_full[va_idx])], verbose=False)
        scores.append(average_precision_score(y_train_full[va_idx], model.predict_proba(X_train_full[va_idx])[:, 1]))
        # Pas de pruning (NopPruner) — chaque essai va jusqu'au bout des 3 folds
    return float(np.mean(scores))

tracker = make_tracker("study_heavy")
tracker.start()
study_heavy = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED), pruner=NopPruner())
study_heavy.optimize(objective_heavy, n_trials=40)
_emissions_kg_heavy = tracker.stop()
```

### 2.2 Étude frugale — espace resserré, 15 essais, pruning agressif

```python
X_train_frugal = X_train_full[::2]  # sous-échantillon 50%, stride systématique

def objective_frugal(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 80, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.95),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 8),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 3.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 3.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, RATIO_FRUGAL),
    }
    tscv = TimeSeriesSplit(n_splits=2)
    scores = []
    for fold_i, (tr_idx, va_idx) in enumerate(tscv.split(X_train_frugal)):
        model = xgb.XGBClassifier(**params, tree_method="hist", n_jobs=1, eval_metric="aucpr",
                                    early_stopping_rounds=15, random_state=SEED)
        model.fit(X_train_frugal[tr_idx], y_train_frugal[tr_idx],
                  eval_set=[(X_train_frugal[va_idx], y_train_frugal[va_idx])], verbose=False)
        scores.append(average_precision_score(y_train_frugal[va_idx], model.predict_proba(X_train_frugal[va_idx])[:, 1]))
        # Pruning agressif : dès le 1er fold, un essai clairement mauvais est coupé
        trial.report(float(np.mean(scores)), step=fold_i)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(scores))

tracker = make_tracker("study_frugal")
tracker.start()
study_frugal = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED),
                                     pruner=MedianPruner(n_warmup_steps=0, n_startup_trials=2))
study_frugal.optimize(objective_frugal, n_trials=15)
_emissions_kg_frugal = tracker.stop()
```

### 2.3 Évaluation à armes égales sur la validation

Les scores CV ne sont pas directement comparables à la baseline (protocoles différents). On réentraîne chaque meilleur essai sur `X_train` complet et on évalue sur `X_val` — même protocole que la baseline.

```python
def retrain_and_eval(best_params, run_name):
    tracker = make_tracker(run_name)
    tracker.start()
    model = xgb.XGBClassifier(**best_params, tree_method="hist", n_jobs=1, eval_metric="aucpr", random_state=SEED)
    model.fit(X_train, y_train)
    emissions_kg = tracker.stop()
    prauc = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
    return {"emissions_kg": emissions_kg, "kwh": get_last_energy_kwh(run_name), "prauc_val": prauc}

retrain_heavy = retrain_and_eval(study_heavy.best_params, "retrain_heavy_best")
retrain_frugal = retrain_and_eval(study_frugal.best_params, "retrain_frugal_best")
```

### 2.4 Coût par point de PR-AUC gagné

Coût total = recherche (étude complète) + réentraînement final, par stratégie.

```python
def cost_per_point(total_value, gain, unit):
    if gain <= 0:
        return f"non défini (aucun gain — {unit} dépensé pour rien ou une régression)"
    return f"{total_value / gain:.4f} {unit} par point de PR-AUC"

comparison["gCO2eq par point PR-AUC"] = comparison.apply(
    lambda r: r["gCO2eq"] / r["gain vs baseline"] if r["gain vs baseline"] > 0 else np.nan, axis=1)
```

### 2.5 Rendu — performance vs CO2

```python
fig, ax = plt.subplots(figsize=(8, 5.5))
colors = {"Baseline": "#55A868", "Frugale": "#DD8452", "Lourde": "#C44E52"}
for _, row in comparison.iterrows():
    ax.scatter(row["gCO2eq"], row["PR-AUC (val)"], s=180, color=colors[row["stratégie"]])
ax.set_xscale("log")
ax.set_xlabel("Émissions (gCO2eq, échelle log)")
ax.set_ylabel("PR-AUC (validation)")
```

### 2.6 Réponses

**L'étude lourde apporte-t-elle un gain proportionné à son coût ?**

Comparer directement `gain vs baseline` et `gCO2eq` des deux stratégies. Si l'étude lourde consomme (très grossièrement) `N_TRIALS_HEAVY / N_TRIALS_FRUGAL` fois plus d'essais et un espace plus large sans pruning, son coût total est structurellement plus élevé — la question est de savoir si le gain de PR-AUC l'est dans les mêmes proportions. Dans la plupart des cas pratiques sur ce type de problème (modèle déjà proche de son plafond avec des hyperparamètres raisonnables), **le gain marginal d'une recherche 2-3x plus coûteuse est généralement très inférieur au facteur de coût**. La colonne `gCO2eq par point PR-AUC` confirme chiffre à l'appui si le coût n'est pas proportionné au gain.

**Quand la parcimonie (modèle plus simple, budget réduit) est-elle la bonne décision Green AI ?**

1. **Quand le gain de la stratégie lourde ne dépasse pas la variance inter-essais** — un gain non distinguable du bruit ne justifie jamais un surcoût carbone.
2. **Quand le `gCO2eq par point de PR-AUC` de la stratégie frugale est du même ordre (ou meilleur) que celui de la lourde** — la parcimonie n'est alors pas un compromis, c'est la meilleure décision sur les deux axes.
3. **Quand le contexte de déploiement ne valorise pas économiquement le dernier point de performance** — un gain qui ne change pas la décision opérationnelle en aval ne justifie pas un budget disproportionné.
4. **Quand l'itération rapide compte plus que l'optimum global** — en phase d'exploration, une étude frugale donne un signal directionnel utile en une fraction du temps/carbone.

À l'inverse, une étude lourde reste justifiable quand le gain a une valeur métier explicite et quantifiée — la parcimonie n'est pas une règle absolue, c'est un défaut à justifier de s'en écarter, pas l'inverse.

---

## 3. Temps de traitement global

Optimisations déjà appliquées (mêmes principes que le notebook Optuna) : `tree_method="hist"`, `early_stopping_rounds`, sous-échantillonnage pour la recherche frugale, pruning agressif, `n_estimators` borné par l'early stopping.

```python
_total_elapsed = time.time() - _notebook_start_time
```

**Fichiers de sortie** (`artifacts/ingestions/output/`) :
- `emissions.csv` — détail CodeCarbon, une ligne par étape instrumentée.
- `steps_summary.csv` — tableau étape → durée/kWh/gCO2eq/perf.
- `comparaison_lourd_vs_frugal.csv`
- `emissions_par_etape.png`
- `performance_vs_co2.png`
