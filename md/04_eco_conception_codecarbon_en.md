# Eco-Design with CodeCarbon — ML & DL

This document accompanies `eco_conception_codecarbon.ipynb`. **Part 2** of the reasoned optimization approach (follows the Optuna/XGBoost notebook). Goal: measure, not assume, the energy and carbon cost of training, and use it as a decision criterion alongside performance.

1. **Instrumentation**: `EmissionsTracker` around ML training (baseline XGBoost) and DL training (SSIM autoencoder) — same wrapper and conventions for both.
2. **Heavy vs frugal study**: two Optuna strategies compared on performance **and** carbon footprint, cost per PR-AUC point gained.
3. Answers to: is the gain proportionate? when is parsimony the right Green AI decision?

All output files (`emissions.csv`, tables, figures) are written to `artifacts/ingestions/output/`.

---

## Shared configuration

```python
OUTPUT_DIR = "artifacts/ingestions/output"   # relative to project root
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Forced carbon intensity rather than automatic geolocation (observed unreliable/non-reproducible
# from one run to another) — order of magnitude for the French electricity mix.
FRANCE_GRID_INTENSITY = 56.0  # gCO2eq/kWh (source ADEME/RTE, largely decarbonized French mix)

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

steps_log = []  # accumulates: step, duration (s), kWh, gCO2eq, performance achieved

def get_last_energy_kwh(project_name):
    """Re-reads emissions.csv to get the authentic kWh value from the last row of a given project."""
    emissions_df = pd.read_csv(os.path.join(OUTPUT_DIR, "emissions.csv"))
    return emissions_df[emissions_df.project_name == project_name].iloc[-1]["energy_consumed"]
```

**Technical note**: `EmissionsTracker` has no direct `country_iso_code` parameter in the installed version. Automatic geolocation proved inconsistent from one run to the next (different country detected minutes apart in the same environment) — `force_carbon_intensity_g_co2e_kwh` is used instead for a reproducible result, with the French electricity mix value as reference.

---

## 1. Instrumenting the training

### 1.1 Loading ML data (`gold_dataset`)

Same column exclusions and same horizon (`label_failure_next_24h`) as the previous Optuna notebook.

### 1.2 Instrumented ML training (baseline XGBoost)

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
    "step": "ML — baseline XGBoost", "duration_s": _duration_ml,
    "kWh": _kwh_ml, "gCO2eq": _emissions_kg_ml * 1000,
    "performance": baseline_prauc, "perf_metric": "PR-AUC (val)",
})
```

### 1.3 Instrumented DL training (convolutional autoencoder, SSIM loss)

Same `EmissionsTracker` wrapper, same `emissions.csv` file — to directly compare ML vs DL cost in the same table. `bottle` dataset (MVTec-AD), architecture and SSIM loss already validated in earlier work (symmetric convolutional autoencoder, Albumentations augmentation on normal images only).

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
    "step": "DL — SSIM autoencoder", "duration_s": _duration_dl,
    "kWh": _kwh_dl, "gCO2eq": _emissions_kg_dl * 1000,
    "performance": dl_val_ssim_final, "perf_metric": "SSIM (val)",
})
```

### 1.4 Render — table `step → duration, kWh, gCO2eq, performance achieved`

```python
steps_table = pd.DataFrame(steps_log)
steps_table.to_csv(os.path.join(OUTPUT_DIR, "steps_summary.csv"), index=False)
```

### 1.5 Relevant charts

Two complementary views: (1) kWh/gCO2eq breakdown per step (very different scales between tabular ML and image DL — log-scale bar), (2) duration per step.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(steps_table["step"], steps_table["gCO2eq"], color=["#4C72B0", "#DD8452"])
axes[0].set_yscale("log")
axes[0].set_title("Emissions per step")
axes[1].bar(steps_table["step"], steps_table["duration_s"], color=["#4C72B0", "#DD8452"])
axes[1].set_title("Duration per step")
```

---

## 2. Heavy study vs frugal study

Two Optuna strategies, on the same ML problem (XGBoost, 24h horizon), instrumented the same way:

| | **Heavy** study | **Frugal** study |
|---|---|---|
| Search space | Wide (bounds ~2-3x wider) | Tight ("reasoned" bounds) |
| Trials | 40 | 15 |
| Pruning | None (`NopPruner`) | Aggressive (`MedianPruner`, cuts from the 1st fold) |
| CV folds | 3, full data | 2, 50% subsample |

Processing time optimization applied to both: `tree_method="hist"`, `early_stopping_rounds`, `n_estimators` bounded by early stopping rather than fixed arbitrarily high.

### 2.1 Heavy study — wide space, 40 trials, no pruning

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
        # No pruning (NopPruner) — every trial runs all 3 folds to completion
    return float(np.mean(scores))

tracker = make_tracker("study_heavy")
tracker.start()
study_heavy = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED), pruner=NopPruner())
study_heavy.optimize(objective_heavy, n_trials=40)
_emissions_kg_heavy = tracker.stop()
```

### 2.2 Frugal study — tight space, 15 trials, aggressive pruning

```python
X_train_frugal = X_train_full[::2]  # 50% subsample, systematic stride

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
        # Aggressive pruning: a clearly bad trial is cut from the 1st fold
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

### 2.3 Fair evaluation on validation

CV scores aren't directly comparable to the baseline (different protocols). Each best trial is retrained on the full `X_train` and evaluated on `X_val` — same protocol as the baseline.

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

### 2.4 Cost per PR-AUC point gained

Total cost = search (full study) + final retrain, per strategy.

```python
def cost_per_point(total_value, gain, unit):
    if gain <= 0:
        return f"undefined (no gain — {unit} spent for nothing or a regression)"
    return f"{total_value / gain:.4f} {unit} per PR-AUC point"

comparison["gCO2eq per PR-AUC point"] = comparison.apply(
    lambda r: r["gCO2eq"] / r["gain vs baseline"] if r["gain vs baseline"] > 0 else np.nan, axis=1)
```

### 2.5 Render — performance vs CO2

```python
fig, ax = plt.subplots(figsize=(8, 5.5))
colors = {"Baseline": "#55A868", "Frugal": "#DD8452", "Heavy": "#C44E52"}
for _, row in comparison.iterrows():
    ax.scatter(row["gCO2eq"], row["PR-AUC (val)"], s=180, color=colors[row["strategy"]])
ax.set_xscale("log")
ax.set_xlabel("Emissions (gCO2eq, log scale)")
ax.set_ylabel("PR-AUC (validation)")
```

### 2.6 Answers

**Does the heavy study bring a gain proportionate to its cost?**

Directly compare `gain vs baseline` and `gCO2eq` for both strategies. If the heavy study consumes (very roughly) `N_TRIALS_HEAVY / N_TRIALS_FRUGAL` times more trials over a wider space without pruning, its total cost is structurally higher — the question is whether the PR-AUC gain is proportionate. In most practical cases on this type of problem (model already close to its ceiling with reasonable hyperparameters), **the marginal gain from a 2-3x more expensive search is generally far below the cost factor**. The `gCO2eq per PR-AUC point` column confirms numerically whether the cost is disproportionate to the gain.

**When is parsimony (simpler model, reduced budget) the right Green AI decision?**

1. **When the heavy strategy's gain doesn't exceed inter-trial variance** — a gain indistinguishable from noise never justifies a carbon surcharge.
2. **When the frugal strategy's `gCO2eq per PR-AUC point` is of the same order (or better) than the heavy one's** — parsimony is then not a trade-off, it's simply the better decision on both axes.
3. **When the deployment context doesn't economically value the last point of performance** — a gain that doesn't change the downstream operational decision doesn't justify a disproportionate budget.
4. **When fast iteration matters more than the global optimum** — during exploration, a frugal study gives a useful directional signal for a fraction of the time/carbon.

Conversely, a heavy study remains justifiable when the performance gain has an explicit, quantified business value — parsimony isn't an absolute rule, it's a default that deviating from must justify, not the other way around.

---

## 3. Overall processing time

Optimizations already applied (same principles as the Optuna notebook): `tree_method="hist"`, `early_stopping_rounds`, subsampling for the frugal search, aggressive pruning, `n_estimators` bounded by early stopping.

```python
_total_elapsed = time.time() - _notebook_start_time
```

**Output files** (`artifacts/ingestions/output/`):
- `emissions.csv` — CodeCarbon detail, one row per instrumented step.
- `steps_summary.csv` — step → duration/kWh/gCO2eq/performance table.
- `comparaison_lourd_vs_frugal.csv`
- `emissions_par_etape.png`
- `performance_vs_co2.png`
