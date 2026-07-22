# Reasoned Optimization with Optuna — XGBoost on `gold_dataset` (24h horizon)

This document accompanies `optuna_xgboost_predictive_maintenance.ipynb`. A **reasoned** hyperparameter optimization approach, not a blind search over a huge space:

1. Framing (model, horizon, metric, validation protocol) + reference baseline.
2. Bounded, plausible search space for XGBoost.
3. Objective function: average PR-AUC over temporal cross-validation.
4. Optuna study (TPE + pruning + bounded budget, fixed seed for reproducibility).
5. Post-optimization analysis (history, HP importance, stability, tuning-overfitting risk).
6. Final validation on the test set (never seen during tuning), compared to the baseline.

Everything is tracked in MLflow (local SQLite, `mlflow/mlflow.db`).

---

## 1. Framing: model, horizon, metric, protocol — and reference baseline

**Choices fixed before any tuning** (to avoid mixing "which problem to solve" with "how to optimize it"):

- **Model**: XGBoost (`XGBClassifier`).
- **Horizon**: `label_failure_next_24h` — recall/anticipation trade-off already identified in earlier work.
- **Metric**: **PR-AUC**, not accuracy — consistent with the class imbalance (~17% positives).
- **Validation protocol**: `TimeSeriesSplit` on train (respects temporal order — no leakage).
- **Features**: same exclusions as the rest of the project (identifiers, temporal-leakage columns, other horizons' labels).

```python
GOLD_PATH = "artifacts/ingestions/datas/gold_dataset.parquet"  # relative to project root

EXCLUDE_BASE = [
    'machine_id_std', 'window_start', 'window_end',
    'future_incident_count_6h', 'label_failure_next_6h',
    'future_incident_count_12h', 'label_failure_next_12h',
    'future_incident_count_24h', 'label_failure_next_24h',
    'future_incident_count_48h', 'label_failure_next_48h',
    'incident_count_1h', 'incident_max_severity_1h',
    'incident_count_prev_24h', 'incident_max_severity_prev_24h',
    'incident_count_prev_7d', 'hours_since_last_incident',
    'type_surchauffe_count_prev_24h', 'type_baisse_pression_count_prev_24h',
    'type_vibration_count_prev_24h', 'type_bruit_mecanique_count_prev_24h',
    'type_surconsommation_count_prev_24h', 'type_blocage_mecanique_count_prev_24h',
    'type_alarme_capteur_count_prev_24h', 'type_arret_urgence_count_prev_24h',
    'type_defaut_qualite_count_prev_24h',
    'days_since_last_maintenance', 'maintenance_count_prev_30d',
    'split_set',
]
HORIZON = 'label_failure_next_24h'
```

**Processing time optimization (1/5)**: CPU parallelism assigned differently depending on context.

```python
N_CORES = os.cpu_count() or 1
XGB_N_JOBS_SINGLE = N_CORES   # baseline and final model (one training at a time)
XGB_N_JOBS_SEARCH = 1         # during Optuna search — parallelism happens at the trial level
OPTUNA_N_JOBS = 1              # cautious default — see detailed note in step 5
```

Two different CPU uses, two different settings: a standalone training run (baseline, final model) benefits from all cores; during the search, a multi-threaded XGBoost fit that already takes only a few seconds has more synchronization overhead than real gain — parallelizing the trials themselves (`OPTUNA_N_JOBS`) is preferred instead.

### Baseline model (reasonable default hyperparameters, no tuning)

Serves as a **reference point** to judge whether Optuna tuning brings a real gain, not just a higher number on an isolated run.

```python
BASELINE_PARAMS = dict(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    subsample=0.9, colsample_bytree=0.9, min_child_weight=1,
    reg_lambda=1.0, reg_alpha=0.0,
)

with mlflow.start_run(run_name="baseline_xgboost_default") as run_baseline:
    mlflow.log_params({**BASELINE_PARAMS, "horizon": HORIZON, "model": "xgboost_baseline"})
    baseline_model = xgb.XGBClassifier(**BASELINE_PARAMS, tree_method="hist", n_jobs=XGB_N_JOBS_SINGLE,
                                        eval_metric="aucpr", random_state=SEED)
    baseline_model.fit(X_train, y_train)
    val_proba_baseline = baseline_model.predict_proba(X_val)[:, 1]
    baseline_prauc = average_precision_score(y_val, val_proba_baseline)
    baseline_auc = roc_auc_score(y_val, val_proba_baseline)
```

### Render — reference table

```python
reference_table = pd.DataFrame([
    {"model": "baseline (reasonable defaults)", "PR-AUC (val)": baseline_prauc, "AUC (val)": baseline_auc},
])
```

---

## 2-3. Search space, sampler and pruner — recap of choices

- **Sampler**: `TPESampler` — good exploration/exploitation trade-off for a medium-sized space (9 hyperparameters).
- **Pruner**: `MedianPruner` — stops a trial if its intermediate performance is below the median of previous trials at the same stage.
- **Pruning applied at the cross-validation fold level**, not at each individual boosting iteration: after each fold, the running mean PR-AUC is reported to Optuna, which decides whether to continue or stop the trial before the next fold. Deliberate choice — the official `XGBoostPruningCallback` (reports at every boosting iteration) conflicts with multi-fold CV (each fold restarts from iteration 0, which disrupts Optuna's step tracking — observed in practice, not just assumed). Fold-level pruning remains driven by XGBoost's actual training progress, while staying robust.

---

## 4. Bounded search space for XGBoost

Bounds chosen based on context (134k rows, ~71 features, strong class imbalance), not generic ranges.

| Hyperparameter | Range | Justification |
|---|---|---|
| `n_estimators` | 100 – 300 | Bounded high by internal early stopping; tightened to 300 for compute time |
| `max_depth` | 3 – 8 | Above 8, near-systematic overfitting at this data volume |
| `learning_rate` | 0.01 – 0.3 (log) | One order of magnitude below/above the usual 0.1 |
| `subsample` | 0.6 – 1.0 | Below 0.6, too much variance |
| `colsample_bytree` | 0.6 – 1.0 | Same, per column |
| `min_child_weight` | 1 – 10 | Regularizes minimum leaf size |
| `reg_lambda` | 1e-3 – 10 (log) | L2, from almost none to strong regularization |
| `reg_alpha` | 1e-3 – 10 (log) | L1, same |
| `scale_pos_weight` | 1 – neg/pos ratio | Upper bound = full rebalancing |

### Why bound rather than open wide?

1. **An unbounded space isn't more "objective," just more expensive to explore** — Optuna (TPE) needs enough sampled points to build a reliable density model; a 10x larger space with the same trial budget gives 10x sparser exploration, not a better guarantee of finding the optimum.
2. **Bounds encode domain knowledge, not an arbitrary bias** — e.g. `max_depth > 10` on 71 features and 94k training rows leads to near-systematic overfitting.
3. **A space that's too large also complicates post-hoc analysis** — HP importance and slice plots are more readable with tightened ranges around plausible values.
4. **The opposite risk also exists**: bounds that are too tight can exclude the true best value — hence ranges left wide by a factor of ~5-10x around usual values rather than narrow ranges centered on a single hypothesis.

In short: bounding isn't a constraint one endures, it's a design choice that makes the search more efficient **and** more interpretable.

### Objective function: average PR-AUC over temporal cross-validation

**Processing time optimization (2/5)**: subsampling for the search only (systematic stride, preserves temporal order) — only the final model is retrained on the full data.

```python
N_SPLITS = 3
SEARCH_SAMPLE_FRACTION = 0.5   # 1.0 = no subsampling
_stride = max(1, round(1 / SEARCH_SAMPLE_FRACTION))

X_train_arr = X_train.values[::_stride]
y_train_arr = y_train.values[::_stride]

def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),   # optimization (4/5): capped at 300, not 600
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, SCALE_POS_WEIGHT_MAX),
    }
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_scores = []
    for fold_i, (tr_idx, va_idx) in enumerate(tscv.split(X_train_arr)):
        model = xgb.XGBClassifier(**params, tree_method="hist", n_jobs=XGB_N_JOBS_SEARCH, eval_metric="aucpr",
                                    early_stopping_rounds=20, random_state=SEED)
        model.fit(X_train_arr[tr_idx], y_train_arr[tr_idx],
                  eval_set=[(X_train_arr[va_idx], y_train_arr[va_idx])], verbose=False)
        proba = model.predict_proba(X_train_arr[va_idx])[:, 1]
        fold_scores.append(average_precision_score(y_train_arr[va_idx], proba))

        # Fold-level pruning
        trial.report(float(np.mean(fold_scores)), step=fold_i)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(fold_scores))
```

---

## 5. Optuna study: TPE + pruning + bounded budget, fixed seed

**Reproducibility**: `TPESampler(seed=SEED)` — two runs with the same seed propose the same sequence of trials.

**Bounded budget**: `n_trials` **and** `timeout` — whichever is reached first stops the study, so a slower or faster machine doesn't silently change the scope of the search.

**Per-trial MLflow tracking**: each Optuna trial is logged as a distinct MLflow run.

**Processing time optimization (5/5)**: parallelize trials (`OPTUNA_N_JOBS`) rather than XGBoost threads — cautiously set to `1` by default:
1. **MLflow + SQLite handles concurrent writes poorly** — several trials logging at the same time can trigger `database is locked` errors.
2. **Consistency with `XGB_N_JOBS_SEARCH=1`** — on a machine with N cores, set `OPTUNA_N_JOBS = N_CORES` to take advantage.

```python
N_TRIALS = 50
TIMEOUT_SECONDS = 3600

sampler = TPESampler(seed=SEED)
pruner = MedianPruner(n_warmup_steps=1)
study = optuna.create_study(study_name="xgboost_gold_24h", direction="maximize", sampler=sampler, pruner=pruner)

def objective_with_mlflow(trial):
    with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=False):
        score = objective(trial)
        mlflow.log_params(trial.params)
        mlflow.log_metric("prauc_cv", score)
        return score

study.optimize(objective_with_mlflow, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS, n_jobs=OPTUNA_N_JOBS)
```

### The best trial beats (or doesn't) the baseline — both cases are informative

```python
best_model = xgb.XGBClassifier(**study.best_params, tree_method="hist", n_jobs=1,
                                 eval_metric="aucpr", random_state=SEED)
best_model.fit(X_train, y_train)
val_proba_best = best_model.predict_proba(X_val)[:, 1]
best_prauc_val = average_precision_score(y_val, val_proba_best)
gain_prauc = best_prauc_val - baseline_prauc
```

**Reading it — both outcomes are informative:**

- **If the best trial clearly beats the baseline**: tuning brings real value — default hyperparameters weren't suited to this specific dataset. Verify next that the gain holds on the test set (step 7) before trusting it definitively.
- **If the best trial doesn't beat (or barely beats) the baseline**: **not a method failure** — the model is already close to its performance ceiling with reasonable hyperparameters, remaining gaps likely come from elsewhere (feature engineering, data volume, label quality). Continuing to search with a larger budget would then be a poor use of compute time.

### When does the improvement no longer "justify" the compute spent?

No universal threshold, but concrete markers:

1. **Diminishing returns visible on the optimization history**: if the last 10 trials no longer notably improve the best score, the remaining budget is very likely to be spent for marginal gain.
2. **Compare the gain to measurement noise**: if the observed gain is of the same order as the variance across folds for a single trial, it's probably not a real gain.
3. **Weigh against the actual cost**: each trial here costs ~3 XGBoost trainings. 50 trials already represent a significant budget — multiplying it by 10 for +0.005 PR-AUC generally isn't justifiable except for an explicit business case.
4. **Pragmatic rule**: stop investing in tuning once the incremental gain per additional trial becomes smaller than the metric's own measurement uncertainty — beyond that, you're optimizing noise.

---

## 6. Post-optimization analysis

Understand *why* a combination works, not just which one.

### Optimization history

```python
ax = ovm.plot_optimization_history(study)
```

### Hyperparameter importance

```python
importances = optuna.importance.get_param_importances(study)
ax = ovm.plot_param_importances(study)
```

### Slice plot — top 2-3 dominant hyperparameters

Shows, for each trial, the hyperparameter value on the x-axis and PR-AUC on the y-axis — reveals whether a value zone stands out clearly, or whether the point cloud is scattered without a clear trend.

```python
top_hps = list(importances.keys())[:3]
axes = ovm.plot_slice(study, params=top_hps)
```

### Reading stability (variance across nearby trials) and which HPs really matter

```python
top10_values = values[np.argsort(values)[-10:]]
# Compare top10_values.std() to the observed gain_prauc
```

**Point of attention — a flattering `val` gap but erratic HP importance = sign of tuning overfitting**

If the best trial shows a net gain, but HP importance reveals no clearly dominant hyperparameter and/or the best trials share no common value zone in the slice plot — then Optuna probably didn't find a real zone of good hyperparameters, but **exploited a lucky combination on this specific validation split**. This is overfitting at a different level than an individual model: it's the **search process itself** that adapted to the specificity of one split.

**Concrete example**: 50 trials where the best gets PR-AUC=0.42 with `max_depth=7`, but the 2nd best (0.419, nearly identical) has `max_depth=3`, and the 3rd best `max_depth=8` with a `learning_rate` ten times higher than the other two. If `max_depth`'s importance still comes out dominant (fANOVA captures global variance across the whole space, including bad trials), that's misleading: the **best** trials don't converge on any common value, signaling that the best trial's gain is probably a split artifact rather than a robust optimum.

**How to guard against it**:
1. Never pick a hyperparameter based solely on the best trial's value — check whether the top 5-10 trials converge on a coherent zone.
2. Revalidate with several seeds or splits before locking in a choice.
3. Be wary of a search space too large relative to the trial budget (implicit multiple testing).

---

## 7. Final validation — no tuning overfitting

1. Retrain with the best hyperparameters on **train + validation combined**.
2. Evaluate **once** on the **test set**, never seen during tuning nor during best-trial selection.
3. Explicitly compare to the baseline retrained the same way.

```python
X_trainval = pd.concat([X_train, X_val], axis=0)
y_trainval = pd.concat([y_train, y_val], axis=0)

final_model = xgb.XGBClassifier(**study.best_params, tree_method="hist", n_jobs=XGB_N_JOBS_SINGLE,
                                  eval_metric="aucpr", random_state=SEED)
final_model.fit(X_trainval, y_trainval)
final_prauc_test = average_precision_score(y_test, final_model.predict_proba(X_test)[:, 1])

baseline_final_model = xgb.XGBClassifier(**BASELINE_PARAMS, tree_method="hist", n_jobs=XGB_N_JOBS_SINGLE,
                                           eval_metric="aucpr", random_state=SEED)
baseline_final_model.fit(X_trainval, y_trainval)
baseline_prauc_test = average_precision_score(y_test, baseline_final_model.predict_proba(X_test)[:, 1])
```

```python
gain_test_prauc = final_prauc_test - baseline_prauc_test
# If the gap between gain_test_prauc and the validation gain exceeds 50% relative:
# signal of tuning overfitting to watch.
```

---

## 8. Overall processing time

```python
_total_elapsed = time.time() - _notebook_start_time
```

Global timer (whole notebook) and Optuna study detail (`_study_elapsed`) displayed separately.
