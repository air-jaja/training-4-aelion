# Optimisation raisonnée avec Optuna — XGBoost sur `gold_dataset` (horizon 24h)

Ce document accompagne `optuna_xgboost_predictive_maintenance.ipynb`. Démarche d'optimisation d'hyperparamètres **raisonnée**, pas une recherche aveugle sur un grand espace :

1. Cadrage (modèle, horizon, métrique, protocole de validation) + baseline de référence.
2. Espace de recherche borné et plausible pour XGBoost.
3. Fonction objectif : PR-AUC moyenne en validation croisée temporelle.
4. Étude Optuna (TPE + pruning + budget borné, seed fixée pour la reproductibilité).
5. Analyse post-optimisation (historique, importance des HP, stabilité, risque de sur-ajustement du tuning).
6. Validation finale sur le test set (jamais vu pendant le tuning), comparée à la baseline.

Tout est suivi dans MLflow (SQLite locale, `mlflow/mlflow.db`).

---

## 1. Cadrage : modèle, horizon, métrique, protocole — et baseline de référence

**Choix figés avant tout tuning** (pour ne pas mélanger "quel problème résoudre" et "comment l'optimiser") :

- **Modèle** : XGBoost (`XGBClassifier`).
- **Horizon** : `label_failure_next_24h` — compromis rappel/anticipation déjà identifié dans les travaux précédents.
- **Métrique** : **PR-AUC**, pas l'accuracy — cohérent avec le déséquilibre de classe (~17 % de positifs).
- **Protocole de validation** : `TimeSeriesSplit` sur le train (respect de l'ordre temporel — pas de fuite).
- **Features** : mêmes exclusions que le reste du projet (identifiants, colonnes de fuite temporelle, labels des autres horizons).

```python
GOLD_PATH = "artifacts/ingestions/datas/gold_dataset.parquet"  # chemin relatif à la racine du projet

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

**Optimisation temps de traitement (1/5)** : parallélisme CPU réparti différemment selon le contexte.

```python
N_CORES = os.cpu_count() or 1
XGB_N_JOBS_SINGLE = N_CORES   # baseline et modèle final (un seul entraînement à la fois)
XGB_N_JOBS_SEARCH = 1         # pendant la recherche Optuna — la parallélisation se fait au niveau des essais
OPTUNA_N_JOBS = 1              # défaut prudent — voir note détaillée étape 5
```

Deux usages différents du CPU, deux réglages différents : un entraînement isolé (baseline, modèle final) profite de tous les cœurs ; pendant la recherche, un XGBoost multi-thread sur un modèle qui s'entraîne déjà en quelques secondes a plus d'overhead de synchronisation que de gain réel — on préfère paralléliser les essais eux-mêmes (`OPTUNA_N_JOBS`).

### Modèle baseline (hyperparamètres par défaut raisonnables, sans tuning)

Sert de **point de comparaison** pour juger si le tuning Optuna apporte un gain réel, pas juste un chiffre plus élevé sur un run isolé.

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

### Rendu — tableau de référence

```python
reference_table = pd.DataFrame([
    {"modèle": "baseline (défauts raisonnables)", "PR-AUC (val)": baseline_prauc, "AUC (val)": baseline_auc},
])
```

---

## 2-3. Espace de recherche, sampler et pruner — rappel des choix

- **Sampler** : `TPESampler` — bon compromis exploration/exploitation pour un espace de taille moyenne (9 hyperparamètres).
- **Pruner** : `MedianPruner` — arrête un essai si sa performance intermédiaire est sous la médiane des essais précédents au même stade.
- **Pruning appliqué au niveau des folds de validation croisée**, pas à chaque itération de boosting individuelle : après chaque fold, on rapporte la PR-AUC moyenne courante à Optuna, qui décide de continuer ou d'arrêter l'essai avant le fold suivant. Choix délibéré — le callback officiel `XGBoostPruningCallback` (rapporte à chaque itération de boosting) entre en conflit avec une CV à plusieurs folds (chaque fold repart de l'itération 0, ce qui perturbe le suivi des étapes côté Optuna — observé en pratique, pas juste supposé). Le pruning par fold reste piloté par la progression réelle de l'entraînement XGBoost, tout en restant robuste.

---

## 4. Espace de recherche borné pour XGBoost

Bornes choisies en fonction du contexte (134k lignes, ~71 features, fort déséquilibre de classe), pas des plages génériques.

| Hyperparamètre | Plage | Justification |
|---|---|---|
| `n_estimators` | 100 – 300 | Borné haut par l'early stopping interne ; resserré à 300 pour le temps de calcul |
| `max_depth` | 3 – 8 | Au-delà de 8, sur-apprentissage quasi systématique sur ce volume |
| `learning_rate` | 0.01 – 0.3 (log) | Un ordre de grandeur en dessous/au-dessus de la valeur usuelle 0.1 |
| `subsample` | 0.6 – 1.0 | Sous 0.6, trop de variance |
| `colsample_bytree` | 0.6 – 1.0 | Idem, par colonne |
| `min_child_weight` | 1 – 10 | Régularise la taille minimale des feuilles |
| `reg_lambda` | 1e-3 – 10 (log) | L2, de quasi rien à une régularisation forte |
| `reg_alpha` | 1e-3 – 10 (log) | L1, idem |
| `scale_pos_weight` | 1 – ratio nég/pos | Borne haute = rééquilibrage complet |

### Pourquoi borner plutôt qu'ouvrir grand ?

1. **Un espace non borné n'est pas plus "objectif", juste plus coûteux à explorer** — Optuna (TPE) doit échantillonner suffisamment de points pour construire un modèle de densité fiable ; un espace 10x plus large avec le même budget d'essais donne une exploration 10x plus clairsemée, pas une meilleure garantie de trouver l'optimum.
2. **Les bornes encodent de la connaissance du domaine, pas un biais arbitraire** — ex. `max_depth > 10` sur 71 features et 94k lignes mène quasi systématiquement à du sur-apprentissage.
3. **Un espace trop large complique aussi l'analyse a posteriori** — importance des HP et slice plots plus lisibles avec des plages resserrées.
4. **Le risque inverse existe aussi** : des bornes trop resserrées peuvent exclure la vraie meilleure valeur — d'où des plages larges d'un facteur ~5-10x autour des valeurs usuelles plutôt que des plages étroites centrées sur une seule hypothèse.

En résumé : borner n'est pas une contrainte qu'on subit, c'est un choix de conception qui rend la recherche plus efficace **et** plus interprétable.

### Fonction objectif : PR-AUC moyenne en validation croisée temporelle

**Optimisation temps de traitement (2/5)** : sous-échantillonnage pour la recherche uniquement (stride systématique, préserve l'ordre temporel) — seul le modèle final est réentraîné sur l'intégralité des données.

```python
N_SPLITS = 3
SEARCH_SAMPLE_FRACTION = 0.5   # 1.0 = pas de sous-échantillonnage
_stride = max(1, round(1 / SEARCH_SAMPLE_FRACTION))

X_train_arr = X_train.values[::_stride]
y_train_arr = y_train.values[::_stride]

def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),   # optimisation (4/5) : borné à 300, pas 600
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

        # Pruning au niveau du fold
        trial.report(float(np.mean(fold_scores)), step=fold_i)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(fold_scores))
```

---

## 5. Étude Optuna : TPE + pruning + budget borné, seed fixée

**Reproductibilité** : `TPESampler(seed=SEED)` — deux exécutions avec le même seed proposent la même séquence d'essais.

**Budget borné** : `n_trials` **et** `timeout` — le premier des deux atteints arrête l'étude, pour qu'un poste plus lent ou plus rapide ne change pas silencieusement la portée de la recherche.

**Suivi MLflow par essai** : chaque essai Optuna est loggé comme un run MLflow distinct.

**Optimisation temps de traitement (5/5)** : paralléliser les essais (`OPTUNA_N_JOBS`) plutôt que les threads XGBoost — réglé prudemment à `1` par défaut :
1. **MLflow + SQLite gère mal les écritures concurrentes** — plusieurs essais en parallèle qui loggent en même temps peuvent provoquer des erreurs `database is locked`.
2. **Cohérence avec `XGB_N_JOBS_SEARCH=1`** — sur une machine à N cœurs, mettre `OPTUNA_N_JOBS = N_CORES` pour en tirer parti.

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

### Le meilleur essai bat (ou non) la baseline — les deux cas sont instructifs

```python
best_model = xgb.XGBClassifier(**study.best_params, tree_method="hist", n_jobs=1,
                                 eval_metric="aucpr", random_state=SEED)
best_model.fit(X_train, y_train)
val_proba_best = best_model.predict_proba(X_val)[:, 1]
best_prauc_val = average_precision_score(y_val, val_proba_best)
gain_prauc = best_prauc_val - baseline_prauc
```

**Lecture — les deux cas de figure sont instructifs :**

- **Si le meilleur essai bat nettement la baseline** : le tuning apporte une vraie valeur — les hyperparamètres par défaut n'étaient pas adaptés à ce dataset précis. Vérifier ensuite que le gain se confirme sur le test (étape 7) avant de le croire définitivement.
- **Si le meilleur essai ne bat pas (ou à peine) la baseline** : **pas un échec de la méthode** — le modèle est déjà proche de son plafond de performance avec des hyperparamètres raisonnables, les gaps restants viennent probablement d'ailleurs (feature engineering, volume de données, qualité des labels). Continuer à chercher un budget plus grand serait alors un mauvais usage du temps de calcul.

### À partir de quand l'amélioration ne « vaut » plus le calcul dépensé ?

Pas de seuil universel, mais des repères concrets :

1. **Rendement décroissant visible sur l'historique d'optimisation** : si les 10 derniers essais n'améliorent plus le meilleur score notablement, le budget restant a de fortes chances d'être dépensé pour un gain marginal.
2. **Comparer le gain au bruit de mesure** : si le gain observé est du même ordre que la variance entre folds pour un même essai, ce n'est probablement pas un vrai gain.
3. **Mettre en regard le coût réel** : chaque essai coûte ~3 entraînements XGBoost. 50 essais représentent déjà un budget significatif — le multiplier par 10 pour +0.005 PR-AUC n'est généralement pas justifiable sauf enjeu métier explicite.
4. **Règle pragmatique** : arrêter d'investir dès que le gain incrémental par essai devient plus petit que l'incertitude de mesure elle-même — au-delà, on optimise du bruit.

---

## 6. Analyse post-optimisation

Comprendre *pourquoi* une combinaison marche, pas seulement laquelle.

### Historique d'optimisation

```python
ax = ovm.plot_optimization_history(study)
```

### Importance des hyperparamètres

```python
importances = optuna.importance.get_param_importances(study)
ax = ovm.plot_param_importances(study)
```

### Slice plot — 2-3 hyperparamètres dominants

Affiche, pour chaque essai, la valeur de l'hyperparamètre en abscisse et la PR-AUC en ordonnée — permet de voir si une zone de valeurs se démarque clairement ou si le nuage est dispersé.

```python
top_hps = list(importances.keys())[:3]
axes = ovm.plot_slice(study, params=top_hps)
```

### Lecture de la stabilité (variance entre essais proches) et des HP qui comptent vraiment

```python
top10_values = values[np.argsort(values)[-10:]]
# Comparer top10_values.std() au gain_prauc observé
```

**Point d'attention — un écart `val` flatteur mais une importance d'HP erratique = signe de sur-ajustement du tuning**

Si le meilleur essai affiche un gain net, mais que l'importance des HP ne dégage aucun hyperparamètre clairement dominant et/ou que les meilleurs essais ne partagent aucune zone de valeurs commune dans le slice plot — alors Optuna n'a probablement pas trouvé une vraie zone de bons hyperparamètres, mais a **exploité une combinaison chanceuse sur ce split de validation précis**. C'est un sur-ajustement d'un niveau différent de celui d'un modèle individuel : c'est le **processus de recherche lui-même** qui s'est ajusté à la spécificité d'un split.

**Exemple concret** : 50 essais où le meilleur obtient PR-AUC=0.42 avec `max_depth=7`, le 2ᵉ meilleur (0.419, quasi identique) a `max_depth=3`, le 3ᵉ meilleur `max_depth=8` avec un `learning_rate` dix fois plus élevé. Si l'importance de `max_depth` ressort pourtant comme dominante (fANOVA capte une variance globale sur tout l'espace, y compris les mauvais essais), c'est trompeur : les **meilleurs** essais ne convergent vers aucune valeur commune, signal que le gain est probablement un artefact du split plutôt qu'un optimum robuste.

**Comment s'en prémunir** :
1. Ne jamais choisir le HP sur la seule valeur du meilleur essai — regarder si les 5-10 meilleurs convergent vers une zone cohérente.
2. Revalider avec plusieurs seeds ou plusieurs splits avant de figer un choix.
3. Se méfier d'un espace de recherche trop large par rapport au nombre d'essais (multiple testing implicite).

---

## 7. Validation finale — pas de sur-ajustement au tuning

1. Réentraîner avec les meilleurs hyperparamètres sur **train + validation combinés**.
2. Évaluer **une seule fois** sur le **test set**, jamais vu ni pendant le tuning ni pendant la sélection du meilleur essai.
3. Comparer explicitement à la baseline réentraînée de la même façon.

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
# Si l'écart entre gain_test_prauc et le gain observé en validation dépasse 50% relatif :
# signal de sur-ajustement du tuning à surveiller.
```

---

## 8. Temps de traitement global

```python
_total_elapsed = time.time() - _notebook_start_time
```

Chronométrage global (notebook entier) et détail de l'étude Optuna (`_study_elapsed`) affichés séparément.
