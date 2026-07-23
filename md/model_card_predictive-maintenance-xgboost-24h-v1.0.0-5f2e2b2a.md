---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
language: fr
license: proprietary
tags:
  - tabular-classification
  - predictive-maintenance
  - xgboost
model_id: predictive-maintenance-xgboost-24h-v1.0.0-5f2e2b2a
model_version: v1.0.0
---

# Model Card for predictive-maintenance-xgboost-24h-v1.0.0-5f2e2b2a

<!-- Provide a quick summary of what the model is/does. -->

Ce modèle prédit si une machine industrielle risque de tomber en panne dans les **24 prochaines heures**, à partir de mesures de capteurs (température, tension électrique, vitesse de rotation, pression, volume produit). Il est destiné à déclencher des alertes de maintenance préventive, pas à remplacer une inspection humaine.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

Modèle de classification binaire (XGBoost) entraîné pour prédire la probabilité qu'une machine tombe en panne dans les 24 heures suivant une mesure, à partir de 71 variables décrivant l'état récent de la machine (températures, tensions, vibrations, pression, volume de production, historique de maintenance à moyen terme). Le modèle a été sélectionné après une recherche raisonnée d'hyperparamètres (Optuna, voir section Training Procedure), puis documenté avec une analyse d'explicabilité (SHAP) et une mesure de son coût énergétique d'entraînement (CodeCarbon).

- **Developed by:** Équipe data / IA — projet de maintenance prédictive (voir Model Card Authors)
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** [More Information Needed]
- **Model type:** Classification binaire supervisée sur données tabulaires (gradient boosting, XGBoost)
- **Language(s) (NLP):** Non applicable — modèle sur données numériques de capteurs, pas de traitement de langage naturel
- **License:** À définir selon la politique de l'organisation (proprietary par défaut)
- **Finetuned from model [optional]:** Aucun — entraîné from scratch, pas un modèle pré-entraîné affiné

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** training-4-aelion (branche develop)
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

Ce modèle est destiné à être utilisé **tel quel** pour générer un score de risque de panne à 24h par machine et par heure, à partir des mêmes variables que celles utilisées à l'entraînement (mêmes capteurs, même fréquence de mesure). Usage prévu : alimenter un tableau de bord de maintenance préventive, ou déclencher une alerte à destination d'un technicien lorsque le score dépasse le seuil calibré (voir section Evaluation). **Le modèle ne doit jamais prendre seul une décision d'arrêt de machine** — il sert à prioriser l'attention humaine, pas à s'y substituer.

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

Ré-entraînement possible sur un parc de machines différent, à condition de vérifier que la distribution des capteurs et le taux de pannes restent comparables au dataset d'origine (voir Bias, Risks, and Limitations). Un ajustement du seuil de décision (actuellement calibré au 95e percentile des scores sur machines saines) est recommandé si le contexte opérationnel change (coût d'une panne manquée vs coût d'une fausse alerte).

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

**Ce modèle ne doit pas être utilisé** :

- Pour des machines d'un type différent de celles du parc d'entraînement (autres capteurs, autre régime de fonctionnement), sans revalidation complète — aucune garantie de généralisation à un contexte industriel différent.
- Pour prédire un horizon différent de 24h sans ré-entraînement spécifique (un modèle entraîné sur l'horizon 24h n'est pas fiable pour prédire à 6h ou 48h — voir les notebooks d'exploration des autres horizons).
- Comme seule source de décision pour arrêter une machine, engager une intervention coûteuse, ou toute décision ayant un impact sur la sécurité des personnes — le rappel du modèle est **partiel** (voir Limitations), une partie des pannes réelles ne sera pas détectée.
- En dehors du contexte de maintenance industrielle pour lequel il a été conçu (aucune vocation à être réutilisé sur un autre type de problème de classification, même à structure de données similaire, sans revalidation).

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->


**Signal faible et diffus.** L'analyse d'explicabilité (SHAP) montre qu'aucune variable ne domine la décision du modèle : les 5 variables les plus influentes n'expliquent qu'une partie modérée de l'impact total, et il faut regrouper une dizaine de variables pour couvrir environ 54% de l'impact (temp_mean_24h, pressure_mean_24h, temp_max_24h, entre autres). Autrement dit, le modèle combine de nombreux indices faibles plutôt que de s'appuyer sur un signal fort et unique — un profil cohérent avec la nature progressive d'une dégradation mécanique, mais qui limite la performance atteignable.

**Faux négatifs (pannes manquées).** Au seuil de décision retenu, le modèle détecte environ 31% des vraies pannes sur le jeu de test — ce qui signifie qu'environ 69% des pannes réelles à 24h **ne sont pas détectées** par le modèle. C'est la limitation la plus importante à communiquer à tout utilisateur : ce modèle réduit le risque, il ne l'élimine pas.

**Faux positifs (fausses alertes).** À l'inverse, environ 6% des machines saines déclenchent une alerte à tort à ce même seuil — un coût opérationnel (temps d'inspection sans panne trouvée) à mettre en balance avec le coût d'une panne manquée.

**Instabilité observée lors des tests de robustesse.** Des essais antérieurs sur ce même type de modèle ont montré que les résultats (quel algorithme domine, quelle importance de variable ressort) peuvent varier sensiblement selon la taille du split de validation et l'initialisation aléatoire — un signal de prudence sur la robustesse du classement exact des variables, même si le modèle retenu ici reste cohérent sur ses métriques globales.

**Portée des données d'entraînement.** Le modèle n'a vu que les machines, la période et les conditions opérationnelles présentes dans `gold_dataset`. Aucune garantie de performance sur un autre parc de machines, une autre période de l'année, ou des conditions de fonctionnement inhabituelles (arrêts prolongés, changements de process) non représentées dans les données d'entraînement.


### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Utiliser ce modèle comme **aide à la priorisation**, jamais comme décision automatique unique. Toute alerte doit être confirmée par une inspection humaine avant action. Le seuil de décision doit être révisé périodiquement (nouvelles données, changement de coût métier entre panne manquée et fausse alerte). Un ré-entraînement périodique est recommandé pour éviter une dérive du modèle si les conditions opérationnelles évoluent. Ne pas extrapoler la performance mesurée ici à un parc de machines ou un horizon de prédiction différents sans revalidation.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
import xgboost as xgb

model = xgb.XGBClassifier()
model.load_model("predictive-maintenance-xgboost-24h-v1.0.0-5f2e2b2a.json")

# X : DataFrame avec les mêmes 71 colonnes que celles utilisées à l'entraînement
proba = model.predict_proba(X)[:, 1]
alerte = proba > 0.3941  # seuil calibré, voir section Evaluation
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

`gold_dataset` — mesures capteurs agrégées par machine et par heure (température, tension, vitesse de rotation, pression, volume produit, taux d'utilisation), avec un historique de maintenance à moyen terme. 93990 lignes d'entraînement, 20145 de validation, 20145 de test, découpées dans le temps (le test correspond toujours à la période la plus récente, jamais mélangée avec l'entraînement). 71 variables conservées après exclusion des colonnes qui donneraient au modèle une information sur le futur (fuite de données) — voir la checklist anti-fuite du projet.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

Exclusion systématique des identifiants, des colonnes dérivées du futur par construction, et des labels des autres horizons de prédiction (6h, 12h, 48h). Pas de normalisation des variables (XGBoost n'en a pas besoin, à la différence des réseaux de neurones). Rééquilibrage du déséquilibre de classe géré par le poids `scale_pos_weight`, pas par sur/sous-échantillonnage des données.


#### Training Hyperparameters

- **Training regime:** CPU, précision flottante standard (fp32) — pas d'entraînement sur GPU, pas de calcul en précision réduite. <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

Entraînement du modèle final : 4.83 secondes sur un seul cœur de processeur (0.000018 kWh consommés). Les hyperparamètres eux-mêmes ont été obtenus via une recherche Optuna antérieure (20 essais, ~3 minutes) — non refaite ici, réutilisée telle quelle (voir `optuna_xgboost_predictive_maintenance.ipynb`).

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

Sous-ensemble `test` de `gold_dataset` (20145 lignes), correspondant à la période la plus récente, jamais utilisée ni pendant l'entraînement ni pendant la calibration du seuil de décision.

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

Évaluation globale sur l'ensemble du jeu de test — pas de décomposition par machine individuelle ou par sous-période dans cette version de la model card. Un futur travail pourrait vérifier la stabilité de la performance par machine.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

**PR-AUC** (aire sous la courbe précision-rappel) — métrique principale, adaptée au déséquilibre de classe (pannes rares). **AUC-ROC** — métrique complémentaire, moins sensible au déséquilibre. **Rappel, précision et taux de faux positifs** au seuil de décision retenu — pour une lecture opérationnelle directe (combien de pannes détectées, combien de fausses alertes).

### Results


| Métrique | Valeur (jeu de test) |
|---|---|
| PR-AUC | 0.4552 |
| AUC-ROC | 0.6938 |
| Seuil de décision | 0.3941 (percentile 95 des scores sur machines saines de validation) |
| Rappel (pannes détectées) | 30.7% |
| Précision (alertes confirmées) | 52.8% |
| Faux positifs (fausses alertes) | 5.7% |


#### Summary

Le modèle détecte environ 31% des pannes réelles à 24h, avec un taux de fausses alertes d'environ 6% sur les machines saines, au seuil retenu. La performance est modeste mais honnête au regard du signal disponible (voir Model Examination) — un compromis assumé plutôt qu'une sur-promesse.

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->


Une analyse d'explicabilité (SHAP, voir `shap_explainability_ml.ipynb`) a été menée sur ce modèle :

- **Variables dominantes** : temp_mean_24h, pressure_mean_24h, temp_max_24h, voltage_mean_24h, rotation_max_24h — des agrégats sur 24h de température, tension et rotation, cohérents avec une dégradation mécanique progressive plutôt qu'un pic isolé.
- **Concentration de l'impact** : diffus plutôt que concentré — les 10 premières variables couvrent environ 54% de l'impact total sur les prédictions, aucune variable ne domine à elle seule.
- **Vérification anti-fuite** : aucune des variables dominantes ne présente de corrélation suspecte avec les colonnes explicitement exclues pour fuite de données — les variables qui pilotent la décision sont mécaniquement plausibles, pas des raccourcis statistiques douteux.
- **Cohérence métier** : les variables dominantes correspondent à des mesures capteur reconnaissables (température, tension, rotation, pression, volume produit), pas à des artefacts de construction du dataset.


## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** CPU (calcul mono-cœur, pas de GPU utilisé)
- **Hours used:** 0.00134 heures (4.83 secondes) pour l'entraînement du modèle final
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** France (intensité carbone du calcul forcée sur le mix électrique français, voir notebook éco-conception)
- **Carbon Emitted:** 0.0010 grammes de CO2 équivalent (mesuré avec CodeCarbon, 0.000018 kWh consommés)

## Technical Specifications [optional]

### Model Architecture and Objective

XGBoost (gradient boosting sur arbres de décision) — n_estimators=135, max_depth=3, learning_rate=0.1399. Objectif : classification binaire (panne / pas de panne à 24h), loss = log-vraisemblance binaire pondérée (`scale_pos_weight=2.127` pour compenser le déséquilibre de classe).

### Compute Infrastructure

Poste de calcul standard (CPU), pas d'infrastructure distribuée ni de GPU nécessaire pour ce type de modèle.

#### Hardware

Minimal — inférence quasi instantanée sur CPU, pas de carte graphique requise, empreinte mémoire réduite (modèle < 1 Mo).

#### Software

Python, XGBoost, scikit-learn, Optuna (recherche d'hyperparamètres), SHAP (explicabilité), CodeCarbon (mesure d'impact), MLflow (suivi des expériences).

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```bibtex
@misc{predictive_maintenance_xgboost_24h_v1.0.0_5f2e2b2a,
  title = {Predictive Maintenance XGBoost Model (24h horizon)},
  author = {[More Information Needed]},
  year = {2026},
  howpublished = {Internal model card},
  note = {Model ID: predictive-maintenance-xgboost-24h-v1.0.0-5f2e2b2a}
}
```

**APA:**

[More Information Needed] (2026). Predictive Maintenance XGBoost Model (24h horizon), predictive-maintenance-xgboost-24h-v1.0.0-5f2e2b2a.

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->


- **PR-AUC** : aire sous la courbe précision-rappel — mesure la capacité à repérer les cas rares (pannes) sans trop de fausses alertes.
- **AUC-ROC** : aire sous la courbe ROC — mesure classique de séparation entre classes, moins sensible au déséquilibre.
- **Rappel** : proportion de vraies pannes effectivement détectées par le modèle.
- **Précision** : proportion d'alertes du modèle réellement suivies d'une panne.
- **Faux positif** : une alerte déclenchée sur une machine qui, en réalité, ne tombera pas en panne.
- **Faux négatif** : une panne réelle que le modèle n'a pas détectée.
- **SHAP** : méthode d'explicabilité qui attribue à chaque variable sa contribution à une prédiction donnée.
- **Fuite de données (data leakage)** : quand une variable d'entraînement contient, par erreur, une information sur le futur que le modèle ne devrait pas connaître au moment de la prédiction.
- **kWh / gCO2eq** : unités de consommation électrique et d'impact climatique équivalent.


## More Information [optional]

Voir les notebooks associés du projet : `optuna_xgboost_predictive_maintenance.ipynb` (recherche d'hyperparamètres), `eco_conception_codecarbon.ipynb` (mesure d'impact comparée ML/DL), `shap_explainability_ml.ipynb` (explicabilité détaillée).

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]