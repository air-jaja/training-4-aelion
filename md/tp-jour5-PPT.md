<!-- marp: true -->
<!-- theme: gaia -->
<!-- paginate: true -->

<!-- _class: lead -->

# **Algorithmes de Prédiction de Pannes**
## Horizon 24h sur Dataset Complexe avec Séries Temporelles

**Contexte** : Dataset prêt à l'entraînement avec séries temporelles et multiples features pour anticiper les défaillances.

---

# **1. Algorithmes Basés sur les Arbres**
### Random Forest & XGBoost

#### **Principe**
- **Random Forest** : Ensemble d'arbres de décision entraînés sur des sous-échantillons aléatoires (bagging).
- **XGBoost** : Boosting extrême (arbres séquentiels) avec régularisation pour éviter le surapprentissage.
- **Adaptés** : Classification binaire (panne vs. non-panne) avec features temporelles agrégées.

#### **Avantages**
✅ Robustesse aux outliers et aux données bruitées
✅ Gestion native des features catégorielles (XGBoost)
✅ Interprétabilité relative (importance des features)
✅ Performances élevées avec peu de réglages

#### **Inconvénients**
❌ Moins adapté aux **dépendances temporelles longues** (nécessite feature engineering manuel)
❌ Complexité croissante avec le nombre de features

#### **Bonnes Pratiques**
- **Feature Engineering** : Créer des lag features (valeurs passées), rolling statistics (moyenne mobile, écart-type).
- **Hyperparamètres** : `max_depth`, `n_estimators`, `learning_rate` (XGBoost).
- **Validation** : Time-based split (pas de shuffle aléatoire).

---

# **2. Réseaux de Neurones Récurrents (RNN)**
### LSTM & GRU

#### **Principe**
- **LSTM** (Long Short-Term Memory) : Cellules mémorielles avec portes (input, forget, output) pour capturer les dépendances temporelles.
- **GRU** : Variante simplifiée (moins de paramètres) avec portes update/reset.
- **Idéal** : Modélisation directe des séries temporelles **sans feature engineering manuel**.

#### **Avantages**
- ✅ Capture automatique des **motifs temporels complexes** (séquences longues)
- ✅ Adapté aux données **multivariées** (plusieurs features simultanées)
- ✅ Flexibilité architecturale (couches empilées, attention mechanisms)

#### **Inconvénients**
❌ **Données massives** requises pour éviter le surapprentissage
❌ Lents à entraîner / Coûteux en ressources
❌ **Boîte noire** (difficile à interpréter)

#### **Bonnes Pratiques**
- **Prétraitement** : Normalisation (`MinMaxScaler` ou `StandardScaler`).
- **Architecture** : 1-2 couches LSTM (50-200 unités), dropout (0.2-0.5).
- **Séquences** : Fenêtre glissante (`window_size=24` pour horizon 24h).
- **Optimiseur** : Adam avec `learning_rate=0.001`.
- **Early Stopping** : Surveillance de la validation loss.

---

# **3. Détection d'Anomalies & Autres Approches**

### **Isolation Forest**
- **Principe** : Isole les anomalies (pannes) en les identifiant comme "faciles à isoler" dans un espace de features.
- **Adapté** : Détection non supervisée si peu de labels disponibles.

#### **Avantages**
- ✅ **Non supervisé** (pas besoin de labels de pannes)
- ✅ Très rapide à entraîner
- ✅ Efficace pour les données **hautement dimensionnelles**

#### **Inconvénients**
- ❌ Moins performant si les pannes **ne sont pas des anomalies** (ex : usure normale)
- ❌ Difficile à régler pour des seuils précis

#### **Bonnes Pratiques**
- **Paramètre** : `contamination` (proportion attendue d'anomalies).
- **Features** : Utiliser des statistiques temporelles (moyenne, variance sur N heures).
- **Combinaison** : Post-traitement avec des règles métiers.

### **SVM (Support Vector Machines)**
- **Principe** : Trouve l'hyperplan optimal séparant les classes (pannes vs. non-pannes).
- **Kernel** : Linéaire ou RBF pour les données non linéaires.

#### **Avantages**
- ✅ Efficace en **haute dimension** (beaucoup de features)
- ✅ Robuste au surapprentissage (avec régularisation)

#### **Inconvénients**
- ❌ **Sensible au scaling** des données (normalisation obligatoire)
- ❌ Moins adapté aux **séries temporelles brutes** (nécessite feature engineering)

---

# **4. Synthèse & Recommandations**

| **Algorithme**       | **Type**          | **Force**                          | **Faiblesse**                     | **Cas d'Usage**                     |
|----------------------|-------------------|------------------------------------|-----------------------------------|-------------------------------------|
| **XGBoost**          | Classification    | Robuste, interprétable             | Feature engineering manuel       | Features agrégées, peu de données   |
| **LSTM**             | RNN               | Modélise le temps nativement       | Données massives requises         | Séries temporelles brutes           |
| **Isolation Forest** | Détection         | Non supervisé, rapide              | Pannes ≠ anomalies                 | Peu de labels, exploration           |
| **SVM**              | Classification    | Haute dimension                    | Sensible au scaling                | Features bien préparées             |

### **Recommandation pour votre Dataset**
1. **Commencer par XGBoost** : Baseline solide avec feature engineering temporel.
2. **Tester LSTM** : Si données suffisantes et motifs temporels complexes.
3. **Combiner** : Ensemble methods (ex : XGBoost + LSTM) pour maximiser la performance.
4. **Validation** : Toujours utiliser un **time-based split** (train/test chronologique).

---

# **5. Prochaines Étapes**

### **Plan d'Action**
1. **Implémenter XGBoost** avec feature engineering (lags, rolling stats).
2. **Benchmark** : Comparer avec LSTM sur un sous-ensemble de données.
3. **Optimiser** : Hyperparamètres via `Optuna` ou `GridSearchCV`.
4. **Déployer** : Modèle le plus performant avec monitoring des prédictions.

### **Ressources Utiles**
- [Guide Algorithmes ML - Jedha](https://www.jedha.co/formation-ia/algorithmes-machine-learning)
- [Types de ML - IBM](https://www.ibm.com/fr-fr/think/topics/machine-learning-types)
- [Cas d'Usage ML - DataCamp](https://www.datacamp.com/fr/blog/top-machine-learning-use-cases-and-algorithms)
