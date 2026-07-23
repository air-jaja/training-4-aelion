# Optimisation raisonnée avec Optuna & Éco-conception — Document de synthèse complet

Ce document consolide l'ensemble des échanges depuis la demande initiale sur l'optimisation raisonnée avec Optuna : la démarche proposée, son implémentation détaillée dans un notebook, l'optimisation du temps de traitement, puis son extension à l'éco-conception avec CodeCarbon (ML & DL). Pas de code ici — uniquement les choix faits et leur justification.

## Sommaire

1. [Démarche proposée initialement](#1-démarche-proposée-initialement)
2. [Notebook 1 — Optimisation Optuna/XGBoost](#2-notebook-1--optimisation-optunaxgboost)
3. [Optimisation du temps de traitement](#3-optimisation-du-temps-de-traitement)
4. [Notebook 2 — Éco-conception avec CodeCarbon](#4-notebook-2--éco-conception-avec-codecarbon)
5. [Résultats réels obtenus lors des tests](#5-résultats-réels-obtenus-lors-des-tests)
6. [Glossaire accessible (non-spécialiste)](#6-glossaire-accessible-non-spécialiste)
7. [Fichiers livrés](#7-fichiers-livrés)

---

## 1. Démarche proposée initialement

Point de départ : construire une optimisation d'hyperparamètres **raisonnée**, pas une recherche aveugle sur un grand espace. Huit étapes proposées :

1. **Cadrer l'objectif avant de toucher à Optuna** — un seul modèle et un seul horizon (XGBoost, horizon 24h), une métrique adaptée au déséquilibre de classe (PR-AUC plutôt qu'accuracy), un protocole de validation respectant l'ordre temporel.
2. **Espace de recherche réfléchi, pas exhaustif** — hyperparamètres à fort impact connu, bornes informées par le contexte du dataset plutôt que des plages génériques.
3. **Sampler et pruner adaptés** — un algorithme de recherche qui apprend des essais précédents pour proposer intelligemment le suivant, et un mécanisme qui arrête tôt les essais peu prometteurs.
4. **Fonction objectif propre** — construit les hyperparamètres à tester, entraîne sur des découpages temporels des données, retourne une métrique moyenne, sans jamais toucher au jeu de test final.
5. **Budget d'essais raisonné** — un nombre maximal d'essais et/ou une limite de temps fixés à l'avance, pas au feeling.
6. **Suivi et traçabilité (MLflow)** — chaque essai enregistré, comparable dans une interface dédiée.
7. **Analyse post-optimisation** — historique des essais, importance de chaque réglage, vérifier que le meilleur essai n'est pas un cas isolé.
8. **Validation finale, pas de sur-ajustement au tuning** — réentraînement sur l'ensemble des données disponibles avant le test, évaluation unique, comparaison honnête à la baseline.

---

## 2. Notebook 1 — Optimisation Optuna/XGBoost

Fichier : `optuna_xgboost_predictive_maintenance.ipynb`. Implémentation détaillée des 8 points, avec des choix explicitement demandés et justifiés à chaque étape.

### 2.1 Cadrage et baseline

- **Modèle** : XGBoost, un algorithme qui combine de nombreuses petites règles de décision.
- **Horizon** : prédiction de panne à 24h (~17 % de cas positifs dans les données).
- **Métrique** : PR-AUC plutôt que le pourcentage de bonnes réponses (accuracy) — cohérente avec le déséquilibre de classe, qui rendrait l'accuracy trompeuse.
- **Protocole** : découpage des données en respectant strictement l'ordre chronologique, jamais d'information du futur utilisée pour prédire le passé.
- **Features** : mêmes exclusions que le reste du projet (identifiants, colonnes qui renseigneraient sur le futur, étiquettes des autres horizons — 71 variables au total conservées).

**Rendu demandé** : un tableau de référence `modèle baseline → PR-AUC (validation), AUC (validation)`, obtenu avec des réglages par défaut raisonnables, sans aucune recherche d'hyperparamètres — sert de point de comparaison pour juger si Optuna apporte un gain réel.

**Résultat réel obtenu** :

| Modèle | PR-AUC (validation) | AUC (validation) |
|---|---|---|
| Baseline (réglages raisonnables) | 0.4330 | 0.6881 |

### 2.2 Espace de recherche, sampler, pruner (points 2-3, validés tels quels)

- **Sampler** : l'algorithme qui propose intelligemment la prochaine combinaison de réglages à tester, en s'appuyant sur les essais précédents plutôt qu'en tirant au hasard.
- **Pruner** : le mécanisme qui arrête un essai avant sa fin s'il est déjà clairement moins bon que les précédents. Appliqué ici **au niveau de chaque bloc de validation croisée** (pas à chaque étape interne de l'entraînement du modèle) — choix fait après avoir constaté que la méthode officielle prévue pour un suivi plus fin entrait en conflit avec la présence de plusieurs blocs de validation successifs (chaque bloc recommençant son propre comptage interne, ce qui brouille le suivi de progression côté Optuna — un problème rencontré concrètement en testant, pas anticipé à l'avance).

### 2.3 Espace borné pour XGBoost (point 4)

Neuf réglages (hyperparamètres) ont été retenus, chacun avec une plage de valeurs justifiée par le contexte du dataset (134 000 lignes, ~71 variables, fort déséquilibre de classe) :

- **Nombre d'arbres construits** : borné par le mécanisme d'arrêt anticipé (qui coupe l'entraînement dès que le modèle cesse de progresser), donc pas besoin d'aller très haut — resserré volontairement pour limiter le temps de calcul.
- **Profondeur maximale des arbres** : au-delà d'un certain seuil, le modèle apprend des règles trop spécifiques aux données déjà vues (sur-apprentissage quasi systématique sur ce volume de données).
- **Vitesse d'apprentissage** : une plage large, environ un facteur 30 entre le bas et le haut, centrée sur la valeur habituellement recommandée.
- **Fraction de lignes et de colonnes utilisées à chaque étape** : en dessous d'un certain seuil, le modèle devient trop instable d'un entraînement à l'autre.
- **Poids minimal requis pour couper une branche de décision** : évite que le modèle ne découpe les données sur du simple bruit statistique.
- **Deux niveaux de régularisation** (des freins qui empêchent le modèle de devenir trop complexe) : une plage large, de quasi aucune régularisation à une régularisation forte.
- **Poids donné à la classe minoritaire** (les pannes) : borné par le déséquilibre réel observé dans les données d'entraînement.

**Pourquoi borner plutôt qu'ouvrir grand** (réponse explicitement demandée) :
1. Un espace non borné n'est pas plus "objectif", juste plus coûteux à explorer — à budget d'essais égal, une recherche dans un espace deux fois plus large explore deux fois moins densément chaque zone, sans garantie supplémentaire de trouver le bon réglage.
2. Les bornes choisies reflètent une connaissance du domaine (ce qui marche généralement sur ce type et ce volume de données), pas un choix arbitraire.
3. Un espace trop large complique aussi l'analyse une fois la recherche terminée — les graphiques qui montrent l'importance de chaque réglage sont plus lisibles quand les plages testées restent resserrées autour de valeurs plausibles.
4. Le risque inverse existe aussi : des bornes trop serrées peuvent exclure la vraie meilleure valeur — d'où le choix de plages qui restent larges (facteur 5 à 10) autour des valeurs usuelles, plutôt que des plages étroites centrées sur une seule hypothèse.

### 2.4 Fonction objectif : PR-AUC moyenne en validation croisée temporelle (point 4)

Pour chaque essai, un modèle est entraîné avec une combinaison de réglages proposée par l'algorithme de recherche, puis évalué sur plusieurs blocs de données découpés dans le temps (un peu comme un examen blanc répété sur différentes périodes). La performance moyenne obtenue sur ces blocs sert de score à l'essai. Après chaque bloc, ce score intermédiaire est communiqué à Optuna, qui peut décider d'arrêter l'essai avant les blocs suivants s'il apparaît déjà clairement décevant.

### 2.5 Étude Optuna : recherche intelligente + arrêt anticipé + budget limité + reproductibilité (point 5)

- **Reproductibilité** : un nombre de départ fixé pour le générateur aléatoire — deux exécutions avec ce même nombre de départ proposent exactement la même séquence d'essais.
- **Budget borné** : un nombre maximal d'essais **et** une limite de temps, le premier des deux atteints arrête la recherche — pour qu'un ordinateur plus lent ou plus rapide ne change pas silencieusement l'ampleur de la recherche effectuée.
- **Suivi par essai** : chaque essai est enregistré individuellement (réglages testés + score obtenu), consultable et comparable après coup.

**Résultat réel obtenu** (run de validation, 20 essais maximum) :

- Étude terminée en 191,4 s (3,2 min) — 11 essais complets, **9 essais arrêtés en cours de route par le mécanisme d'arrêt anticipé**, sur 20 essais lancés au total.
- Meilleur essai : PR-AUC (validation croisée) = 0,4002.
- Réévalué sur la validation : **PR-AUC = 0,4523** contre 0,4330 pour la baseline.

| Modèle | PR-AUC (validation) | AUC (validation) |
|---|---|---|
| Baseline | 0.4330 | 0.6881 |
| Meilleur essai Optuna | 0.4523 | 0.7009 |

**Gain PR-AUC (validation) vs baseline : +0,0192 (+4,4 %).**

**Le meilleur essai bat (ou non) la baseline — les deux cas sont instructifs** (réponse demandée) :
- S'il bat nettement la baseline : la recherche de réglages apporte une vraie valeur — les réglages par défaut n'étaient pas adaptés à ce dataset précis. À confirmer ensuite sur le jeu de test.
- S'il ne bat pas (ou à peine) la baseline : **ce n'est pas un échec de la méthode** — le modèle est déjà proche de sa performance maximale avec des réglages raisonnables, et les limites restantes viennent probablement d'ailleurs (qualité des données, quantité de données, construction des variables) plutôt que du réglage fin des hyperparamètres.

Sur ce run, c'est le premier cas de figure : le gain est net et — comme le montre la section 2.7 ci-dessous — il ne s'agit pas d'un simple coup de chance.

**À partir de quand l'amélioration ne « vaut » plus le calcul dépensé ?** (réponse demandée) :
1. Rendement décroissant visible dans l'historique des essais — si les derniers essais n'améliorent plus le meilleur score de façon notable, le budget restant a de fortes chances d'être dépensé pour un gain marginal.
2. Comparer le gain observé au bruit de mesure naturel (la variance entre les différents blocs de validation pour un même essai) — un gain du même ordre de grandeur que ce bruit n'est probablement pas un vrai gain.
3. Mettre en regard le coût réel déjà engagé — chaque essai représente plusieurs entraînements complets du modèle ; multiplier ce budget par dix pour un gain minime n'est généralement pas justifiable, sauf enjeu métier explicite où chaque point de performance a une valeur monétaire connue.
4. Règle pragmatique : arrêter d'investir du temps de calcul en réglage dès que le gain apporté par un essai supplémentaire devient plus petit que l'incertitude de mesure elle-même — au-delà de ce point, on optimise du bruit statistique, pas de la vraie performance.

### 2.6 Suivi MLflow

Chaque essai est enregistré comme une entrée distincte (réglages testés, score obtenu), de même que la baseline, le meilleur essai réévalué sur la validation, et l'évaluation finale sur le test — tous consultables et comparables dans une interface dédiée après coup.

### 2.7 Analyse post-optimisation (point 7)

Rendus demandés et produits :
- **Historique des essais** : un graphique montrant l'évolution du meilleur score au fil du temps.
- **Importance des réglages** : quel réglage a le plus influencé le résultat final.
- **Graphique de détail sur les 2-3 réglages les plus influents** : montre, pour chaque valeur testée d'un réglage donné, quel score a été obtenu — permet de voir si une zone de valeurs se démarque clairement ou si le nuage de points est dispersé sans tendance nette.
- **Lecture de stabilité** : comparaison de la variabilité observée entre les meilleurs essais au gain obtenu par rapport à la baseline.

**Résultats réels obtenus :**

![Historique d'optimisation](figures/opt_history.png)

L'historique montre le meilleur score progresser puis se stabiliser — cohérent avec un espace de recherche borné qui ne laisse pas une infinité de combinaisons à explorer.

![Importance des hyperparamètres](figures/param_importances.png)

![Slice plot des hyperparamètres dominants](figures/slice_plot.png)

**Lecture de stabilité, chiffrée** :

| | Valeur |
|---|---|
| PR-AUC (CV) — moyenne des 10 meilleurs essais | 0.3917 |
| Écart-type des 10 meilleurs essais | 0.0064 |
| PR-AUC (CV) — moyenne sur tous les essais complets (n=11) | 0.3893 (± 0.0097) |
| Gain observé vs baseline | 0.0192 |

**Le gain observé (0,0192) dépasse nettement l'écart-type inter-essais du top 10 (0,0064)** — signal que ce gain est probablement réel, pas un artefact du hasard d'échantillonnage. C'est exactement le type de vérification décrite dans le point d'attention ci-dessous, appliqué ici à un cas où le signal se révèle solide.

**Point d'attention demandé — sur-ajustement du tuning, avec exemple** :

> Un écart flatteur en validation mais une importance des réglages erratique est un signe de sur-ajustement du processus de recherche lui-même. Si le meilleur essai affiche un gain net, mais qu'aucun réglage ne ressort clairement comme déterminant, et que les meilleurs essais ne partagent aucune zone de valeurs commune pour les réglages importants — alors la recherche a probablement exploité une combinaison chanceuse propre à ce découpage précis des données, plutôt que trouvé une vraie zone de bons réglages.
>
> **Exemple concret** : parmi 50 essais, le meilleur obtient un très bon score avec une profondeur d'arbre élevée, mais le deuxième meilleur essai (un score quasiment identique) utilise une profondeur d'arbre bien plus faible, et le troisième meilleur une profondeur différente encore avec une vitesse d'apprentissage dix fois supérieure aux deux autres. Si le graphique d'importance des réglages désigne pourtant la profondeur d'arbre comme le facteur dominant (parce que le calcul d'importance prend en compte tous les essais, y compris les mauvais), c'est trompeur : les **meilleurs** essais, eux, ne convergent vers aucune valeur commune — signe que le gain du meilleur essai est probablement un hasard lié à ce découpage particulier des données, pas un réglage réellement robuste.
>
> **Comment s'en prémunir** : ne jamais choisir un réglage sur la seule valeur du meilleur essai isolé ; vérifier que les 5 à 10 meilleurs essais convergent vers une zone de valeurs cohérente ; revalider avec plusieurs découpages différents des données avant de figer un choix ; se méfier d'un espace de recherche trop large par rapport au nombre d'essais réalisés (plus l'espace est grand, plus il est facile de "trouver" par hasard une combinaison qui a l'air excellente sans l'être vraiment).

### 2.8 Validation finale (point 8)

1. Réentraînement du modèle, avec les meilleurs réglages trouvés, sur l'ensemble des données disponibles avant le jeu de test (entraînement + validation réunis).
2. Évaluation **une seule fois** sur le jeu de test, jamais utilisé ni pendant la recherche de réglages, ni pendant la comparaison à la baseline.
3. Comparaison explicite à la baseline, réentraînée de la même façon, pour un chiffre final honnête.
4. Vérification de cohérence : si l'écart entre le gain observé en validation et celui observé sur le test est important, c'est un signal de sur-ajustement du processus de tuning à prendre au sérieux.

**Résultat réel obtenu :**

| | Gain PR-AUC observé |
|---|---|
| En validation (étape 2.5) | +0,0192 (+4,4 %) |
| Sur le test, après réentraînement final | **+0,0169 (+3,8 %)** |

Le gain se confirme de façon cohérente entre validation et test (écart relatif faible, pas de chute brutale) — signal fiable : ce n'est pas un cas de sur-ajustement du tuning, le gain trouvé se généralise bien à des données jamais vues pendant toute la démarche.

---

## 3. Optimisation du temps de traitement

Suite à la question *"as-tu optimisé le temps de traitement du notebook ?"* — réponse initiale : non, réglages par défaut raisonnables seulement. Cinq optimisations identifiées puis implémentées et testées :

1. **Répartition du calcul entre cœurs du processeur adaptée au contexte** : tous les cœurs disponibles pour les entraînements isolés (baseline, modèle final, un seul entraînement à la fois) ; un seul cœur par entraînement pendant la recherche de réglages, où l'on préfère faire tourner plusieurs essais en parallèle plutôt que d'accélérer chaque essai individuellement (moins de gaspillage lié à la coordination entre cœurs pour des entraînements déjà rapides).
2. **Réduction du volume de données utilisé pendant la recherche uniquement** (garder une ligne sur deux, réparties sur toute la période pour ne pas casser l'ordre chronologique) — seul le modèle final est réentraîné sur l'intégralité des données disponibles.
3. *(Option documentée, non appliquée par défaut)* réduire le nombre de blocs de validation croisée.
4. **Nombre maximal d'arbres réduit de moitié** par rapport à la première version — le mécanisme d'arrêt anticipé coupe presque toujours l'entraînement bien avant cette limite de toute façon (observé en pratique : l'arrêt intervient généralement bien avant le maximum autorisé).
5. **Parallélisation des essais eux-mêmes** plutôt que du calcul interne de chaque essai — réglée prudemment à un seul essai à la fois par défaut, en raison d'un risque de conflit d'écriture si plusieurs essais tentent d'enregistrer leurs résultats simultanément dans le même fichier de suivi ; à augmenter en connaissance de cause sur une machine disposant de plusieurs cœurs.

**Gain mesuré** (même budget d'essais, avant/après) : environ **2 fois plus rapide**, à qualité de résultat quasiment inchangée.

---

## 4. Notebook 2 — Éco-conception avec CodeCarbon

Fichier : `eco_conception_codecarbon.ipynb`. Extension de la démarche précédente à la mesure de l'empreinte énergétique et carbone, pour le Machine Learning **et** le Deep Learning.

### 4.1 Instrumenter l'entraînement

Un outil de mesure d'émissions encadre chaque entraînement à mesurer : il démarre juste avant l'entraînement, s'arrête juste après, et renvoie la quantité d'électricité consommée ainsi que l'équivalent en émissions de CO2 sur cette période précise.

**Point technique découvert en cours de route** : l'outil de mesure n'a pas de réglage direct pour indiquer le pays dans la version installée, et sa détection automatique de localisation s'est montrée incohérente d'une exécution à l'autre (pays différent détecté à quelques minutes d'intervalle, sur la même machine, sans changement de configuration). Solution retenue : indiquer manuellement une valeur représentative de l'intensité carbone du mix électrique français (très largement décarboné grâce au nucléaire), pour un résultat reproductible plutôt qu'une estimation automatique incertaine.

La même instrumentation sert pour :
- **ML** : entraînement de la baseline XGBoost sur le dataset de maintenance prédictive.
- **DL** : entraînement d'un autoencodeur convolutionnel (un réseau de neurones qui apprend à reconstruire des images pour détecter des anomalies visuelles) sur un dataset de photos de bouteilles.

**Rendus produits** :
- Un fichier détaillant chaque mesure d'émission, une ligne par étape instrumentée.
- Un tableau récapitulatif `étape → durée, énergie consommée, émissions de CO2, performance obtenue`.
- Des graphiques : émissions par étape (à échelle logarithmique, car ML et DL sont d'ordres de grandeur très différents) et durée par étape.

**Résultat réel obtenu :**

| Étape | Durée (s) | kWh | gCO2eq | Performance obtenue |
|---|---|---|---|---|
| ML — baseline XGBoost | 7,9 | 0,0000291 | 0,0016 | PR-AUC (val) = 0,4330 |
| DL — autoencodeur SSIM | 43,1 | 0,0001497 | 0,0084 | SSIM (val) = 0,4485 |

Pour un entraînement DL pourtant réduit (3 cycles d'apprentissage seulement, sur un tout petit dataset d'images), la durée est déjà **5,5 fois plus longue** et l'empreinte carbone **5,1 fois plus élevée** que la baseline ML complète — confirmation chiffrée que le deep learning sur images coûte structurellement plus cher que le machine learning sur données tabulaires, même à volume d'entraînement réduit.

![Émissions et durée par étape](figures/emissions_par_etape.png)

### 4.2 Étude lourde vs étude frugale

Deux stratégies de recherche de réglages comparées sur le même problème (XGBoost, horizon 24h) :

- **L'étude lourde** : un espace de recherche large (bornes environ 2 à 3 fois plus larges que la version raisonnée), un nombre d'essais élevé (40), et aucun arrêt anticipé des essais en cours de route — chaque essai va jusqu'au bout de son évaluation complète.
- **L'étude frugale** : un espace de recherche resserré autour des valeurs déjà jugées raisonnables, un arrêt anticipé agressif des essais visiblement mauvais dès le tout début de leur évaluation, un nombre d'essais réduit (15), et un volume de données réduit de moitié pour la recherche.

Chaque étude est instrumentée avec son propre suivi d'émissions. Les deux meilleurs essais sont ensuite réentraînés sur l'ensemble complet des données d'entraînement et évalués sur la validation — même protocole que la baseline, pour une comparaison à armes égales.

**Coût par point de performance gagné** : pour chaque stratégie, on divise le coût total (recherche + réentraînement final) par le gain de performance obtenu par rapport à la baseline. Si une stratégie n'apporte aucun gain (ou une régression), ce coût par point n'est pas défini — ce qui est en soi une information importante (de l'énergie dépensée sans aucun bénéfice mesurable).

**Résultat réel obtenu** (run de validation, 4 essais par stratégie) :

| Stratégie | Essais | Durée (s) | gCO2eq | PR-AUC (val) | Gain vs baseline | gCO2eq par point de PR-AUC |
|---|---|---|---|---|---|---|
| Baseline | 1 | 7,9 | 0,0016 | 0,4330 | — | — |
| **Frugale** | 4 | 19,2 | 0,0042 | **0,4528** | **+0,0198** | **0,2106** |
| **Lourde** | 4 | 89,2 | 0,0193 | 0,4409 | +0,0078 | 2,4659 |

Sur ce run, la stratégie **frugale gagne davantage** que la lourde (+0,0198 contre +0,0078), **pour moins de cinq fois moins d'énergie dépensée** — la lourde coûte environ **11,7 fois plus par point de PR-AUC gagné**. Un exemple concret et chiffré du symptôme de rendement décroissant : ouvrir un espace de recherche plus large et multiplier les essais sans arrêt anticipé n'a, ici, apporté ni un meilleur score, ni un score proportionné à la dépense engagée.

**Rendu** : un graphique positionnant la baseline, la stratégie frugale et la stratégie lourde selon deux axes — la performance obtenue et l'empreinte carbone associée.

![Performance vs empreinte carbone](figures/performance_vs_co2.png)

### 4.3 Réponses aux questions posées

**L'étude lourde apporte-t-elle un gain proportionné à son coût ?**

En comparant le gain de performance et l'empreinte carbone des deux stratégies : si l'étude lourde consomme grossièrement deux à trois fois plus d'essais dans un espace plus large sans arrêt anticipé, son coût total est structurellement plus élevé — la question est de savoir si le gain de performance l'est dans les mêmes proportions. **Sur le run réel obtenu (section 4.1), la réponse est clairement non** : la stratégie lourde a coûté environ 4,6 fois plus cher (en gCO2eq) que la frugale, tout en obtenant un gain de performance 2,5 fois plus faible — soit un coût par point de performance gagné **11,7 fois supérieur**. Ce n'est pas un cas limite ni un artefact isolé : c'est l'illustration directe du symptôme de rendement décroissant évoqué en théorie (section 2.5) — ouvrir davantage l'espace de recherche et retirer l'arrêt anticipé n'a, ici, rien apporté de plus, bien au contraire.

**Quand la parcimonie est-elle la bonne décision Green AI ?**

1. Quand le gain de la stratégie lourde ne dépasse pas la variabilité naturelle observée entre essais proches — un gain qui n'est pas distinguable du bruit statistique ne justifie jamais un surcoût énergétique, quel qu'il soit.
2. Quand le coût par point de performance de la stratégie frugale est du même ordre de grandeur (voire meilleur) que celui de la stratégie lourde — **exactement le cas observé ici** : la parcimonie n'est alors pas un compromis, c'est simplement la meilleure décision sur les deux plans à la fois.
3. Quand le contexte d'utilisation finale ne valorise pas économiquement le tout dernier point de performance gagné — un gain qui ne change pas la décision opérationnelle en aval ne justifie pas un budget de calcul disproportionné.
4. Quand l'itération rapide compte plus que la recherche du optimum absolu — en phase d'exploration d'un nouveau projet, une recherche frugale donne un signal directionnel utile pour une fraction du temps et de l'énergie dépensés par une recherche exhaustive.

À l'inverse, une recherche plus poussée reste justifiable quand le gain de performance a une valeur métier explicite et quantifiée (par exemple, un point de rappel supplémentaire sur la détection de panne qui évite un coût de maintenance non planifiée bien supérieur au coût énergétique de la recherche) — la parcimonie n'est pas une règle absolue, c'est un choix par défaut qu'il faut savoir justifier de ne pas suivre, pas l'inverse.

### 4.4 Optimisation du temps de traitement (reprise dans ce notebook)

Mêmes principes que le notebook 1, appliqués aux deux études : méthode d'entraînement rapide, arrêt anticipé des entraînements individuels, réduction du volume de données pour la recherche frugale, arrêt anticipé agressif des essais, nombre maximal d'arbres borné par l'arrêt anticipé plutôt que fixé arbitrairement haut.

### 4.5 Sortie

Tous les fichiers de résultats sont écrits dans le répertoire de sortie demandé explicitement (`artifacts/ingestions/output/`) : le détail des émissions, le tableau récapitulatif par étape, le tableau de comparaison lourd vs frugal, et les deux graphiques (émissions par étape, performance vs empreinte carbone).

---

## 5. Résultats réels obtenus lors des tests

Chiffres issus des exécutions réelles de validation (budgets réduits pour rester dans les contraintes de temps de test — mécanique strictement identique à celle des budgets réels prévus par défaut dans les notebooks livrés) :

- **Baseline XGBoost** : PR-AUC (validation) = 0,4330, obtenue en 7,9 secondes, pour une empreinte quasi négligeable (0,0016 gCO2eq).
- **Recherche Optuna (20 essais maximum, 11 aboutis)** : meilleur réglage trouvé avec PR-AUC (validation) = 0,4523, soit un gain de +4,4 % par rapport à la baseline. Ce gain (0,0192) dépasse nettement la variabilité naturelle entre les meilleurs essais (écart-type de 0,0064) — signal solide, pas un coup de chance. Confirmé sur le test jamais vu pendant le tuning : gain final de +3,8 %, cohérent avec le gain en validation.
- **Optimisation du temps de traitement (notebook 1)** : gain mesuré d'environ **2 fois plus rapide** sur la durée totale de la recherche de réglages (test antérieur, budget identique avant/après), à qualité de résultat quasiment inchangée.
- **Autoencodeur (Deep Learning), 3 cycles d'apprentissage seulement** : déjà 5,5 fois plus long et 5,1 fois plus d'émissions que la baseline ML complète — écart structurel entre apprentissage sur données tabulaires (capteurs) et apprentissage sur images, confirmé par la mesure plutôt que supposé.
- **Étude lourde vs frugale (4 essais chacune)** : la stratégie frugale l'emporte sur les deux plans à la fois — un gain de performance 2,5 fois supérieur à la stratégie lourde, pour un coût en énergie 4,6 fois inférieur. Coût par point de performance gagné : **11,7 fois plus élevé pour la stratégie lourde**. Le cas illustré ici n'est pas un compromis entre performance et écologie — la parcimonie est ici strictement meilleure sur les deux critères.

---

## 6. Glossaire accessible (non-spécialiste)

### Définitions des métriques utilisées

- **PR-AUC** (aire sous la courbe précision-rappel) : une note entre 0 et 1 qui mesure la capacité d'un modèle à bien repérer les cas rares (ici, les pannes) sans se tromper trop souvent en déclenchant de fausses alertes. Plus la valeur est proche de 1, mieux le modèle sépare les vrais cas positifs des cas négatifs. Contrairement au simple pourcentage de bonnes réponses, elle reste fiable même quand l'événement recherché est rare — c'est pour ça qu'elle est utilisée ici plutôt que l'accuracy.
- **AUC (aire sous la courbe ROC)** : une autre note entre 0 et 1, plus classique, qui mesure la capacité du modèle à classer correctement les cas positifs au-dessus des cas négatifs, peu importe le seuil de décision choisi. 0,5 correspond à un modèle qui ne fait pas mieux que le hasard, 1 à un modèle parfait. Complémentaire à la PR-AUC, moins sensible qu'elle au déséquilibre entre classes rares et fréquentes.
- **Gain vs baseline** : la différence entre le score obtenu par un modèle amélioré et le score du modèle de référence (baseline), exprimée à la fois en valeur brute et en pourcentage relatif. Permet de juger si un effort de recherche a vraiment apporté quelque chose.
- **Écart-type inter-essais** : une mesure de la dispersion des scores obtenus par les meilleurs essais d'une recherche de réglages. Sert de référence pour juger si un gain observé est un vrai signal ou seulement une variation due au hasard.
- **kWh (kilowattheure)** : l'unité qui mesure la quantité d'électricité consommée — la même unité que celle affichée sur une facture d'électricité domestique.
- **gCO2eq (grammes de CO2 équivalent)** : l'unité qui mesure l'impact climatique d'une quantité d'électricité consommée, convertie en une quantité de CO2 qui aurait le même effet sur le climat — permet de comparer des sources d'énergie différentes sur une même échelle.
- **Coût par point de performance gagné** : le coût total (en énergie ou en émissions) divisé par le gain de score obtenu par rapport à la baseline. Une façon de répondre à la question "est-ce que ce gain valait la dépense engagée pour l'obtenir ?", plutôt que de regarder le gain ou le coût séparément.

### Autres notions clés

Résumé des explications déjà fournies en langage simple pour les principaux concepts :

- **Hyperparamètres** : les réglages qu'on fixe avant l'apprentissage du modèle, comme la température d'un four avant d'enfourner un gâteau.
- **Optuna** : l'outil qui teste automatiquement de nombreux réglages différents, en apprenant au fur et à mesure où chercher plutôt qu'en tirant au hasard.
- **Arrêt anticipé d'un essai (pruning)** : couper un essai en cours de route s'il est déjà clairement mauvais, pour économiser du temps de calcul — comme un chef qui goûte un plat en cours de cuisson et le jette avant la fin s'il est manifestement raté.
- **PR-AUC** : une façon de juger un modèle plus fiable que le simple pourcentage de bonnes réponses, adaptée aux cas où l'événement à détecter (une panne) est rare.
- **Découpage temporel des données** : tester le modèle en respectant strictement l'ordre chronologique — jamais utiliser des informations du futur pour prédire le passé.
- **Baseline** : le modèle de référence "sans effort particulier de réglage", qui sert à juger si toute la recherche entreprise apporte vraiment un gain.
- **Suivi des essais (MLflow)** : le carnet de bord numérique qui enregistre chaque essai réalisé, pour pouvoir les comparer après coup.
- **Sur-ajustement du tuning** : quand la recherche de réglages trouve une combinaison qui a l'air excellente par pure chance sur un découpage précis des données, pas parce qu'elle est vraiment bonne — comme un élève qui, en révisant sur les annales, tombe par hasard sur exactement les mêmes questions à l'examen.
- **CodeCarbon** : le compteur électrique du code, qui mesure la consommation réelle pendant qu'un programme tourne et la convertit en équivalent CO2.
- **kWh et grammes de CO2 équivalent** : les unités de consommation électrique et d'impact climatique, les mêmes que celles qui apparaissent sur une facture d'électricité domestique.
- **Étude lourde vs frugale** : chercher beaucoup, largement et sans limite de temps par essai (lourde) contre chercher peu, de façon ciblée et en arrêtant tôt les mauvaises pistes (frugale).
- **Green AI** : l'idée de juger un modèle d'intelligence artificielle non seulement sur sa performance, mais aussi sur son coût énergétique et environnemental — comme on ne choisirait pas une voiture uniquement sur sa vitesse maximale, sans regarder sa consommation d'essence.

---

## 7. Fichiers livrés

| Fichier | Contenu |
|---|---|
| `optuna_xgboost_predictive_maintenance.ipynb` | Notebook complet : cadrage, baseline, espace borné, étude Optuna, analyse, validation finale |
| `optuna_xgboost_predictive_maintenance_fr.md` / `_en.md` | Documentation du notebook ci-dessus, FR et EN |
| `eco_conception_codecarbon.ipynb` | Notebook complet : instrumentation CodeCarbon, étude lourde vs frugale |
| `eco_conception_codecarbon_fr.md` / `_en.md` | Documentation du notebook ci-dessus, FR et EN |
| `explication_non_specialiste.md` | Glossaire explicatif complet pour un public non technique |
| `optimisation_optuna_synthese_complete.md` | Ce document — synthèse de l'ensemble des échanges, sans code |
