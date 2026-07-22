# Comprendre les deux notebooks — explication pour non-spécialiste

Ce document reprend, un par un, les éléments techniques mis en avant dans les deux fichiers `.md` précédents (`optuna_xgboost_predictive_maintenance` et `eco_conception_codecarbon`), et les explique en langage simple. L'objectif : pouvoir suivre la logique globale sans connaître le vocabulaire du machine learning.

---

## Partie 1 — Le notebook Optuna (recherche du meilleur réglage)

### Le problème de départ

On a un modèle qui essaie de prédire si une machine va tomber en panne dans les 24 prochaines heures, à partir de mesures capteurs (température, vibrations, etc.). Le modèle utilisé s'appelle **XGBoost** — un algorithme très répandu qui construit sa prédiction en combinant plein de petites règles de décision simples ("si la température dépasse X et que les vibrations dépassent Y, alors risque élevé"), empilées les unes sur les autres.

### Les "hyperparamètres" — les boutons de réglage du modèle

Un modèle comme XGBoost ne se contente pas d'apprendre à partir des données : il a aussi des **réglages** qu'on doit fixer soi-même avant de le laisser apprendre, un peu comme les réglages d'un four (température, durée) avant d'enfourner un gâteau. Ces réglages s'appellent des **hyperparamètres**. Exemples concrets dans notre cas :

- **`max_depth`** : jusqu'à quel niveau de détail le modèle peut affiner ses règles de décision. Trop haut, il invente des règles trop précises qui ne marcheront que sur les données déjà vues (voir plus bas "sur-apprentissage").
- **`learning_rate`** : à quelle vitesse le modèle corrige ses erreurs à chaque étape d'apprentissage. Trop vite, il risque de "zapper" la bonne solution ; trop lentement, il met un temps infini à apprendre.
- **`n_estimators`** : combien de petites règles de décision on empile au total.

Le problème : on ne sait pas à l'avance quels réglages donnent le meilleur résultat sur **ce** dataset précis. Il faut essayer plusieurs combinaisons et comparer.

### Optuna — l'outil qui teste les réglages à notre place

**Optuna** est un outil qui automatise cette recherche : au lieu de tester les réglages à la main un par un (des dizaines d'heures de travail manuel), il teste lui-même plein de combinaisons différentes, en apprenant au fur et à mesure quelles zones semblent prometteuses. Chaque combinaison testée s'appelle un **essai** (trial en anglais).

- **`TPESampler`** : la stratégie qu'Optuna utilise pour choisir la prochaine combinaison à tester. Plutôt que de tester au hasard, il regarde les essais précédents et devine intelligemment où chercher ensuite — un peu comme un joueur de "plus ou moins" qui affine ses hypothèses au fil des indices.
- **La "seed" (graine aléatoire)** : un nombre de départ qui fixe le hasard utilisé partout dans le calcul. La fixer (`SEED = 42`) garantit que si on relance exactement le même code, on obtient exactement les mêmes résultats — indispensable pour pouvoir comparer deux expériences de façon fiable, et pour que quelqu'un d'autre puisse reproduire ce qu'on a fait.

### Le "pruning" — arrêter les essais visiblement mauvais avant la fin

Tester une combinaison d'hyperparamètres prend du temps (il faut entraîner un modèle complet). Le **pruning** ("élagage") consiste à surveiller un essai en cours de route et à l'arrêter avant la fin s'il est déjà clairement moins bon que les précédents — comme un chef qui goûte un plat en cours de cuisson et le jette avant la fin s'il est manifestement raté, plutôt que d'attendre la cuisson complète. Ça économise du temps de calcul (et donc de l'énergie).

### "Borner" l'espace de recherche — donner des limites raisonnables

Pour chaque hyperparamètre, on ne laisse pas Optuna chercher n'importe quelle valeur entre moins l'infini et plus l'infini : on lui donne une **plage raisonnable** (par exemple, `max_depth` entre 3 et 8, pas entre 1 et 1000). Pourquoi ? Parce qu'un espace de recherche trop large, avec le même nombre d'essais, revient à chercher une aiguille dans une botte de foin plus grande — moins de chances de bien la trouver, pour le même effort. Borner intelligemment (en s'appuyant sur ce qu'on sait déjà du problème) rend la recherche plus efficace.

### PR-AUC — comment on juge si un réglage est "bon"

Il faut un critère chiffré pour dire "ce réglage est meilleur que cet autre". On utilise la **PR-AUC** plutôt que le pourcentage de bonnes réponses (l'"accuracy") classique, pour une raison importante : dans notre dataset, les pannes sont rares (~17 % des cas). Un modèle "bête" qui prédirait toujours "pas de panne" aurait déjà 83 % de bonnes réponses sans avoir rien appris d'utile ! La PR-AUC est une métrique qui reste pertinente même dans ce genre de situation déséquilibrée — elle mesure la capacité du modèle à bien distinguer les vraies pannes des fausses alertes, peu importe leur rareté.

### Validation croisée temporelle (`TimeSeriesSplit`) — ne jamais tricher avec le temps

Pour savoir si un modèle marche bien, on doit le tester sur des données qu'il n'a jamais vues pendant son apprentissage — sinon, c'est comme donner à un élève l'énoncé ET le corrigé de son examen avant l'épreuve. Ici, comme les données sont mesurées dans le temps (capteurs relevés heure par heure), il y a une règle supplémentaire : **on ne doit jamais utiliser des données du futur pour prédire le passé**. Le `TimeSeriesSplit` découpe donc les données en respectant strictement l'ordre chronologique : le modèle apprend sur une période, puis est testé sur la période qui suit — jamais l'inverse.

### La "baseline" — le point de comparaison de référence

Avant de chercher le meilleur réglage possible, on entraîne un modèle avec des réglages "raisonnables par défaut", sans optimisation particulière. Ce modèle de référence s'appelle la **baseline**. Il sert à répondre à la question : "est-ce que tout ce travail de recherche d'hyperparamètres apporte vraiment un gain, ou est-ce qu'on aurait obtenu un résultat presque identique sans effort ?"

### MLflow — le carnet de bord des expériences

**MLflow** est un outil qui enregistre automatiquement chaque essai réalisé : quels réglages ont été testés, quel score a été obtenu, combien de temps ça a pris. C'est l'équivalent d'un carnet de laboratoire numérique — sans lui, on perdrait la trace de ce qui a été essayé, et on ne pourrait pas comparer facilement des dizaines d'essais entre eux.

### L'analyse après coup — comprendre *pourquoi* ça marche

Une fois les 50 essais terminés, on ne se contente pas de prendre "le meilleur réglage trouvé" — on regarde en détail :

- **L'historique d'optimisation** : un graphique qui montre l'évolution du meilleur score au fil des essais. S'il continue à s'améliorer jusqu'au bout, ça veut dire qu'on aurait pu aller plus loin ; s'il stagne depuis longtemps, la recherche a atteint ses limites.
- **L'importance des hyperparamètres** : quel réglage a le plus influencé le résultat final. Utile pour savoir sur quoi se concentrer si on doit recommencer une recherche plus tard.
- **Le "slice plot"** : un graphique qui montre, pour chaque valeur testée d'un hyperparamètre donné, quel score a été obtenu. Permet de voir si une zone de valeurs se dégage clairement comme "la bonne zone", ou si c'est le hasard qui domine.

### Le piège du "sur-ajustement du tuning" — quand la recherche se trompe elle-même

Il y a un piège spécifique à surveiller : parfois, Optuna trouve un réglage qui a l'air excellent, mais seulement parce qu'il "colle" par chance aux particularités des données de test utilisées à ce moment précis — pas parce que c'est un réglage vraiment robuste. C'est un peu comme un élève qui, en révisant sur les annales des années précédentes, tombe par hasard sur exactement les mêmes questions à l'examen : son excellente note ne prouve pas qu'il maîtrise vraiment la matière. Le signe d'alerte : si les meilleurs essais ont des réglages très différents les uns des autres (pas de zone de valeurs cohérente), c'est suspect. La parade : vérifier le résultat sur un jeu de données totalement différent (le "test") avant de faire confiance au gain observé.

### La validation finale — le vrai test, à ne faire qu'une fois

Une fois le meilleur réglage choisi, on l'évalue **une seule fois** sur des données mises de côté depuis le début et jamais utilisées pendant toute la recherche — le **test set**. C'est le "vrai" examen final, celui qui donne le chiffre à annoncer honnêtement. On ne doit jamais l'utiliser plusieurs fois pour ajuster ses réglages, sinon on retombe dans le piège décrit juste au-dessus.

---

## Partie 2 — Le notebook éco-conception (CodeCarbon)

### Pourquoi mesurer l'empreinte carbone d'un entraînement ?

Entraîner un modèle consomme de l'électricité — donc a un coût environnemental, même si celui-ci reste souvent invisible (pas de facture affichée en direct). L'idée de ce notebook : **mesurer ce coût**, plutôt que de le supposer négligeable ou de l'ignorer, et l'utiliser comme un critère de décision au même titre que la performance du modèle.

### CodeCarbon — le compteur électrique du code

**CodeCarbon** est un outil qui, pendant qu'un programme tourne, mesure combien d'électricité l'ordinateur consomme (processeur, mémoire), puis convertit cette consommation en équivalent d'émissions de CO2 — en tenant compte du fait que produire de l'électricité pollue plus ou moins selon le pays (un mix électrique très nucléaire comme en France pollue beaucoup moins par kWh qu'un mix très charbon). On l'utilise ici comme un chronomètre, mais qui mesure l'énergie au lieu du temps.

- **kWh (kilowattheure)** : l'unité qui mesure la quantité d'électricité consommée — la même unité que celle affichée sur une facture d'électricité domestique.
- **gCO2eq (grammes de CO2 équivalent)** : l'unité qui mesure l'impact climatique de cette électricité consommée, convertie en une quantité de CO2 qui aurait le même effet sur le climat.

### Pourquoi le pays choisi (France) change le résultat

Produire un kWh d'électricité ne pollue pas pareil partout dans le monde (dépend du mix énergétique : nucléaire, charbon, renouvelables...). Dans ce notebook, on a fixé manuellement une valeur représentative du mix électrique français (très peu carboné grâce au nucléaire) plutôt que de laisser l'outil deviner automatiquement le pays — cette détection automatique s'est révélée peu fiable (elle donnait des résultats différents à quelques minutes d'intervalle sur la même machine), donc on a préféré fixer une valeur de référence fiable et vérifiable plutôt qu'une estimation automatique incertaine.

### Étude "lourde" vs étude "frugale" — deux façons de chercher, deux coûts différents

Reprenant l'idée de recherche d'hyperparamètres du premier notebook, on compare ici deux stratégies extrêmes :

- **L'étude "lourde"** : on autorise Optuna à explorer une très large plage de réglages possibles, sans jamais arrêter un essai en cours de route (pas de pruning), avec beaucoup d'essais (40). C'est la stratégie "on met les moyens", potentiellement plus susceptible de trouver un excellent réglage, mais aussi beaucoup plus gourmande en calcul (et donc en électricité).
- **L'étude "frugale"** : on resserre la plage de recherche autour de valeurs déjà jugées raisonnables, on coupe les essais visiblement mauvais dès le début (pruning agressif), et on limite le nombre d'essais (15). C'est la stratégie "on va à l'essentiel".

### Le "coût par point de performance gagné" — la vraie question à se poser

Comparer juste "qui gagne, la lourde ou la frugale ?" ne suffit pas — il faut aussi regarder **combien ça a coûté** pour obtenir ce gain. Le notebook calcule donc : *combien de grammes de CO2 ont été dépensés pour chaque point de performance (PR-AUC) gagné par rapport à la baseline ?* Si l'étude lourde ne gagne que très peu de performance en plus pour un coût en énergie très supérieur, ce n'est pas un bon calcul, même si son score final est légèrement meilleur.

### La notion de "Green AI" — l'écologie comme critère de décision, pas comme contrainte accessoire

Le "Green AI" est l'idée qu'on ne devrait pas juger un modèle d'intelligence artificielle uniquement sur sa performance brute, mais aussi sur son coût énergétique et environnemental — un peu comme on ne choisit pas une voiture uniquement sur sa vitesse maximale, sans regarder sa consommation d'essence. La question posée dans ce notebook ("la parcimonie est-elle la bonne décision ?") invite à se demander, à chaque projet, si le gain de performance espéré justifie vraiment le coût (calcul, temps, énergie, argent) qu'on est prêt à y mettre — et à choisir la solution la plus simple qui suffit, plutôt que systématiquement la plus complexe.

### Pourquoi comparer aussi le Machine Learning (XGBoost) et le Deep Learning (l'autoencodeur d'images)

Le notebook mesure aussi la consommation d'un modèle de deep learning (un "autoencodeur", un type de réseau de neurones qui apprend à reconstruire des images pour détecter des anomalies visuelles sur des photos de bouteilles). L'idée : montrer concrètement que le deep learning sur des images est généralement beaucoup plus gourmand en calcul (et donc en énergie) que le machine learning classique sur des données tabulaires (comme les capteurs de machines) — une intuition qu'on vérifie ici avec de vrais chiffres mesurés, plutôt que de la supposer.

---

## En résumé — les questions auxquelles ces deux notebooks répondent

1. **Comment trouver le meilleur réglage d'un modèle sans y passer un temps infini ni tester au hasard ?** → Optuna, avec une recherche bornée et intelligente (TPE), qui s'arrête tôt sur les mauvais essais (pruning).
2. **Comment être sûr que le gain trouvé est réel, et pas un coup de chance ?** → Comparer à une baseline, vérifier la cohérence des meilleurs essais, valider une seule fois sur des données jamais vues.
3. **Combien coûte, en électricité et en CO2, le fait de chercher "le meilleur" plutôt que de se contenter de "raisonnable" ?** → CodeCarbon mesure ce coût précisément, plutôt que de le laisser invisible.
4. **Vaut-il toujours le coup de chercher plus loin ?** → Pas automatiquement — le gain de performance doit être mis en regard du coût réel (temps, calcul, énergie) pour décider si ça en vaut la peine.
