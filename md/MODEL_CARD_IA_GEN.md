---
language: fr
tags:
  - education
  - pedagogy
  - speech-to-text
  - text-enrichment
  - retrieval-augmented-generation
  - pipeline
  - french
pipeline_tag: text-generation
---

# Model Card — XXX · Enrichisseur de supports de cours

Service d'enrichissement automatique de supports pédagogiques à partir de
l'enregistrement audio du cours. Le système transcrit l'audio, l'aligne sur le
support écrit, puis ré-injecte dans le format d'origine les explications,
exemples et anecdotes apportés à l'oral par l'enseignant.

## Détails du système

### Description

- **Développé par :** PredexIA
- **Financé / porté par :** projet XXX
- **Type de système :** pipeline multi-étapes ASR → alignement → enrichissement
  LLM → rendu, avec garde-fous anti-hallucination
- **Langue(s) :** français (contenu traité et interface)
- **Modèles sous-jacents (par défaut, configurables) :**
  - Transcription : **Voxtral** (`voxtral-mini-latest`, Mistral) — alternative :
    **Whisper** (Scaleway)
  - Enrichissement : **Mistral** (`mistral-large-latest`) — alternatives :
    **Anthropic** (`claude-sonnet-4-6`), **OpenAI** (`gpt-4o`)

### Sources

- **Dépôt :** XXX
- **Documentation d'architecture :** `.claude/plans/bubbly-wiggling-knuth.md` (interne)
- **README :** [README.md](README.md)

## Usages

### Public visé

Cette carte s'adresse à la fois :

- **aux enseignants et établissements** — utilisateurs finaux qui déposent un
  support et son enregistrement audio pour récupérer un support enrichi ;
- **aux développeurs et intégrateurs** — équipes qui déploient, configurent
  (choix des providers ASR/LLM, niveau de détail) ou étendent le service.

### Usage direct

Enrichir un support de cours écrit avec ce qui a été dit à l'oral pendant la
séance. Le système :

1. **Transcrit** chaque fichier audio (concaténé avec offsets temporels) ;
2. **Construit un glossaire** des noms propres canoniques à partir du support
   écrit (autorité orthographique servant à corriger les variantes mal
   transcrites par l'ASR) ;
3. **Aligne** les segments du transcript sur les sections du support (mapping
   section → segments, réalisé par le LLM) ;
4. **Enrichit** chaque section : le LLM produit des apports (`explanation`,
   `example`, `anecdote`, `reformulation`) **strictement tirés du transcript** ;
5. **Rend** le résultat dans le format d'origine.

**Formats pris en charge :**

| Entrée support | Sortie |
|---|---|
| XML Scenari/Opale (item seul) | XML enrichi validé (XSD/RNG ou profil inféré) |
| Archive Scenari `.scar` | Archive `.scar` enrichie, structure et binaires préservés |
| LaTeX (`.tex`) | LaTeX enrichi compilable (mono- ou multi-fichiers) |
| PowerPoint (`.pptx`) | Word (`.docx`) : contenu des slides + apports oraux |

Audio accepté : `mp3`, `m4a`, `wav`, `ogg`, `flac`, `aac` ; vidéos (`mp4`, `mov`,
`mkv`, `webm`) dont seule la piste audio est utilisée.

**Niveaux de détail** (pilotent la quantité/longueur des apports, sans jamais
relâcher les garde-fous qualité) : `concise` (≤ 2 apports/section),
`balanced` (défaut, ≤ 3), `exhaustive` (≤ 6).

### Usage en aval

- Génération de polycopiés ou de notes de révision fidèles au cours donné.
- Constitution de supports accessibles et complets (notamment pour des publics
  ayant besoin d'une trace écrite exhaustive de l'oral).
- Intégration dans une chaîne éditoriale Scenari/Opale ou LaTeX existante.

### Hors périmètre

- **Ce n'est pas un système de résumé ni de reformulation du support** : il
  n'ajoute que ce que l'oral apporte de neuf ; il ne réécrit pas l'écrit existant.
- **Pas de génération de contenu ex nihilo** : toute sortie non ancrée dans le
  transcript est, par conception, à proscrire.
- Pas conçu pour la transcription verbatim brute, le sous-titrage synchronisé,
  ni la traduction.
- Pas évalué pour d'autres langues que le français.
- Pas un outil de décision ni d'évaluation des élèves.

## Biais, risques et limites

- **Dépendance à la qualité audio.** Un enregistrement bruité, lointain ou
  inaudible dégrade la transcription et donc l'enrichissement. Les sections dont
  l'oral est trop court sont volontairement ignorées (seuil configurable par
  niveau).
- **Erreurs d'ASR sur les noms propres et termes techniques.** Le glossaire
  dérivé du support atténue ce risque (correction des variantes phonétiques,
  ex. « Antipi Frémy » → « Atypie-Friendly ») mais ne le supprime pas : un terme
  absent du support ne sera pas corrigé.
- **Erreurs d'alignement.** Le mapping segment → section est produit par un LLM ;
  un apport oral peut être rattaché à la mauvaise section ou omis.
- **Hallucinations résiduelles.** Malgré des consignes strictes
  (interdiction d'inventer, ancrage obligatoire dans le transcript, fidélité du
  sens), aucun garde-fou LLM n'est infaillible. Une reformulation peut, à la
  marge, altérer le propos d'origine.
- **Fidélité du sens.** Le prompt impose de conserver les relations qui lèvent
  l'ambiguïté (qui fait quoi, dans quel ordre, à quel moment) ; ce contrôle reste
  probabiliste.
- **Biais des modèles tiers.** Les biais des modèles ASR et LLM sous-jacents
  (Voxtral/Whisper, Mistral/Anthropic/OpenAI) se répercutent sur les sorties.
- **Confidentialité.** L'audio et le support sont envoyés à des API tierces
  pour traitement. L'audio original et les WAV normalisés sont purgés du disque
  dès la fin de la transcription, mais le traitement distant implique les
  politiques de rétention des providers choisis.

### Recommandations

- **Relecture humaine obligatoire** avant diffusion : l'enrichi est une aide à
  la production, pas une sortie de confiance à publier telle quelle.
- Vérifier en priorité les **noms propres, chiffres, formules et enchaînements
  logiques** des passages enrichis.
- Choisir les providers ASR/LLM en cohérence avec les **contraintes de
  confidentialité** de l'établissement (données élèves, RGPD).
- Fournir un **audio de bonne qualité** (micro proche de l'enseignant).

## Prise en main

```bash
# 1. Cloner et configurer
cp .env.example .env
# Éditer .env : MISTRAL_API_KEY (ou autre provider), DOMAIN, etc.

# 2. Démarrer la stack + migrations + compte admin
./init.sh admin@example.fr 'mot-de-passe-fort'

# 3. Accéder à l'application
# https://<votre-domaine>/
```

Variables clés (`.env`) :

- `TRANSCRIPTION_PROVIDER` (`voxtral_mistral` | `whisper_scaleway`) et
  `VOXTRAL_MODEL`
- `LLM_PROVIDER` (`mistral` | `anthropic` | `openai`) et `LLM_MODEL`
- Clés API correspondantes (`MISTRAL_API_KEY`, `ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`, `SCALEWAY_API_KEY`…)

Voir [README.md](README.md) pour le mode développement et le déploiement.

## Détails « d'entraînement » (modèles sous-jacents)

Aucun entraînement ni fine-tuning n'est réalisé dans ce projet. Le comportement
est déterminé par :

- les **modèles pré-entraînés** listés plus haut, appelés via API ;
- les **prompts maîtres** ([enrich_prompts.py](backend/app/pipeline/enrich_prompts.py)),
  qui portent les consignes d'enrichissement, d'alignement et les garde-fous
  anti-hallucination ;
- les **paramètres d'inférence** de l'étape d'enrichissement : `temperature=0.3`,
  `json_mode=True`, `max_tokens` selon le niveau de détail (2048 / 3072 / 6144) ;
- la **logique de post-traitement** déterministe (déduplication des apports,
  plancher anti-apport creux de 30 caractères, troncature à la longueur max du
  niveau, filtrage des sections à oral trop court).

## Évaluation

- **Méthode actuelle :** validation qualitative par **relecture humaine** des
  paires de test (audios + supports) du dossier `data/`. Pas de métrique
  automatisée à ce stade.
- **Critères observés :** fidélité au transcript (absence d'invention),
  pertinence de l'alignement section/segment, correction des noms propres via le
  glossaire, conservation du sens dans les reformulations, propreté du rendu dans
  le format cible.
- **Métriques quantitatives :** *non disponibles.*

### Évaluation recommandée (prochaines étapes)

- Jeu de test annoté (support + audio + enrichi attendu) par format.
- Taux d'hallucination (apports non ancrés) mesuré par échantillonnage.
- Précision/rappel de l'alignement section → segments.
- Taux de correction correcte / sur-correction des noms propres.
- Validité formelle des sorties (compilation LaTeX, validation XSD/RNG, ouverture
  `.docx`/`.scar`).

## Impact environnemental

L'empreinte provient essentiellement des **appels d'inférence aux API tierces**
(ASR + LLM), non quantifiés ici. Le facteur principal est la **durée d'audio
transcrit** et le **volume de texte enrichi** (croissant avec le niveau de
détail). L'hébergement applicatif recommandé est une instance unique de type
Scaleway DEV1-M (voir README).

## Aspects techniques

- **Architecture :** SvelteKit → Caddy (TLS) → FastAPI (+ PostgreSQL) → Celery
  (Redis) → Worker exécutant le pipeline `transcribe → align → enrich → render`.
- **Stack :** FastAPI + Celery + Redis + PostgreSQL (Python 3.12), SvelteKit,
  Docker Compose.
- **Abstractions multi-provider :** [transcription/factory.py](backend/app/transcription/factory.py),
  [llm/factory.py](backend/app/llm/factory.py).
- **Point d'entrée du pipeline :** [pipeline/runner.py](backend/app/pipeline/runner.py).

## Contact

- **Organisation :** Predexia
- **Contact :** marine.lannes@predexia.com

---

