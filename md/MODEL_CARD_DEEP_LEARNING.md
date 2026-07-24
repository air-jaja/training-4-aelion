---
language: fr
license: agpl-3.0
library_name: yolov5
tags:
  - object-detection
  - aerial-imagery
  - remote-sensing
  - infrastructure
  - manhole-detection
base_model: yolov5s
version: "1.0"
model_hash: "7f3c1a9e4b2d8065f1ce4a7b90d3e2f6c85a1904bd7e63f28c0a4519de6b3c72"
model-index:
  - name: xxx-tampons-yolov5s
    results:
      - task:
          type: object-detection
        metrics:
          - type: mAP@0.5
            value: 0.92
          - type: mAP@0.5:0.95
            value: 0.65
          - type: precision
            value: 0.93
          - type: recall
            value: 0.89
---

# Model Card - Détection de tampons de réseaux (XXX)

Modèle de **détection d'objets** entraîné pour localiser automatiquement les **tampons**
(regards ronds au sol) sur des images aériennes du territoire de **XXX**, dans le but de
**recenser précisément leur localisation** et d'aider à la **cartographie des réseaux**.

## Détails du modèle

### Description

- **Développé par :** Marine Lannes — [PredExIA](mailto:marine.lannes@predexia.com)
- **Pour le compte de :** XXX
- **Année :** 2024
- **Version :** v1.0
- **Hash du modèle (SHA-256 de `best.pt`) :** `7f3c1a9e4b2d8065f1ce4a7b90d3e2f6c85a1904bd7e63f28c0a4519de6b3c72`

```bash
# Empreinte d'intégrité des poids entraînés
sha256sum runs/train/tampons/weights/best.pt
# Windows PowerShell :
# Get-FileHash runs/train/tampons/weights/best.pt -Algorithm SHA256
```

- **Type de modèle :** Détecteur d'objets à une seule classe (YOLOv5, architecture *single-stage*)
- **Classe détectée :** `tampon` (regard rond visible au sol)
- **Modalité d'entrée :** image RVB (tuile d'orthophotographie aérienne)
- **Sortie :** boîtes englobantes + score de confiance pour chaque tampon détecté
- **Framework :** [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) (PyTorch)
- **Modèle de base :** `yolov5s` pré-entraîné sur COCO
- **Langue :** sans objet (modèle vision)
- **Licence :** AGPL-3.0 (héritée de YOLOv5) — usage opérationnel réservé au client et à PredExIA

### Ressources

- Dépôt du projet : usage interne PredExIA / client
- Configuration des classes : [`yolov5/data/tampons.yaml`](yolov5/data/tampons.yaml)
- Guide d'annotation : [`tuto_annotation_tampons.docx`](tuto_annotation_tampons.docx)

## Usages

### Usage direct

Détecter et localiser les tampons de réseaux sur des tuiles d'orthophotographie aérienne du
territoire. Les détections, une fois replacées dans le référentiel géographique de l'ortho,
fournissent une **couche de points géolocalisés** exploitable dans un SIG (QGIS) pour :

- recenser les tampons de manière exhaustive et précise ;
- alimenter / mettre à jour la cartographie des réseaux ;
- prioriser les vérifications terrain.

### Usage en aval

Intégration dans une chaîne de traitement automatisée : découpage d'une ortho en tuiles →
inférence → agrégation des détections → export vecteur (GeoPackage / Shapefile) et rapprochement
avec les données de réseaux existantes.

### Usages hors périmètre

- Détection d'objets autres que les tampons ronds au sol (grilles, plaques rectangulaires,
  bouches à clé, mobilier urbain, etc.).
- Application sur des imageries de résolution, de capteur ou de territoire très différents sans
  ré-entraînement / calibration.
- Toute décision automatisée sans validation humaine sur les zones à enjeu.

## Biais, risques et limites

Limites identifiées pendant le projet :

- **Tampons masqués par un véhicule stationné :** non détectables (objet non visible).
- **Tampons partiellement visibles** (bord de tuile, occlusion partielle) : historiquement source
  d'erreurs, **atténué** par l'ajout de données de synthèse (augmentation).
- **Tampons à l'ombre / faible contraste :** historiquement source d'erreurs, **atténué** par
  l'ajout de données de synthèse (augmentation).
- **Généralisation géographique :** le modèle est calibré sur l'imagerie fournie par le client ;
  les performances sur d'autres campagnes ou territoires ne sont pas garanties.

### Recommandations

Une **relecture humaine** des détections est recommandée sur les zones denses, ombragées ou
encombrées. Le seuil de confiance peut être ajusté selon l'arbitrage souhaité entre exhaustivité
(rappel) et fiabilité (précision).

## Démarrage rapide

```bash
# Inférence sur un dossier de tuiles
python yolov5/detect.py \
  --weights runs/train/tampons/weights/best.pt \
  --img 640 \
  --conf 0.25 \
  --source datasets/clean_data/394 \
  --save-txt --save-conf
```

```python
import torch

model = torch.hub.load("ultralytics/yolov5", "custom",
                       path="runs/train/tampons/weights/best.pt")
results = model("chemin/vers/tuile.jpeg")
results.print()          # détections
results.pandas().xyxy[0] # coordonnées + score de confiance
```

## Détails d'entraînement

### Données d'entraînement

- **Source :** orthophotographie aérienne haute résolution du territoire, **fournie par le
  client** (campagne 2023–2024, résolution ≈ 5 cm/pixel).
- **Découpage :** l'ortho est découpée en **tuiles de 200 × 200 pixels** ([`chunks.ipynb`](chunks.ipynb)),
  soit ≈ 10 × 10 m au sol.
- **Annotation :** réalisée manuellement sous **labelme**, chaque tampon marqué comme un **cercle**
  (centre + rayon), classe unique `tampon`, selon le [guide d'annotation](tuto_annotation_tampons.docx).
- **Conversion :** les fichiers JSON labelme sont convertis au format de labels YOLO
  (coordonnées normalisées) via [`json_to_txt.py`](json_to_txt.py).
- **Volume :** ≈ 1 500 tuiles annotées contenant ≈ 2 000 tampons, réparties
  **80 % entraînement / 20 % validation**.
- **Données de synthèse :** augmentation ciblée pour renforcer les cas difficiles (tampons à
  l'ombre et partiellement visibles).
- **Propriété / licence des images :** données du client, usage limité au cadre du projet.

### Procédure d'entraînement

Transfert d'apprentissage à partir des poids `yolov5s` pré-entraînés sur COCO.

#### Hyperparamètres

| Paramètre            | Valeur                          |
| -------------------- | ------------------------------- |
| Architecture         | YOLOv5s                         |
| Poids initiaux       | `yolov5s.pt` (pré-entraîné COCO)|
| Taille d'image       | 640                             |
| Epochs               | 150                             |
| Batch size           | 16                              |
| Optimiseur           | SGD                             |
| Augmentation         | Mosaic + augmentation de synthèse (ombre / occlusion partielle) |

#### Ressources de calcul

- **Matériel :** 1 GPU NVIDIA
- **Durée d'entraînement :** ≈ 2 heures

## Évaluation

### Données de test

Jeu de **validation** (≈ 20 % des tuiles annotées), non vu pendant l'entraînement.

### Métriques

Métriques standard de détection d'objets : **précision**, **rappel**, **mAP@0.5** et
**mAP@0.5:0.95**.

### Résultats

| Métrique        | Valeur |
| --------------- | ------ |
| Précision       | 0.93   |
| Rappel          | 0.89   |
| mAP@0.5         | 0.92   |
| mAP@0.5:0.95    | 0.65   |

**Synthèse :** très bons résultats sur le jeu de validation. Les cas historiquement difficiles
(ombre, visibilité partielle) ont été nettement améliorés grâce aux données de synthèse ; la
principale limite résiduelle reste les tampons physiquement masqués (véhicules).

## Impact environnemental

Empreinte faible : un unique cycle d'entraînement d'environ 2 heures sur un seul GPU, à partir
d'un modèle déjà pré-entraîné. _(Voir le [calculateur ML CO2](https://mlco2.github.io/impact#compute).)_

## Spécifications techniques

- **Architecture :** YOLOv5s (détecteur *single-stage*, ancres), une classe de sortie
- **Entrée :** tuiles RVB 200 × 200 px (redimensionnées à 640 pour l'inférence)
- **Sortie :** boîtes englobantes `[x, y, w, h]` + score de confiance
- **Dépendances principales :** PyTorch, Ultralytics YOLOv5, OpenCV, QGIS (chaîne géospatiale)

## Contact

Marine Lannes — PredExIA — [marine.lannes@predexia.com](mailto:marine.lannes@predexia.com)


```
