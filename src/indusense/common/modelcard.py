"""Génération de model card (format Hugging Face) — commune ML et DL.

Extrait de `pipeleine_model_card_generation.ipynb`. Chemin du template et version lus
depuis `configs/modelcard.yaml`. `RANDOM_SEED` réutilisé pour l'identifiant unique.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jinja2

from indusense.common.config import CONFIG_DIR, OUTPUT_DIR, PROJECT_ROOT, load_config


def make_model_id(prefix: str, hyperparameters: dict[str, Any], version: str | None = None) -> str:
    """Identifiant unique et reproductible : `<prefix>-<version>-<hash des hyperparamètres>`.

    Un hash calculé à partir des hyperparamètres eux-mêmes (pas d'un tirage aléatoire) —
    deux exécutions avec les mêmes réglages produisent le même identifiant.
    """
    cfg = load_config("modelcard", domain="common")
    version = version or cfg["model_version"]

    params_hash = hashlib.sha256(json.dumps(hyperparameters, sort_keys=True).encode()).hexdigest()[:8]
    return f"{prefix}-{version}-{params_hash}"


def render_model_card(context: dict[str, Any], model_id: str) -> Path:
    """Rend `templates/modelcard_template.md` (Jinja2) avec le contexte fourni.

    Écrit dans `OUTPUT_DIR/model_card_<model_id>.md` — jamais un chemin en dur.
    """
    cfg = load_config("modelcard", domain="common")
    template_path = PROJECT_ROOT / cfg["template_path"]

    with open(template_path, encoding="utf-8") as f:
        template_source = f.read()

    rendered = jinja2.Template(template_source).render(**context)

    output_path = OUTPUT_DIR / f"model_card_{model_id}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered)

    return output_path
