"""Résolution centralisée des chemins et paramètres globaux du projet.

Priorité de résolution pour toute valeur : variable d'environnement > fichier YAML
de l'activité > valeur par défaut définie ici. Aucun autre module du package ne doit
contenir de chemin en dur — tout doit passer par ce module.

Constantes exposées : RAW_DIR, MODELS_DIR, OUTPUT_DIR, FIGURES_DIR, RANDOM_SEED.
`FIGURES_DIR` est toujours un sous-répertoire de `OUTPUT_DIR` (sauf si explicitement
surchargé par sa propre variable d'environnement).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml


def _find_project_root(start: Optional[Path] = None) -> Path:
    """Remonte depuis `start` (ou le cwd) jusqu'au dossier contenant pyproject.toml.

    Robuste à l'endroit d'où le notebook/script est lancé (racine du projet ou un
    sous-dossier comme `notebooks/`) — même logique que celle déjà validée dans le
    pipeline DL pour résoudre `src/`.
    """
    current = start or Path.cwd()
    while not (current / "pyproject.toml").exists() and current != current.parent:
        current = current.parent
    return current


def _env_or_default(env_var: str, default: str) -> str:
    return os.environ.get(env_var, default)


PROJECT_ROOT = _find_project_root()
CONFIG_DIR = PROJECT_ROOT / "configs"

# --- Constantes globales : env var > défaut. Résolues une seule fois à l'import. ---
RAW_DIR = Path(_env_or_default("INDUSENSE_RAW_DIR", str(PROJECT_ROOT / "artifacts/ingestions/datas")))
MODELS_DIR = Path(_env_or_default("INDUSENSE_MODELS_DIR", str(PROJECT_ROOT / "artifacts/models")))
OUTPUT_DIR = Path(_env_or_default("INDUSENSE_OUTPUT_DIR", str(PROJECT_ROOT / "artifacts/ingestions/output")))

# FIGURES_DIR dérive de OUTPUT_DIR par défaut (garantit qu'il en reste un sous-répertoire
# même si OUTPUT_DIR est surchargé), mais reste individuellement surchargeable si besoin.
FIGURES_DIR = Path(_env_or_default("INDUSENSE_FIGURES_DIR", str(OUTPUT_DIR / "figures")))

RANDOM_SEED = int(_env_or_default("INDUSENSE_RANDOM_SEED", "42"))

# Création automatique des dossiers de sortie (pas de service d'initialisation séparé :
# le simple fait d'importer ce module garantit que ces dossiers existent).
for _d in (MODELS_DIR, OUTPUT_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def load_config(activity: str, domain: Optional[str] = None) -> dict[str, Any]:
    """Charge `configs/<activity>.yaml`.

    Si `domain` ("ml" ou "dl") est précisé, retourne uniquement cette section du fichier.
    Toute clé de la section retournée peut être surchargée par une variable d'environnement
    `INDUSENSE_<ACTIVITY>_<DOMAIN>_<CLE>` (tout en majuscules) — utile en CI/CD sans
    avoir à modifier le YAML.

    Exemples :
        load_config("train", domain="ml")   -> section "ml" de configs/train.yaml
        load_config("emissions")            -> fichier configs/emissions.yaml entier
    """
    path = CONFIG_DIR / f"{activity}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config introuvable : {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    section = data.get(domain, {}) if domain else data
    if domain and domain not in data:
        raise KeyError(f"Domaine '{domain}' absent de {path} (clés disponibles : {list(data.keys())})")

    return _apply_env_overrides(section, activity, domain)


def _coerce(raw: str, reference: Any) -> Any:
    """Convertit la valeur d'environnement (toujours une chaîne) vers le type de la valeur YAML d'origine."""
    if isinstance(reference, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(reference, int):
        return int(raw)
    if isinstance(reference, float):
        return float(raw)
    return raw


def _apply_env_overrides(section: dict[str, Any], activity: str, domain: Optional[str]) -> dict[str, Any]:
    prefix = f"INDUSENSE_{activity.upper()}_{(domain or '').upper()}_".replace("__", "_")
    for key in list(section.keys()):
        env_key = f"{prefix}{key.upper()}"
        if env_key in os.environ:
            section[key] = _coerce(os.environ[env_key], section[key])
    return section
