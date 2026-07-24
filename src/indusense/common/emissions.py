"""Instrumentation CodeCarbon — mesure d'émissions, commune ML et DL.

Extrait de `pipeline_eco_conception_codecarbon.ipynb`. Intensité carbone, intervalle de
mesure et nom de fichier lus depuis `configs/emissions.yaml`. Chemin de sortie via
`indusense.common.config.OUTPUT_DIR` — jamais codé en dur.
"""
from __future__ import annotations

import pandas as pd
from codecarbon import EmissionsTracker

from indusense.common.config import OUTPUT_DIR, load_config


def make_tracker(project_name: str) -> EmissionsTracker:
    """Construit un `EmissionsTracker` prêt à l'emploi (`.start()` / `.stop()`).

    Intensité carbone **forcée** (pas de géolocalisation automatique) — observée peu fiable
    et non reproductible d'une exécution à l'autre dans nos tests ; valeur de secours
    représentative du mix électrique français, surchargeable via `configs/emissions.yaml`.
    """
    cfg = load_config("emissions", domain="common")
    return EmissionsTracker(
        project_name=project_name,
        output_dir=str(OUTPUT_DIR),
        output_file=cfg["emissions_filename"],
        force_carbon_intensity_g_co2e_kwh=cfg["carbon_intensity_g_co2e_kwh"],
        log_level="error",
        measure_power_secs=cfg["measure_power_secs"],
        allow_multiple_runs=True,
    )


def get_last_energy_kwh(project_name: str) -> float:
    """Relit `emissions.csv` pour récupérer le kWh authentique de la dernière ligne d'un projet."""
    cfg = load_config("emissions", domain="common")
    emissions_df = pd.read_csv(OUTPUT_DIR / cfg["emissions_filename"])
    return float(emissions_df[emissions_df.project_name == project_name].iloc[-1]["energy_consumed"])
