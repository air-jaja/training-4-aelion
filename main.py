import os
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ----------------------------
# Configure environment variables
# ----------------------------
os.environ["INPUT_DATA_DIR"] = os.getenv("INPUT_DATA_DIR", "./artifacts/ingestions/datas")
os.environ["OUTPUT_DIR"] = os.getenv("OUTPUT_DIR", "./artifacts/outputs")
os.environ["ANONYMIZE_COLUMNS"] = os.getenv("ANONYMIZE_COLUMNS", "operator_name|operator_badge")
os.environ["VISION_DATA_DIR"] = os.getenv("VISION_DATA_DIR", "./data")
os.environ["VISION_OUTPUT_DIR"] = os.getenv("VISION_OUTPUT_DIR", "./artifacts/outputs/vision")

# ----------------------------
# Define command-line arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Data Processing Pipeline for InduSense Dataset")
parser.add_argument("--anonymize", action="store_true", help="Run anonymization module only")
parser.add_argument("--load-bronze", action="store_true", help="Load data into Bronze layer (PostgreSQL). Automatically runs anonymization first.")
parser.add_argument("--drop-tables", action="store_true", help="Drop existing tables before loading data (requires --load-bronze)")
parser.add_argument("--analyze-incidents", action="store_true", help="Run incidents analysis")
parser.add_argument("--analyze-anomalies", action="store_true", help="Run anomalies analysis")
parser.add_argument("--analyze-telemetry", action="store_true", help="Run telemetry analysis")
parser.add_argument("--load-gold", action="store_true", help="Load gold_dataset and gold_dataset_csv tables")
parser.add_argument("--dl-dataset", action="store_true", help="Prépare le dataset de vision (téléchargement si besoin, inventaire, split train/val, pipelines tf.data)")
parser.add_argument("--dl-albumentation", action="store_true", help="Génère un aperçu comparatif original vs augmenté (Albumentations) pour le dataset de vision")
parser.add_argument("--all", action="store_true", help="Run all modules in order: anonymize -> load-bronze -> load-gold -> analyze-incidents -> analyze-anomalies -> analyze-telemetry")

# ----------------------------
# Contextes ML & DL (package indusense) — liste ouverte, à affiner au fur et à mesure
# ----------------------------
CONTEXT_CHOICES = ["preprocessing", "ml", "dl", "tune", "explain", "emissions", "modelcard", "test", "all"]
parser.add_argument(
    "--context", nargs="+", choices=CONTEXT_CHOICES, default=None,
    help=(
        "Active une ou plusieurs activités du package indusense (ML & DL), indépendamment "
        "du pipeline IT-ops ci-dessus. Ex : --context preprocessing ml, ou --context all "
        "(all = preprocessing + ml + dl, sans tune/explain/modelcard qui restent à la demande)."
    ),
)

args = parser.parse_args()

# If no arguments are provided, run all modules by default
if not any(vars(args).values()):
    args.all = True

def run_module(module_name, module_function):
    """Helper function to run a module and log its execution."""
    print(f"\n🔹 {module_name}...")
    try:
        module_function()
        print(f"✅ {module_name} completed successfully.")
        return True
    except Exception as e:
        print(f"❌ Error in {module_name}: {e}")
        raise

def drop_tables():
    """Drop existing tables in the database."""
    from modules.database.database_loader import drop_tables as db_drop_tables
    print("\n🔹 Dropping existing tables...")
    try:
        db_drop_tables()
        print("✅ Tables dropped successfully.")
    except Exception as e:
        print(f"❌ Error dropping tables: {e}")
        raise

def load_bronze_data_with_anonymization():
    """Load bronze data with automatic anonymization first."""
    # Run anonymization first
    from modules.anonymize_data import anonymize_data
    run_module("Anonymizing data", anonymize_data)

    # Then load bronze data
    from modules.database.database_loader import load_bronze_data
    run_module("Loading [Bronze] data into PostgreSQL database", load_bronze_data)


def run_dl_dataset():
    """Prépare le dataset de vision (bottle/MVTec-AD) : téléchargement si absent,
    inventaire, split train/validation, pipelines tf.data (sans augmentation)."""
    from indusense.vision.dataset import prepare_bottle_pipeline

    data_dir = os.environ["VISION_DATA_DIR"]
    pipeline = prepare_bottle_pipeline(data_dir=data_dir, category="bottle")

    print(f"  Train (saines) : {len(pipeline.train_paths)} images")
    print(f"  Validation (saines) : {len(pipeline.val_paths)} images")
    print(f"  Classes de défauts (test) : {pipeline.defect_names}")
    for name, masks in pipeline.test_defect_masks.items():
        print(f"    - {name}: {masks.shape[0]} images / masques {masks.shape[1:]} ")


def run_dl_albumentation():
    """Génère un aperçu visuel comparant images saines originales vs augmentées
    (Albumentations), sauvegardé dans VISION_OUTPUT_DIR."""
    from indusense.vision.dataset import inventory, load_and_resize_image
    from indusense.vision.augment import build_train_augmentations, compare_original_vs_augmented

    data_dir = os.environ["VISION_DATA_DIR"]
    output_dir = os.environ["VISION_OUTPUT_DIR"]

    inv = inventory(f"{data_dir}/bottle")
    transform = build_train_augmentations()
    save_path = f"{output_dir}/augmentation_check.png"
    compare_original_vs_augmented(
        inv.train_good[:4], transform, load_and_resize_image, n_draws=3, save_path=save_path,
    )
    print(f"  Aperçu sauvegardé : {save_path}")

# ----------------------------
# Contextes ML & DL (package indusense)
# ----------------------------

def run_context_preprocessing():
    """Charge et valide les données préparées pour ML (gold_dataset) et DL (pipeline bottle)."""
    from indusense.ml.data import load_gold_dataset, temporal_split

    df, feats, horizon = load_gold_dataset()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = temporal_split(df, feats, horizon)
    print(f"  ML — {len(feats)} features, horizon {horizon} — train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    from indusense.vision.train import load_bottle_pipeline

    pipeline = load_bottle_pipeline()
    print(f"  DL — train={len(pipeline.train_paths)} val={len(pipeline.val_paths)} classes de défauts={pipeline.defect_names}")


def run_context_ml():
    """Entraîne et évalue le modèle ML (baseline + modèle retenu) sur le test set."""
    from indusense.ml.data import load_gold_dataset, temporal_split
    from indusense.ml.train import build_baseline_model, train_final_model, save_model
    from indusense.ml.evaluate import calibrate_threshold, evaluate_at_threshold
    from indusense.common.modelcard import make_model_id
    from indusense.common.config import load_config

    df, feats, horizon = load_gold_dataset()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = temporal_split(df, feats, horizon)

    baseline = build_baseline_model()
    baseline.fit(X_train, y_train)
    threshold_baseline = calibrate_threshold(baseline, X_val, y_val)
    metrics_baseline = evaluate_at_threshold(y_val, threshold_baseline, model=baseline, X=X_val)
    print(f"  Baseline — PR-AUC (val) = {metrics_baseline['prauc']:.4f}")

    final_model = train_final_model(X_train, y_train, X_val, y_val)
    threshold = calibrate_threshold(final_model, X_val, y_val)  # même modèle ici, calibration cohérente pour ce contexte rapide
    metrics_test = evaluate_at_threshold(y_test, threshold, model=final_model, X=X_test)
    print(f"  Modèle final — PR-AUC (test) = {metrics_test['prauc']:.4f}  AUC (test) = {metrics_test['auroc']:.4f}")

    cfg = load_config("train", domain="ml")
    model_id = make_model_id("predictive-maintenance-xgboost-24h", cfg["hyperparameters"])
    path = save_model(final_model, model_id)
    print(f"  Modèle sauvegardé : {path}")


def run_context_dl():
    """Entraîne et évalue l'autoencodeur DL (seuil calibré, IoU, AUC-ROC/AUC-PR)."""
    from indusense.vision.train import load_bottle_pipeline, train_autoencoder, save_autoencoder
    from indusense.vision.evaluate import calibrate_threshold, evaluate_image_level, evaluate_pixel_level
    from indusense.common.modelcard import make_model_id

    pipeline = load_bottle_pipeline()
    model, history = train_autoencoder(pipeline)
    print(f"  Autoencodeur entraîné — {len(history.history['loss'])} epochs — val_loss final = {history.history['val_loss'][-1]:.4f}")

    threshold = calibrate_threshold(model, pipeline)
    img_metrics = evaluate_image_level(model, pipeline, threshold)
    pixel_metrics = evaluate_pixel_level(model, pipeline)
    print(f"  Rappel = {img_metrics['recall']:.1%}  FP = {img_metrics['fpr']:.1%}  "
          f"IoU moyen = {pixel_metrics['mean_iou']:.4f}")

    model_id = make_model_id("bottle-autoencoder-ssim", {"threshold": threshold})
    path = save_autoencoder(model, model_id)
    print(f"  Modèle sauvegardé : {path}")


def run_context_tune():
    """Lance la recherche d'hyperparamètres Optuna (ML)."""
    from indusense.ml.data import load_gold_dataset, temporal_split
    from indusense.ml.tuning import run_optuna_study

    df, feats, horizon = load_gold_dataset()
    (X_train, y_train), _, _ = temporal_split(df, feats, horizon)

    study = run_optuna_study(X_train, y_train)
    print(f"  Meilleur essai : PR-AUC (CV) = {study.best_value:.4f}")
    print(f"  Hyperparamètres : {study.best_params}")
    print("  ⚠️ Penser à reporter ces valeurs dans configs/train.yaml (clé ml.hyperparameters) si retenues.")


def run_context_explain():
    """Explicabilité SHAP sur le modèle ML retenu (top features, anti-fuite)."""
    from indusense.ml.data import load_gold_dataset, temporal_split
    from indusense.ml.train import build_tuned_model
    from indusense.ml.explain import compute_shap_values, top_features_with_direction, concentration_share, leakage_check

    df, feats, horizon = load_gold_dataset()
    (X_train, y_train), (X_val, y_val), _ = temporal_split(df, feats, horizon)

    model = build_tuned_model()
    model.fit(X_train, y_train)

    shap_values, X_sample = compute_shap_values(model, X_val)
    top_features = top_features_with_direction(shap_values, X_sample, feats)
    print(top_features.to_string(index=False))
    print(f"\n  Concentration (top 10) : {concentration_share(shap_values, 10):.1%}")

    future_cols = [c for c in df.columns if c.startswith("future_incident_count_")]
    leak = leakage_check(top_features["feature"].tolist(), df[df.split_set == "train"], future_cols)
    print(f"  Features suspectes (anti-fuite) : {leak['à_vérifier'].sum()}/{len(leak)}")


def run_context_emissions():
    """Instrumente CodeCarbon autour d'un entraînement ML de référence (baseline)."""
    from indusense.ml.data import load_gold_dataset, temporal_split
    from indusense.ml.train import build_baseline_model
    from indusense.common.emissions import make_tracker, get_last_energy_kwh

    df, feats, horizon = load_gold_dataset()
    (X_train, y_train), _, _ = temporal_split(df, feats, horizon)

    tracker = make_tracker("main_ml_baseline")
    tracker.start()
    model = build_baseline_model()
    model.fit(X_train, y_train)
    emissions_kg = tracker.stop()
    kwh = get_last_energy_kwh("main_ml_baseline")

    print(f"  Émissions : {emissions_kg * 1000:.4f} gCO2eq  ({kwh:.6f} kWh)")


def run_context_modelcard():
    """Génère la model card du modèle ML retenu (entraînement + métriques + rendu Jinja2)."""
    from indusense.ml.data import load_gold_dataset, temporal_split
    from indusense.ml.train import build_tuned_model, train_final_model
    from indusense.ml.evaluate import calibrate_threshold, evaluate_at_threshold
    from indusense.common.config import load_config
    from indusense.common.modelcard import make_model_id, render_model_card

    df, feats, horizon = load_gold_dataset()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = temporal_split(df, feats, horizon)

    calib_model = build_tuned_model()
    calib_model.fit(X_train, y_train)
    threshold = calibrate_threshold(calib_model, X_val, y_val)

    final_model = train_final_model(X_train, y_train, X_val, y_val)
    metrics_test = evaluate_at_threshold(y_test, threshold, model=final_model, X=X_test)

    cfg = load_config("train", domain="ml")
    model_id = make_model_id("predictive-maintenance-xgboost-24h", cfg["hyperparameters"])

    context = {
        "model_id": model_id,
        "model_summary": "Prédiction de panne machine à 24h (XGBoost) — voir configs/ pour les réglages.",
        "results": f"PR-AUC (test) = {metrics_test['prauc']:.4f} ; AUC (test) = {metrics_test['auroc']:.4f} ; seuil = {threshold:.4f}",
    }
    path = render_model_card(context, model_id)
    print(f"  Model card générée : {path}")


def run_context_test():
    """Smoke test rapide de chaque contexte (budgets minimaux) — pensé pour la CI.

    Volontairement minimal pour l'instant : vérifie que chaque étape s'exécute sans erreur,
    pas que les résultats sont bons. À enrichir avec de vraies assertions au fur et à mesure.
    """
    import os

    os.environ["INDUSENSE_TRAIN_DL_EPOCHS"] = "2"
    os.environ["INDUSENSE_TUNE_ML_N_TRIALS"] = "2"
    os.environ["INDUSENSE_TUNE_ML_TIMEOUT_SECONDS"] = "60"

    print("  [test] preprocessing..."); run_context_preprocessing()
    print("  [test] ml..."); run_context_ml()
    print("  [test] dl..."); run_context_dl()
    print("  [test] tune..."); run_context_tune()
    print("  [test] explain..."); run_context_explain()


CONTEXT_RUNNERS = {
    "preprocessing": run_context_preprocessing,
    "ml": run_context_ml,
    "dl": run_context_dl,
    "tune": run_context_tune,
    "explain": run_context_explain,
    "emissions": run_context_emissions,
    "modelcard": run_context_modelcard,
    "test": run_context_test,
}
CONTEXT_ALL = ["preprocessing", "ml", "dl"]  # tune/explain/modelcard restent à la demande, pas dans "all"


# ----------------------------
# Execute modules based on arguments
# ----------------------------
if __name__ == "__main__":
    print("🚀 Starting data processing...")

    # Track if we've already run anonymization
    anonymization_done = False

    # 1. Drop tables if requested (only if --load-bronze or --all is specified)
    if args.drop_tables and (args.load_bronze or args.all):
        drop_tables()

    # 2. Handle anonymization and bronze loading
    if args.all or args.load_bronze:
        load_bronze_data_with_anonymization()
        anonymization_done = True
    elif args.anonymize:
        if not anonymization_done:
            from modules.anonymize_data import anonymize_data
            run_module("Anonymizing data", anonymize_data)
            anonymization_done = True
        else:
            print("⚠️ Anonymization has already been completed in this run. Skipping.")

    # 3. Load gold datasets (typed and CSV string tables)
    if args.all or args.load_gold:
        from modules.load_gold_dataset import load_gold_datasets
        run_module("Loading [Gold] datasets into PostgreSQL database", load_gold_datasets)

    # 4. Analyze incidents
    if args.all or args.analyze_incidents:
        from modules.analyze_incidents import analyze_incidents
        run_module("Analyzing incidents", analyze_incidents)

    # 5. Analyze anomalies in telemetry data
    if args.all or args.analyze_anomalies:
        from modules.analyze_anomalies import analyze_anomalies
        run_module("Analyzing anomalies in telemetry data", analyze_anomalies)

    # 6. Analyze telemetry
    if args.all or args.analyze_telemetry:
        from modules.analyze_telemetry import analyze_telemetry
        run_module("Analyzing telemetry", analyze_telemetry)

    # 7. Préparation du dataset de vision (bottle/MVTec-AD)
    if args.dl_dataset:
        run_module("Préparation du dataset de vision (dl-dataset)", run_dl_dataset)

    # 8. Aperçu de l'augmentation (Albumentations) sur les images saines
    if args.dl_albumentation:
        run_module("Aperçu d'augmentation (dl-albumentation)", run_dl_albumentation)

    # 9. Contextes ML & DL (package indusense)
    if args.context:
        contexts = CONTEXT_ALL if "all" in args.context else args.context
        for ctx in contexts:
            run_module(f"Contexte '{ctx}'", CONTEXT_RUNNERS[ctx])

    print("\n✅ All selected processes completed successfully.")
