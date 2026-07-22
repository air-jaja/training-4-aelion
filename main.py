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

    print("\n✅ All selected processes completed successfully.")
