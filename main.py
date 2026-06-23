import os
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

# ----------------------------
# Execute modules
# ----------------------------
if __name__ == "__main__":
    print("🚀 Starting data processing...")

    # 1. Anonymize data
    print("\n🔹 Anonymizing data...")
    from modules.anonymize_data import anonymize_data
    anonymize_data()

    # 2. Load data into PostgreSQL database
    print("\n🔹 Loading [Bronze] data into PostgreSQL database...")
    from modules.database.database_loader import load_bronze_data 
    load_bronze_data()

    # 3. Analyze incidents
    print("\n🔹 Analyzing incidents...")
    from modules.analyze_incidents import analyze_incidents
    analyze_incidents()

    # 4. Analyze anomalies in telemetry data
    print("\n🔹 Analyzing anomalies in telemetry data...")
    from modules.analyze_anomalies import analyze_anomalies
    analyze_anomalies()

    # 5. Analyze telemetry
    print("\n🔹 Analyzing telemetry...")
    from modules.analyze_telemetry import analyze_telemetry
    analyze_telemetry()

    print("\n✅ All processes completed successfully.")