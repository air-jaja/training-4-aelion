import os
import pandas as pd
from sqlalchemy import create_engine, URL, text
from sqlalchemy.orm import sessionmaker
from .models import Base, IncidentBronze, TelemetryBronze, Machine, Maintenance, GoldDataset
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_db_engine():
    """Create and return a SQLAlchemy engine for PostgreSQL."""
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "your_database_name")
    db_user = os.getenv("DB_USER", "your_database_user")
    db_password = os.getenv("DB_PASSWORD", "your_database_password")

    # Construct the database URL using SQLAlchemy's URL class
    db_url = URL.create(
        drivername="postgresql+psycopg2",
        username=db_user,
        password=db_password,
        host=db_host,
        port=db_port,
        database=db_name
    )

    logger.debug(f"Database URL: {db_url}")

    # Create the SQLAlchemy engine
    engine = create_engine(db_url)
    return engine

def drop_tables():
    """Drop existing tables if they exist."""
    engine = get_db_engine()
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS incidents_bronze CASCADE"))
        logger.info("Dropped 'incidents_bronze' table if it existed.")
        conn.execute(text("DROP TABLE IF EXISTS telemetry_bronze CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS gold_dataset CASCADE"))
        logger.info("Dropped 'gold_dataset' table if it existed.")
        conn.execute(text("DROP TABLE IF EXISTS gold_dataset_csv CASCADE"))
        logger.info("Dropped 'gold_dataset_csv' table if it existed.")
        logger.info("Dropped 'telemetry_bronze' table if it existed.")

def create_tables():
    """Create all tables defined in the models."""
    engine = get_db_engine()
    Base.metadata.create_all(engine)
    logger.info("✅ Database tables 'incidents_bronze' and 'telemetry_bronze' created successfully.")

def find_latest_anonymized_file(directory, suffix="_anonymised.csv"):
    """
    Find the latest anonymized file in the given directory.
    """
    latest_file = None
    latest_time = 0

    if not os.path.exists(directory):
        logger.warning(f"Directory {directory} does not exist.")
        return None

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                file_path = os.path.join(root, file)
                file_time = os.path.getmtime(file_path)
                if file_time > latest_time:
                    latest_time = file_time
                    latest_file = file_path

    return latest_file

def load_incidents_data():
    """
    Load anonymized incidents data from CSV into the 'incidents_bronze' table.
    Uses the latest anonymized file from the 'incidents' directory.
    """
    # Look for anonymized files in the 'incidents' subdirectory
    input_dir = os.path.dirname(os.getenv("ANONYMIZED_FILE_PATH"))
    latest_incidents_file = find_latest_anonymized_file(input_dir)

    logger.info(f"Env variable for incidents ANONYMIZED_FILE_PATH =  {input_dir}.")
    logger.info(f"Aonymized data from {latest_incidents_file}.")
    
    if not latest_incidents_file:
        logger.warning(f"No anonymized incidents file found in {input_dir}.")
        return 0

    engine = get_db_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Load the CSV file
        df = pd.read_csv(latest_incidents_file)

        # Define the mapping between CSV columns and Incident model fields
        csv_to_model_mapping = {
            "date": "date",
            "shift": "shift",
            "machine_id": "machine_id",
            "severity": "severity",
            "type_surchauffe": "surchauffe",
            "type_baisse_pression": "baisse_pression",
            "vibration": "vibration",
            "type_bruit_mecanique": "bruit_mecanique",
            "type_surconsommation": "surconsommation",
            "type_blocage_mecanique": "blocage_mecanique",
            "type_alarme_capteur": "alarme_capteur",
            "type_arret_urgence": "arret_urgence",
            "type_defaut_qualite": "defaut_qualite",
            "comment": "comment",
            "flag_ok": "flag_ok"
        }

        # Get all model fields (excluding 'id')
        model_fields = [col.name for col in IncidentBronze.__table__.columns if col.name != "id"]

        row_count = 0
        for _, row in df.iterrows():
            # Initialize with empty strings for all model fields
            incident_data = {field: "" for field in model_fields}
            incident_data["flag_ok"] = "1"  # Default value

            # Map CSV columns to model fields (force string conversion)
            for csv_col, model_field in csv_to_model_mapping.items():
                if csv_col in df.columns:
                    value = row[csv_col]
                    incident_data[model_field] = str(value) if pd.notna(value) else ""

            # Create and insert the incident record (id is auto-generated)
            incident = IncidentBronze(**incident_data)
            session.add(incident)
            row_count += 1

        session.commit()
        logger.info(f"✅ Successfully loaded {row_count} incidents into 'incidents_bronze'.")
        return row_count

    except Exception as e:
        session.rollback()
        logger.error(f"❌Error loading incidents data: {e}")
        return 0
    finally:
        session.close()

def load_telemetry_data():
    """
    Load telemetry data from CSV into the 'telemetry_bronze' table.
    Uses the 'telemetry.csv' file from the 'datas' directory.
    """
    # Look for telemetry.csv in the 'datas' directory
    input_dir = os.getenv("INPUT_DATA_DIR", "./artifacts/ingestions/datas")
    telemetry_file = os.path.join(input_dir, "telemetry.csv")

    logger.info(f"Env variable for telemetry  INPUT_DATA_DIR =  {input_dir}.")
    logger.info(f"Telemetry data from {telemetry_file}.")
    
    if not os.path.exists(telemetry_file):
        logger.warning(f"Telemetry file not found at {telemetry_file}.")
        return 0

    engine = get_db_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Load the CSV file
        df = pd.read_csv(telemetry_file)

        # Define the mapping between CSV columns and Telemetry model fields
        csv_to_model_mapping = {
            "timestamp": "timestamp",
            "machine_id": "machine_id",
            "temperature_c": "temperature_c",
            "pressure_bar": "pressure_bar",
            "voltage_mean_v": "voltage_mean_v",
            "rotation_mean_rpm": "rotation_mean_rpm",
            "pieces_produced": "pieces_produced",
            "flag_ok": "flag_ok"
        }

        # Get all model fields (excluding 'id')
        model_fields = [col.name for col in TelemetryBronze.__table__.columns if col.name != "id"]

        row_count = 0
        for _, row in df.iterrows():
            # Initialize with empty strings for all model fields
            telemetry_data = {field: "" for field in model_fields}
            telemetry_data["flag_ok"] = "1"  # Default value

            # Map CSV columns to model fields (force string conversion)
            for csv_col, model_field in csv_to_model_mapping.items():
                if csv_col in df.columns:
                    value = row[csv_col]
                    telemetry_data[model_field] = str(value) if pd.notna(value) else ""

            # Create and insert the telemetry record (id is auto-generated)
            telemetry = TelemetryBronze(**telemetry_data)
            session.add(telemetry)
            row_count += 1

        session.commit()
        logger.info(f"✅ Successfully loaded {row_count} telemetry records into 'telemetry_bronze'.")
        return row_count

    except Exception as e:
        session.rollback()
        logger.error(f"❌ Error loading telemetry data: {e}")
        return 0
    finally:
        session.close()

def load_bronze_data():
    """Load both incidents and telemetry data into the database and log row counts."""
    # Drop existing tables
    drop_tables()

    # Create tables
    create_tables()

    # Load incidents data from anonymized files
    incidents_row_count = load_incidents_data()
    logger.info(f"✅ Total incidents inserted into 'incidents_bronze': {incidents_row_count}")

    # Load telemetry data from telemetry.csv
    telemetry_row_count = load_telemetry_data()
    logger.info(f"✅ Total telemetry records inserted into 'telemetry_bronze': {telemetry_row_count}")

    # Log summary
    logger.info(f"✅ Data loading complete. Incidents: {incidents_row_count}, Telemetry: {telemetry_row_count}")
    