import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from .models import Base, Incident, Telemetry
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

    # Construct the database URL
    db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    # Create the SQLAlchemy engine
    engine = create_engine(db_url)
    return engine

def create_tables():
    """Create all tables defined in the models."""
    engine = get_db_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables created successfully.")

def load_incidents_data():
    """Load anonymized incidents data from CSV into the database."""
    input_dir = os.getenv("INPUT_DATA_DIR", "./artifacts/ingestions")
    engine = get_db_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Find the latest anonymized incidents file
        latest_incidents_file = None
        latest_time = 0

        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith("_anonymised.csv"):
                    file_path = os.path.join(root, file)
                    file_time = os.path.getmtime(file_path)
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_incidents_file = file_path

        if not latest_incidents_file:
            logger.warning("No anonymized incidents file found.")
            return

        # Load the CSV file
        df = pd.read_csv(latest_incidents_file)

        # Check required columns
        required_columns = ["date", "shift", "machine_id", "severity"]
        for column in required_columns:
            if column not in df.columns:
                logger.error(f"Required column '{column}' not found in {latest_incidents_file}.")
                return

        # Insert data into the database
        for _, row in df.iterrows():
            # Create a dictionary for the incident data
            incident_data = {
                "date": row["date"],
                "shift": row["shift"],
                "machine_id": row["machine_id"],
                "severity": row["severity"]
            }

            # Add dynamic type_* columns
            type_columns = [col for col in df.columns if col.startswith("type_")]
            for col in type_columns:
                incident_data[col] = row[col]

            # Create and insert the incident record
            incident = Incident(**incident_data)
            session.add(incident)

        session.commit()
        logger.info(f"Successfully loaded incidents data from {latest_incidents_file}.")

    except Exception as e:
        session.rollback()
        logger.error(f"Error loading incidents data: {e}")
    finally:
        session.close()

def load_telemetry_data():
    """Load telemetry data from CSV into the database."""
    input_dir = os.getenv("INPUT_DATA_DIR", "./artifacts/ingestions")
    engine = get_db_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Find the telemetry.csv file
        telemetry_file = None
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file == "telemetry.csv":
                    telemetry_file = os.path.join(root, file)
                    break

        if not telemetry_file:
            logger.warning("No telemetry file found.")
            return

        # Load the CSV file
        df = pd.read_csv(telemetry_file)

        # Check required columns
        required_columns = ["timestamp", "machine_id", "temperature_c", "pressure_bar", "voltage_mean_v", "rotation_mean_rpm", "pieces_produced"]
        for column in required_columns:
            if column not in df.columns:
                logger.error(f"Required column '{column}' not found in {telemetry_file}.")
                return

        # Insert data into the database
        for _, row in df.iterrows():
            telemetry = Telemetry(
                timestamp=row["timestamp"],
                machine_id=row["machine_id"],
                temperature_c=row["temperature_c"],
                pressure_bar=row["pressure_bar"],
                voltage_mean_v=row["voltage_mean_v"],
                rotation_mean_rpm=row["rotation_mean_rpm"],
                pieces_produced=row["pieces_produced"]
            )
            session.add(telemetry)

        session.commit()
        logger.info(f"Successfully loaded telemetry data from {telemetry_file}.")

    except Exception as e:
        session.rollback()
        logger.error(f"Error loading telemetry data: {e}")
    finally:
        session.close()

def load_bronze_data():
    """Load both incidents and telemetry data into the database."""
    create_tables()
    load_incidents_data()
    load_telemetry_data()