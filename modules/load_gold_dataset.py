import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def get_db_engine():
    """Create and return a SQLAlchemy engine for PostgreSQL using environment variables."""
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "indusense_db")
    db_user = os.getenv("DB_USER", "indusense_user")
    db_password = os.getenv("DB_PASSWORD", "ThEPssW0rd")

    # Construct the database URL
    db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    logger.debug(f"Database URL: {db_url}")

    # Create the SQLAlchemy engine
    engine = create_engine(db_url)
    return engine


def load_gold_dataset():
    """
    Load the gold_dataset from a CSV file into PostgreSQL with proper data types.
    The CSV file is expected to be in the INPUT_DATA_DIR (default: ./artifacts/ingestions/datas).
    The table 'gold_dataset' will be created with typed columns (TIMESTAMP, FLOAT, INTEGER, BOOLEAN, VARCHAR).
    """
    # Get the input directory from environment variables
    input_dir = os.getenv("INPUT_DATA_DIR", "./artifacts/ingestions/datas")
    csv_file = os.path.join(input_dir, "gold_dataset.csv")

    logger.info(f"Loading gold_dataset from: {csv_file}")

    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        logger.error(f"CSV file not found: {csv_file}")
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    # --- 1. Read the CSV file ---
    try:
        df = pd.read_csv(csv_file, sep="\t")  # Use tab separator as per the original code
        logger.info(f"CSV file read successfully. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise

    # --- 2. Clean the data ---
    # Replace empty strings with None to avoid SQL errors
    df.replace("", None, inplace=True)

    # Convert date columns to datetime
    date_columns = ["window_start", "window_end"]
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], format="%d/%m/%Y %H:%M")
                logger.info(f"Converted column '{col}' to datetime.")
            except Exception as e:
                logger.warning(f"Error converting column '{col}' to datetime: {e}")

    # --- 3. Create a connection to PostgreSQL ---
    engine = get_db_engine()

    # --- 4. Create the 'gold_dataset' table (if it doesn't exist) ---
    create_table_query = """
    CREATE TABLE IF NOT EXISTS gold_dataset (
        machine_id_std VARCHAR(50),
        window_start TIMESTAMP,
        temp_mean_1h FLOAT,
        temp_max_1h FLOAT,
        pressure_mean_1h FLOAT,
        pressure_max_1h FLOAT,
        voltage_mean_1h FLOAT,
        voltage_max_1h FLOAT,
        rotation_mean_1h FLOAT,
        rotation_max_1h FLOAT,
        pieces_produced_sum_1h INTEGER,
        window_end TIMESTAMP,
        temp_mean_6h FLOAT,
        temp_max_6h FLOAT,
        temp_std_6h FLOAT,
        pressure_mean_6h FLOAT,
        pressure_max_6h FLOAT,
        pressure_std_6h FLOAT,
        voltage_mean_6h FLOAT,
        voltage_max_6h FLOAT,
        voltage_std_6h FLOAT,
        rotation_mean_6h FLOAT,
        rotation_max_6h FLOAT,
        rotation_std_6h FLOAT,
        temp_mean_12h FLOAT,
        temp_max_12h FLOAT,
        temp_std_12h FLOAT,
        pressure_mean_12h FLOAT,
        pressure_max_12h FLOAT,
        pressure_std_12h FLOAT,
        voltage_mean_12h FLOAT,
        voltage_max_12h FLOAT,
        voltage_std_12h FLOAT,
        rotation_mean_12h FLOAT,
        rotation_max_12h FLOAT,
        rotation_std_12h FLOAT,
        temp_mean_24h FLOAT,
        temp_max_24h FLOAT,
        temp_std_24h FLOAT,
        pressure_mean_24h FLOAT,
        pressure_max_24h FLOAT,
        pressure_std_24h FLOAT,
        voltage_mean_24h FLOAT,
        voltage_max_24h FLOAT,
        voltage_std_24h FLOAT,
        rotation_mean_24h FLOAT,
        rotation_max_24h FLOAT,
        rotation_std_24h FLOAT,
        temp_trend_6h FLOAT,
        pressure_trend_6h FLOAT,
        voltage_trend_6h FLOAT,
        rotation_trend_6h FLOAT,
        temp_zscore_24h FLOAT,
        temp_delta_1h FLOAT,
        temp_delta_3h FLOAT,
        pressure_delta_1h FLOAT,
        pressure_delta_3h FLOAT,
        rotation_delta_1h FLOAT,
        rotation_delta_3h FLOAT,
        voltage_delta_1h FLOAT,
        voltage_delta_3h FLOAT,
        temp_zscore_machine FLOAT,
        pressure_zscore_machine FLOAT,
        pieces_produced_sum_24h INTEGER,
        capacity_utilization_pct FLOAT,
        incident_count_1h INTEGER,
        incident_max_severity_1h INTEGER,
        incident_count_prev_24h INTEGER,
        incident_max_severity_prev_24h INTEGER,
        incident_count_prev_7d INTEGER,
        hours_since_last_incident FLOAT,
        type_surchauffe BOOLEAN,
        type_baisse_pression BOOLEAN,
        type_vibration BOOLEAN,
        type_bruit_mecanique BOOLEAN,
        type_surconsommation BOOLEAN,
        type_blocage_mecanique BOOLEAN,
        type_alarme_capteur BOOLEAN,
        type_arret_urgence BOOLEAN,
        type_defaut_qualite BOOLEAN,
        type_surchauffe_count_prev_24h INTEGER,
        type_baisse_pression_count_prev_24h INTEGER,
        type_vibration_count_prev_24h INTEGER,
        type_bruit_mecanique_count_prev_24h INTEGER,
        type_surconsommation_count_prev_24h INTEGER,
        type_blocage_mecanique_count_prev_24h INTEGER,
        type_alarme_capteur_count_prev_24h INTEGER,
        type_arret_urgence_count_prev_24h INTEGER,
        type_defaut_qualite_count_prev_24h INTEGER,
        days_since_last_maintenance FLOAT,
        maintenance_count_prev_30d INTEGER,
        future_incident_count_6h INTEGER,
        label_failure_next_6h BOOLEAN,
        future_incident_count_12h INTEGER,
        label_failure_next_12h BOOLEAN,
        future_incident_count_24h INTEGER,
        label_failure_next_24h BOOLEAN,
        future_incident_count_48h INTEGER,
        label_failure_next_48h BOOLEAN,
        split_set VARCHAR(50)
    );
    """

    try:
        with engine.connect() as conn:
            # Drop the table if it exists (optional)
            conn.execute(text("DROP TABLE IF EXISTS gold_dataset CASCADE;"))
            conn.commit()
            logger.info("Dropped 'gold_dataset' table if it existed.")

            # Create the table
            conn.execute(text(create_table_query))
            conn.commit()
            logger.info("Created 'gold_dataset' table successfully.")
    except Exception as e:
        logger.error(f"Error creating 'gold_dataset' table: {e}")
        raise

    # --- 5. Load data into PostgreSQL ---
    try:
        df.to_sql(
            name="gold_dataset",
            con=engine,
            if_exists="append",  # Append data without overwriting the table
            index=False,
            method="multi",  # Optimize bulk insertion
        )
        logger.info("Data loaded successfully into 'gold_dataset' table!")
    except Exception as e:
        logger.error(f"Error loading data into 'gold_dataset': {e}")
        raise


def load_gold_dataset_csv():
    """
    Load the gold_dataset from a CSV file into PostgreSQL with all columns as VARCHAR.
    The CSV file is expected to be in the INPUT_DATA_DIR (default: ./artifacts/ingestions/datas).
    The table 'gold_dataset_csv' will be created with all columns as VARCHAR(255).
    """
    # Get the input directory from environment variables
    input_dir = os.getenv("INPUT_DATA_DIR", "./artifacts/ingestions/datas")
    csv_file = os.path.join(input_dir, "gold_dataset.csv")

    logger.info(f"Loading gold_dataset_csv from: {csv_file}")

    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        logger.error(f"CSV file not found: {csv_file}")
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    # --- 1. Read the CSV file ---
    try:
        df = pd.read_csv(csv_file, sep="\t")  # Use tab separator as per the original code
        logger.info(f"CSV file read successfully. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise

    # --- 2. Convert all columns to strings ---
    # Replace empty strings with None to avoid SQL errors
    df.replace("", None, inplace=True)
    # Convert all columns to strings
    df = df.astype(str)
    # Replace 'nan' strings with None
    df.replace("nan", None, inplace=True)

    # --- 3. Create a connection to PostgreSQL ---
    engine = get_db_engine()

    # --- 4. Create the 'gold_dataset_csv' table (if it doesn't exist) ---
    # Generate the CREATE TABLE query dynamically with all columns as VARCHAR(255)
    columns = df.columns.tolist()
    column_definitions = [f"{col} VARCHAR(255)" for col in columns]
    create_table_query = f"CREATE TABLE IF NOT EXISTS gold_dataset_csv ({', '.join(column_definitions)});"

    try:
        with engine.connect() as conn:
            # Drop the table if it exists (optional)
            conn.execute(text("DROP TABLE IF EXISTS gold_dataset_csv CASCADE;"))
            conn.commit()
            logger.info("Dropped 'gold_dataset_csv' table if it existed.")

            # Create the table
            conn.execute(text(create_table_query))
            conn.commit()
            logger.info("Created 'gold_dataset_csv' table successfully.")
    except Exception as e:
        logger.error(f"Error creating 'gold_dataset_csv' table: {e}")
        raise

    # --- 5. Load data into PostgreSQL ---
    try:
        df.to_sql(
            name="gold_dataset_csv",
            con=engine,
            if_exists="append",  # Append data without overwriting the table
            index=False,
            method="multi",  # Optimize bulk insertion
        )
        logger.info("Data loaded successfully into 'gold_dataset_csv' table!")
    except Exception as e:
        logger.error(f"Error loading data into 'gold_dataset_csv': {e}")
        raise


def load_gold_datasets():
    """
    Load both 'gold_dataset' (typed columns) and 'gold_dataset_csv' (all VARCHAR) tables.
    This is the main function to call from main.py.
    """
    logger.info("Starting gold datasets loading...")

    # Load the typed gold_dataset table
    load_gold_dataset()

    # Load the all-VARCHAR gold_dataset_csv table
    load_gold_dataset_csv()

    logger.info("Both 'gold_dataset' and 'gold_dataset_csv' tables loaded successfully!")
