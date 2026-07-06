import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging
from dotenv import load_dotenv
from .database.models import Base, GoldDataset

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
    Load the gold_dataset from a CSV file into PostgreSQL using SQLAlchemy ORM.
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

    # --- 3. Create a connection to PostgreSQL and create tables ---
    engine = get_db_engine()

    # Create all tables defined in the models (including gold_dataset)
    try:
        Base.metadata.create_all(engine)
        logger.info("Database tables created/verified successfully.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

    # --- 4. Drop and recreate the gold_dataset table to ensure clean state ---
    try:
        with engine.connect() as conn:
            # Drop the table if it exists (to avoid conflicts with existing data)
            conn.execute(text("DROP TABLE IF EXISTS gold_dataset CASCADE;"))
            conn.commit()
            logger.info("Dropped 'gold_dataset' table if it existed.")

            # Recreate the table using SQLAlchemy ORM
            Base.metadata.create_all(engine)
            logger.info("Recreated 'gold_dataset' table successfully.")
    except Exception as e:
        logger.error(f"Error recreating 'gold_dataset' table: {e}")
        raise

    # --- 5. Load data into PostgreSQL using SQLAlchemy ORM ---
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Convert DataFrame to dictionary of records
        records = df.to_dict(orient="records")

        # Insert records into the gold_dataset table
        for record in records:
            # Create a new GoldDataset instance
            gold_record = GoldDataset(**record)
            session.add(gold_record)

        session.commit()
        logger.info(f"✅ {len(records)} records loaded successfully into 'gold_dataset' table!")
    except Exception as e:
        session.rollback()
        logger.error(f"Error loading data into 'gold_dataset': {e}")
        raise
    finally:
        session.close()


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
        logger.info("✅ Data loaded successfully into 'gold_dataset_csv' table!")
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

    logger.info("✅ Both 'gold_dataset' and 'gold_dataset_csv' tables loaded successfully!")
