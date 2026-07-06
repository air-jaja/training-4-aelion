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
        # Try reading with comma separator first (most common)
        df = pd.read_csv(csv_file, sep=",")
        logger.info(f"CSV file read successfully with comma separator. Shape: {df.shape}")
    except Exception as e:
        logger.warning(f"Failed to read CSV with comma separator: {e}")
        try:
            # Fallback to tab separator
            df = pd.read_csv(csv_file, sep="\t")
            logger.info(f"CSV file read successfully with tab separator. Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to read CSV with tab separator: {e}")
            raise

    # Debug: Print the first few columns to verify
    logger.info(f"CSV columns: {list(df.columns[:5])}...")  # Print first 5 columns

    # --- 2. Clean the data ---
    # Replace empty strings with None to avoid SQL errors
    df.replace("", None, inplace=True)

    # Convert date columns to datetime (if they exist)
    date_columns = ["window_start", "window_end"]
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], format="%d/%m/%Y %H:%M")
                logger.info(f"Converted column '{col}' to datetime.")
            except Exception as e:
                logger.warning(f"Error converting column '{col}' to datetime: {e}")

    # --- 3. Map CSV columns to GoldDataset model columns ---
    # Get the column names from the GoldDataset model (excluding 'id')
    gold_dataset_columns = [col.name for col in GoldDataset.__table__.columns if col.name != "id"]
    logger.info(f"GoldDataset model columns: {gold_dataset_columns[:5]}...")  # Print first 5 model columns

    # Check if CSV columns match model columns
    csv_columns = list(df.columns)
    
    # If CSV has a single column with all names concatenated (common error with wrong separator)
    if len(csv_columns) == 1:
        # Try to split the single column name by comma
        single_col_name = csv_columns[0]
        if "," in single_col_name:
            logger.warning(f"CSV appears to have a single column with concatenated names. Splitting by comma...")
            # Split the column name and use as new column names
            new_columns = [col.strip() for col in single_col_name.split(",")]
            df.columns = new_columns
            csv_columns = new_columns
            logger.info(f"Split into {len(new_columns)} columns: {new_columns[:5]}...")

    # Check for missing or extra columns
    model_columns = set(gold_dataset_columns)
    csv_columns_set = set(csv_columns)
    missing_columns = model_columns - csv_columns_set
    extra_columns = csv_columns_set - model_columns

    if missing_columns:
        logger.warning(f"Columns in GoldDataset model but not in CSV: {missing_columns}")
    if extra_columns:
        logger.warning(f"Columns in CSV but not in GoldDataset model: {extra_columns}")

    # If there are missing columns, try to map CSV columns to model columns
    # This handles cases where CSV column names are slightly different
    if missing_columns or extra_columns:
        logger.info("Attempting to map CSV columns to GoldDataset model...")
        # Create a mapping from CSV column names to model column names
        # This is a simple example - you may need to customize this based on your actual CSV
        column_mapping = {
            # Add any known mappings here, e.g.:
            # "machine_id": "machine_id_std",
            # "timestamp": "window_start",
        }
        # Apply mapping
        df.rename(columns=column_mapping, inplace=True)
        # Update CSV columns after mapping
        csv_columns = list(df.columns)
        csv_columns_set = set(csv_columns)
        missing_columns = model_columns - csv_columns_set
        extra_columns = csv_columns_set - model_columns

    # Keep only the columns that exist in the model
    available_columns = [col for col in gold_dataset_columns if col in csv_columns]
    df_filtered = df[available_columns].copy()
    logger.info(f"Filtered DataFrame to {len(available_columns)} columns: {available_columns[:5]}...")

    # --- 4. Create a connection to PostgreSQL and create tables ---
    engine = get_db_engine()

    # Create all tables defined in the models (including gold_dataset)
    try:
        Base.metadata.create_all(engine)
        logger.info("Database tables created/verified successfully.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

    # --- 5. Drop and recreate the gold_dataset table to ensure clean state ---
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

    # --- 6. Load data into PostgreSQL using SQLAlchemy ORM ---
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Convert filtered DataFrame to dictionary of records
        records = df_filtered.to_dict(orient="records")

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
        # Try reading with comma separator first
        df = pd.read_csv(csv_file, sep=",")
        logger.info(f"CSV file read successfully with comma separator. Shape: {df.shape}")
    except Exception as e:
        logger.warning(f"Failed to read CSV with comma separator: {e}")
        try:
            # Fallback to tab separator
            df = pd.read_csv(csv_file, sep="\t")
            logger.info(f"CSV file read successfully with tab separator. Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to read CSV with tab separator: {e}")
            raise

    # Debug: Print the first few columns to verify
    logger.info(f"CSV columns: {list(df.columns[:5])}...")

    # Handle case where CSV has a single column with concatenated names
    if len(df.columns) == 1:
        single_col_name = df.columns[0]
        if "," in single_col_name:
            logger.warning(f"CSV appears to have a single column with concatenated names. Splitting by comma...")
            new_columns = [col.strip() for col in single_col_name.split(",")]
            df.columns = new_columns
            logger.info(f"Split into {len(new_columns)} columns: {new_columns[:5]}...")

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
