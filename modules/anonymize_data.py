import os
import pandas as pd
from datetime import datetime

def anonymize_data():
    # Get environment variables
    input_file_path = os.getenv("INPUT_FILE_PATH", "./artifacts/ingestions/datas/releves_incidents.csv")
    output_base_dir = os.getenv("OUTPUT_DIR", "./artifacts/ingestions")
    anonymize_columns_str = os.getenv("ANONYMIZE_COLUMNS", "operator_name|operator_badge")

    # Load the CSV file
    df = pd.read_csv(input_file_path)

    # Anonymize specified columns
    columns_to_anonymize = anonymize_columns_str.split("|")
    for column in columns_to_anonymize:
        if column in df.columns:
            df[column] = "ANONYMOUS"
        else:
            print(f"⚠️ Warning: Column '{column}' not found. Skipping.")

    # Create output directory with timestamp
    date_time_output = datetime.now().strftime("%Y%m%d%H%M")
    output_dir = os.path.join(output_base_dir, date_time_output)
    os.makedirs(output_dir, exist_ok=True)

    # Save the anonymized file
    output_file_path = os.path.join(output_dir, "releves_incidents_anonymised.csv")
    df.to_csv(output_file_path, index=False)

    # Set the output file path as an environment variable for the next module
    os.environ["ANONYMIZED_FILE_PATH"] = output_file_path
    os.environ["INPUT_DATA_DIR"] = os.path.dirname(output_file_path)  # Set the directory containing the anonymized file

    print(f"✅ Anonymized file saved at: {output_file_path}")