import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Store the original INPUT_DATA_DIR value to reset it later
original_input_data_dir = "./artifacts/ingestions/datas"

def analyze_incidents():
    # Configure columns
    date_column = "date"
    shift_column = "shift"
    machine_column = "machine_id"
    severity_column = "severity"

    # Get input directory from environment variable (set by anonymize_data.py)
    input_dir = os.path.dirname(os.getenv("ANONYMIZED_FILE_PATH"))

    # Find all _anonymised.csv files in the input directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith("_anonymised.csv"):
                file_path = os.path.join(root, file)
                print(f"✅ File in used: {file_path}....")
                process_incidents_file(file_path, date_column, shift_column, machine_column, severity_column)

    # Reset INPUT_DATA_DIR to its original value for the next module
    os.environ["INPUT_DATA_DIR"] = original_input_data_dir

def process_incidents_file(file_path, date_column, shift_column, machine_column, severity_column):
    df = pd.read_csv(file_path)

    # Check for required columns
    required_columns = [date_column, shift_column, machine_column, severity_column]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' is missing in {file_path}.")

    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    # Identify all "type_*" columns
    type_columns = [col for col in df.columns if col.startswith("type_")]
    if not type_columns:
        raise ValueError(f"No columns starting with 'type_' found in {file_path}.")

    # Filter valid incidents
    valid_pannes_mask = (df[type_columns] == 1).any(axis=1)
    valid_pannes_df = df[valid_pannes_mask]

    # Create output directory
    output_dir = os.getenv("OUTPUT_DIR", "./artifacts/ingestions")
    output_graph_dir = os.path.join(output_dir, "incidents", datetime.now().strftime("%Y%m%d%H%M"),"graphs")
    os.makedirs(output_graph_dir, exist_ok=True)

    # Generate graphs
    generate_incidents_graphs(df, valid_pannes_df, type_columns, severity_column, machine_column, shift_column, date_column, output_graph_dir)
    print(f"✅ Incident graphs generated in: {output_graph_dir}")

def generate_incidents_graphs(df, valid_pannes_df, type_columns, severity_column, machine_column, shift_column, date_column, output_graph_dir):
    # Graph 1: Distribution by Day
    df["day"] = df[date_column].dt.date
    daily_distribution = df["day"].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    plt.plot(daily_distribution.index, daily_distribution.values, marker="o", linestyle="-", color="b")
    plt.title("Distribution of Incidents by Day")
    plt.xlabel("Day")
    plt.ylabel("Number of Incidents")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_graph_dir, "incidents_by_day.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Graph 2: Distribution by Week
    df["year"] = df[date_column].dt.year
    df["week_number"] = df[date_column].dt.isocalendar().week
    df["year_week"] = df["year"].astype(str).str[-2:] + "-W" + df["week_number"].astype(str).str.zfill(2)
    weekly_distribution = df["year_week"].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_distribution.index, weekly_distribution.values, marker="o", linestyle="-", color="g")
    plt.title("Distribution of Incidents by Week")
    plt.xlabel("Week")
    plt.ylabel("Number of Incidents")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_graph_dir, "incidents_by_week.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Graph 3: Distribution by Shift
    shift_distribution = df[shift_column].value_counts()
    plt.figure(figsize=(10, 6))
    shift_distribution.plot(kind="bar", color="r", edgecolor="black")
    plt.title("Distribution of Incidents by Shift")
    plt.xlabel("Shift")
    plt.ylabel("Number of Incidents")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_graph_dir, "incidents_by_shift.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Graph 4: Histogram by Machine
    machine_distribution = valid_pannes_df[machine_column].value_counts()
    plt.figure(figsize=(12, 6))
    machine_distribution.plot(kind="bar", color="orange", edgecolor="black")
    plt.title("Histogram of Incidents by Machine (Valid Only)")
    plt.xlabel("Machine ID")
    plt.ylabel("Number of Incidents")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_graph_dir, "incidents_by_machine.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Graph 5: Histogram by Signal
    signal_distribution = pd.Series(dtype=int)
    for type_col in type_columns:
        signal_counts = valid_pannes_df[type_col].value_counts().get(1, 0)
        signal_distribution[type_col] = signal_counts
    plt.figure(figsize=(12, 6))
    signal_distribution.plot(kind="bar", color="purple", edgecolor="black")
    plt.title("Histogram of Incidents by Signal (Valid Only)")
    plt.xlabel("Signal (Type of Failure)")
    plt.ylabel("Number of Incidents")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_graph_dir, "incidents_by_signal.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Graph 6: Correlation Heatmap
    severity_and_signals = valid_pannes_df[[severity_column] + type_columns]
    correlation_matrix = severity_and_signals.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Severity by Signal")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_graph_dir, "severity_correlation_by_signal.png"), dpi=300, bbox_inches="tight")
    plt.close()