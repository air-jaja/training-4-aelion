import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_anomalies():
    # Define columns for telemetry data
    machine_column = "machine_id"
    properties = {
        "temperature_c": "Temperature (C)",
        "pressure_bar": "Pressure (Bar)",
        "voltage_mean_v": "Voltage (V)",
        "rotation_mean_rpm": "Rotation (RPM)",
        "pieces_produced": "Pieces Produced"
    }

    # Get input directory from environment variable
    input_dir = os.getenv("INPUT_DATA_DIR", "./artifacts/ingestions/datas")

    # Find telemetry.csv file
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file == "telemetry.csv":
                file_path = os.path.join(root, file)
                print(f"✅ File in used: {file_path}....")
                process_anomalies_file(file_path, machine_column, properties)

def process_anomalies_file(file_path, machine_column, properties):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Check if the required columns exist
    required_columns = [machine_column] + list(properties.keys())
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"⚠️ Warning: The following columns are missing: {missing_columns}. Skipping anomaly analysis.")
        return

    # Create output directory for anomaly results
    output_dir = os.getenv("OUTPUT_DIR", "./artifacts/ingestions")
    output_anomaly_dir = os.path.join(output_dir, "telemetry", datetime.now().strftime("%Y%m%d%H%M"), "datas_generated")
    output_graph_dir = os.path.join(output_dir, "telemetry", datetime.now().strftime("%Y%m%d%H%M"), "graphs")
    os.makedirs(output_anomaly_dir, exist_ok=True)
    os.makedirs(output_graph_dir, exist_ok=True)

    # Calculate outliers for each property and machine
    anomaly_results = {}
    for property_name, property_display in properties.items():
        outliers_by_machine = detect_outliers_by_machine(df, machine_column, property_name)
        anomaly_results[property_display] = outliers_by_machine

    # Save results to a CSV file
    results_df = pd.DataFrame(anomaly_results)
    results_df.index.name = "Machine ID"
    results_file_path = os.path.join(output_anomaly_dir, "anomalies_by_machine.csv")
    results_df.to_csv(results_file_path)

    # Generate graphs for each property
    generate_anomaly_graphs(anomaly_results, output_graph_dir, machine_column)

    print(f"✅ Anomaly analysis results saved at: {results_file_path}")
    print(f"✅ Anomaly analysis graphics saved in: {output_graph_dir}")

def detect_outliers_by_machine(df, machine_column, property_name):
    """
    Detect outliers for a given property by machine using the IQR method.
    Returns a dictionary with machine IDs as keys and outlier counts as values.
    """
    outliers_by_machine = {}

    # Group data by machine
    grouped = df.groupby(machine_column)[property_name]

    for machine_id, group in grouped:
        # Calculate Q1, Q3, and IQR
        Q1 = group.quantile(0.25)
        Q3 = group.quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers
        outliers = ((group < lower_bound) | (group > upper_bound)).sum()
        outliers_by_machine[machine_id] = outliers

    return outliers_by_machine

def generate_anomaly_graphs(anomaly_results, output_anomaly_dir, machine_column):
    """
    Generate bar graphs for the distribution of outliers by machine for each property.
    """
    for property_display, outliers_by_machine in anomaly_results.items():
        # Create a DataFrame for the current property
        df = pd.DataFrame({
            "Machine ID": list(outliers_by_machine.keys()),
            "Outlier Count": list(outliers_by_machine.values())
        })

        # Plot the graph
        plt.figure(figsize=(12, 6))
        plt.bar(df["Machine ID"], df["Outlier Count"], color="skyblue", edgecolor="black")
        plt.title(f"Distribution of Outliers by Machine - {property_display}")
        plt.xlabel("Machine ID")
        plt.ylabel("Number of Outliers")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the graph
        graph_filename = f"outliers_by_machine_{property_display.replace(' ', '_').replace('(', '').replace(')', '').lower()}.png"
        plt.savefig(os.path.join(output_anomaly_dir, graph_filename), dpi=300, bbox_inches="tight")
        plt.close()