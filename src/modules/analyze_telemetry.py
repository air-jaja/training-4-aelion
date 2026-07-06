import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_telemetry():
    # Configure columns
    time_column = "timestamp"
    temperature_column = "temperature_c"
    pressure_column = "pressure_bar"
    tension_column = "voltage_mean_v"
    rotation_column = "rotation_mean_rpm"
    production_column = "pieces_produced"

    # Get input directory
    input_dir = os.getenv("INPUT_DATA_DIR", "./artifacts/ingestions/datas")
    print(f"✅ Input directory: {input_dir}")

    # Find telemetry.csv file
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file == "telemetry.csv":
                file_path = os.path.join(root, file)
                print(f"✅ File in used: {file_path}....")
                process_telemetry_file(file_path, time_column, temperature_column, pressure_column, tension_column, rotation_column, production_column)

def process_telemetry_file(file_path, time_column, temperature_column, pressure_column, tension_column, rotation_column, production_column):
    df = pd.read_csv(file_path)

    # Check for required columns
    required_columns = [time_column, temperature_column, pressure_column, tension_column, rotation_column, production_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"⚠️ Warning: The following columns are missing: {missing_columns}. Skipping telemetry graphs.")
        return

    # Convert time column to datetime
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")

    # Create output directory
    output_dir = os.getenv("OUTPUT_DIR", "./artifacts/ingestions")
    output_graph_dir = os.path.join(output_dir, "telemetry", datetime.now().strftime("%Y%m%d%H%M"), "graphs")
    os.makedirs(output_graph_dir, exist_ok=True)

    # Generate graphs
    generate_telemetry_graphs(df, time_column, temperature_column, pressure_column, tension_column, rotation_column, production_column, output_graph_dir)
    print(f"✅ Telemetry graphs generated in: {output_graph_dir}")

def generate_telemetry_graphs(df, time_column, temperature_column, pressure_column, tension_column, rotation_column, production_column, output_graph_dir):
    # Graph 1: Temperature over Time
    plt.figure(figsize=(12, 6))
    plt.plot(df[time_column], df[temperature_column], marker="o", linestyle="-", color="red", label="Temperature")
    plt.title("Temperature Distribution Over Time")
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_graph_dir, "temperature_over_time.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Graph 2: Pressure over Time
    plt.figure(figsize=(12, 6))
    plt.plot(df[time_column], df[pressure_column], marker="o", linestyle="-", color="blue", label="Pressure")
    plt.title("Pressure Distribution Over Time")
    plt.xlabel("Time")
    plt.ylabel("Pressure")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_graph_dir, "pressure_over_time.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Graph 3: Tension over Time
    plt.figure(figsize=(12, 6))
    plt.plot(df[time_column], df[tension_column], marker="o", linestyle="-", color="green", label="Tension")
    plt.title("Tension Distribution Over Time")
    plt.xlabel("Time")
    plt.ylabel("Tension")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_graph_dir, "tension_over_time.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Graph 4: Rotation over Time
    plt.figure(figsize=(12, 6))
    plt.plot(df[time_column], df[rotation_column], marker="o", linestyle="-", color="purple", label="Rotation")
    plt.title("Rotation Distribution Over Time")
    plt.xlabel("Time")
    plt.ylabel("Rotation")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_graph_dir, "rotation_over_time.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Graph 5: Pieces Produced over Time
    plt.figure(figsize=(12, 6))
    plt.plot(df[time_column], df[production_column], marker="o", linestyle="-", color="orange", label="Pieces Produced")
    plt.title("Pieces Produced Over Time")
    plt.xlabel("Time")
    plt.ylabel("Pieces Produced")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_graph_dir, "pieces_produced_over_time.png"), dpi=300, bbox_inches="tight")
    plt.close()