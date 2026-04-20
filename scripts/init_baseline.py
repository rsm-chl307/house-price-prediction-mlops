import pandas as pd
import os
import shutil

def initialize_baseline():
    """
    Sets the current raw data as the 'Golden Baseline' for future drift detection.
    This script should typically be run once at the start of the project or 
    whenever the baseline needs to be officially updated.
    """
    raw_data_path = "data/raw/house_price.csv"
    baseline_path = "data/raw/house_price_baseline.csv"

    # 1. Check if raw data exists
    if not os.path.exists(raw_data_path):
        print(f"Error: {raw_data_path} not found. Please run 'prepare_data.py' first.")
        return

    # 2. Check if baseline already exists to prevent accidental overwrite
    if os.path.exists(baseline_path):
        response = input(f"Baseline already exists at {baseline_path}. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Initialization cancelled.")
            return

    # 3. Create the baseline file
    try:
        shutil.copy(raw_data_path, baseline_path)
        print(f"Successfully initialized baseline dataset at: {baseline_path}")
        
        # 4. Optional: Display baseline summary for record keeping
        df = pd.read_csv(baseline_path)
        print("\n--- Baseline Summary Statistics (MedInc) ---")
        print(df['MedInc'].describe())
        
    except Exception as e:
        print(f"An error occurred during initialization: {e}")

if __name__ == "__main__":
    initialize_baseline()