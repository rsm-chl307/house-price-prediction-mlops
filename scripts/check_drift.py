import pandas as pd
from scipy import stats
import json
import os
import shutil

def check_drift():
    # Define paths for current data and the static baseline dataset
    current_path = "data/raw/house_price.csv"
    baseline_path = "data/raw/house_price_baseline.csv"

    # Ensure the current data exists
    if not os.path.exists(current_path):
        print("Error: Current data file not found. Run prepare_data.py first.")
        return False

    # 1. Initialize Baseline if it doesn't exist
    # In a real MLOps workflow, the baseline is usually the data used for the first production model.
    if not os.path.exists(baseline_path):
        print("--- Initializing Baseline Dataset ---")
        shutil.copy(current_path, baseline_path)
        print(f"Created baseline at: {baseline_path}")
        # We return True here as there is no comparison needed for the first run
        return True

    print("--- Running Data Drift Detection (K-S Test) ---")
    
    # 2. Load both datasets for comparison
    baseline_df = pd.read_csv(baseline_path)
    current_df = pd.read_csv(current_path)
    
    # 3. Perform Statistical Test (K-S Test)
    # We focus on 'MedInc' as it is the most influential feature for house prices
    feature_to_test = 'MedInc'
    baseline_data = baseline_df[feature_to_test]
    current_data = current_df[feature_to_test]

    # Perform two-sample Kolmogorov-Smirnov test
    d_stat, p_value = stats.ks_2samp(baseline_data, current_data)
    
    # A p-value below 0.05 indicates the distributions are significantly different
    drift_detected = bool(p_value < 0.05)
    
    print(f"Feature Monitored: {feature_to_test}")
    print(f"P-Value: {p_value:.4f}")
    
    if drift_detected:
        print("⚠️ WARNING: Significant Data Drift Detected compared to the baseline!")
    else:
        print("✅ Status: Data Distribution is consistent with the baseline.")

    # 4. Save results to a report for pipeline integration or audit trails
    report = {
        "feature": feature_to_test,
        "p_value": p_value, 
        "drift_detected": drift_detected,
        "baseline_used": baseline_path
    }
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/drift_report.json", "w") as f:
        json.dump(report, f)
    
    return True

if __name__ == "__main__":
    check_drift()