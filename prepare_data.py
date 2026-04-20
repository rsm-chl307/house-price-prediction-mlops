import pandas as pd
from sklearn.datasets import fetch_california_housing
import os

# Ensure the directory exists
os.makedirs("data/raw", exist_ok=True)

print("Fetching dataset...")
data = fetch_california_housing(as_frame=True)
df = data.frame

# --- GENERATING MONTH 2 (Data with Drift) ---
# 1. Take a different 10% sample to simulate new incoming data
# Changing random_state ensures this batch is different from Month 1
batch_2 = df.sample(frac=0.1, random_state=7)

# 2. INJECTING DATA DRIFT (Professional MLOps Simulation)
# We simulate a 20% increase in Median Income (inflation or regional growth)
# This will trigger the K-S test in your check_drift.py
batch_2['MedInc'] = batch_2['MedInc'] * 1.2 

# 3. Save as the main data file (Overwriting the previous month)
file_path = "data/raw/house_price.csv"
batch_2.to_csv(file_path, index=False)

print(f"Month 2 data (with Drift) saved to {file_path}! (Total: {len(batch_2)} records)")