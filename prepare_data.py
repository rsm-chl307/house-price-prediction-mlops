import pandas as pd
from sklearn.datasets import fetch_california_housing
import os

os.makedirs("data/raw", exist_ok=True)

print("Fetching dataset...")
data = fetch_california_housing(as_frame=True)
df = data.frame

# --- SIMULATING MONTH 2 ---
# 1. We take another 10% sample, but we'll exclude the data used in Batch 1
# To keep it simple for this exercise, we use a different random_state
batch_2 = df.sample(frac=0.1, random_state=7)

# 2. INJECTING DATA DRIFT (Optional but Professional)
# Let's pretend inflation happened: increase MedInc by 20%
batch_2['MedInc'] = batch_2['MedInc'] * 1.2 

# 3. OVERWRITE the previous file to simulate a data update
file_path = "data/raw/house_price.csv"
batch_2.to_csv(file_path, index=False)

print(f"Month 2 data (with Drift) saved to {file_path}! (Total: {len(batch_2)} records)")