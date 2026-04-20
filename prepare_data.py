import pandas as pd
from sklearn.datasets import fetch_california_housing
import os

# Ensure the directory exists
os.makedirs("data/raw", exist_ok=True)

print("Fetching dataset...")
data = fetch_california_housing(as_frame=True)
df = data.frame

# --- GENERATING MONTH 1 (Baseline) ---
# We take 10% of the data as our initial batch
batch_1 = df.sample(frac=0.1, random_state=42)

# Save as the main data file
file_path = "data/raw/house_price.csv"
batch_1.to_csv(file_path, index=False)

print(f"Month 1 data (Baseline) saved to {file_path}! (Total: {len(batch_1)} records)")