import pandas as pd
from sklearn.datasets import fetch_california_housing
import os

# 1. Create directory for raw data
os.makedirs("data/raw", exist_ok=True)

# 2. Fetch the California housing dataset from Scikit-Learn
print("Fetching dataset...")
data = fetch_california_housing(as_frame=True)
df = data.frame

# 3. Simulate time-slicing: This is our first batch of data (Month 1)
# We take a 20% random sample as the initial training set
batch_1 = df.sample(frac=0.2, random_state=42)

# 4. Save to CSV
file_path = "data/raw/house_price.csv"
batch_1.to_csv(file_path, index=False)

print(f"Batch 1 successfully saved to {file_path}! (Total: {len(batch_1)} records)")