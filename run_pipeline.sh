#!/bin/bash
# Exit immediately if a command fails
set -e

echo " Starting MLOps Pipeline..."

# 1. Update Data
echo "Step 1: Updating data..."
python prepare_data.py

# 2. Drift Detection
echo "Step 2: Checking for Data Drift..."
python check_drift.py

# 3. Train Model
echo "Step 3: Training the model and logging to MLflow..."
python train.py

# 4. Version Control (DVC)
echo "Step 4: Tracking changes with DVC..."
dvc add data/raw/house_price.csv
dvc add data/raw/house_price_baseline.csv
dvc add model.json

# 5. Sync with Remote
echo "Step 5: Pushing metadata and code..."
git add .
git commit -m "Auto-update: Pipeline executed on $(date) [skip ci]" || echo "No changes to commit"
git push
dvc push

echo "Pipeline completed successfully!"