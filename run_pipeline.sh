#!/bin/bash

# --- MLOps Pipeline Automation Script ---

echo " Starting MLOps Pipeline..."

# 1. Update Data
echo "Step 1: Generating and updating data..."
python prepare_data.py

# 2. Train Model
echo "Step 2: Training the model and logging to MLflow..."
python train.py

# 3. Version Control (DVC)
echo "Step 3: Tracking new model and data with DVC..."
dvc add data/raw/house_price.csv
dvc add model.json

# 4. Sync with Remote (Git & DagsHub)
echo "Step 4: Pushing code and metadata to DagsHub..."
git add .
git commit -m "Auto-update: Pipeline executed on $(date)"
git push
dvc push

echo " Pipeline completed successfully!"
echo "You can now run 'streamlit run ui.py' to see the results."