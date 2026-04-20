#!/bin/bash
# Exit immediately if a command fails
set -e

echo " Starting MLOps Pipeline..."

# 1. Update Data
echo "Step 1: Updating data..."
python scripts/prepare_data.py

# 2. Drift Detection
echo "Step 2: Checking for Data Drift..."
python scripts/check_drift.py

# 3. Train Model
echo "Step 3: Training the model and logging to MLflow..."
python scripts/train.py

# 4. Version Control (DVC)
echo "Step 4: Tracking changes with DVC..."
dvc add data/raw/house_price.csv
dvc add data/raw/house_price_baseline.csv
dvc add model.json

dvc add reports/drift_report.json
dvc add reports/MLOps_System_Final_Report.pdf

# 5. Sync with Remote
echo "Step 5: Pushing metadata and code..."
git add .
git commit -m "Auto-update: Pipeline executed on $(date) [skip ci]" || echo "No changes to commit"
git push
dvc push

echo "Pipeline completed successfully!"


# option: open UI
echo "------------------------------------------"
read -p "Do you want to launch the Streamlit UI now? (y/n): " launch_ui

if [[ "$launch_ui" == "y" || "$launch_ui" == "Y" ]]; then
    echo " Launching UI..."
    streamlit run scripts/ui.py
else
    echo " Pipeline finished. You can launch the UI later using: streamlit run scripts/ui.py"
fi