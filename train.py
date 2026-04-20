import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import os
import dagshub
import hashlib

def get_data_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

dagshub.init(repo_owner='rsm-chl307', repo_name='house-price-prediction-mlops', mlflow=True)


# 1. Initialize DagsHub and MLflow
# connect local environment to the DagsHub remote tracking server
dagshub.init(repo_owner='rsm-chl307', repo_name='house-price-prediction-mlops', mlflow=True)

import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import os
import dagshub
import hashlib

# Initialize DagsHub and MLflow tracking
dagshub.init(repo_owner='rsm-chl307', repo_name='house-price-prediction-mlops', mlflow=True)

def get_data_hash(filepath):
    """
    Generates an MD5 hash of the data file to ensure data version traceability.
    This acts as a unique fingerprint for the specific dataset used in this run.
    """
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        # Read in chunks to handle large files efficiently if needed
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def train_model():
    # 1. Data Loading and Hashing
    data_path = "data/raw/house_price.csv"
    if not os.path.exists(data_path):
        print("Error: Data file not found. Please run prepare_data.py first.")
        return

    print("--- Loading Data and Generating Version Fingerprint ---")
    df = pd.read_csv(data_path)
    
    # Calculate data hash before training to ensure reproducibility
    data_version = get_data_hash(data_path)
    print(f"Data Version (MD5): {data_version}")

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # 2. Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Start MLflow Run
    with mlflow.start_run(run_name="Production_Ready_Train"):
        # Tag the run with the data version hash for audit trails
        mlflow.set_tag("data_version", data_version)
        mlflow.log_param("data_path", data_path)

        # Define Hyperparameters
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "reg:squarederror",
            "random_state": 42
        }
        
        # Log parameters to MLflow
        mlflow.log_params(params)

        # 4. Model Training
        print("--- Training XGBoost Model ---")
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        # 5. Evaluation
        predictions = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, predictions)
        
        # Log metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        print(f"Training complete. RMSE: {rmse:.4f}")

        # 6. Model Artifact Management
        
        # A. Save model locally for immediate API/UI use
        model_filename = "model.json"
        model.save_model(model_filename)
        print(f"Model saved locally as {model_filename}")

        # B. Log and Register the Model
        # This ties the specific data version (via tags) to the registered model version
        mlflow.xgboost.log_model(
            model, 
            artifact_path="model",
            registered_model_name="california-housing-model"
        )
        print("Model registered to MLflow Model Registry with data version tag!")

if __name__ == "__main__":
    train_model()