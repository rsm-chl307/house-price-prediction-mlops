import os
import hashlib
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# 1. Global Configuration & Initialization
REPO_OWNER = 'rsm-chl307'
REPO_NAME = 'house-price-prediction-mlops'
DATA_PATH = "data/raw/house_price.csv"
MODEL_NAME = "california-housing-model"

dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

def get_data_hash(filepath):
    """
    Generates an MD5 hash of the data file to ensure data version traceability.
    """
    hasher = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        return None

def train_model():
    # 2. Data Preparation
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Please run prepare_data.py first.")
        return

    print("--- Loading Data and Generating Version Fingerprint ---")
    df = pd.read_csv(DATA_PATH)
    data_version = get_data_hash(DATA_PATH)
    print(f"Data Version (MD5): {data_version}")

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. MLflow Experiment Tracking
    with mlflow.start_run(run_name="Production_Ready_Train"):
        # Log metadata
        mlflow.set_tag("data_version", data_version)
        mlflow.log_param("data_path", DATA_PATH)

        # Define and log Hyperparameters
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "reg:squarederror",
            "random_state": 42
        }
        mlflow.log_params(params)

        # 4. Model Training
        print("--- Training XGBoost Model ---")
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        # 5. Evaluation
        predictions = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, predictions)
        mlflow.log_metric("rmse", rmse)
        print(f"Training complete. RMSE: {rmse:.4f}")

        # 6. Artifact & Model Registry Management
        # Local save
        model.save_model("model.json")
        print("Model saved locally as model.json")

        # Remote Log and Registration
        mlflow.xgboost.log_model(
            model, 
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )
        print(f"Model versioned and registered as '{MODEL_NAME}' on DagsHub!")

if __name__ == "__main__":
    train_model()