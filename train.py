import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import os
import dagshub

# 1. Initialize DagsHub and MLflow
# This connects your local environment to the DagsHub remote tracking server
dagshub.init(repo_owner='rsm-chl307', repo_name='house-price-prediction-mlops', mlflow=True)

def train_model():
    # 2. Data Loading
    data_path = "data/raw/house_price.csv"
    if not os.path.exists(data_path):
        print("Error: Data file not found. Please run prepare_data.py first.")
        return

    print("--- Loading Data ---")
    df = pd.read_csv(data_path)
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # 3. Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Start MLflow Run
    # We use a consistent experiment name to track progress
    with mlflow.start_run(run_name="Production_Ready_Train"):
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

        # 5. Model Training
        print("--- Training XGBoost Model ---")
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        # 6. Evaluation
        predictions = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, predictions)
        
        # Log metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        print(f"Training complete. RMSE: {rmse:.4f}")

        # 7. Model Artifact Management (CRITICAL FOR MLOPS)
        
        # A. Save model locally for immediate API/UI use
        model_filename = "model.json"
        model.save_model(model_filename)
        print(f"Model saved locally as {model_filename}")

        # B. Log model to MLflow and Register it in the Model Registry
        # Setting 'registered_model_name' automatically creates a versioned model in DagsHub
        mlflow.xgboost.log_model(
            model, 
            artifact_path="model",
            registered_model_name="california-housing-model"
        )
        print("Model registered to MLflow Model Registry on DagsHub!")

if __name__ == "__main__":
    train_model()