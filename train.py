import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import os

# Setup MLflow tracking UI (DagsHub provides this automatically)
# The 'dagshub login' you did earlier handles the authentication 
import dagshub
dagshub.init(repo_owner='rsm-chl307', repo_name='house-price-prediction-mlops', mlflow=True)

def train_model():
    #  Load the data tracked by DVC
    data_path = "data/raw/house_price.csv"
    if not os.path.exists(data_path):
        print("Error: Data file not found. Please run prepare_data.py first.")
        return

    df = pd.read_csv(data_path)
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    #  Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #  Start MLflow experiment
    with mlflow.start_run(run_name="Month_1_Baseline"):
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

        #  Train XGBoost Model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        #  Evaluation
        predictions = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, predictions)
        
        # Log metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        print(f"Training complete. RMSE: {rmse:.4f}")

        #  Save and Log the Model itself
        mlflow.xgboost.log_model(model, "model")
        print("Model and metrics logged to DagsHub via MLflow!")

if __name__ == "__main__":
    train_model()