from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import xgboost as xgb
import pandas as pd
import os

# Initialize FastAPI
app = FastAPI(
    title="California Housing Prediction Service",
    description="Real-time inference API for California housing prices using XGBoost.",
    version="1.1.0"
)

# 1. Load the trained model globally when the API starts
MODEL_PATH = "model.json"

if os.path.exists(MODEL_PATH):
    # Load the Booster object for real-time inference
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
else:
    model = None
    print(f"Warning: {MODEL_PATH} not found. Prediction will return mock data.")

# 2. Define the input data structure with examples
class HouseFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float = Field(..., description="Median house age in block group")
    AveRooms: float = Field(..., description="Average number of rooms per household")
    AveBedrms: float = Field(..., description="Average number of bedrooms per household")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average number of household members")
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "MedInc": 3.87,
                    "HouseAge": 28.0,
                    "AveRooms": 5.0,
                    "AveBedrms": 1.0,
                    "Population": 1000.0,
                    "AveOccup": 3.0,
                    "Latitude": 34.05,
                    "Longitude": -118.24
                }
            ]
        }
    }

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "api_version": "1.1.0"
    }

@app.post("/predict")
def predict_price(features: HouseFeatures):
    """
    Performs real-time prediction based on the input features.
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model file not found on server. Please run training first."
        )

    try:
        # Convert input Pydantic model to DataFrame
        input_df = pd.DataFrame([features.model_dump()])
        
        # XGBoost prediction requires a DMatrix
        dmatrix = xgb.DMatrix(input_df)
        
        # Perform prediction (returns a list, we take the first element)
        prediction = model.predict(dmatrix)[0]
        
        # Ensure the output is a standard float for JSON serialization
        prediction_float = float(prediction)

        return {
            "prediction": {
                "median_house_value": round(prediction_float, 4),
                "currency": "USD",
                "scale": "100,000s",
                "formatted_price": f"${prediction_float * 100000:,.0f}"
            },
            "metadata": {
                "model_type": "XGBoost Regressor",
                "features_used": list(input_df.columns)
            },
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)