import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import time

# 1. Page Configuration
st.set_page_config(
    page_title="CA Housing ML Dashboard",
    page_icon="🏡",
    layout="wide"
)

# 2. Header and Description
st.title(" California House Price Prediction Dashboard")
st.markdown("""
This interactive dashboard allows users to predict median house values in California using a trained **XGBoost** model. 
By adjusting the features in the sidebar, you can see how different factors impact housing prices in real-time.
""")

# 3. Model Loading Logic
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "model.json")
    
    if os.path.exists(model_path):
        model = xgb.Booster()
        model.load_model(model_path)
        mod_time = time.ctime(os.path.getmtime(model_path))
        return model, mod_time
    else:
        return None, None

model, model_date = load_model()

# 4. Sidebar for User Inputs
st.sidebar.header(" Input Features")
st.sidebar.info("Adjust these values to simulate different area profiles.")

def get_user_inputs():
    med_inc = st.sidebar.slider("Median Income (10k USD)", 0.5, 15.0, 3.87, help="Median income for households within a block group.")
    house_age = st.sidebar.slider("Median House Age", 1.0, 52.0, 28.0)
    ave_rooms = st.sidebar.slider("Average Rooms", 1.0, 10.0, 5.0)
    population = st.sidebar.number_input("Total Population", value=1500, step=100)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Map Location")
    lat = st.sidebar.slider("Latitude", 32.0, 42.0, 34.05)
    lon = st.sidebar.slider("Longitude", -124.0, -114.0, -118.24)
    
    data = {
        'MedInc': med_inc,
        'HouseAge': house_age,
        'AveRooms': ave_rooms,
        'AveBedrms': 1.0, 
        'Population': population,
        'AveOccup': 3.0,   
        'Latitude': lat,
        'Longitude': lon
    }
    return pd.DataFrame([data])

input_df = get_user_inputs()

# 5. Main Content Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(" Prediction Results")
    if model:
        dmatrix = xgb.DMatrix(input_df)
        prediction = model.predict(dmatrix)[0]
        
        st.metric(
            label="Predicted Median House Value", 
            value=f"${prediction * 100000:,.0f}",
            delta=f"Based on {input_df['MedInc'].iloc[0]} Income Level"
        )
        
        st.success(f" Model Loaded Successfully")
        with st.expander("Show Model Metadata"):
            st.write(f"**Model Type:** XGBoost Regressor")
            st.write(f"**Last Updated:** {model_date}")
            st.write(f"**Source:** Local `model.json` (Synced via DVC)")
    else:
        st.error(" Model file `model.json` not found in project root. Please run `scripts/train.py` first.")

with col2:
    st.subheader(" Location Preview")
    map_data = pd.DataFrame({'lat': input_df['Latitude'], 'lon': input_df['Longitude']})
    st.map(map_data)

# 6. Data Summary Table
st.markdown("---")
st.subheader(" Current Input Parameters")
st.table(input_df[['MedInc', 'HouseAge', 'AveRooms', 'Population', 'Latitude', 'Longitude']])