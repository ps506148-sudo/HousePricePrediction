import streamlit as st
import joblib
import numpy as np
from pathlib import Path
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction System")
st.write("Predict the estimated house price using the trained XGBoost model.")

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "Model" / "house_price_model.pkl"

model = joblib.load(model_path)

# -----------------------------
# User Inputs
# -----------------------------
st.header("Enter House Details")

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=20,
    value=3
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1.0,
    max_value=10.0,
    value=2.0,
    step=0.5,
    format="%.1f"
)

living_area = st.number_input(
    "Living Area (sq ft)",
    min_value=200,
    value=2000
)

lot_area = st.number_input(
    "Lot Area (sq ft)",
    min_value=500,
    value=5000
)

floors = st.number_input(
    "Floors",
    min_value=1.0,
    max_value=5.0,
    value=1.0,
    step=0.5,
    format="%.1f"
)

condition = st.slider(
    "Condition",
    1,
    5,
    3
)

grade = st.slider(
    "Grade",
    1,
    10,
    6
)

sqft_above = st.number_input(
    "Area of House(Excluding Basement) (sq ft)",
    min_value=0,
    value=1800
)

sqft_basement = st.number_input(
    "Basement Area (sq ft)",
    min_value=0,
    value=200
)

year_built = st.number_input(
    "Year Built",
    min_value=1900,
    max_value=2026,
    value=2000
)

distance_airport = st.number_input(
    "Distance from Airport (km)",
    min_value=0.0,
    value=10.0,
    step=0.5
)

nearby_schools = st.number_input(
    "Number of Schools Nearby",
    min_value=0,
    value=5
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Price"):

    # Construct DataFrame matching the exact feature names and order from training
    feature_dict = {
        'number of bedrooms': [bedrooms],
        'number of bathrooms': [bathrooms],
        'living area': [living_area],
        'lot area': [lot_area],
        'number of floors': [floors],
        'condition of the house': [condition],
        'grade of the house': [grade],
        'Area of the house(excluding basement)': [sqft_above],
        'Area of the basement': [sqft_basement],
        'Built Year': [year_built],
        'Number of schools nearby': [nearby_schools],
        'Distance from the airport': [distance_airport]
    }
    
    features_df = pd.DataFrame(feature_dict)

    st.write("Features Sent to Model:")
    st.dataframe(features_df)

    # Predict using the DataFrame
    prediction = model.predict(features_df)

    st.write("Raw Prediction:", prediction)

    st.success(f"🏡 Estimated House Price: ₹ {prediction[0]:,.2f}")

    st.info("Model Used: XGBoost")