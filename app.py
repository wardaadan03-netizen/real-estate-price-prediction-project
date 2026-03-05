import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# CONFIGURE PAGE
# -----------------------------
st.set_page_config(
    page_title="Melbourne House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Melbourne House Price Predictor")
st.write("Predict house prices in Melbourne based on property features.")

# -----------------------------
# MODEL PATH
# -----------------------------
MODEL_PATH = "models/model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model not found! Please make sure 'model.pkl' is in the 'models/' folder.")
    st.stop()

# -----------------------------
# LOAD MODEL (CACHED)
# -----------------------------
@st.cache_data
def load_model(path):
    return joblib.load(path)

model = load_model(MODEL_PATH)

# Automatically get feature names
columns = model.feature_names_in_

# -----------------------------
# DEFINE DEFAULT VALUES
# -----------------------------
defaults = {
    "Rooms": 3,
    "Type": 1,  # You can change to dropdown if needed
    "Postcode": 3000,
    "Bedroom2": 3,
    "Bathroom": 2,
    "Car": 1,
    "Landsize": 200.0,
    "BuildingArea": 100.0,
    "YearBuilt": 2000,
    "Propertycount": 1000
}

# Columns that should be int vs float
int_cols = ["Rooms", "Type", "Postcode", "Bedroom2", "Bathroom", "Car", "YearBuilt", "Propertycount"]
float_cols = ["Landsize", "BuildingArea"]

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Input Property Features")

input_data = {}

for col in columns:
    if col in int_cols:
        input_data[col] = st.sidebar.number_input(
            label=col,
            value=int(defaults.get(col, 0)),
            min_value=0,
            step=1
        )
    elif col in float_cols:
        input_data[col] = st.sidebar.number_input(
            label=col,
            value=float(defaults.get(col, 0.0)),
            min_value=0.0,
            step=1.0
        )

# -----------------------------
# PREDICT BUTTON
# -----------------------------
input_df = pd.DataFrame([input_data])

if st.button("Predict Price"):
    try:
        prediction = model.predict(input_df)
        st.success(f"🏡 Estimated Price: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Made with ❤️ for Melbourne Real Estate Project")