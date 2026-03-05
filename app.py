import streamlit as st
import pandas as pd
import joblib

# Load pipeline
pipeline = joblib.load("models/pipeline.pkl")

st.title("🏠 Melbourne House Price Predictor")

# Collect input
input_dict = {}
for feature in pipeline.named_steps['preprocessor'].transformers_[0][2] + \
               pipeline.named_steps['preprocessor'].transformers_[1][2]:
    if feature in pipeline.named_steps['preprocessor'].transformers_[0][2]:  # numeric
        input_dict[feature] = st.number_input(feature, value=0.0)
    else:  # categorical
        input_dict[feature] = st.text_input(feature, value='Unknown')

input_df = pd.DataFrame([input_dict])

if st.button("Predict Price"):
    prediction = pipeline.predict(input_df)
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")