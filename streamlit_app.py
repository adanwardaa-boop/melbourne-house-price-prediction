import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Melbourne House Price Predictor")

st.title("🏠 Melbourne House Price Prediction")

model_info = joblib.load("models/model.pkl")

model = model_info["model"]
scaler = model_info["scaler"]
columns = model_info["columns"]

st.sidebar.header("Input Features")

input_data = {}

for col in columns:
    input_data[col] = st.sidebar.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data])

if st.button("Predict Price"):
    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")
