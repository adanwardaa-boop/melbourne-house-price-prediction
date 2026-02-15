import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import gdown

# -----------------------------
# DOWNLOAD MODEL IF NOT EXISTS
# -----------------------------

MODEL_PATH = "models/model.pkl"

if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    
    url = "https://drive.google.com/uc?id=1Yt_cFwhAKGgilBUHM3ybuI9vQEDq1CQH"
    gdown.download(url, MODEL_PATH, quiet=False)

# -----------------------------
# LOAD MODEL
# -----------------------------

model_info = joblib.load(MODEL_PATH)

model = model_info["model"]
scaler = model_info["scaler"]
columns = model_info["columns"]

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.set_page_config(page_title="Melbourne House Price Predictor")

st.title("🏠 Melbourne House Price Prediction")

st.sidebar.header("Input Features")

input_data = {}

for col in columns:
    input_data[col] = st.sidebar.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data])

if st.button("Predict Price"):
    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")
