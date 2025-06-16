import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load model and preprocessors
model = joblib.load("lasso_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Streamlit app setup
st.set_page_config(page_title="Weather Predictor", page_icon="🌤️")
st.title("🌦️ Weather Temperature Predictor")
st.markdown("Predict temperature using machine learning models trained on historical weather data.")

# Sidebar for input
st.sidebar.header("Enter Weather Details:")
humidity = st.sidebar.slider("Humidity", 0.0, 1.0, 0.75)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 60.0, 10.0)
pressure = st.sidebar.slider("Pressure (millibars)", 900.0, 1050.0, 1015.0)
apparent_temp = st.sidebar.slider("Apparent Temp (°C)", -20.0, 40.0, 20.0)
visibility = st.sidebar.slider("Visibility (km)", 0.0, 16.0, 10.0)
summary = st.sidebar.selectbox("Weather Summary", ["Clear", "Overcast", "Foggy", "Rain", "Partly Cloudy"])
precip_type = st.sidebar.selectbox("Precipitation Type", ["rain", "snow", "None"])
date = st.sidebar.date_input("Date", datetime.now())
hour = st.sidebar.slider("Hour", 0, 23, 12)

# Input dictionary
input_dict = {
    'Humidity': humidity,
    'Wind Speed (km/h)': wind_speed,
    'Pressure (millibars)': pressure,
    'Apparent Temperature (C)': apparent_temp,
    'Visibility (km)': visibility,
    'Summary': summary,
    'Precip Type': precip_type,
    'Year': pd.to_datetime(date).year,
    'Month': pd.to_datetime(date).month,
    'Day': pd.to_datetime(date).day,
    'Hour': hour
}

# Input preparation
def prepare_input(input_data):
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df, columns=['Summary', 'Precip Type'], drop_first=True)

    # Add missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    return scaler.transform(df)

# Prediction
if st.button("Predict Temperature"):
    X_input = prepare_input(input_dict)
    prediction = model.predict(X_input)[0]
    st.success(f"🌡️ Predicted Temperature: **{prediction:.2f} °C**")
