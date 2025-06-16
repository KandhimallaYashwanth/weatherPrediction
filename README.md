
# 🌦️ Weather Prediction Using Lasso and Random Forest Regression

This project is part of an AICTE virtual internship. It aims to predict temperature based on weather-related features using machine learning techniques like **Lasso Regression** and **Random Forest Regressor**.

🚀 **Live Demo**:  (https://weatherprediction-l.streamlit.app/)

## 📌 Project Objectives

- Predict temperature (in °C) based on environmental data
- Compare performance between Lasso and Random Forest models
- Deploy a user-friendly web interface using **Streamlit**
- Enable real-time prediction and visualization


## 🗃️ Dataset

- Source: [Kaggle Historical Weather Dataset](https://www.kaggle.com/datasets/budincsevity/szeged-weather/data)
- Features: Humidity, Wind Speed, Visibility, Pressure, Summary, Precip Type, Date & Time
- Target: Temperature (°C)



## 🛠️ Technologies Used

- **Python** (pandas, numpy, scikit-learn, matplotlib)
- **Machine Learning**: Lasso Regression, Random Forest Regressor
- **Model Serialization**: joblib
- **Web App**: Streamlit
- **Deployment**: Streamlit Cloud



## 📊 Model Evaluation

| Model            | MSE     | R² Score |
|------------------|---------|----------|
| Lasso Regression | 0.9219  | 0.9900   |
| Random Forest    | 0.0021  | 0.99997  |

✅ **Random Forest outperformed Lasso**, showing near-perfect accuracy.

# google collab link: https://colab.research.google.com/drive/14qMnSftvQVlPlTsjEjXIf-x8voknNyZz?usp=drive_link![image](https://github.com/user-attachments/assets/d12f5859-a66e-4113-8970-57062e9d89f4)



