import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Sales Forecasting App",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting Dashboard")
st.write("This app predicts unit sales using the champion model: Tuned XGBoost.")

# Load model and data
model = joblib.load("tuned_xgboost.pkl")

df = pd.read_csv("feature_engineered.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")
df = df.dropna()

st.subheader("Dataset Preview")
st.dataframe(df.tail())

# Latest prediction
last_row = df.drop(columns=["unit_sales"]).iloc[-1:]
prediction = model.predict(last_row)[0]

st.subheader("Latest Sales Prediction")
st.metric("Predicted Unit Sales", f"{prediction:.2f}")

# Historical chart
st.subheader("Historical Unit Sales")
st.line_chart(df["unit_sales"])

st.subheader("Champion Model")
st.write("Tuned XGBoost was selected as the champion model because it achieved strong metrics and produced realistic forecasts.")

st.write({
    "MAE": 89.77,
    "RMSE": 136.31,
    "R²": 0.4872
})