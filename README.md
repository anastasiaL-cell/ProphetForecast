## Sales Forecasting Project
Project Overview

End-to-end time-series forecasting project comparing statistical, machine learning, and deep learning approaches for daily retail sales prediction. Includes feature engineering, hyperparameter optimization, MLflow experiment tracking, champion model selection, and Streamlit deployment.

### Tech Stack

Python | Pandas | NumPy | scikit-learn | XGBoost | Prophet | Statsmodels | TensorFlow/Keras | Hyperopt | MLflow | Streamlit | Matplotlib

Data Preparation
Loaded and cleaned the time series dataset.
Converted the date column to a datetime format.
Created time-based features, including:
Lag features (1, 7, and 30 days)
Rolling average
Day of week
Weekend indicator
Split the data into:
Training set: 01.01.2013 – 31.12.2013
Test set: 01.01.2014 – 31.03.2014
Statistical Models

The following statistical forecasting models were implemented and evaluated:

ARIMA
SARIMA
Exponential Smoothing
Prophet
Machine Learning Models

The following machine learning models were trained and compared:

Linear Regression
Random Forest Regressor
XGBoost Regressor

For multi-step forecasting, an iterative forecasting strategy was implemented, where each predicted value was used as input for the next prediction.

Hyperparameter Optimization

Hyperparameter tuning was performed for:

XGBoost (using Hyperopt)
LSTM (using Keras Tuner)
Deep Learning

A Long Short-Term Memory (LSTM) neural network was implemented and optimized using Keras Tuner. Due to the relatively small size of the dataset, the LSTM model showed signs of underfitting and did not outperform the simpler machine learning models.

Model Evaluation

Models were evaluated using the following metrics:

Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Coefficient of Determination (R²)

### Model Performance Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 90.85 | 134.22 | 0.5028 |
| Random Forest | 88.88 | 141.29 | 0.4490 |
| XGBoost | 91.40 | 143.47 | 0.4319 |
| Tuned XGBoost | 89.77 | 136.31 | 0.4872 |
| ARIMA | 144.33 | 186.45 | 0.0405 |
| Exponential Smoothing | 98.43 | 150.41 | 0.3756 |
| SARIMA | 98.89 | 151.02 | 0.3705 |
| Prophet | 98.36 | 150.47 | 0.3751 |
| Tuned LSTM | 141.30 | 188.45 | 0.0198 |

Forecasts were also visually compared with actual sales values to assess prediction quality.

## Results

Linear Regression and Tuned XGBoost showed the strongest overall performance. While Linear Regression achieved the lowest RMSE and highest R², its long-term forecasts showed unrealistic extrapolation. Tuned XGBoost provided competitive metrics with more stable forecast behavior and was therefore selected as the champion model.

### Champion Model – Tuned XGBoost

<img width="1437" height="670" alt="Tuned XGBoost - Actual vs Predicted" src="https://github.com/user-attachments/assets/0b932314-a198-45b5-ab02-86913b256883" />

## Model Tracking and Deployment

MLflow was used to track experiments, model parameters, evaluation metrics, and trained models.

The final Tuned XGBoost model was deployed with Streamlit as an interactive application for visualizing historical sales data and generating forecasts.
