# Retail Sales Forecasting

## Project Overview

End-to-end time-series forecasting project comparing statistical, machine learning, and deep learning approaches for daily retail sales prediction. The project covers feature engineering, model comparison, hyperparameter optimization, MLflow experiment tracking, champion model selection, and Streamlit deployment.

## Tech Stack

**Python | Pandas | NumPy | scikit-learn | XGBoost | Prophet | Statsmodels | TensorFlow/Keras | Hyperopt | MLflow | Streamlit | Matplotlib**

## Data Preparation & Feature Engineering

The time-series dataset was cleaned and prepared for daily sales forecasting. Features included:

- Lag features: 1, 7, and 30 days
- Rolling average
- Day of week
- Weekend indicator

**Training period:** Jan 2013 – Dec 2013  
**Test period:** Jan 2014 – Mar 2014

## Models

Three forecasting approaches were compared:

- **Statistical:** ARIMA, SARIMA, Exponential Smoothing, Prophet
- **Machine Learning:** Linear Regression, Random Forest, XGBoost
- **Deep Learning:** LSTM

For machine-learning models, an iterative multi-step forecasting strategy was used. XGBoost was tuned with Hyperopt and the LSTM network with Keras Tuner.

## Model Evaluation

Models were evaluated using **MAE, RMSE, and R²**.

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

## Results

Linear Regression and Tuned XGBoost showed the strongest overall performance. While Linear Regression achieved the lowest RMSE and highest R², its long-term forecasts showed unrealistic extrapolation.

Tuned XGBoost provided competitive metrics with more stable forecast behavior and was therefore selected as the champion model.

### Champion Model – Tuned XGBoost

<img width="1437" height="670" alt="Tuned XGBoost - Actual vs Predicted" src="https://github.com/user-attachments/assets/8b0a2082-c404-4061-bcd0-7389716577e0" />

The tuned XGBoost model follows the main sales patterns and captures recurring peaks more effectively while avoiding the unstable long-term extrapolation observed with Linear Regression.

The tuned LSTM showed clear underfitting, demonstrating that increased model complexity did not improve forecasting performance on the relatively small training dataset.

## Model Tracking & Deployment

**MLflow** was used to track experiments, model parameters, evaluation metrics, and trained models.

The final **Tuned XGBoost** model was deployed with **Streamlit** as an interactive application for visualizing historical sales data and generating forecasts.

### Streamlit Application

The final Tuned XGBoost model was integrated into an interactive Streamlit application for visualizing historical sales data and generating model predictions.

<img width="1935" height="1159" alt="streamlit_sales_forecasting_app" src="https://github.com/user-attachments/assets/6849a95b-4b49-42e7-b70b-bcd9ac7ade5c" />

