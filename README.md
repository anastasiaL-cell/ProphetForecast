Sales Forecasting Project
Project Overview

This project focuses on forecasting daily unit sales using statistical, machine learning, and deep learning approaches. The objective was to compare different forecasting techniques and evaluate their predictive performance on unseen data.

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

Forecasts were visualized and compared with the actual sales values to assess prediction quality.

Results

Among all evaluated models, Linear Regression and the tuned XGBoost model achieved the best overall forecasting performance on this dataset. Statistical models provided reasonable baseline results, while the LSTM model was less effective because of the limited amount of training data.

Although Linear Regression achieved the lowest RMSE and the highest R², visual inspection of the forecast revealed unrealistic long-term extrapolation. Since forecasting models should not only minimize prediction error but also produce plausible future trends, the tuned XGBoost model was selected as the champion model. It achieved competitive error metrics while generating substantially more realistic forecasts.