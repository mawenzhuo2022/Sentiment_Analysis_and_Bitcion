# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/21 5:43
# @Function:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from math import sqrt

from project_paths import data_path

# Data preprocessing and model configuration
n_steps = 60  # longer lookback window
n_features = 1

# Load data
data = pd.read_csv(data_path('Bitcoin_Price', 'bitcoin_2018-01-01_2020-12-31.csv'))
data['Date'] = pd.to_datetime(data['Start'])
data.set_index('Date', inplace=True)

# Normalize close prices
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Prepare input/output pairs
X, y = [], []
for i in range(len(data_scaled) - n_steps):
    X.append(data_scaled[i:i+n_steps])
    y.append(data_scaled[i+n_steps])
X, y = np.array(X), np.array(y)

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Build the LSTM model
model = Sequential([
    LSTM(100, activation='relu', input_shape=(n_steps, n_features)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, verbose=1)

# Predict
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Bring the actual values back to the original scale
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate performance
mse = mean_squared_error(actual, predictions)
rmse = sqrt(mse)
mae = mean_absolute_error(actual, predictions)

print("Performance Evaluation:")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)

# Visualize the most recent portion of the series
plt.figure(figsize=(12, 6))
plt.plot(actual[-180:], label='Actual Prices', color='blue')  # show only the last 180 days
plt.plot(predictions[-180:], label='Predicted Prices', color='red')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
