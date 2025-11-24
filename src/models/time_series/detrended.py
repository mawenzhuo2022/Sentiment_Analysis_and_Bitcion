# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/21 1:53
# @Function:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import signal

from project_paths import data_path, ensure_directory

# Load data
data = pd.read_csv(data_path('Bitcoin_Price', 'bitcoin_2021-02-05_2022-12-27.csv'))
data['Date'] = pd.to_datetime(data['Start'])
data.set_index('Date', inplace=True)

# Prepare the linear regression model
X = np.array(range(len(data))).reshape(-1, 1)  # day index as the independent variable
y = data['Close'].values  # closing price as the dependent variable

# Fit the linear model
model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)

# Remove the trend
detrended = y - trend

# Plot original data and trend line
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(data.index, y, label='Original')
plt.plot(data.index, trend, label='Trend', color='red')
plt.legend()
plt.title('Original Data and Linear Trend')

# Plot the detrended series
plt.subplot(212)
plt.plot(data.index, detrended, label='Detrended')
plt.legend()
plt.title('Detrended Data')

plt.tight_layout()
plt.show()

# Inspect the detrended structure
plt.figure(figsize=(12, 6))
plt.acorr(detrended, maxlags=20, usevlines=True)
plt.title('Autocorrelation of Detrended Data')
plt.show()


# Create a copy of the original data
extended_data = data.copy()

# Add the detrended data as a new column
extended_data['Detrended'] = detrended

# Save the extended data to a new CSV file
output_file = data_path('detrended', 'detrended.csv')
ensure_directory(output_file.parent)
extended_data.to_csv(output_file)
