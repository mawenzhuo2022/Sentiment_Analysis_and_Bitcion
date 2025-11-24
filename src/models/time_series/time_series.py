# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/20
# @Function: Time Series module

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.dates import DateFormatter, AutoDateLocator

from project_paths import data_path

# Load data
data = pd.read_csv(data_path('Bitcoin_Price', 'bitcoin_2021-02-05_2022-12-27.csv'))
data['Start'] = pd.to_datetime(data['Start'])
data.set_index('Start', inplace=True)

# Compute log returns
data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))

# Resample data to derive average close prices and log returns for each period
resampled_data = {
    'Daily': data['Close'].resample('D').mean(),
    'Weekly': data['Close'].resample('W').mean(),
    'Monthly': data['Close'].resample('ME').mean(),
    'Quarterly': data['Close'].resample('QE').mean(),
}

log_returns_resampled = {
    'Daily': data['Log_Returns'].resample('D').mean(),
    'Weekly': data['Log_Returns'].resample('W').mean(),
    'Monthly': data['Log_Returns'].resample('ME').mean(),
    'Quarterly': data['Log_Returns'].resample('QE').mean(),
}

# Seasonal decomposition configuration
periods = {'Daily': 1, 'Weekly': 7, 'Monthly': 30, 'Quarterly': 91}

# Perform seasonal decomposition and store de-noised data
for key, period in periods.items():
    result = seasonal_decompose(data['Close'].dropna(), model='additive', period=period)
    data[f'{key}_Data_Without_Noise'] = result.trend + result.seasonal

# Persist the de-noised data
data.reset_index().to_csv(data_path('without_noise', 'without_noise.csv'), index=False)

# Seasonal decomposition plots
fig, axes = plt.subplots(nrows=len(periods), ncols=1, figsize=(10, 20))

# Plot each seasonal decomposition component
for i, (key, period) in enumerate(periods.items()):
    result = seasonal_decompose(data['Close'].dropna(), model='additive', period=period)
    axes[i].plot(result.trend, label='Trend')
    axes[i].plot(result.seasonal, label='Seasonal', linestyle='--')
    axes[i].plot(result.resid, label='Residual', linestyle=':')
    axes[i].set_title(f'Seasonal Decomposition - {key}')
    axes[i].legend()

fig.tight_layout()
plt.show()

# Plot time series and histograms for each aggregation period
fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(18, 30))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

date_format = DateFormatter("%Y-%m")
locator = AutoDateLocator()

for i, (key, values) in enumerate(resampled_data.items()):
    # Average close price series
    ax = axes[2*i, 0]
    ax.plot(values.index, values, label=f'{key} Average Close Price', color='blue')
    ax.set_title(f'{key} Average Close Price Time Series')
    ax.set_ylabel('Average Close Price')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(date_format)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)
    ax.legend()

    # Log-return series
    ax = axes[2*i+1, 0]
    ax.plot(values.index, log_returns_resampled[key], label=f'{key} Log Returns', color='green')
    ax.set_title(f'{key} Log Returns Time Series')
    ax.set_ylabel('Log Returns')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(date_format)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)
    ax.legend()

    # Log-return histogram
    ax_hist = axes[2*i, 1]
    ax_hist.hist(log_returns_resampled[key].dropna(), bins=50, alpha=0.7, color='red')
    ax_hist.set_title(f'{key} Log Returns Histogram')
    ax_hist.set_xlabel('Log Returns')
    ax_hist.set_ylabel('Frequency')
    ax_hist.grid(True)

    # Keep layout aligned by disabling the extra subplot
    axes[2*i+1, 1].axis('off')

plt.show()
