# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/20
# @Function: D Linear module

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

from project_paths import data_path
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn

class DLinear(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DLinear, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        return self.linear2(x)

# Read and prepare the data
data = pd.read_csv(data_path('Bitcoin_Price', 'bitcoin_2021-02-05_2022-12-27.csv'))
data['Start'] = pd.to_datetime(data['Start'])
data.set_index('Start', inplace=True)
data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
data.dropna(inplace=True)

# Seasonal decomposition to extract trends and seasonality
result = seasonal_decompose(data['Close'], model='additive', period=30)
data['Trend'] = result.trend.fillna(method='bfill').fillna(method='ffill')
data['Seasonal'] = result.seasonal.fillna(method='bfill').fillna(method='ffill')
data.dropna(inplace=True)

# Prepare data for training
features = data[['Trend', 'Seasonal']].values
targets = data['Close'].values

# Convert to PyTorch tensors
features = torch.tensor(features, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

# Creating datasets and dataloaders
dataset = TensorDataset(features, targets)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize the DLinear model
model = DLinear(input_size=2, hidden_size=64, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}')

# Testing the model
model.eval()
predictions = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predictions.extend(outputs.detach().numpy())

# Resampling for different frequencies and plotting
results = pd.DataFrame({
    'Predicted': np.array(predictions).flatten(),
    'Actual': data.iloc[train_size:]['Close'].values
}, index=data.iloc[train_size:].index)

frequencies = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly'}
for freq, title in frequencies.items():
    resampled = results.resample(freq).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(resampled.index, resampled['Predicted'], label='Predicted')
    plt.plot(resampled.index, resampled['Actual'], label='Actual')
    plt.title(f'Bitcoin Price Forecast - {title}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
