# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/20
# @Function: Dlinear module

import torch
import torch.nn as nn
import torch.optim as optim

class ImprovedDLinear(nn.Module):
    def __init__(self, input_size):
        super(ImprovedDLinear, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.output(x)
        return x

# Initialize the model
model = ImprovedDLinear(input_size=2)
criterion = nn.MSELoss()  # consider nn.HuberLoss() if the series is noisy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and evaluation code remains unchanged; only the model definition differs
