# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/20
# @Function: Sentiment Statistic module

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode, norm
import numpy as np

# Load the data
output_csv = '..\\data\\Sentiment_Analysis\\sentiment_analysis.csv'
data = pd.read_csv(output_csv)
sentiment_scores = data.iloc[:, -1]

# Plot boxplot
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.boxplot(sentiment_scores)
plt.title('Boxplot of Sentiment Scores')

# Plot histogram with normal distribution fit
plt.subplot(122)
bin_edges = np.arange(-1.0, 1.1, 0.1)
plt.hist(sentiment_scores, bins=bin_edges, density=True, alpha=0.6, color='g')

# Normal distribution fit
mu, std = norm.fit(sentiment_scores)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title('Fit results: mu = %.2f, std = %.2f' % (mu, std))
plt.xlabel('Sentiment Score')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# Calculate statistics
mean_score = sentiment_scores.mean()
median_score = sentiment_scores.median()
mode_result = mode(sentiment_scores)

negative_count = (sentiment_scores < 0).sum()
positive_count = (sentiment_scores > 0).sum()

print(f'Mean: {mean_score}')
print(f'Median: {median_score}')
print(f'Mode: {mode_result}')
print(f'Negative scores: {negative_count}')
print(f'Positive scores: {positive_count}')
