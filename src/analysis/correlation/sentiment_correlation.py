# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/9
# @Function: Using sentiment score and various features is price data to explore the correlation,

import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from project_paths import data_path

def sentiment_correlation_analysis(tweet_data, price_data):
    # Load the CSV file into a DataFrame
    tweet_df = pd.read_csv(tweet_data)
    price_df = pd.read_csv(price_data)
    price_df = price_df.sort_values(by='Start', ascending=True).reset_index(drop=True)

    # Convert the 'date' column to a datetime format
    tweet_df['date'] = pd.to_datetime(tweet_df['date']).dt.date
    price_df['Start'] = pd.to_datetime(price_df['Start']).dt.date

    # Merge the tweet data and price data that share the same date
    sentiment_data = tweet_df.groupby('date')['sentiment_score'].mean().reset_index()
    price_df_filtered = price_df[price_df['Start'].isin(sentiment_data['date'])]
    price_sentiment_data = pd.merge(price_df_filtered, sentiment_data, left_on='Start', right_on='date', how='inner')
    price_sentiment_data = price_sentiment_data.drop(columns=['date'])
    price_sentiment_data.to_csv(data_path("Bitcoin_Price", "price_sentiment_data.csv"), index=False)

    # Plot the heatmap
    correlation_matrix = price_sentiment_data.drop(['Start', 'End'], axis=1).corr()
    sentiment_corr = correlation_matrix[['sentiment_score']].drop(index='sentiment_score')
    plt.figure(figsize=(10, 8))
    sns.heatmap(sentiment_corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap Between Sentiment Score and Other Features")

    plt.savefig('sentiment_heatmap.png', format='png', dpi=300)
    plt.show()

# Run the function
tweet_data = data_path("Location", "location.csv")
price_data = data_path('Bitcoin_Price', 'bitcoin_2021-02-05_2022-12-27.csv')
sentiment_correlation_analysis(tweet_data, price_data)
