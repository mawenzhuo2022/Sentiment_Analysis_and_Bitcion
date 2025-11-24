# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/10
# @Function: Location Sentiment module

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from project_paths import data_path

def location_sentiment_correlation(tweet_data):
    # Load the tweet data
    tweet_df = pd.read_csv(tweet_data)

    # Drop the rows with unknown location (continent)
    tweet_df = tweet_df[tweet_df["Continent"] != "Unknown"]
    tweet_df = tweet_df[tweet_df["Continent"] != "Error"]

    # Convert the 'date' column to a datetime format if it's not already
    tweet_df['date'] = pd.to_datetime(tweet_df['date']).dt.date

    # Generate continent score info with average sentiment scores in each continent
    continent_sentiment = tweet_df.groupby(['date', 'Continent'])['sentiment_score'].mean().reset_index()

    # Generate histogram
    plt.figure(figsize=(10, 6))
    bar = sns.barplot(data=continent_sentiment, x="date", y="sentiment_score", errorbar=None, hue='Continent')
    bar.set(title = "Sentiment Scores by Date", xlabel = "Date", ylabel = "Sentiment Score")
    plt.savefig('sentiment_location.png', format='png', dpi=300)
    plt.show()

tweet_data = data_path("Location", "location.csv")
location_sentiment_correlation(tweet_data)
