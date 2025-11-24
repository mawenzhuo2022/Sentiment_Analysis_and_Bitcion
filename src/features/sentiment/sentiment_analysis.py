# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/1 8:10
# @Function: Script to perform feature transformation and sentiment analysis on processed text data.

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

from project_paths import data_path

# Ensure the NLTK sentiment analysis package is downloaded
nltk.download('vader_lexicon')

def feature_transformation_and_sentiment_analysis(input_file, output_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Check if the 'processed_text' column exists in the DataFrame
    if 'processed_text' not in df.columns:
        print("Error: The CSV file does not contain a 'processed_text' column.")
        return

    # Handle missing values by filling them with empty strings
    df['processed_text'] = df['processed_text'].fillna("")  # Alternative: df.dropna(subset=['processed_text']) to remove rows

    # Feature transformation - BoW (Bag of Words)
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(df['processed_text'])

    # Feature transformation - TF-IDF (Term Frequency-Inverse Document Frequency)
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(bow_matrix)

    # Perform sentiment analysis using NLTK's VADER
    sid = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['processed_text'].apply(lambda text: sid.polarity_scores(text)['compound'])

    # Remove rows where the sentiment score is 0
    df = df[df['sentiment_score'] != 0]

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

# Specify the input and output file paths
input_csv = data_path('Sentiment_Analysis', 'text_process.csv')
output_csv = data_path('Sentiment_Analysis', 'sentiment_analysis.csv')

# Call the function to process the data
feature_transformation_and_sentiment_analysis(input_csv, output_csv)
