# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/1 7:54
# @Function: Processes text data in a CSV file by converting it to lowercase, removing punctuation,
#            and filtering out stop words and Bitcoin-related terms.

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

from project_paths import data_path

# Ensure that NLTK stopwords and tokenizer resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert the text to lowercase
    text = text.lower()
    # Tokenize the text (split into individual words)
    tokens = word_tokenize(text)
    # Remove punctuation from tokens
    tokens = [word for word in tokens if word not in string.punctuation]
    # Retrieve English stop words and add terms related to Bitcoin
    stop_words = set(stopwords.words('english'))
    bitcoin_related_terms = {'bitcoin', 'btc', 'cryptocurrency', 'crypto'}
    stop_words.update(bitcoin_related_terms)
    # Filter out stop words and Bitcoin-related terms from the tokens
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Return the processed text as a single string
    return ' '.join(filtered_tokens)

def process_csv(input_file, output_file):
    # Read the input CSV file
    df = pd.read_csv(input_file)
    # Check if the 'text' column exists in the dataframe
    if 'text' in df.columns:
        # Apply the preprocess_text function to each entry in the 'text' column
        df['processed_text'] = df['text'].apply(preprocess_text)
        # Write the modified dataframe with the new 'processed_text' column to a new CSV file
        df.to_csv(output_file, index=False)
    else:
        # Print an error message if 'text' column is not found in the input file
        print("Error: The CSV file does not contain a 'text' column.")

# Specify the input and output file paths
input_csv = data_path('Tweet_Preprocessing', 'clean_tweets.csv')
output_csv = data_path('Sentiment_Analysis', 'text_process.csv')

# Call the function to process the CSV file
process_csv(input_csv, output_csv)
