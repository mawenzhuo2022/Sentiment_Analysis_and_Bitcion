# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/20
# @Function: Sample Raw Data module

import numpy as np
import pandas as pd
import os

from project_paths import data_path

# Read the data
nRowsRead = 5000000
raw_tweets = data_path('Tweet_Preprocessing', 'Bitcoin_tweets.csv')
df1 = pd.read_csv(raw_tweets, delimiter=',', nrows=nRowsRead, low_memory=False)
print("initial: ", df1.shape)


def clean_and_sample_by_date(df, n_samples=100, date_column='date'):
    """
    Clean date column and sample up to n_samples rows per day from a DataFrame.
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Print some diagnostic information
    print("Sample of problematic dates before cleaning:")
    print(df[pd.to_datetime(df[date_column], errors='coerce').isna()][date_column].head())

    # Convert date column to datetime, setting invalid dates to NaT
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # Remove rows with invalid dates
    df = df.dropna()

    print(f"\nNumber of rows after date cleaning: {len(df)}")
    print(f"Date range: {df[date_column].min()} to {df[date_column].max()}")

    # Add date-only column for grouping
    df['date_only'] = df[date_column].dt.date
    print("the number of unique dates :", len(df["date_only"].unique()))
    # Function to sample groups
    def sample_group(group):
        if len(group) <= n_samples:
            return group
        return group.sample(n=n_samples, random_state=42)

    # Group by date and apply sampling
    columns_to_keep = [col for col in df.columns if col != 'date_only']
    sampled_df = df.groupby('date_only')[columns_to_keep].apply(sample_group)

    # Print final statistics
    print("\nFinal dataset statistics:")
    print(f"Number of rows in sampled data: {len(sampled_df)}")
    print("\nNumber of rows per year:")
    print(sampled_df[date_column].dt.year.value_counts().sort_index())

    return sampled_df


# Clean and sample the data
df1 = clean_and_sample_by_date(df1)


# Save the results
output_file = data_path('Tweet_Preprocessing', 'sampled_tweets.csv')
df1.to_csv(output_file, index=False)
