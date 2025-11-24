# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/20
# @Function: Extract Combined module

import pandas as pd
import numpy as np
from pathlib import Path

from project_paths import data_path, ensure_directory


def extract_features(csv_path, output_path):
    # Read data
    df = pd.read_csv(csv_path)
    df.set_index(df.columns[0], inplace=True)

    # Get all unique feature base names (e.g., 'Open', 'Close')
    base_features = set()
    for col in df.columns:
        if '_Trend_' in col:
            base_name = col.split('_Trend_')[0]
            base_features.add(base_name)

    # Initialize dictionary for processed data
    processed_data = {}

    # Process each base feature
    for feature in base_features:
        for period in ['Daily', 'Weekly', 'Monthly', 'Quarterly']:
            # Combine trend and seasonal
            trend_col = f"{feature}_Trend_{period}"
            seasonal_col = f"{feature}_Seasonal_{period}"
            if trend_col in df.columns and seasonal_col in df.columns:
                processed_data[f"{feature}_Combined_{period}"] = df[trend_col] + df[seasonal_col]

            # Keep log returns
            log_returns_col = f"{feature}_Log_Returns_{period}"
            if log_returns_col in df.columns:
                processed_data[f"{feature}_Log_Returns_{period}"] = df[log_returns_col]

    # Create and save result DataFrame
    result_df = pd.DataFrame(processed_data)
    ensure_directory(Path(output_path).parent)
    result_df.to_csv(output_path)


# Usage
extract_features(
    data_path('without_noise', 'all_features_processed.csv'),
    data_path('without_noise', 'all_features_final.csv')
)
