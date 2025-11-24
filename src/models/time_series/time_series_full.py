# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/20
# @Function: Time Series Full module

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.dates import DateFormatter, AutoDateLocator
from pathlib import Path

from project_paths import data_path, ensure_directory


def process_and_plot_features(source_file, save_path_plots, save_path_csv):
    save_path_plots = ensure_directory(save_path_plots)
    ensure_directory(Path(save_path_csv).parent)

    # Read data
    data = pd.read_csv(source_file)
    data['Start'] = pd.to_datetime(data['Start'])
    data.set_index('Start', inplace=True)

    periods = {
        'Daily': 'D',
        'Weekly': 'W',
        'Monthly': 'ME',
        'Quarterly': 'QE'
    }

    period_days = {'Daily': 1, 'Weekly': 7, 'Monthly': 30, 'Quarterly': 91}
    processed_data = {}
    numeric_columns = data.select_dtypes(include=[np.number]).columns

    for column in numeric_columns:
        fig, axes = plt.subplots(nrows=len(periods), ncols=1, figsize=(15, 20))
        fig.suptitle(f'{column} Analysis', fontsize=16)

        data[f'{column}_Log_Returns'] = np.log(data[column] / data[column].shift(1))

        for i, (period_name, period_code) in enumerate(periods.items()):
            mean_values = data[column].resample(period_code).mean()
            log_returns = data[f'{column}_Log_Returns'].resample(period_code).mean()

            processed_data[f'{column}_{period_name}'] = mean_values
            processed_data[f'{column}_Log_Returns_{period_name}'] = log_returns

            try:
                result = seasonal_decompose(
                    data[column].dropna(),
                    model='additive',
                    period=period_days[period_name]
                )

                processed_data[f'{column}_Without_Noise_{period_name}'] = result.trend + result.seasonal

                axes[i].plot(result.trend + result.seasonal, label='Trend + Seasonal')
                axes[i].plot(data[column], label='Original', alpha=0.5)
                axes[i].set_title(f'{period_name} Analysis')
                axes[i].legend()
                axes[i].xaxis.set_major_formatter(DateFormatter('%Y-%m'))
                axes[i].tick_params(axis='x', rotation=45)

            except Exception as e:
                print(f"Error processing {column} - {period_name}: {str(e)}")

        plt.tight_layout()
        plt.savefig(save_path_plots / f'{column}_analysis.png')
        plt.close()

    result_df = pd.DataFrame(processed_data)
    result_df.to_csv(save_path_csv)


# Usage
process_and_plot_features(
    data_path('Bitcoin_Price', 'bitcoin_2021-02-05_2022-12-27.csv'),
    data_path('without_noise', 'plots'),
    data_path('without_noise', 'features_without_noise.csv')
)
