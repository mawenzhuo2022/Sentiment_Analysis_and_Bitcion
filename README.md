### Main report

- **Primary report (PDF)**: `reports/Bitcoin_Sentiment_Analysis_Report.pdf`  
  This document contains the full problem statement, methods, experiments, and discussion of results.

---

## Bitcoin_Price_Analysis

End-to-end code and data for analyzing how Twitter sentiment relates to Bitcoin prices, and for building forecasting models (Random Forest, classical time series, and LSTM).

See the PDF report above for the full write-up; this repository focuses on making the
code and data easy to run and extend.

---

### High-level summary

- **Goal**: Study how Twitter sentiment and basic price/volume features relate to Bitcoin price movements, and test whether sentiment adds predictive power.
- **Data**:
  - Bitcoin OHLCV and market cap time series.
  - Sampled Bitcoin-related tweets with timestamps, text, and (optional) user locations.
  - Derived sentiment scores (lexicon-based) and engineered time-series features.
- **Main methods**:
  - Text preprocessing and lexicon-based sentiment (VADER).
  - Correlation and feature-importance analysis with Random Forests.
  - Time-series analysis (seasonal decomposition, detrending, log-returns).
  - LSTM models on raw series and on processed feature sets.

---

### Installation & environment

- **Python**: 3.9–3.11 recommended.
- **Dependencies**: listed in `requirements.txt`.

Basic setup:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

You can run everything on CPU. A GPU (with compatible PyTorch / TensorFlow) will make
the LSTM experiments faster but is not required.

---

### Project structure

```
.
├── project_paths.py              # Small helpers to build project paths
├── data/                         # Raw and processed datasets plus generated figures
├── reports/                      # Final report(s) and presentation material
└── src/
    ├── analysis/                 # Correlation studies and visualization scripts
    ├── features/                 # Feature engineering and sentiment modeling
    └── models/                   # Forecasting models (LSTM, classical TS, etc.)
```

**Directory highlights**

- `src/analysis/correlation`  
  Random-forest experiments, correlation heatmaps, and sentiment–price fusion.
- `src/analysis/location`  
  Geolocation-based sentiment aggregation by region.
- `src/features/sentiment`, `src/features/tweet_preprocessing`  
  Text cleaning and sentiment feature generation.
- `src/models/price`  
  A standalone LSTM price forecaster built directly on price data.
- `src/models/time_series`  
  Seasonal decomposition, detrending, DLinear baselines, and feature extraction utilities.
- `src/models/lstm`  
  LSTM variants working on raw and processed feature sets.
- `data/`  
  Input CSV files and intermediate artifacts (kept so that key results are reproducible).
- `reports/`  
  The main PDF report and any slides.

---

### Path utilities (`project_paths.py`)

To avoid hard-coded `../data/...` segments, scripts import path helpers:

```python
from project_paths import data_path

price_file = data_path("Bitcoin_Price", "price_sentiment_data.csv")
```

- `data_path(*parts)`: join any path inside the `data/` directory.
- `ensure_directory(path)`: create a directory tree before writing files.

All code in `src/` is written assuming the working directory is the project root.

---

### Data files and how they are used

Below is a short map from **code modules** to the **key data files** they read or write.

- **Raw price and sentiment alignment**
  - Input:
    - `data/Bitcoin_Price/bitcoin_2021-02-05_2022-12-27.csv`
    - `data/Sentiment_Analysis/sentiment_analysis.csv`
  - Code:
    - `src/analysis/correlation/sentiment_correlation.py`  
      Merges price data with aggregated sentiment and writes:
      - `data/Bitcoin_Price/price_sentiment_data.csv`
      - correlation heatmaps in `data/Time_Series/` and `src/analysis/correlation/`.

- **Tweet preprocessing and sentiment**
  - Input:
    - Raw tweet CSVs (used originally to create `data/Tweet_Preprocessing/clean_tweets.csv`
      and `data/Sentiment_Analysis/text_process.csv`).
  - Code:
    - `src/features/tweet_preprocessing/sample_raw_data.py`  
      Balances and cleans raw tweets, writes sampled tweet CSVs.
    - `src/features/tweet_preprocessing/text_preprocessing.py`  
      Tokenization, normalization, and basic text cleaning.
    - `src/features/sentiment/text_process.py`  
      Produces `data/Sentiment_Analysis/text_process.csv`.
    - `src/features/sentiment/sentiment_analysis.py`  
      Reads `text_process.csv`, computes VADER / bag-of-words / TF-IDF sentiment,
      writes `data/Sentiment_Analysis/sentiment_analysis.csv`.

- **Location and regional sentiment**
  - Input:
    - `data/Location/location.csv`
  - Code:
    - `src/analysis/location/location.py`  
      Geocodes free-text locations into continents/regions using `geopy`,
      then aggregates sentiment by region and saves location-level outputs.

- **Correlation and Random Forest experiments**
  - Input:
    - `data/Bitcoin_Price/price_sentiment_data.csv`
    - `data/without_noise/features_without_noise.csv`
  - Code:
    - `src/analysis/correlation/random_forest_prediction.py`  
      Uses lagged price, volume, market cap, and sentiment to predict close prices
      with Random Forests. Saves `random_forest_prediction*.png` plots.
    - `src/analysis/correlation/random_forest_no_sentiment.py`  
      Same idea but without sentiment features (baseline).
    - `src/analysis/correlation/RF_*(without_noise).py`  
      Daily/weekly/monthly/quarterly Random Forests on the engineered feature set
      in `data/without_noise/features_without_noise.csv`, writing RF forecast PNGs.

- **Time-series feature engineering**
  - Input:
    - `data/Bitcoin_Price/bitcoin_2021-02-05_2022-12-27.csv`
  - Code:
    - `src/models/time_series/time_series.py`  
      Seasonal decomposition and diagnostic plots, saved into `data/Time_Series/`.
    - `src/models/time_series/detrended.py`  
      Fits linear trends, exports detrended series to `data/detrended/detrended.csv`.
    - `src/models/time_series/time_series_full.py`  
      Builds a larger panel of trend/seasonal/log-return features.
    - `src/models/time_series/extract_combined.py`  
      Consolidates features into:
      - `data/without_noise/all_features_final.csv`
      - `data/without_noise/all_features_processed.csv`
    - `src/models/time_series/d_linear.py`, `DLinear.py`  
      PyTorch-based time-series baselines using the engineered features.

- **LSTM models on price and features**
  - Inputs:
    - Raw price CSVs in `data/Bitcoin_Price/`
    - Processed feature sets in `data/without_noise/all_features_final.csv`
  - Code:
    - `src/models/price/bitcoin_lstm.py`  
      Standalone LSTM on close prices from `bitcoin_2018-01-01_2020-12-31.csv`.
    - `src/models/lstm/raw_bit/*`  
      LSTM models using raw price-based features (with noise).
    - `src/models/lstm/raw_sent/*`  
      LSTM models using raw sentiment-enriched series.
    - `src/models/lstm/processed/lstm_price.py`, `lstm_trend.py`, `lstm_predict.py`  
      LSTMs on `all_features_final.csv` for price and trend prediction.
    - `src/models/lstm/*/test_*.py`  
      Small driver scripts to experiment with lookback windows, hyperparameters,
      and to produce evaluation plots.

---

### How to run the main pipelines

All commands assume the working directory is the project root (`Bitcoin_Price_Analysis_and_Prediction-main`).

1. **Rebuild sentiment and intermediate data (if you want to regenerate from scratch)**

   ```bash
   # Tweet sampling and text preprocessing
   python src/features/tweet_preprocessing/sample_raw_data.py
   python src/features/sentiment/text_process.py
   python src/features/sentiment/sentiment_analysis.py

   # Fuse sentiment with price and build time-series features
   python src/analysis/correlation/sentiment_correlation.py
   python src/models/time_series/time_series_full.py
   python src/models/time_series/extract_combined.py
   ```

2. **Run correlation and Random Forest models**

   ```bash
   python src/analysis/correlation/random_forest_prediction.py
   python src/analysis/correlation/random_forest_no_sentiment.py

   # Optional: RF variants on different frequencies
   python src/analysis/correlation/RF_daily(without_noise).py
   python src/analysis/correlation/RF_weekly(without_noise).py
   python src/analysis/correlation/RF_monthly(without_noise).py
   python src/analysis/correlation/RF_quarterly(without_noise).py
   ```

3. **Run time-series and LSTM models**

   ```bash
   # Classical time-series diagnostics
   python src/models/time_series/time_series.py
   python src/models/time_series/detrended.py

   # LSTM on processed features
   python src/models/lstm/processed/lstm_price.py
   python src/models/lstm/processed/lstm_trend.py

   # Example LSTM on raw series
   python src/models/lstm/raw_bit/test_price.py
   ```

4. **(Optional) Additional experiments**

   The original course project also explored BERT-based sentiment models in Colab,
   but those notebooks/scripts are not required for reproducing the core results in
   this repository. The main pipelines here rely on lexicon-based sentiment
   (`src/features/sentiment/*.py`), correlation analysis, and LSTM/time-series models.

---

### Results (very short summary)

Full tables, plots, and a detailed discussion are in `reports/Bitcoin_Sentiment_Analysis_Report.pdf`.  
At a high level:

- Including sentiment features generally improves Random Forest performance over price-only baselines.
- LSTM models capture short-term dynamics reasonably well, especially when trained on the processed
  feature set rather than purely raw series.
- Location-based analysis shows noticeable regional differences in average sentiment, which can be
  used as additional context for price movements.

---

### Notes

- This repository keeps intermediate CSVs and figures in `data/` so that reviewers can
  inspect outputs without re-running every step.
- Scripts are organized to be small and readable rather than heavily engineered.
- Path handling is centralized in `project_paths.py` so the code can be run from the
  project root on different machines.
- The original work started as a group course project on Bitcoin price prediction; this
  reorganized repository, additional experiments, and the documentation here reflect my
  own follow-up work.
