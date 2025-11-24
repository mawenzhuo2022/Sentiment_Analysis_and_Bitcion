# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/20
# @Function: Test Price module

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras import models, layers, optimizers, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from project_paths import data_path


class FinancialPredictor:
    def __init__(self, target_column):
        self.target_column = target_column
        self.scaler = MinMaxScaler()
        self.model = None
        self.history = None
        self.target_idx = None

    def prepare_data(self, data, feature_columns, lookback=1):
        """
        Prepare sequences with NaN handling for multiple features
        """
        # Check for and handle NaN values
        data_cleaned = data[feature_columns].fillna(method='ffill').fillna(method='bfill')

        # Find target index
        self.target_idx = feature_columns.index(self.target_column)

        # Scale features
        scaled_data = self.scaler.fit_transform(data_cleaned)

        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i - lookback:i])
            y.append(scaled_data[i, self.target_idx])

        return np.array(X), np.array(y)

    def create_model(self, input_shape, lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
        model = models.Sequential([
            layers.LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(dropout_rate),
            layers.LSTM(lstm_units),
            layers.Dropout(dropout_rate),
            layers.Dense(1)
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
        return model

    def grid_search(self, X_train, y_train, X_val, y_val, feature_columns):
        param_grid = {
            'lstm_units': [32, 50, 64],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01],
        }

        best_val_loss = float('inf')
        best_params = {
            'lstm_units': 50,
            'dropout_rate': 0.2,
            'learning_rate': 0.001
        }

        print("\nPerforming Grid Search...")
        total_combinations = len(param_grid['lstm_units']) * len(param_grid['dropout_rate']) * len(
            param_grid['learning_rate'])
        current = 0

        for units in param_grid['lstm_units']:
            for dropout in param_grid['dropout_rate']:
                for lr in param_grid['learning_rate']:
                    current += 1
                    print(f"\nTrying combination {current}/{total_combinations}:")
                    print(f"LSTM units: {units}, Dropout: {dropout}, Learning rate: {lr}")

                    model = self.create_model(
                        input_shape=(X_train.shape[1], len(feature_columns)),
                        lstm_units=units,
                        dropout_rate=dropout,
                        learning_rate=lr
                    )

                    early_stopping = callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )

                    try:
                        history = model.fit(
                            X_train,
                            y_train,
                            validation_data=(X_val, y_val),
                            epochs=30,
                            batch_size=32,
                            callbacks=[early_stopping],
                            verbose=0
                        )

                        val_loss = min(history.history['val_loss'])
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_params = {
                                'lstm_units': units,
                                'dropout_rate': dropout,
                                'learning_rate': lr
                            }
                    except:
                        print(f"Error with this combination, skipping...")
                        continue

        print("\nBest parameters found:")
        print(best_params)
        print(f"Best validation loss: {best_val_loss}")
        return best_params

    def train(self, X_train, y_train, feature_columns, epochs=30, batch_size=32, validation_split=0.1,
              lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
        self.model = self.create_model(
            input_shape=(X_train.shape[1], len(feature_columns)),
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )

        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        return self.history

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        dummy = np.zeros((predictions.shape[0], X_test.shape[2]))
        dummy[:, self.target_idx] = predictions.ravel()
        transformed = self.scaler.inverse_transform(dummy)
        return transformed[:, self.target_idx:self.target_idx + 1]

    def evaluate(self, y_true, y_pred):
        dummy_true = np.zeros((y_true.shape[0], self.scaler.scale_.shape[0]))
        dummy_true[:, self.target_idx] = y_true
        y_true_orig = self.scaler.inverse_transform(dummy_true)[:, self.target_idx]

        mask = ~np.isnan(y_true_orig) & ~np.isnan(y_pred.ravel())
        y_true_orig = y_true_orig[mask]
        y_pred = y_pred.ravel()[mask]

        if len(y_true_orig) == 0:
            print("Warning: No valid predictions after removing NaN values")
            return {'MSE': float('inf'), 'MAE': float('inf'), 'R2': float('-inf')}

        mse = mean_squared_error(y_true_orig, y_pred)
        mae = mean_absolute_error(y_true_orig, y_pred)
        r2 = r2_score(y_true_orig, y_pred)
        return {'MSE': mse, 'MAE': mae, 'R2': r2}

    def plot_results(self, y_true, y_pred, dates, title, lookback):
        dummy_true = np.zeros((y_true.shape[0], self.scaler.scale_.shape[0]))
        dummy_true[:, self.target_idx] = y_true
        y_true_orig = self.scaler.inverse_transform(dummy_true)[:, self.target_idx]

        mask = ~np.isnan(y_true_orig) & ~np.isnan(y_pred.ravel())
        y_true_orig = y_true_orig[mask]
        y_pred = y_pred.ravel()[mask]
        plot_dates = dates[lookback:][mask]

        plt.figure(figsize=(15, 7))
        plt.plot(plot_dates, y_true_orig, label='Actual', linewidth=2)
        plt.plot(plot_dates, y_pred, label='Predicted', linewidth=2)
        plt.title(f'{title} - {self.target_column}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print("\nPrediction Statistics:")
        print(f"Mean Actual Value: {np.mean(y_true_orig):.2f}")
        print(f"Mean Predicted Value: {np.mean(y_pred):.2f}")
        print(f"Min Actual Value: {np.min(y_true_orig):.2f}")
        print(f"Max Actual Value: {np.max(y_true_orig):.2f}")

    def plot_training_history(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss - {self.target_column}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        loss_improvement = np.diff(self.history.history['loss'])
        plt.plot(loss_improvement, label='Loss Improvement')
        plt.title('Loss Improvement Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Difference')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.legend()

        plt.tight_layout()
        plt.show()


def train_and_evaluate(data, target_column, feature_columns, lookback=5, epochs=30, do_grid_search=False):
    data.index = pd.to_datetime(data.index)
    dates = data.index.values

    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    train_dates = dates[:train_size]
    test_dates = dates[train_size:]

    predictor = FinancialPredictor(target_column)

    X_train, y_train = predictor.prepare_data(train_data, feature_columns, lookback=lookback)
    X_test, y_test = predictor.prepare_data(test_data, feature_columns, lookback=lookback)

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    if do_grid_search:
        train_idx = int(len(X_train) * 0.8)
        X_train_gs, X_val = X_train[:train_idx], X_train[train_idx:]
        y_train_gs, y_val = y_train[:train_idx], y_train[train_idx:]

        best_params = predictor.grid_search(X_train_gs, y_train_gs, X_val, y_val, feature_columns)
        history = predictor.train(X_train, y_train, feature_columns, epochs=epochs, **best_params)
    else:
        history = predictor.train(X_train, y_train, feature_columns, epochs=epochs)

    predictions = predictor.predict(X_test)
    metrics = predictor.evaluate(y_test, predictions)
    print(f"\nMetrics for {target_column}:")
    print(metrics)

    predictor.plot_results(y_test, predictions, test_dates, "Bitcoin Price Prediction", lookback)
    predictor.plot_training_history()

    return predictor, metrics, test_dates

def try_different_lookbacks(data, target_column, feature_columns, lookbacks=[3, 5, 7, 10], epochs=30,
                          do_grid_search=False):
    results = {}
    best_lookback = None
    best_mse = float('inf')

    print("\nTesting different lookback periods...")

    for lookback in lookbacks:
        print(f"\n{'=' * 50}")
        print(f"Testing lookback period: {lookback}")
        print(f"{'=' * 50}")

        predictor, metrics, _ = train_and_evaluate(
            data, target_column, feature_columns,
            lookback=lookback, epochs=epochs,
            do_grid_search=do_grid_search
        )

        results[lookback] = {
            'metrics': metrics,
            'predictor': predictor
        }

        if metrics['MSE'] < best_mse:
            best_mse = metrics['MSE']
            best_lookback = lookback

    print("\nComparison of Different Lookback Periods:")
    print("\nLookback | MSE | MAE | R2")
    print("-" * 50)
    for lookback in lookbacks:
        metrics = results[lookback]['metrics']
        print(f"{lookback:8d} | {metrics['MSE']:.2f} | {metrics['MAE']:.2f} | {metrics['R2']:.4f}")

    print(f"\nBest lookback period: {best_lookback} (MSE: {best_mse:.2f})")

    plt.figure(figsize=(12, 6))
    mse_values = [results[lb]['metrics']['MSE'] for lb in lookbacks]
    r2_values = [results[lb]['metrics']['R2'] for lb in lookbacks]

    plt.subplot(1, 2, 1)
    plt.plot(lookbacks, mse_values, 'o-')
    plt.title('MSE vs Lookback Period')
    plt.xlabel('Lookback Period')
    plt.ylabel('MSE')

    plt.subplot(1, 2, 2)
    plt.plot(lookbacks, r2_values, 'o-')
    plt.title('R² vs Lookback Period')
    plt.xlabel('Lookback Period')
    plt.ylabel('R²')

    plt.tight_layout()
    plt.show()

    return results, best_lookback

if __name__ == "__main__":
    try:
        data = pd.read_csv(data_path("without_noise", "all_features_final.csv"), index_col=0)
    except FileNotFoundError:
        print("Default processed-features file not found. Please provide the path:")
        file_path = input().strip()
        data = pd.read_csv(file_path, index_col=0)

    # First choose the time period
    periods = ['Daily', 'Weekly', 'Monthly', 'Quarterly']
    print("\nChoose the time period for analysis:")
    for i, period in enumerate(periods):
        print(f"{i + 1}. {period}")
    period_choice = int(input("\nEnter your choice (1-4): "))
    selected_period = periods[period_choice - 1]

    # Filter features for selected period
    period_features = [col for col in data.columns if selected_period in col]

    # Get base features (without period suffix)
    base_features = sorted(list(set(col.split('_Combined_')[0] for col in period_features if '_Combined_' in col)))

    print("\nWhat would you like to predict? Available options:")
    for i, feature in enumerate(base_features):
        print(f"{i + 1}. {feature}")

    choice = int(input("\nEnter the number of your choice: "))
    base_feature = base_features[choice - 1]

    # Get target and related features for the period
    target = f"{base_feature}_Combined_{selected_period}"
    related_features = []

    # Include target feature's components
    related_features.extend([col for col in period_features if base_feature in col])

    # Include other price indicators for the same period
    for other_feature in base_features:
        if other_feature != base_feature:
            related_features.extend([col for col in period_features if other_feature in col])

    print("\nSelected features for prediction:")
    for feature in related_features:
        print(f"- {feature}")

    do_grid_search = input("\nWould you like to perform grid search for hyperparameter tuning? (y/n): ").lower() == 'y'

    print("\nHow would you like to handle lookback periods?")
    print("1. Try multiple lookback periods (recommended)")
    print("2. Use single lookback period")
    lookback_choice = input("Enter your choice (1 or 2): ")

    if lookback_choice == '1':
        print("\nEnter lookback periods to try (comma-separated numbers, e.g., '3,5,7,10')")
        lookbacks_input = input("Lookback periods: ")
        lookbacks = [int(x.strip()) for x in lookbacks_input.split(',')]

        results, best_lookback = try_different_lookbacks(
            data, target, related_features,
            lookbacks=lookbacks, epochs=30,
            do_grid_search=do_grid_search
        )

        retrain = input(
            f"\nWould you like to retrain the model with the best lookback period ({best_lookback})? (y/n): ").lower() == 'y'

        if retrain:
            print(f"\nRetraining with best lookback period: {best_lookback}")
            final_predictor, final_metrics, _ = train_and_evaluate(
                data, target, related_features,
                lookback=best_lookback, epochs=30,
                do_grid_search=do_grid_search
            )
    else:
        lookback = int(input("\nEnter lookback period (recommended 3-10 for daily data): "))
        predictor, metrics, _ = train_and_evaluate(
            data, target, related_features,
            lookback=lookback, epochs=30,
            do_grid_search=do_grid_search
        )
