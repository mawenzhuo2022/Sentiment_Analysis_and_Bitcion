# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/20
# @Function: Lstm Price Noise module

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

    def prepare_data(self, data, lookback=1):
        """
        Prepare sequences for single target prediction
        """
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))

        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i - lookback:i, 0])
            y.append(scaled_data[i, 0])

        return np.array(X), np.array(y)

    def create_model(self, input_shape, lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
        model = models.Sequential([
            layers.LSTM(lstm_units, return_sequences=True, input_shape=(input_shape, 1)),
            layers.Dropout(dropout_rate),
            layers.LSTM(lstm_units),
            layers.Dropout(dropout_rate),
            layers.Dense(1)
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
        return model

    def grid_search(self, X_train, y_train, X_val, y_val):
        param_grid = {
            'lstm_units': [32, 50, 64],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01],
        }

        best_val_loss = float('inf')
        best_params = None

        print("\nPerforming Grid Search...")
        total_combinations = len(param_grid['lstm_units']) * len(param_grid['dropout_rate']) * \
                             len(param_grid['learning_rate'])
        current = 0

        for units in param_grid['lstm_units']:
            for dropout in param_grid['dropout_rate']:
                for lr in param_grid['learning_rate']:
                    current += 1
                    print(f"\nTrying combination {current}/{total_combinations}:")
                    print(f"LSTM units: {units}, Dropout: {dropout}, Learning rate: {lr}")

                    model = self.create_model(
                        input_shape=X_train.shape[1],
                        lstm_units=units,
                        dropout_rate=dropout,
                        learning_rate=lr
                    )

                    early_stopping = callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )

                    history = model.fit(
                        X_train.reshape(-1, X_train.shape[1], 1),
                        y_train,
                        validation_data=(X_val.reshape(-1, X_val.shape[1], 1), y_val),
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

        print("\nBest parameters found:")
        print(best_params)
        print(f"Best validation loss: {best_val_loss}")
        return best_params

    def train(self, X_train, y_train, epochs=30, batch_size=32, validation_split=0.1,
              lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
        self.model = self.create_model(
            input_shape=X_train.shape[1],
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
            X_train.reshape(-1, X_train.shape[1], 1),
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        return self.history

    def predict(self, X_test):
        predictions = self.model.predict(X_test.reshape(-1, X_test.shape[1], 1))
        return self.scaler.inverse_transform(predictions)

    def evaluate(self, y_true, y_pred):
        y_true_orig = self.scaler.inverse_transform(y_true.reshape(-1, 1))
        mse = mean_squared_error(y_true_orig, y_pred)
        mae = mean_absolute_error(y_true_orig, y_pred)
        r2 = r2_score(y_true_orig, y_pred)
        return {'MSE': mse, 'MAE': mae, 'R2': r2}

    def plot_results(self, y_true, y_pred, dates, title, lookback):  # Added lookback parameter
        y_true_orig = self.scaler.inverse_transform(y_true.reshape(-1, 1))

        plt.figure(figsize=(15, 7))
        plt.plot(dates[lookback:], y_true_orig, label='Actual', linewidth=2)
        plt.plot(dates[lookback:], y_pred, label='Predicted', linewidth=2)
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


def train_and_evaluate(data, target_column, lookback=5, epochs=30, do_grid_search=False):
    data['Start'] = pd.to_datetime(data['Start'])
    data = data.sort_values('Start')

    target_data = data[target_column].values
    dates = data['Start'].values

    train_data, test_data, train_dates, test_dates = train_test_split(
        target_data, dates, test_size=0.2, random_state=42, shuffle=False
    )

    predictor = FinancialPredictor(target_column)

    X_train, y_train = predictor.prepare_data(train_data, lookback=lookback)
    X_test, y_test = predictor.prepare_data(test_data, lookback=lookback)

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    if do_grid_search:
        train_idx = int(len(X_train) * 0.8)
        X_train_gs, X_val = X_train[:train_idx], X_train[train_idx:]
        y_train_gs, y_val = y_train[:train_idx], y_train[train_idx:]

        best_params = predictor.grid_search(X_train_gs, y_train_gs, X_val, y_val)
        history = predictor.train(X_train, y_train, epochs=epochs, **best_params)
    else:
        history = predictor.train(X_train, y_train, epochs=epochs)

    predictions = predictor.predict(X_test)
    metrics = predictor.evaluate(y_test, predictions)
    print(f"\nMetrics for {target_column}:")
    print(metrics)

    predictor.plot_results(y_test, predictions, test_dates, "Bitcoin Price Prediction",
                           lookback)  # Added lookback parameter
    predictor.plot_training_history()

    return predictor, metrics, test_dates


def try_different_lookbacks(data, target_column, lookbacks=[3, 5, 7, 10], epochs=30, do_grid_search=False):
    results = {}
    best_lookback = None
    best_mse = float('inf')

    print("\nTesting different lookback periods...")

    for lookback in lookbacks:
        print(f"\n{'=' * 50}")
        print(f"Testing lookback period: {lookback}")
        print(f"{'=' * 50}")

        predictor, metrics, _ = train_and_evaluate(
            data, target_column, lookback=lookback,
            epochs=epochs, do_grid_search=do_grid_search
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
        data = pd.read_csv(data_path('Bitcoin_Price', 'price_sentiment_data.csv'))
    except FileNotFoundError:
        print("Default dataset not found. Please provide the CSV path:")
        file_path = input().strip()
        data = pd.read_csv(file_path)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']

    print("\nWhat would you like to predict? Available options:")
    for i, feature in enumerate(features):
        print(f"{i + 1}. {feature}")

    choice = int(input("\nEnter the number of your choice (1-7): "))
    target = features[choice - 1]

    do_grid_search = input("\nWould you like to perform grid search for hyperparameter tuning? (y/n): ").lower() == 'y'

    print("\nHow would you like to handle lookback periods?")
    print("1. Try multiple lookback periods (recommended)")
    print("2. Use single lookback period")
    lookback_choice = input("Enter your choice (1 or 2): ")

    if lookback_choice == '1':
        print("\nEnter lookback periods to try (comma-separated numbers, e.g., '3,5,7,10')")
        print("Recommended range: 3-10 for daily data")
        lookbacks_input = input("Lookback periods: ")
        lookbacks = [int(x.strip()) for x in lookbacks_input.split(',')]

        results, best_lookback = try_different_lookbacks(
            data, target, lookbacks=lookbacks,
            epochs=30, do_grid_search=do_grid_search
        )

        retrain = input(
            f"\nWould you like to retrain the model with the best lookback period ({best_lookback})? (y/n): ").lower() == 'y'

        if retrain:
            print(f"\nRetraining with best lookback period: {best_lookback}")
            final_predictor, final_metrics, _ = train_and_evaluate(
                data, target, lookback=best_lookback,
                epochs=30, do_grid_search=do_grid_search
            )
    else:
        lookback = int(input("\nEnter lookback period (recommended 3-10 for daily data): "))
        predictor, metrics, _ = train_and_evaluate(
            data, target, lookback=lookback,
            epochs=30, do_grid_search=do_grid_search
        )
