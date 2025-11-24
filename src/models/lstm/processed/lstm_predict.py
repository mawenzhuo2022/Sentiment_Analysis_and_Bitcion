# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/3/4 10:35
# @Function: LSTM-based Bitcoin Price predictor to model time series data with a grid search for hyperparameter optimization

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras import models, layers, optimizers, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import logging

from project_paths import data_path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FinancialPredictor:
    def __init__(self, target_column):
        """
        Initialize FinancialPredictor class with target column and create scaler instance.

        :param target_column: The column that you want to predict in the dataset.
        """
        self.target_column = target_column
        self.scaler = MinMaxScaler()
        self.model = None
        self.history = None
        logging.info(f"FinancialPredictor initialized with target column: {self.target_column}")

    def prepare_data(self, data, features, lookback=1):
        """
        Prepare sequences of input features and target values for LSTM, handling NaN values.

        :param data: The input dataframe.
        :param features: List of feature columns to be used.
        :param lookback: Number of past time steps used for prediction.
        :return: Feature set (X) and target set (y) for LSTM.
        """
        logging.info("Preparing data for LSTM...")

        # Check for and handle NaN values by forward-filling and back-filling
        data_cleaned = data[features].fillna(method='ffill').fillna(method='bfill')
        logging.info("NaN values handled using forward-fill and back-fill methods.")

        # Find target index to use for extracting target values after scaling
        self.target_idx = features.index(self.target_column)

        # Scale features using MinMaxScaler
        scaled_data = self.scaler.fit_transform(data_cleaned)
        logging.info("Data scaled using MinMaxScaler.")

        # Prepare the sequences for training (features and target)
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i - lookback:i])
            y.append(scaled_data[i, self.target_idx])

        logging.info("Data preparation completed.")
        return np.array(X), np.array(y)

    def create_model(self, input_shape, lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
        """
        Create and compile an LSTM model with given hyperparameters.

        :param input_shape: Shape of input data for the LSTM.
        :param lstm_units: Number of units in each LSTM layer.
        :param dropout_rate: Dropout rate for regularization.
        :param learning_rate: Learning rate for the optimizer.
        :return: Compiled Keras model.
        """
        logging.info(
            f"Creating model with LSTM units: {lstm_units}, Dropout rate: {dropout_rate}, Learning rate: {learning_rate}")
        model = models.Sequential([
            layers.LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(dropout_rate),
            layers.LSTM(lstm_units),
            layers.Dropout(dropout_rate),
            layers.Dense(1)
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
        logging.info("Model compiled successfully.")
        return model

    def grid_search(self, X_train, y_train, X_val, y_val, feature_columns):
        """
        Perform grid search to find the best hyperparameters for the LSTM model.

        :param X_train: Training feature set.
        :param y_train: Training target set.
        :param X_val: Validation feature set.
        :param y_val: Validation target set.
        :param feature_columns: List of feature columns used in training.
        :return: Dictionary with the best hyperparameters.
        """
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
        }  # Default parameters in case nothing better is found

        logging.info("Starting grid search...")
        total_combinations = len(param_grid['lstm_units']) * len(param_grid['dropout_rate']) * len(
            param_grid['learning_rate'])
        current = 0

        # Iterate through all parameter combinations
        for units in param_grid['lstm_units']:
            for dropout in param_grid['dropout_rate']:
                for lr in param_grid['learning_rate']:
                    current += 1
                    logging.info(
                        f"Trying combination {current}/{total_combinations}: LSTM units: {units}, Dropout: {dropout}, Learning rate: {lr}")

                    # Create a model with current parameter combination
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
                        # Fit the model on training data
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
                        logging.info(f"Validation loss for current combination: {val_loss}")
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_params = {
                                'lstm_units': units,
                                'dropout_rate': dropout,
                                'learning_rate': lr
                            }
                    except Exception as e:
                        logging.error(f"Error with this combination, skipping... Error: {e}")
                        continue

        logging.info(f"Best parameters found: {best_params} with validation loss: {best_val_loss}")
        return best_params

    def train(self, X_train, y_train, feature_columns, epochs=30, batch_size=32, validation_split=0.1,
              lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
        """
        Train the LSTM model with the provided hyperparameters.

        :param X_train: Training feature set.
        :param y_train: Training target set.
        :param feature_columns: List of feature columns used in training.
        :param epochs: Number of training epochs.
        :param batch_size: Size of training batches.
        :param validation_split: Proportion of training data to use for validation.
        :param lstm_units: Number of units in each LSTM layer.
        :param dropout_rate: Dropout rate for regularization.
        :param learning_rate: Learning rate for the optimizer.
        :return: Training history object.
        """
        logging.info("Training model...")
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

        # Train the model and store training history
        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        logging.info("Model training completed.")
        return self.history

    def predict(self, X_test):
        """
        Make predictions using the trained LSTM model.

        :param X_test: Test feature set.
        :return: Predicted values, transformed back to original scale.
        """
        logging.info("Making predictions...")
        predictions = self.model.predict(X_test)

        # Create dummy array for inverse transform
        dummy = np.zeros((predictions.shape[0], X_test.shape[2]))
        dummy[:, self.target_idx] = predictions.ravel()

        # Inverse transform predictions
        transformed = self.scaler.inverse_transform(dummy)
        logging.info("Predictions made and transformed back to original scale.")
        return transformed[:, self.target_idx:self.target_idx + 1]

    def evaluate(self, y_true, y_pred):
        """
        Evaluate model predictions using standard metrics (MSE, MAE, R2).

        :param y_true: Actual values.
        :param y_pred: Predicted values.
        :return: Dictionary containing MSE, MAE, and R2 score.
        """
        logging.info("Evaluating model performance...")
        # Create dummy arrays with same number of features as original data for inverse transform
        dummy_true = np.zeros((y_true.shape[0], self.scaler.scale_.shape[0]))
        dummy_true[:, self.target_idx] = y_true

        # Inverse transform actual values
        y_true_orig = self.scaler.inverse_transform(dummy_true)[:, self.target_idx]

        # Remove any NaN values
        mask = ~np.isnan(y_true_orig) & ~np.isnan(y_pred.ravel())
        y_true_orig = y_true_orig[mask]
        y_pred = y_pred.ravel()[mask]

        if len(y_true_orig) == 0:
            logging.warning("No valid predictions after removing NaN values")
            return {'MSE': float('inf'), 'MAE': float('inf'), 'R2': float('-inf')}

        # Calculate metrics
        mse = mean_squared_error(y_true_orig, y_pred)
        mae = mean_absolute_error(y_true_orig, y_pred)
        r2 = r2_score(y_true_orig, y_pred)
        logging.info(f"Evaluation results - MSE: {mse}, MAE: {mae}, R2: {r2}")
        return {'MSE': mse, 'MAE': mae, 'R2': r2}

    def plot_results(self, y_true, y_pred, dates, title, lookback):
        """
        Plot the model's predicted values against actual values.

        :param y_true: Actual target values.
        :param y_pred: Predicted target values.
        :param dates: Dates corresponding to the observations.
        :param title: Title for the plot.
        :param lookback: Number of past time steps used for prediction (to adjust dates).
        """
        logging.info("Plotting results...")
        # Create dummy array for inverse transform
        dummy_true = np.zeros((y_true.shape[0], self.scaler.scale_.shape[0]))
        dummy_true[:, self.target_idx] = y_true
        y_true_orig = self.scaler.inverse_transform(dummy_true)[:, self.target_idx]

        # Remove NaN values for plotting
        mask = ~np.isnan(y_true_orig) & ~np.isnan(y_pred.ravel())
        y_true_orig = y_true_orig[mask]
        y_pred = y_pred.ravel()[mask]
        plot_dates = dates[lookback:][mask]

        if len(y_true_orig) == 0:
            logging.warning("No valid data to plot after removing NaN values")
            return

        # Plotting the actual vs predicted values
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

        logging.info("Results plotted successfully.")
        logging.info(
            f"Prediction Statistics - Mean Actual Value: {np.mean(y_true_orig):.2f}, Mean Predicted Value: {np.mean(y_pred):.2f}, Min Actual Value: {np.min(y_true_orig):.2f}, Max Actual Value: {np.max(y_true_orig):.2f}")
# Define a class for financial prediction
class FinancialPredictor:
    # Function to predict the target values based on the test data
    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        dummy = np.zeros((predictions.shape[0], X_test.shape[2]))
        dummy[:, self.target_idx] = predictions.ravel()
        transformed = self.scaler.inverse_transform(dummy)
        result = transformed[:, self.target_idx:self.target_idx + 1]

        # Check for and report any NaN values
        if np.any(np.isnan(result)):
            print(f"Warning: {np.sum(np.isnan(result))} NaN values in predictions")

        return result

    # Function to plot training history, loss improvement and validation loss
    def plot_training_history(self):
        plt.figure(figsize=(15, 5))
        # Plotting Training and Validation Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss - {self.target_column}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plotting Loss Improvement rate
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

# Training and evaluating the model with hyperparameter tuning and visualization
# Main function to train and evaluate the model

def train_and_evaluate(data, target_column, feature_columns, lookback=5, epochs=30, do_grid_search=False):
    # Set data index as datetime to capture trend
    data.index = pd.to_datetime(data.index)
    dates = data.index.values

    # Splitting dataset into training (80%) and testing (20%) datasets
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    train_dates = dates[:train_size]
    test_dates = dates[train_size:]

    # Initialize FinancialPredictor class with the target column
    predictor = FinancialPredictor(target_column)

    # Preparing the training and testing data based on the lookback period
    X_train, y_train = predictor.prepare_data(train_data, feature_columns, lookback=lookback)
    X_test, y_test = predictor.prepare_data(test_data, feature_columns, lookback=lookback)

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # If grid search is enabled, split validation data and search for best hyperparameters
    if do_grid_search:
        train_idx = int(len(X_train) * 0.8)
        X_train_gs, X_val = X_train[:train_idx], X_train[train_idx:]
        y_train_gs, y_val = y_train[:train_idx], y_train[train_idx:]

        # Get best hyperparameters via grid search
        best_params = predictor.grid_search(X_train_gs, y_train_gs, X_val, y_val, feature_columns)
        history = predictor.train(X_train, y_train, feature_columns, epochs=epochs, **best_params)
    else:
        # Otherwise, use default hyperparameters to train
        history = predictor.train(X_train, y_train, feature_columns, epochs=epochs)

    # Make predictions based on the testing data
    predictions = predictor.predict(X_test)
    # Evaluate predictions using the testing data
    metrics = predictor.evaluate(y_test, predictions)
    print(f"\nMetrics for {target_column}:")
    print(metrics)

    # Plotting the prediction results and training history
    predictor.plot_results(y_test, predictions, test_dates, "Bitcoin Price Prediction", lookback)
    predictor.plot_training_history()

    return predictor, metrics, test_dates

# Function to test different lookback periods to determine the optimal lookback value

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

        # Store metrics and predictors for each lookback value
        results[lookback] = {
            'metrics': metrics,
            'predictor': predictor
        }

        # Update best MSE and best lookback if current lookback yields better result
        if metrics['MSE'] < best_mse:
            best_mse = metrics['MSE']
            best_lookback = lookback

    # Printing and plotting the comparison results for different lookback periods
    print("\nComparison of Different Lookback Periods:")
    print("\nLookback | MSE | MAE | R2")
    print("-" * 50)
    for lookback in lookbacks:
        metrics = results[lookback]['metrics']
        print(f"{lookback:8d} | {metrics['MSE']:.2f} | {metrics['MAE']:.2f} | {metrics['R2']:.4f}")

    print(f"\nBest lookback period: {best_lookback} (MSE: {best_mse:.2f})")

    # Plotting MSE and R2 scores for different lookback periods
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
        # Load data from CSV file containing processed features
        data = pd.read_csv(data_path("without_noise", "all_features_final.csv"), index_col=0)
    except FileNotFoundError:
        print("Default processed-features file not found. Please provide the path:")
        file_path = input().strip()
        data = pd.read_csv(file_path, index_col=0)

    # First choose the time period for the prediction analysis
    periods = ['Daily', 'Weekly', 'Monthly', 'Quarterly']
    print("\nChoose the time period for analysis:")
    for i, period in enumerate(periods):
        print(f"{i + 1}. {period}")
    period_choice = int(input("\nEnter your choice (1-4): "))
    selected_period = periods[period_choice - 1]

    # Filtering out features related to the selected period
    period_features = [col for col in data.columns if selected_period in col]

    # Getting the base feature names
    base_features = sorted(list(set(col.split('_Combined_')[0] for col in period_features if '_Combined_' in col)))

    print("\nWhat would you like to predict? Available options:")
    for i, feature in enumerate(base_features):
        print(f"{i + 1}. {feature}")

    choice = int(input("\nEnter the number of your choice: "))
    base_feature = base_features[choice - 1]

    # Get the target column and all the related features for the chosen period
    target = f"{base_feature}_Combined_{selected_period}"
    related_features = []

    # Including all components related to the chosen target feature
    related_features.extend([col for col in period_features if base_feature in col])

    # Adding other indicators for the same period
    for other_feature in base_features:
        if other_feature != base_feature:
            related_features.extend([col for col in period_features if other_feature in col])

    print("\nSelected features for prediction:")
    for feature in related_features:
        print(f"- {feature}")

    # Check whether to perform grid search for hyperparameter tuning
    do_grid_search = input("\nWould you like to perform grid search for hyperparameter tuning? (y/n): ").lower() == 'y'

    # Handle multiple lookback periods to determine the best lookback
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
            data, target, related_features,
            lookbacks=lookbacks, epochs=30,
            do_grid_search=do_grid_search
        )

        # Retrain model using the best lookback period obtained
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