# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/20
# @Function: Test Trend module

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras import models, layers, optimizers, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

from project_paths import data_path


class TrendPredictor:
    def __init__(self, price_column, trend_threshold=0.01):
        self.price_column = price_column
        self.trend_threshold = trend_threshold
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None
        self.classes_ = None

    def calculate_trends(self, prices):
        """Convert price series into trend categories with momentum"""
        price_changes = np.diff(prices) / prices[:-1]

        # For shorter-term trend detection, use smaller window
        window = 2  # Reduced from 3
        smoothed_changes = np.convolve(price_changes, np.ones(window) / window, mode='valid')

        trends = []
        # Add initial points that we couldn't calculate due to the window
        trends.extend(['sideways'] * (len(price_changes) - len(smoothed_changes) + 1))

        for change in smoothed_changes:
            if abs(change) < self.trend_threshold:
                trends.append('sideways')
            elif change > 0:
                trends.append('up')
            else:
                trends.append('down')

        self.classes_ = np.unique(trends)
        return trends

    def prepare_data(self, data, lookback=1):
        """Prepare sequences for trend prediction using all available features including target lags"""
        df = data.copy()

        # Calculate technical indicators for target column
        df['returns'] = df[self.price_column].pct_change()
        df['momentum_1'] = df['returns'].rolling(window=2).mean()
        df['momentum_2'] = df['returns'].rolling(window=3).mean()
        df['MA3'] = df[self.price_column].rolling(window=3).mean()
        df['MA5'] = df[self.price_column].rolling(window=5).mean()
        df['volatility'] = df['returns'].rolling(window=3).std()
        df['roc'] = df[self.price_column].pct_change(periods=lookback)
        df['price_level'] = df[self.price_column] / df[self.price_column].rolling(window=3).mean()

        # Create lagged versions of all available features
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
        for col in features:
            for i in range(1, lookback + 1):
                df[f'{col}_lag_{i}'] = df[col].shift(i)

        # Fill NaN values
        df = df.fillna(method='bfill')

        # Combine base features and lagged features
        base_features = ['returns', 'momentum_1', 'momentum_2', 'MA3', 'MA5',
                         'volatility', 'roc', 'price_level']
        lagged_features = [f'{col}_lag_{i}'
                           for col in features
                           for i in range(1, lookback + 1)]

        all_features = base_features + lagged_features

        # Scale features
        scaled_features = self.scaler.fit_transform(df[all_features])

        # Calculate trends
        trends = self.calculate_trends(df[self.price_column].values)
        encoded_trends = self.label_encoder.fit_transform(trends)

        # Prepare sequences
        X, y = [], []
        for i in range(lookback, len(scaled_features)):
            X.append(scaled_features[i - lookback:i])
            y.append(encoded_trends[i])

        print(f"\nFeatures used ({len(all_features)} total):")
        print("Base features:", ', '.join(base_features))
        print("Lagged features:", ', '.join(lagged_features))

        return np.array(X), np.array(y)

    def create_model(self, input_shape, num_classes, lstm_units=32, dropout_rate=0.2, learning_rate=0.001):
        """Modified model architecture better suited for shorter sequences"""
        model = models.Sequential([
            layers.LSTM(lstm_units, input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(16, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, epochs=30, batch_size=32, validation_split=0.2,
              lstm_units=32, dropout_rate=0.2, learning_rate=0.001):
        num_classes = len(self.classes_)
        print(f"Training model with {num_classes} classes: {self.classes_}")

        # Calculate class weights to handle imbalanced data
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))

        self.model = self.create_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            num_classes=num_classes,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )

        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,  # Reduced from 15
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks_list,
            class_weight=class_weight_dict,
            verbose=1
        )
        return self.history

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return np.argmax(predictions, axis=1)

    def evaluate(self, y_true, y_pred):
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.classes_,
            output_dict=True
        )
        conf_matrix = confusion_matrix(y_true, y_pred)
        return report, conf_matrix

    def plot_results(self, y_true, y_pred, dates):
        plt.figure(figsize=(15, 12))

        # Plot 1: Confusion Matrix
        plt.subplot(3, 1, 1)
        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.imshow(conf_matrix, cmap='Blues')
        plt.title('Confusion Matrix')
        tick_marks = np.arange(len(self.classes_))
        plt.xticks(tick_marks, self.classes_, rotation=45)
        plt.yticks(tick_marks, self.classes_)

        for i in range(len(self.classes_)):
            for j in range(len(self.classes_)):
                plt.text(j, i, conf_matrix[i, j],
                         ha="center", va="center")

        # Plot 2: Training History
        plt.subplot(3, 1, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot 3: Trend Comparison
        plt.subplot(3, 1, 3)
        plt.plot(dates, self.label_encoder.inverse_transform(y_true), label='Actual Trend', marker='o', markersize=4)
        plt.plot(dates, self.label_encoder.inverse_transform(y_pred), label='Predicted Trend', marker='x', markersize=4)
        plt.title('Trend Prediction Comparison')
        plt.xlabel('Date')
        plt.ylabel('Trend')
        plt.legend()
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def grid_search(self, X_train, y_train, X_val, y_val):
        param_grid = {
            'lstm_units': [32, 64, 128],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01],
        }

        best_val_accuracy = float('-inf')
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
                        input_shape=(X_train.shape[1], X_train.shape[2]),
                        num_classes=len(self.classes_),
                        lstm_units=units,
                        dropout_rate=dropout,
                        learning_rate=lr
                    )

                    early_stopping = callbacks.EarlyStopping(
                        monitor='val_accuracy',
                        patience=10,
                        restore_best_weights=True
                    )

                    history = model.fit(
                        X_train,
                        y_train,
                        validation_data=(X_val, y_val),
                        epochs=30,
                        batch_size=32,
                        callbacks=[early_stopping],
                        verbose=0
                    )

                    val_accuracy = max(history.history['val_accuracy'])
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        best_params = {
                            'lstm_units': units,
                            'dropout_rate': dropout,
                            'learning_rate': lr
                        }

        print("\nBest parameters found:")
        print(best_params)
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
        return best_params

    def plot_results(self, y_true, y_pred, dates):
        plt.figure(figsize=(15, 12))

        # Plot 1: Confusion Matrix
        plt.subplot(3, 1, 1)
        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.imshow(conf_matrix, cmap='Blues')
        plt.title('Confusion Matrix')
        tick_marks = np.arange(len(self.classes_))
        plt.xticks(tick_marks, self.classes_, rotation=45)
        plt.yticks(tick_marks, self.classes_)

        for i in range(len(self.classes_)):
            for j in range(len(self.classes_)):
                plt.text(j, i, conf_matrix[i, j],
                         ha="center", va="center")

        # Plot 2: Training History
        plt.subplot(3, 1, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot 3: Trend Comparison with ordered y-axis labels
        plt.subplot(3, 1, 3)
        actual_trends = self.label_encoder.inverse_transform(y_true)
        predicted_trends = self.label_encoder.inverse_transform(y_pred)

        plt.plot(dates, actual_trends, label='Actual Trend', marker='o', markersize=4)
        plt.plot(dates, predicted_trends, label='Predicted Trend', marker='x', markersize=4)
        plt.title('Trend Prediction Comparison')
        plt.xlabel('Date')
        plt.ylabel('Trend')
        plt.yticks(['up', 'sideways', 'down'])
        plt.legend()
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()


def train_and_evaluate_trend(data, price_column, lookback=1, epochs=30, trend_threshold=0.01):  # Modified defaults
    data['Start'] = pd.to_datetime(data['Start'])
    data = data.sort_values('Start')

    predictor = TrendPredictor(price_column, trend_threshold)

    X, y = predictor.prepare_data(data, lookback=lookback)

    # Split the data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    test_dates = data['Start'].values[train_size + lookback:]

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Class distribution in training set:")
    for cls in np.unique(y_train):
        print(f"Class {predictor.label_encoder.inverse_transform([cls])[0]}: {np.sum(y_train == cls)}")

    predictor.train(X_train, y_train, epochs=epochs)
    predictions = predictor.predict(X_test)

    report, conf_matrix = predictor.evaluate(y_test, predictions)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=predictor.classes_))

    predictor.plot_results(y_test, predictions, test_dates)

    return predictor, report, conf_matrix


if __name__ == "__main__":
    try:
        data = pd.read_csv(data_path('Bitcoin_Price', 'price_sentiment_data.csv'))
    except FileNotFoundError:
        print("Default dataset not found. Please provide the CSV path:")
        file_path = input().strip()
        data = pd.read_csv(file_path)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap', 'sentiment_score']

    # Select target column
    print("\nWhat would you like to predict trends for? Available options:")
    for i, feature in enumerate(features):
        print(f"{i + 1}. {feature}")

    choice = int(input("\nEnter the number of your choice (1-7): "))
    target = features[choice - 1]

    print(f"\nSelected target: {target}")
    print("Will use all available features including lagged values of the target")

    print("\nStarting trend prediction...")
    predictor, report, conf_matrix = train_and_evaluate_trend(
        data=data,
        price_column=target,
        lookback=1,
        epochs=30,
        trend_threshold=0.01
    )
