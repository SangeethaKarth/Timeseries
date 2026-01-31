import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.statespace.sarimax import SARIMAX

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Layer
import tensorflow.keras.backend as K
import optuna

# ------------------------------------------------
# helper functions
# ------------------------------------------------

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def make_sequences(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# ------------------------------------------------
# loading the data
# ------------------------------------------------

df = yf.download("NFLX", start="2015-01-01")
df = df[['Close']]
df.fillna(method="ffill", inplace=True)

# feature engineering (before training)
df["lag_1"] = df["Close"].shift(1)
df["day_of_week"] = df.index.dayofweek
df.dropna(inplace=True)

values = df[["Close"]].values

scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(values)

split = int(len(scaled_values) * 0.8)
train_data = scaled_values[:split]
test_data = scaled_values[split:]


# ------------------------------------------------
# SARIMA baseline
# ------------------------------------------------

sarima_model = SARIMAX(
    train_data,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 5)
)

sarima_fit = sarima_model.fit(disp=False)
sarima_preds = sarima_fit.forecast(steps=len(test_data))

sarima_mae = mean_absolute_error(test_data, sarima_preds)
sarima_rmse = np.sqrt(mean_squared_error(test_data, sarima_preds))
sarima_mape = mape(test_data, sarima_preds)


# ------------------------------------------------
# prepare sequences for deep learning
# ------------------------------------------------

X_train, y_train = make_sequences(train_data)
X_test, y_test = make_sequences(test_data)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# ------------------------------------------------
# Optuna hyperparameter tuning (LSTM)
# ------------------------------------------------

def objective(trial):
    units = trial.suggest_int("units", 32, 128, step=32)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    model = Sequential()
    model.add(LSTM(units, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=batch_size,
        verbose=0
    )

    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

best_params = study.best_params


# ------------------------------------------------
# final LSTM using best parameters
# ------------------------------------------------

lstm_model = Sequential()
lstm_model.add(
    LSTM(best_params["units"], input_shape=(X_train.shape[1], 1))
)
lstm_model.add(Dense(1))

lstm_model.compile(optimizer="adam", loss="mse")
history_lstm = lstm_model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=best_params["batch_size"],
    verbose=0
)

lstm_preds = lstm_model.predict(X_test)

lstm_mae = mean_absolute_error(y_test, lstm_preds)
lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_preds))
lstm_mape = mape(y_test, lstm_preds)


# ------------------------------------------------
# attention layer
# ------------------------------------------------

class TemporalAttention(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="normal",
            trainable=True
        )
        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

    def call(self, x):
        score = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        weights = K.softmax(score)
        self.last_attention_weights = weights
        weights = K.expand_dims(weights, axis=-1)
        return K.sum(x * weights, axis=1)


# ------------------------------------------------
# attention-based LSTM
# ------------------------------------------------

inputs = Input(shape=(X_train.shape[1], 1))
lstm_out = LSTM(64, return_sequences=True)(inputs)
att_out = TemporalAttention()(lstm_out)
output = Dense(1)(att_out)

att_model = Model(inputs, output)
att_model.compile(optimizer="adam", loss="mse")
history_att = att_model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    verbose=0
)

att_preds = att_model.predict(X_test)

att_mae = mean_absolute_error(y_test, att_preds)
att_rmse = np.sqrt(mean_squared_error(y_test, att_preds))
att_mape = mape(y_test, att_preds)


# ------------------------------------------------
# final comparison table
# ------------------------------------------------

print("\nFINAL MODEL COMPARISON\n")
print(f"{'Model':<18} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
print("-" * 50)
print(f"{'SARIMA':<18} {sarima_mae:.4f} {sarima_rmse:.4f} {sarima_mape:.2f}%")
print(f"{'LSTM':<18} {lstm_mae:.4f} {lstm_rmse:.4f} {lstm_mape:.2f}%")
print(f"{'Attention LSTM':<18} {att_mae:.4f} {att_rmse:.4f} {att_mape:.2f}%")


# ------------------------------------------------
# plots
# ------------------------------------------------

# Actual vs predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test[:100], label="Actual", linewidth=2)
plt.plot(lstm_preds[:100], label="LSTM")
plt.plot(att_preds[:100], label="Attention LSTM")
plt.title("Actual vs Predicted (Scaled)")
plt.legend()
plt.show()

# training loss
plt.figure(figsize=(6, 4))
plt.plot(history_lstm.history["loss"], label="LSTM Loss")
plt.plot(history_att.history["loss"], label="Attention Loss")
plt.title("Training Loss")
plt.legend()
plt.show()

# attention weights
attention_weights = att_model.layers[2].last_attention_weights.numpy()

plt.figure(figsize=(8, 4))
plt.plot(attention_weights[0])
plt.title("Attention Weights Across Time Steps")
plt.xlabel("Time Step")
plt.ylabel("Weight")
plt.show()
