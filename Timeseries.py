# ============================================================
# Imports
# ============================================================
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import itertools
import optuna
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.statespace.sarimax import SARIMAX

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Layer


# ============================================================
# Metrics
# ============================================================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# ============================================================
# Sequence Builder
# ============================================================
def make_sequences(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# ============================================================
# Load Data
# ============================================================
df = yf.download("NFLX", start="2015-01-01")
df = df[["Close"]].dropna()

split = int(len(df) * 0.8)
train_close = df["Close"].iloc[:split]
test_close = df["Close"].iloc[split:]


# ============================================================
# SARIMA Hyperparameter Search (Unscaled)
# ============================================================
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 5) for x in pdq]

best_aic = np.inf
best_sarima = None

for order in pdq:
    for seasonal_order in seasonal_pdq:
        try:
            model = SARIMAX(
                train_close,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit(disp=False)
            if results.aic < best_aic:
                best_aic = results.aic
                best_sarima = results
        except:
            continue

sarima_preds = best_sarima.forecast(len(test_close))
sarima_rmse = rmse(test_close, sarima_preds)
sarima_mape = mape(test_close, sarima_preds)


# ============================================================
# Scaling for Deep Learning Models
# ============================================================
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(df[["Close"]].values)

train_scaled = scaled_values[:split]
test_scaled = scaled_values[split:]

X_train, y_train = make_sequences(train_scaled)
X_test, y_test = make_sequences(test_scaled)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# validation split (NO test leakage)
X_tr, X_val = X_train[:-200], X_train[-200:]
y_tr, y_val = y_train[:-200], y_train[-200:]


# ============================================================
# Optuna – Baseline LSTM
# ============================================================
def objective_lstm(trial):
    units = trial.suggest_int("units", 32, 128, step=32)
    batch = trial.suggest_categorical("batch_size", [16, 32])

    model = Sequential([
        LSTM(units, input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_tr, y_tr, epochs=5, batch_size=batch, verbose=0)

    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds)

study_lstm = optuna.create_study(direction="minimize")
study_lstm.optimize(objective_lstm, n_trials=10)
best_lstm = study_lstm.best_params


# ============================================================
# Train Final LSTM
# ============================================================
lstm_model = Sequential([
    LSTM(best_lstm["units"], input_shape=(X_train.shape[1], 1)),
    Dense(1)
])
lstm_model.compile(optimizer="adam", loss="mse")
lstm_model.fit(X_train, y_train,
               epochs=10,
               batch_size=best_lstm["batch_size"],
               verbose=0)

lstm_preds = lstm_model.predict(X_test)

y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
lstm_preds_inv = scaler.inverse_transform(lstm_preds)

lstm_rmse = rmse(y_test_inv, lstm_preds_inv)
lstm_mape = mape(y_test_inv, lstm_preds_inv)


# ============================================================
# Keras-Correct Attention Layer
# ============================================================
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, return_weights=False):
        super().__init__()
        self.return_weights = return_weights

    def build(self, input_shape):
        # input_shape = (batch, time_steps, features)
        self.W = self.add_weight(
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            shape=(input_shape[1],),
            initializer="zeros",
            trainable=True
        )

    def call(self, x):
        # x: (batch, time_steps, features)
        score = tf.tensordot(x, self.W, axes=1) + self.b
        score = tf.nn.tanh(score)

        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(
            x * tf.expand_dims(weights, -1),
            axis=1
        )

        if self.return_weights:
            return context, weights
        return context

# ============================================================
# Optuna – Attention LSTM
# ============================================================
def objective_attention(trial):
    units = trial.suggest_int("units", 32, 128, step=32)
    batch = trial.suggest_categorical("batch_size", [16, 32])

    inp = Input(shape=(X_train.shape[1], 1))
    lstm_out = LSTM(units, return_sequences=True)(inp)
    att_out = TemporalAttention()(lstm_out)
    out = Dense(1)(att_out)

    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_tr, y_tr, epochs=5, batch_size=batch, verbose=0)

    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds)

study_att = optuna.create_study(direction="minimize")
study_att.optimize(objective_attention, n_trials=10)
best_att = study_att.best_params


# ============================================================
# Train Final Attention Model
# ============================================================
inp = Input(shape=(X_train.shape[1], 1))
lstm_out = LSTM(best_att["units"], return_sequences=True)(inp)
att_out, att_weights = TemporalAttention(return_weights=True)(lstm_out)
out = Dense(1)(att_out)

att_model = Model(inp, out)
att_model.compile(optimizer="adam", loss="mse")
att_model.fit(X_train, y_train,
              epochs=10,
              batch_size=best_att["batch_size"],
              verbose=0)

att_preds = att_model.predict(X_test)
att_preds_inv = scaler.inverse_transform(att_preds)

att_rmse = rmse(y_test_inv, att_preds_inv)
att_mape = mape(y_test_inv, att_preds_inv)


# ============================================================
# Results Table (Deliverable 2)
# ============================================================
results = pd.DataFrame({
    "Model": ["SARIMA", "LSTM", "Attention LSTM"],
    "RMSE": [sarima_rmse, lstm_rmse, att_rmse],
    "MAPE (%)": [sarima_mape, lstm_mape, att_mape]
})

print("\nFINAL MODEL COMPARISON\n")
print(results)


# ============================================================
# Robust Attention Analysis (Deliverable 3)
# ============================================================
attention_extractor = Model(att_model.input, att_weights)
weights = attention_extractor.predict(X_test[:50])
mean_attention = weights.mean(axis=0)

plt.figure(figsize=(8,4))
plt.plot(mean_attention)
plt.title("Average Attention Weights Across Test Sequences")
plt.xlabel("Time Step")
plt.ylabel("Weight")
plt.show()