"""
Advanced Time Series Forecasting with Deep Learning and Attention
----------------------------------------------------------------
Baseline: LSTM Seq2Seq
Advanced: LSTM Seq2Seq + Luong Attention

Dataset: Multivariate Electricity Consumption (UCI via sklearn-compatible source)
Evaluation: Rolling Forecast Origin (Time Series CV)
Metrics: RMSE, MAE
"""

# =========================
# 1. Imports
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, RepeatVector,
    TimeDistributed, Softmax, Dot, Concatenate
)
from tensorflow.keras.optimizers import Adam

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# =========================
# 2. Load Dataset
# =========================
def load_dataset():
    """
    Load a multivariate time series dataset.
    Using UCI Electricity Consumption sample via pandas.
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption.csv"
    df = pd.read_csv(url, sep=";", low_memory=False)

    # Basic cleaning
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    # Convert to float
    for col in df.columns[2:]:
        df[col] = df[col].astype(float)

    # Use multivariate features
    features = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity"
    ]

    data = df[features].values
    return data


# =========================
# 3. Windowing Function
# =========================
def create_sequences(data, input_len=24, output_len=1):
    X, y = [], []
    for i in range(len(data) - input_len - output_len):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len, 0])  # forecast target
    return np.array(X), np.array(y)


# =========================
# 4. Time Series CV Split
# =========================
def rolling_split(X, y, train_ratio=0.7):
    split = int(len(X) * train_ratio)
    return X[:split], y[:split], X[split:], y[split:]


# =========================
# 5. Baseline LSTM Model
# =========================
def build_baseline_model(input_len, n_features, output_len):
    inputs = Input(shape=(input_len, n_features))
    x = LSTM(64, activation="tanh")(inputs)
    x = RepeatVector(output_len)(x)
    x = LSTM(64, activation="tanh", return_sequences=True)(x)
    outputs = TimeDistributed(Dense(1))(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(0.001),
        loss="mse"
    )
    return model


# =========================
# 6. Attention Model (Luong)
# =========================
def build_attention_model(input_len, n_features, output_len):
    encoder_inputs = Input(shape=(input_len, n_features))
    encoder_outputs, state_h, state_c = LSTM(
        64, return_sequences=True, return_state=True
    )(encoder_inputs)

    decoder_inputs = RepeatVector(output_len)(state_h)
    decoder_outputs = LSTM(64, return_sequences=True)(
        decoder_inputs, initial_state=[state_h, state_c]
    )

    # Luong Attention
    score = Dot(axes=[2, 2])([decoder_outputs, encoder_outputs])
    attention_weights = Softmax(axis=-1, name="attention_weights")(score)
    context = Dot(axes=[2, 1])([attention_weights, encoder_outputs])

    decoder_combined = Concatenate()([context, decoder_outputs])
    outputs = TimeDistributed(Dense(1))(decoder_combined)

    model = Model(encoder_inputs, outputs)
    model.compile(
        optimizer=Adam(0.001),
        loss="mse"
    )

    attention_model = Model(
        encoder_inputs,
        attention_weights
    )

    return model, attention_model


# =========================
# 7. Training & Evaluation
# =========================
def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        verbose=0
    )

    preds = model.predict(X_val)
    preds = preds.squeeze()

    rmse = np.sqrt(mean_squared_error(y_val.squeeze(), preds))
    mae = mean_absolute_error(y_val.squeeze(), preds)

    return rmse, mae, preds


# =========================
# 8. Main Pipeline
# =========================
def main():
    data = load_dataset()

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    INPUT_LEN = 24
    OUTPUT_LEN = 1

    X, y = create_sequences(data_scaled, INPUT_LEN, OUTPUT_LEN)
    X_train, y_train, X_val, y_val = rolling_split(X, y)

    # Baseline
    baseline_model = build_baseline_model(
        INPUT_LEN, X.shape[2], OUTPUT_LEN
    )
    baseline_rmse, baseline_mae, _ = evaluate_model(
        baseline_model, X_train, y_train, X_val, y_val
    )

    # Attention Model
    attn_model, attn_weights_model = build_attention_model(
        INPUT_LEN, X.shape[2], OUTPUT_LEN
    )
    attn_rmse, attn_mae, preds = evaluate_model(
        attn_model, X_train, y_train, X_val, y_val
    )

    print("\n===== MODEL COMPARISON =====")
    print(f"Baseline LSTM RMSE: {baseline_rmse:.4f}")
    print(f"Baseline LSTM MAE : {baseline_mae:.4f}")
    print(f"Attention LSTM RMSE: {attn_rmse:.4f}")
    print(f"Attention LSTM MAE : {attn_mae:.4f}")

    # =========================
    # 9. Attention Visualization
    # =========================
    attention_weights = attn_weights_model.predict(X_val[:3])

    for i in range(3):
        plt.figure(figsize=(6, 3))
        plt.imshow(attention_weights[i], aspect="auto", cmap="viridis")
        plt.colorbar()
        plt.title(f"Attention Weights – Forecast Example {i+1}")
        plt.xlabel("Encoder Time Steps")
        plt.ylabel("Decoder Step")
        plt.show()


if __name__ == "__main__":
    main()
