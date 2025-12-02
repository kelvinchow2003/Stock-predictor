import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Fetch Stock Data
symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2025-01-01'

print("Downloading stock data...")
data = yf.download(symbol, start=start_date, end=end_date)

# Use 'Close' price and calculate technical indicators (RSI, MACD)
df = data[['Close']].copy()

# RSI Calculation
window_length = 14
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD Calculation
short_window = 12
long_window = 26
signal_window = 9
ema_short = df['Close'].ewm(span=short_window, adjust=False).mean()
ema_long = df['Close'].ewm(span=long_window, adjust=False).mean()
df['MACD'] = ema_short - ema_long
df['Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()

# Fill NaN values
df.fillna(method='bfill', inplace=True)

# Step 2: Preprocess Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Multi-step forecasting: predict next 7 days
sequence_length = 60
forecast_steps = 7
X, y = [], []
for i in range(sequence_length, len(scaled_data) - forecast_steps):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i:i+forecast_steps, 0])  # Only predict Close price

X, y = np.array(X), np.array(y)

# Reshape X for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

# Split into train and test sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 3: Build LSTM Model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(100),
    Dense(forecast_steps)
])
model.compile(optimizer='adam', loss='mean_squared_error')

print("Training model...")
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Step 4: Make Predictions
predictions = model.predict(X_test)

# Inverse transform predictions and actual values
predictions_rescaled = []
actual_rescaled = []
for i in range(len(predictions)):
    dummy_pred = np.zeros((forecast_steps, scaled_data.shape[1]))
    dummy_actual = np.zeros((forecast_steps, scaled_data.shape[1]))
    dummy_pred[:, 0] = predictions[i]
    dummy_actual[:, 0] = y_test[i]
    predictions_rescaled.append(scaler.inverse_transform(dummy_pred)[:, 0])
    actual_rescaled.append(scaler.inverse_transform(dummy_actual)[:, 0])

# Step 5: Visualization (plot first test sample)
plt.figure(figsize=(12, 6))
plt.plot(actual_rescaled[0], color='blue', label='Actual Price')
plt.plot(predictions_rescaled[0], color='red', label='Predicted Price')
plt.title(f'{symbol} Stock Price Prediction (Next {forecast_steps} Days)')
plt.xlabel('Days Ahead')
plt.ylabel('Price')
plt.legend()
plt.show()