import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, LayerNormalization, 
                                     MultiHeadAttention, Dropout, 
                                     GlobalAveragePooling1D, Add, Conv1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Configuration ---
SYMBOL = 'AAPL'
START_DATE = '2020-01-01'
END_DATE = '2025-01-01'
SEQ_LEN = 60       # Lookback window
FORECAST_STEPS = 7 # Days to predict
d_k = 64           # Dimension of Key/Query
d_v = 64           # Dimension of Value
n_heads = 4        # Number of Attention Heads
ff_dim = 128       # Feed Forward Dimension

# --- Step 1: Robust Data Fetching & Engineering ---
print(f"Downloading data for {SYMBOL}...")
# auto_adjust=True fixes some MultiIndex issues in new yfinance versions
data = yf.download(SYMBOL, start=START_DATE, end=END_DATE, auto_adjust=True)

if data.empty:
    raise ValueError("No data downloaded. Check your internet or symbol.")

df = data[['Close', 'Volume']].copy()

# 1. RSI
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# 2. MACD
ema_short = df['Close'].ewm(span=12, adjust=False).mean()
ema_long = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_short - ema_long
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# 3. Bollinger Bands (New)
df['MA20'] = df['Close'].rolling(window=20).mean()
df['STD20'] = df['Close'].rolling(window=20).std()
df['Upper_Band'] = df['MA20'] + (df['STD20'] * 2)
df['Lower_Band'] = df['MA20'] - (df['STD20'] * 2)

# Cleanup
df.dropna(inplace=True)

# --- Step 2: Preprocessing ---
# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Create Sequences
X, y = [], []
for i in range(SEQ_LEN, len(scaled_data) - FORECAST_STEPS + 1):
    X.append(scaled_data[i-SEQ_LEN:i])
    # Target: Predict 'Close' price (column 0) for next FORECAST_STEPS
    y.append(scaled_data[i:i+FORECAST_STEPS, 0])

X, y = np.array(X), np.array(y)

# Split Train/Test (90/10 split for time series)
split_idx = int(len(X) * 0.9)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

# --- Step 3: Transformer Components ---

# Custom Layer: Positional Encoding
# Transformers have no sense of order without this.
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length

    def call(self, inputs):
        # inputs shape: (batch, seq_len, features)
        # Create positions: [0, 1, 2, ..., seq_len-1]
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs]) # SKIP CONNECTION 1

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return Add()([x, res]) # SKIP CONNECTION 2

# --- Step 4: Build Model ---
input_layer = Input(shape=(SEQ_LEN, X.shape[2]))

# Add Positional Information
x = Dense(d_k)(input_layer) # Project input to d_k dimension
x = PositionalEmbedding(SEQ_LEN, d_k)(x)

# Stack Transformer Blocks
x = transformer_encoder(x, d_k, n_heads, ff_dim, dropout=0.1)
x = transformer_encoder(x, d_k, n_heads, ff_dim, dropout=0.1)
x = transformer_encoder(x, d_k, n_heads, ff_dim, dropout=0.1)

# Global Average Pooling often works better than flattening for Time Series
x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
x = Dense(64, activation="relu")(x)
output_layer = Dense(FORECAST_STEPS)(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# --- Step 5: Training with Callbacks ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
]

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50, # Increased epochs (callbacks will stop it early if needed)
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# --- Step 6: Visualization ---
# Select a random sample from test set to visualize
sample_idx = np.random.randint(0, len(X_test))
input_seq = X_test[sample_idx]
actual_future = y_test[sample_idx]

# Predict
pred_future = model.predict(input_seq[np.newaxis, :, :])[0]

# Inverse Scale (We need to handle the extra feature dimensions)
# Create dummy arrays to satisfy the scaler's shape requirement
dummy_pred = np.zeros((FORECAST_STEPS, scaled_data.shape[1]))
dummy_actual = np.zeros((FORECAST_STEPS, scaled_data.shape[1]))
dummy_hist = np.zeros((SEQ_LEN, scaled_data.shape[1]))

# Fill Close price column (index 0)
dummy_pred[:, 0] = pred_future
dummy_actual[:, 0] = actual_future
dummy_hist[:, 0] = input_seq[:, 0]

# Inverse transform
res_pred = scaler.inverse_transform(dummy_pred)[:, 0]
res_actual = scaler.inverse_transform(dummy_actual)[:, 0]
res_hist = scaler.inverse_transform(dummy_hist)[:, 0]

# Plot
plt.figure(figsize=(12, 6))
range_hist = range(0, SEQ_LEN)
range_future = range(SEQ_LEN, SEQ_LEN + FORECAST_STEPS)

plt.plot(range_hist, res_hist, label='History (Past 60 Days)', color='gray')
plt.plot(range_future, res_actual, label='Actual Future', color='blue', marker='o')
plt.plot(range_future, res_pred, label='Predicted Future', color='red', linestyle='--', marker='x')

plt.title(f"Transformer Multi-Step Prediction: {SYMBOL}")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plot Loss
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Training Loss")
plt.legend()
plt.show()