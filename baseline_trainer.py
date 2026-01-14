
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 CNN-LSTM BASELINE - SIMPLE & CLEAN")
print("="*60)

# --- 1. LOAD OR PROCESS DATA ---
if os.path.exists('data/train_data.csv'):
    print("📥 Loading pre-split data...")
    train_df = pd.read_csv('data/train_data.csv')
    test_df = pd.read_csv('data/test_data.csv')
    print(f"✅ Loaded: {len(train_df)} train, {len(test_df)} test")
else:
    print("📥 Loading and processing data...")
    df = pd.read_csv('data/cleaned_data.csv')
    
    # Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Encode protocol
    encoder = LabelEncoder()
    train_df["Main_Protocol"] = encoder.fit_transform(train_df["Main_Protocol"])
    test_df["Main_Protocol"] = encoder.transform(test_df["Main_Protocol"])
    
    # Scale
    scaler = MinMaxScaler()
    cols = ["Total_Length", "Unique_Sources", "Unique_Destinations"]
    train_df[cols] = scaler.fit_transform(train_df[cols])
    test_df[cols] = scaler.transform(test_df[cols])
    
    print(f"✅ Processed: {len(train_df)} train, {len(test_df)} test")

# --- 2. CREATE SEQUENCES ---
def create_sequences(data, window=50, lead=5):
    X, y = [], []
    features = data[["Total_Length", "Unique_Sources", "Unique_Destinations", "Main_Protocol"]].values.astype("float32")
    for i in range(len(data) - window - lead):
        X.append(features[i:i + window])
        y.append(features[i + window + lead, 0])  # Predict Total_Length
    return np.array(X), np.array(y)

WINDOW, LEAD = 50, 5
X_train, y_train = create_sequences(train_df, WINDOW, LEAD)
X_test, y_test = create_sequences(test_df, WINDOW, LEAD)

print(f"\n📊 Sequences created:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# --- 3. BUILD MODEL ---
print("\n🏗️ Building CNN-LSTM model...")
model = models.Sequential([
    layers.Conv1D(64, 3, padding='same', activation="relu", input_shape=(WINDOW, 4)),
    layers.MaxPooling1D(2),
    layers.LSTM(100),
    layers.Dense(50, activation="relu"),
    layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=['mae'])
print("✅ Model ready")

# --- 4. TRAIN ---
print("\n🚀 Training model (10 epochs)...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,  # Fewer epochs for speed
    batch_size=32,
    verbose=1
)
print("✅ Training complete")

# --- 5. PREDICT ---
y_pred = model.predict(X_test, verbose=0).flatten()

# Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\n📊 PERFORMANCE:")
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")

# --- 6. YOUR ORIGINAL VISUALIZATION STYLE ---
def get_extended_forecast(model, start_window, steps=5):
    """Multi-step forecasting"""
    forecast = []
    curr_window = start_window.copy()
    for _ in range(steps):
        p = model.predict(curr_window[np.newaxis, :, :], verbose=0)[0, 0]
        forecast.append(p)
        new_row = curr_window[-1].copy()
        new_row[0] = p  # Update Total_Length
        curr_window = np.vstack([curr_window[1:], new_row])
    return np.array(forecast)

# Get extended forecast
extended_5s = get_extended_forecast(model, X_test[-1], steps=5)

# Create your style plot
plt.figure(figsize=(15, 7))
plt.style.use('ggplot')

# X-axis
PAST_VIEW = 100
x_history = np.arange(PAST_VIEW)
x_extended = np.arange(PAST_VIEW - 1, PAST_VIEW + 5)

# 1. Actual Traffic (Blue)
plt.plot(x_history, y_test[-PAST_VIEW:], 
         label="Actual Traffic (Real Data)", color="royalblue", linewidth=2.5)

# 2. Predicted Traffic (Crimson)
plt.plot(x_history, y_pred[-PAST_VIEW:], 
         label="Predicted Traffic (CNN-LSTM)", color="crimson", linestyle=":", linewidth=2)

# 3. Extended Prediction (Green)
green_line_y = np.concatenate([[y_test[-1]], extended_5s])
plt.plot(x_extended, green_line_y, 
         label="5s Extended Forecast", color="#2ca02c", linewidth=4, marker='s', markersize=7)

# 4. "NOW" separator
plt.axvline(PAST_VIEW - 1, color="black", linestyle="--", alpha=0.3)
plt.text(PAST_VIEW - 1.5, plt.ylim()[1]*0.8, "NOW", rotation=90, fontweight='bold', color='black')

# Labels
plt.title(f"Network Traffic Forecast | 5-Second Prediction | MSE: {mse:.4f}", fontsize=16, pad=15)
plt.xlabel("Time (Relative Seconds)", fontsize=12)
plt.ylabel("Traffic Volume (Min-Max Scaled)", fontsize=12)
plt.legend(loc="upper left", frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('baseline_results.png', dpi=100)
print("\n💾 Graph saved as 'baseline_results.png'")
plt.show()

# --- 7. QUICK LOSS PLOT (Optional) ---
if input("\n📈 Show training loss plot? (y/n): ").lower() == 'y':
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\n" + "="*60)
print("✅ BASELINE COMPLETE!")
print("="*60)