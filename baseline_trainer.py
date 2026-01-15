import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

print("=" * 60)
print("🚀 CNN–LSTM BASELINE WITH EXTENDED FORECAST")
print("=" * 60)

# -----------------------------
# Load preprocessed data
# -----------------------------
data = np.load("data/dataset.npz")
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

WINDOW = X_train.shape[1]
N_FEATURES = X_train.shape[2]

print(f"Train shape: {X_train.shape}")
print(f"Test shape : {X_test.shape}")

# -----------------------------
# Build CNN–LSTM model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        activation="relu",
        input_shape=(WINDOW, N_FEATURES)
    ),
    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.LSTM(64),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)  # LINEAR output (regression)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32,
    verbose=1
)

# -----------------------------
# Predict on test set
# -----------------------------
y_pred = model.predict(X_test, verbose=0).ravel()

# Metrics (scaled space)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n📊 RESULTS")
print(f"R²  : {r2:.4f}")
print(f"MAE : {mae:.4f} (scaled)")

# -----------------------------
# Extended future forecast
# -----------------------------
def forecast_future(model, last_window, steps=20):
    """
    Autoregressive multi-step forecast
    """
    forecasts = []
    current = last_window.copy()

    for _ in range(steps):
        pred = model.predict(current[np.newaxis], verbose=0)[0, 0]
        pred = max(pred, 0)  # safety

        forecasts.append(pred)

        # Shift window left
        current[:-1] = current[1:]

        # Copy previous feature row
        current[-1] = current[-2]

        # Update lag features (last two columns: Lag_1, Lag_5)
        lag1_idx = -2
        lag5_idx = -1

        current[-1, lag1_idx] = pred
        current[-1, lag5_idx] = current[-5, lag1_idx]

    return np.array(forecasts)

# Generate forecast
FORECAST_HORIZON = 5
last_window = X_test[-1]
forecast_scaled = forecast_future(model, last_window, steps=FORECAST_HORIZON)

# -----------------------------
# Better visualization
# -----------------------------
PLOT_LEN = 200

plt.figure(figsize=(14, 5))

# Past actual
plt.plot(
    y_test[:PLOT_LEN],
    label="Actual (past)",
    linewidth=2
)

# Past predicted
plt.plot(
    y_pred[:PLOT_LEN],
    "--",
    label="Predicted (past)",
    linewidth=2
)

# Forecast
forecast_start = PLOT_LEN
plt.plot(
    range(forecast_start, forecast_start + FORECAST_HORIZON),
    forecast_scaled,
    "r-o",
    label="Forecast (future)",
    markersize=4
)

# Separator
plt.axvline(
    forecast_start,
    linestyle=":",
    color="black",
    alpha=0.7,
    label="Forecast start"
)

plt.title(
    f"CNN–LSTM Traffic Forecast (Scaled) | R² = {r2:.3f}",
    fontsize=14
)
plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Traffic")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\n🔮 Forecast (next 5 seconds, scaled):")
for i, v in enumerate(forecast_scaled, 1):
    print(f"t+{i:02d}: {v:.4f}")

print("\n✅ Done.")
