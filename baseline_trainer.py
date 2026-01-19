# train_baseline.py
import os, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Reproducibility
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("🚀 CNN–LSTM BASELINE (MSE report)")
print("=" * 60)

data = np.load("data/dataset.npz", allow_pickle=True)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val     = data["X_val"], data["y_val"]
X_test, y_test   = data["X_test"], data["y_test"]
feature_names    = data["feature_names"].tolist()

WINDOW = X_train.shape[1]
N_FEATURES = X_train.shape[2]

y_scale  = float(data["y_scale"][0])
y_offset = float(data["y_offset"][0])

def inverse_minmax(y_scaled):
    # y_scaled = y * scale + offset  => y = (y_scaled - offset)/scale
    return (y_scaled - y_offset) / y_scale

def log1p_to_bytes(z):
    return np.expm1(z)

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"WINDOW={WINDOW}, N_FEATURES={N_FEATURES}")

# Find indices for Target + Lag cols for forecasting updates
name_to_idx = {n: i for i, n in enumerate(feature_names)}
target_idx = name_to_idx.get("Target", None)
lag_idxs = []
k = 1
while f"Lag_{k}" in name_to_idx:
    lag_idxs.append(name_to_idx[f"Lag_{k}"])
    k += 1

# -----------------------------
# Model (strong baseline, still reasonable size)
# -----------------------------
inputs = tf.keras.Input(shape=(WINDOW, N_FEATURES))

x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(inputs)
x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling1D(2)(x)
x = tf.keras.layers.Dropout(0.25)(x)

x = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
x = tf.keras.layers.LSTM(32, dropout=0.1)(x)

x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, outputs)

# ✅ Train with MSE (as per your prof preference)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse"
)

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best_baseline.keras", monitor="val_loss", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-5),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# Test evaluation (report MSE + RMSE + R²)
# -----------------------------
y_pred = model.predict(X_test, verbose=0).ravel()

mse_scaled = mean_squared_error(y_test, y_pred)
rmse_scaled = float(np.sqrt(mse_scaled))
mae_scaled = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Convert to bytes MAE/MSE (Target is scaled log1p(bytes))
y_test_log = inverse_minmax(y_test)
y_pred_log = inverse_minmax(y_pred)

y_test_bytes = log1p_to_bytes(y_test_log)
y_pred_bytes = log1p_to_bytes(y_pred_log)

mse_bytes = mean_squared_error(y_test_bytes, y_pred_bytes)
rmse_bytes = float(np.sqrt(mse_bytes))
mae_bytes = mean_absolute_error(y_test_bytes, y_pred_bytes)

print("\n📊 TEST RESULTS")
print(f"R² (scaled target)  : {r2:.4f}")
print(f"MSE (scaled target) : {mse_scaled:.6f}")
print(f"RMSE (scaled target): {rmse_scaled:.4f}")
print(f"MAE (scaled target) : {mae_scaled:.4f}")
print(f"MAE (bytes)         : {mae_bytes:.2f}")
print(f"RMSE (bytes)        : {rmse_bytes:.2f}")

# -----------------------------
# Autoregressive forecast (updates Target + lags properly)
# -----------------------------
def forecast_future(model, last_window, steps=20):
    current = last_window.copy()
    out = []

    for _ in range(steps):
        pred = float(model.predict(current[np.newaxis], verbose=0)[0, 0])
        pred = max(pred, 0.0)
        out.append(pred)

        current[:-1] = current[1:]
        current[-1] = current[-2]

        # update Target input feature
        if target_idx is not None:
            current[-1, target_idx] = pred

        # update lags Lag_1..Lag_K
        if lag_idxs:
            current[-1, lag_idxs[0]] = pred
            for j in range(1, len(lag_idxs)):
                current[-1, lag_idxs[j]] = current[-2, lag_idxs[j-1]]

    return np.array(out)

FORECAST_HORIZON = 20
forecast_scaled = forecast_future(model, X_test[-1], steps=FORECAST_HORIZON)

# -----------------------------
# Plot
# -----------------------------
PLOT_LEN = min(300, len(y_test))
plt.figure(figsize=(14, 5))
plt.plot(y_test[:PLOT_LEN], label="Actual (past)", linewidth=2)
plt.plot(y_pred[:PLOT_LEN], "--", label="Predicted (past)", linewidth=2)

forecast_start = PLOT_LEN
plt.plot(
    range(forecast_start, forecast_start + FORECAST_HORIZON),
    forecast_scaled,
    "r-o",
    label="Forecast (future)",
    markersize=3
)
plt.axvline(forecast_start, linestyle=":", color="black", alpha=0.7, label="Forecast start")

plt.title(f"CNN–LSTM Baseline | R²={r2:.3f} (scaled)", fontsize=14)
plt.xlabel("Time windows (250ms)")
plt.ylabel("Target (scaled)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("baseline_result.png", dpi=140)
plt.show()

print("\n✅ Saved plot to baseline_result.png")





# # baseline_trainer.py  (MIMIC REPO)
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import joblib
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, RepeatVector, LSTM, Dense
# from tensorflow.keras.optimizers import Adam

# MODEL_FILE = "trained_model_mimic.keras"
# SCALER_FILE = "data/scaler.gz"
# DATASET_FILE = "data/dataset.npz"

# SEQ_LEN = 5
# FREQ_SECONDS = 10
# FORECAST_STEPS = 15  # 15 * 10s = 150 seconds

# def create_model(seq_len):
#     model = Sequential([
#         Conv1D(filters=64, kernel_size=2, activation="relu", input_shape=(seq_len, 1)),
#         MaxPooling1D(pool_size=2),
#         Flatten(),
#         RepeatVector(1),
#         LSTM(25, activation="relu", return_sequences=True),
#         LSTM(25, activation="relu"),
#         Dense(1)
#     ])
#     model.compile(optimizer=Adam(learning_rate=0.01), loss="mse")
#     return model

# def extend_predictions(model, last_seq, steps):
#     preds = []
#     seq = last_seq.copy()
#     for _ in range(steps):
#         p = float(model.predict(seq.reshape(1, SEQ_LEN, 1), verbose=0)[0, 0])
#         preds.append(p)
#         seq = np.append(seq[1:], p)
#     return np.array(preds, dtype=np.float32)

# def main():
#     data = np.load(DATASET_FILE, allow_pickle=True)
#     X = data["X"].astype(np.float32)
#     y = data["y"].astype(np.float32)
#     ts = pd.to_datetime(data["ts"])

#     print(f"Loaded: X={X.shape}, y={y.shape}")

#     model = create_model(SEQ_LEN)

#     # Mimic repo: use validation_split (not a strict time split)
#     history = model.fit(X, y, epochs=100, validation_split=0.2, verbose=0)

#     y_pred = model.predict(X, verbose=0).ravel()

#     mse = mean_squared_error(y, y_pred)
#     mae = mean_absolute_error(y, y_pred)
#     r2  = r2_score(y, y_pred)

#     print("\n📊 RESULTS (repo-style / optimistic)")
#     print(f"R²  : {r2:.4f}")
#     print(f"MSE : {mse:.6f}")
#     print(f"MAE : {mae:.6f}")

#     # Extended forecast
#     last_seq = X[-1].reshape(SEQ_LEN)
#     ext = extend_predictions(model, last_seq, FORECAST_STEPS)

#     last_t = ts[-1]
#     ext_ts = [last_t + pd.Timedelta(seconds=FREQ_SECONDS*(i+1)) for i in range(FORECAST_STEPS)]

#     # Plot like the reference
#     plt.figure(figsize=(12, 6))
#     plt.plot(ts, y, label="Actual")
#     plt.plot(ts, y_pred, label="Predicted")
#     plt.plot(ext_ts, ext, "--", label="Extended Predictions")
#     plt.xlabel("Time")
#     plt.ylabel("Length (scaled)")
#     plt.title("Actual vs Predicted Values(taking 10s interval)")
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.savefig("prediction_plot_mimic.png", dpi=140)
#     plt.show()

#     model.save(MODEL_FILE)
#     print(f"\n✅ Saved model: {MODEL_FILE}")
#     print("✅ Saved plot : prediction_plot_mimic.png")

# if __name__ == "__main__":
#     main()
