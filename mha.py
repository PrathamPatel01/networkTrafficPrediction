# train_baseline_cnn_lstm_mha_v2.py
import os, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -----------------------------
# Reproducibility
# -----------------------------
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 75)
print("🚀 CNN–LSTM + MHA (stabilized v2) | predicts Target directly | MSE")
print("=" * 75)

# -----------------------------
# Load data
# -----------------------------
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
    return (y_scaled - y_offset) / y_scale

def log1p_to_bytes(z):
    return np.expm1(z)

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"WINDOW={WINDOW}, N_FEATURES={N_FEATURES}")

# -----------------------------
# Find indices for Target + Lag cols (for forecasting updates)
# -----------------------------
name_to_idx = {n: i for i, n in enumerate(feature_names)}
target_idx = name_to_idx.get("Target", None)

lag_idxs = []
k = 1
while f"Lag_{k}" in name_to_idx:
    lag_idxs.append(name_to_idx[f"Lag_{k}"])
    k += 1

print(f"Target idx: {target_idx}")
print(f"Lag idxs  : {lag_idxs} (count={len(lag_idxs)})")

# -----------------------------
# Hyperparams (safer MHA)
# -----------------------------
CNN_FILTERS = 64
LSTM_UNITS = 64

NUM_HEADS = 2
KEY_DIM = 16
ATTN_DROPOUT = 0.15

DROPOUT = 0.30
L2 = 1e-4
LR = 3e-4

# -----------------------------
# Model
# -----------------------------
reg = tf.keras.regularizers.l2(L2)

inputs = tf.keras.Input(shape=(WINDOW, N_FEATURES))

# CNN
x = tf.keras.layers.Conv1D(CNN_FILTERS, 5, padding="same", activation="relu", kernel_regularizer=reg)(inputs)
x = tf.keras.layers.Conv1D(CNN_FILTERS, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
x = tf.keras.layers.MaxPooling1D(2)(x)
x = tf.keras.layers.Dropout(0.25)(x)

# LSTM seq
x = tf.keras.layers.LSTM(LSTM_UNITS, return_sequences=True, dropout=0.20)(x)

# ---- Pre-Norm MHA block (stabilizes training) ----
x_norm = tf.keras.layers.LayerNormalization()(x)

attn = tf.keras.layers.MultiHeadAttention(
    num_heads=NUM_HEADS,
    key_dim=KEY_DIM,
    dropout=ATTN_DROPOUT,
    name="mha"
)(x_norm, x_norm)

# Gated residual: x + g * attn, where g in [0,1]
gate = tf.keras.layers.Dense(1, activation="sigmoid", name="attn_gate")(x_norm)   # (B,T,1)
attn = tf.keras.layers.Multiply()([attn, gate])

x = tf.keras.layers.Add()([x, attn])
x = tf.keras.layers.LayerNormalization()(x)

# Instead of avg/max pooling, use LAST timestep (often best for next-step regression)
x_last = tf.keras.layers.Lambda(lambda t: t[:, -1, :], name="last_timestep")(x)

# Head
h = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=reg)(x_last)
h = tf.keras.layers.Dropout(DROPOUT)(h)
h = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=reg)(h)
h = tf.keras.layers.Dropout(0.20)(h)
outputs = tf.keras.layers.Dense(1)(h)

model = tf.keras.Model(inputs, outputs)
model.summary()

# -----------------------------
# Train
# -----------------------------
opt = tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0)
model.compile(optimizer=opt, loss="mse")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best_baseline_cnn_lstm_mha_v2.keras", monitor="val_loss", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-5),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# Test evaluation
# -----------------------------
y_pred = model.predict(X_test, verbose=0).ravel()

mse_scaled = mean_squared_error(y_test, y_pred)
rmse_scaled = float(np.sqrt(mse_scaled))
mae_scaled = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

y_test_log = inverse_minmax(y_test)
y_pred_log = inverse_minmax(y_pred)

y_test_bytes = log1p_to_bytes(y_test_log)
y_pred_bytes = log1p_to_bytes(y_pred_log)

mse_bytes = mean_squared_error(y_test_bytes, y_pred_bytes)
rmse_bytes = float(np.sqrt(mse_bytes))
mae_bytes = mean_absolute_error(y_test_bytes, y_pred_bytes)

print("\n📊 TEST RESULTS (CNN–LSTM + MHA v2)")
print(f"R² (scaled target)  : {r2:.4f}")
print(f"MSE (scaled target) : {mse_scaled:.6f}")
print(f"RMSE (scaled target): {rmse_scaled:.4f}")
print(f"MAE (scaled target) : {mae_scaled:.4f}")
print(f"MAE (bytes)         : {mae_bytes:.2f}")
print(f"RMSE (bytes)        : {rmse_bytes:.2f}")

# -----------------------------
# Autoregressive forecast (same as baseline)
# -----------------------------
def forecast_future(model, last_window, steps=20, clip_range=(0.0, 1.0)):
    current = last_window.copy()
    out = []

    lag_hist = []
    if lag_idxs:
        for idx in lag_idxs:
            lag_hist.append(float(current[-1, idx]))

    for _ in range(steps):
        pred = float(model.predict(current[np.newaxis], verbose=0)[0, 0])
        pred = float(np.clip(pred, clip_range[0], clip_range[1]))
        out.append(pred)

        current[:-1] = current[1:]
        new_row = current[-2].copy()

        if target_idx is not None:
            new_row[target_idx] = pred

        if lag_idxs:
            new_row[lag_idxs[0]] = pred
            for j in range(1, len(lag_idxs)):
                new_row[lag_idxs[j]] = lag_hist[j-1] if lag_hist else new_row[lag_idxs[j-1]]
            lag_hist = [pred] + (lag_hist[:-1] if lag_hist else [pred]*(len(lag_idxs)-1))

        current[-1] = new_row

    return np.array(out, dtype=np.float32)

FORECAST_HORIZON = 20
PLOT_LEN = min(300, len(y_test))
start_idx = PLOT_LEN - 1

forecast_scaled = forecast_future(model, X_test[start_idx], steps=FORECAST_HORIZON, clip_range=(0.0, 1.0))

print("\n🔎 Forecast sanity check")
print("forecast head:", forecast_scaled[:5])
print("forecast std :", float(np.std(forecast_scaled)))

# -----------------------------
# Plot
# -----------------------------
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

plt.title(f"CNN–LSTM + MHA v2 | R²={r2:.3f} (scaled)", fontsize=14)
plt.xlabel("Time windows (250ms)")
plt.ylabel("Target (scaled)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("baseline_cnn_lstm_mha_v2_result.png", dpi=140)
plt.show()

print("\n✅ Saved plot to baseline_cnn_lstm_mha_v2_result.png")
print("✅ Saved best model to best_baseline_cnn_lstm_mha_v2.keras")

