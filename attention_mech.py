# # train_baseline_delta.py
# # CNN–LSTM baseline but trained as residual (delta) predictor:
# #   delta = y_next - y_last_in_window
# #   y_pred = y_last_in_window + delta_pred
# #
# # This often improves R² when Target + lags are already in the inputs.

# import os, random
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# # -----------------------------
# # Reproducibility
# # -----------------------------
# os.environ["TF_DETERMINISTIC_OPS"] = "1"
# random.seed(42)
# np.random.seed(42)
# tf.random.set_seed(42)

# print("=" * 60)
# print("🚀 CNN–LSTM BASELINE (Residual / Delta Target)")
# print("=" * 60)

# # -----------------------------
# # Load data
# # -----------------------------
# data = np.load("data/dataset.npz", allow_pickle=True)
# X_train, y_train = data["X_train"], data["y_train"]
# X_val, y_val     = data["X_val"], data["y_val"]
# X_test, y_test   = data["X_test"], data["y_test"]
# feature_names    = data["feature_names"].tolist()

# WINDOW = X_train.shape[1]
# N_FEATURES = X_train.shape[2]

# # For inverse transform (scaled -> log1p(bytes))
# y_scale  = float(data["y_scale"][0])
# y_offset = float(data["y_offset"][0])

# def inverse_minmax(y_scaled: np.ndarray) -> np.ndarray:
#     # y_scaled = y * scale + offset  => y = (y_scaled - offset)/scale
#     return (y_scaled - y_offset) / y_scale

# def log1p_to_bytes(z: np.ndarray) -> np.ndarray:
#     return np.expm1(z)

# print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
# print(f"WINDOW={WINDOW}, N_FEATURES={N_FEATURES}")

# # -----------------------------
# # Find indices for Target + Lag cols for forecasting updates
# # -----------------------------
# name_to_idx = {n: i for i, n in enumerate(feature_names)}
# target_idx = name_to_idx.get("Target", None)

# lag_idxs = []
# k = 1
# while f"Lag_{k}" in name_to_idx:
#     lag_idxs.append(name_to_idx[f"Lag_{k}"])
#     k += 1

# print(f"Target idx: {target_idx}")
# print(f"Lag idxs  : {lag_idxs} (count={len(lag_idxs)})")

# if target_idx is None:
#     raise ValueError("❌ 'Target' not found in feature_names. Your preprocess must include it as a feature.")

# # -----------------------------
# # Build residual (delta) targets
# # -----------------------------
# y_last_train = X_train[:, -1, target_idx].astype(np.float32)
# y_last_val   = X_val[:,   -1, target_idx].astype(np.float32)
# y_last_test  = X_test[:,  -1, target_idx].astype(np.float32)

# y_train_delta = (y_train.astype(np.float32) - y_last_train).astype(np.float32)
# y_val_delta   = (y_val.astype(np.float32)   - y_last_val).astype(np.float32)

# print("✅ Training residual target: delta = y_next - y_last_in_window")
# print("   delta stats (train): mean=", float(np.mean(y_train_delta)), "std=", float(np.std(y_train_delta)))

# # -----------------------------
# # Naive baseline (persistence) for reference
# # -----------------------------
# naive_pred = y_last_test  # predict next = last Target in window
# print("\n📉 NAIVE BASELINE (predict next = last Target)")
# print(f"R²  : {r2_score(y_test, naive_pred):.4f}")
# print(f"MSE : {mean_squared_error(y_test, naive_pred):.6f}")
# print(f"MAE : {mean_absolute_error(y_test, naive_pred):.6f}")

# # -----------------------------
# # Model (same as your baseline)
# # -----------------------------
# inputs = tf.keras.Input(shape=(WINDOW, N_FEATURES))

# x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(inputs)
# x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
# x = tf.keras.layers.MaxPooling1D(2)(x)
# x = tf.keras.layers.Dropout(0.25)(x)

# x = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
# x = tf.keras.layers.LSTM(32, dropout=0.1)(x)

# x = tf.keras.layers.Dense(64, activation="relu")(x)
# x = tf.keras.layers.Dropout(0.25)(x)
# x = tf.keras.layers.Dense(32, activation="relu")(x)

# # Output is delta (can be negative), so keep linear Dense(1)
# outputs = tf.keras.layers.Dense(1)(x)

# model = tf.keras.Model(inputs, outputs)

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-3),
#     loss="mse"
# )

# model.summary()

# callbacks = [
#     tf.keras.callbacks.ModelCheckpoint("best_baseline_delta.keras", monitor="val_loss", save_best_only=True),
#     tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-5),
#     tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
# ]

# history = model.fit(
#     X_train, y_train_delta,
#     validation_data=(X_val, y_val_delta),
#     epochs=150,
#     batch_size=32,
#     callbacks=callbacks,
#     verbose=1
# )

# # -----------------------------
# # Test evaluation (reconstruct y_pred)
# # -----------------------------
# delta_pred = model.predict(X_test, verbose=0).ravel().astype(np.float32)
# y_pred = (y_last_test + delta_pred).astype(np.float32)
# y_pred = np.clip(y_pred, 0.0, 1.0)  # since Target is scaled 0..1

# mse_scaled = mean_squared_error(y_test, y_pred)
# rmse_scaled = float(np.sqrt(mse_scaled))
# mae_scaled = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# # Convert to bytes metrics (Target is scaled log1p(bytes))
# y_test_log = inverse_minmax(y_test)
# y_pred_log = inverse_minmax(y_pred)

# y_test_bytes = log1p_to_bytes(y_test_log)
# y_pred_bytes = log1p_to_bytes(y_pred_log)

# mse_bytes = mean_squared_error(y_test_bytes, y_pred_bytes)
# rmse_bytes = float(np.sqrt(mse_bytes))
# mae_bytes = mean_absolute_error(y_test_bytes, y_pred_bytes)

# print("\n📊 TEST RESULTS (Residual model reconstructed to y)")
# print(f"R² (scaled target)  : {r2:.4f}")
# print(f"MSE (scaled target) : {mse_scaled:.6f}")
# print(f"RMSE (scaled target): {rmse_scaled:.4f}")
# print(f"MAE (scaled target) : {mae_scaled:.4f}")
# print(f"MAE (bytes)         : {mae_bytes:.2f}")
# print(f"RMSE (bytes)        : {rmse_bytes:.2f}")

# # -----------------------------
# # Autoregressive forecast (residual version)
# # -----------------------------
# def forecast_future_delta(model, last_window, steps=5, clip_range=(0.0, 1.0)):
#     """
#     Multi-step autoregressive forecast for residual model.
#     Model predicts delta_t = y_{t+1} - y_t(last).
#     We reconstruct y_{t+1} = y_t + delta_pred.
#     Also updates Target and Lag_1..Lag_K consistently.
#     """
#     current = last_window.copy()
#     out = []

#     lag_hist = []
#     if lag_idxs:
#         for idx in lag_idxs:
#             lag_hist.append(float(current[-1, idx]))

#     for _ in range(steps):
#         delta = float(model.predict(current[np.newaxis], verbose=0)[0, 0])
#         last_target = float(current[-1, target_idx])
#         pred = last_target + delta

#         if clip_range is not None:
#             pred = float(np.clip(pred, clip_range[0], clip_range[1]))

#         out.append(pred)

#         # shift window
#         current[:-1] = current[1:]
#         new_row = current[-2].copy()  # carry exogenous features forward

#         # update Target
#         new_row[target_idx] = pred

#         # update Lag_1..Lag_K
#         if lag_idxs:
#             new_row[lag_idxs[0]] = pred
#             for j in range(1, len(lag_idxs)):
#                 new_row[lag_idxs[j]] = lag_hist[j-1] if lag_hist else new_row[lag_idxs[j-1]]
#             lag_hist = [pred] + (lag_hist[:-1] if lag_hist else [pred]*(len(lag_idxs)-1))

#         current[-1] = new_row

#     return np.array(out, dtype=np.float32)

# # -----------------------------
# # Plot (aligned) + forecast
# # -----------------------------
# # Choose horizon: 5 means 1.25s ahead at 250ms; 20 means 5s ahead.
# FORECAST_HORIZON = 5

# PLOT_LEN = min(300, len(y_test))
# start_idx = PLOT_LEN - 1

# forecast_scaled = forecast_future_delta(
#     model,
#     X_test[start_idx],
#     steps=FORECAST_HORIZON,
#     clip_range=(0.0, 1.0)
# )

# print("\n🔎 Forecast sanity check")
# print("forecast head:", forecast_scaled[:5])
# print("forecast std :", float(np.std(forecast_scaled)))

# plt.figure(figsize=(14, 5))
# plt.plot(y_test[:PLOT_LEN], label="Actual (past)", linewidth=2)
# plt.plot(y_pred[:PLOT_LEN], "--", label="Predicted (past)", linewidth=2)

# forecast_start = PLOT_LEN
# plt.plot(
#     range(forecast_start, forecast_start + FORECAST_HORIZON),
#     forecast_scaled,
#     "r-o",
#     label="Forecast (future)",
#     markersize=3
# )
# plt.axvline(forecast_start, linestyle=":", color="black", alpha=0.7, label="Forecast start")

# plt.title(f"CNN–LSTM Residual | R²={r2:.3f} (scaled)", fontsize=14)
# plt.xlabel("Time windows (250ms)")
# plt.ylabel("Target (scaled)")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig("baseline_delta_result.png", dpi=140)
# plt.show()

# print("\n✅ Saved plot to baseline_delta_result.png")
# print("✅ Saved best model to best_baseline_delta.keras")






# train_delta_with_attention.py
# CNN -> (Bi)LSTM(return_sequences) -> ATTENTION -> Dense
# Trains residual target: delta = y_next - y_last_in_window
# Reconstruct: y_pred = y_last + delta_pred
#
# Toggle attention type with ATTENTION_TYPE.

import os, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -----------------------------
# CONFIG
# -----------------------------
ATTENTION_TYPE = "additive"   # "additive" or "mha"
USE_BILSTM = True
FORECAST_HORIZON = 5          # 5 steps = 1.25s ahead if freq=250ms
CLIP_RANGE = (0.0, 1.0)

CKPT_NAME = f"best_delta_{ATTENTION_TYPE}.keras"
PLOT_FILE = f"delta_{ATTENTION_TYPE}_result.png"

# -----------------------------
# Reproducibility
# -----------------------------
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print(f"🔥 RUNNING: train_delta_with_attention.py | ATTENTION={ATTENTION_TYPE}")
print("=" * 60)

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

def inverse_minmax(y_scaled: np.ndarray) -> np.ndarray:
    return (y_scaled - y_offset) / y_scale

def log1p_to_bytes(z: np.ndarray) -> np.ndarray:
    return np.expm1(z)

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"WINDOW={WINDOW}, N_FEATURES={N_FEATURES}")

# -----------------------------
# Find Target + lags indices
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

if target_idx is None:
    raise ValueError("❌ 'Target' not found in feature_names (from preprocess).")

# -----------------------------
# Residual (delta) targets
# -----------------------------
y_last_train = X_train[:, -1, target_idx].astype(np.float32)
y_last_val   = X_val[:,   -1, target_idx].astype(np.float32)
y_last_test  = X_test[:,  -1, target_idx].astype(np.float32)

y_train_delta = (y_train.astype(np.float32) - y_last_train).astype(np.float32)
y_val_delta   = (y_val.astype(np.float32)   - y_last_val).astype(np.float32)

print("✅ Training residual target: delta = y_next - y_last_in_window")
print("   delta stats (train): mean=", float(np.mean(y_train_delta)), "std=", float(np.std(y_train_delta)))

# -----------------------------
# Naive baseline (persistence)
# -----------------------------
naive_pred = y_last_test
print("\n📉 NAIVE BASELINE (predict next = last Target)")
print(f"R²  : {r2_score(y_test, naive_pred):.4f}")
print(f"MSE : {mean_squared_error(y_test, naive_pred):.6f}")
print(f"MAE : {mean_absolute_error(y_test, naive_pred):.6f}")

# -----------------------------
# Attention layers
# -----------------------------
class TemporalAdditiveAttention(tf.keras.layers.Layer):
    """
    Additive attention over time.
    Input:  (B, T, H)
    Output: (B, H) context
    """
    def __init__(self, attn_units=64, **kwargs):
        super().__init__(**kwargs)
        self.W = tf.keras.layers.Dense(attn_units, activation="tanh")
        self.v = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, x):
        score = self.v(self.W(x))               # (B, T, 1)
        w = tf.nn.softmax(score, axis=1)        # (B, T, 1)
        ctx = tf.reduce_sum(w * x, axis=1)      # (B, H)
        return ctx

def build_model(window: int, n_features: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window, n_features))

    # CNN
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # LSTM encoder (must keep return_sequences for attention)
    if USE_BILSTM:
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)
        )(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)
        )(x)
        hidden_dim = 128
    else:
        x = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
        x = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
        hidden_dim = 64

    # Attention
    if ATTENTION_TYPE == "additive":
        ctx = TemporalAdditiveAttention(attn_units=64, name="additive_attention")(x)

    elif ATTENTION_TYPE == "mha":
        # Multi-head self-attention over time, then pool
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=32, dropout=0.1, name="mha"
        )(x, x)
        x2 = tf.keras.layers.Add()([x, attn])
        x2 = tf.keras.layers.LayerNormalization()(x2)

        # Pool to fixed vector
        avg = tf.keras.layers.GlobalAveragePooling1D()(x2)
        mx  = tf.keras.layers.GlobalMaxPooling1D()(x2)
        ctx = tf.keras.layers.Concatenate()([avg, mx])

    else:
        raise ValueError("ATTENTION_TYPE must be 'additive' or 'mha'.")

    # Head to predict delta (can be negative)
    h = tf.keras.layers.Dense(128, activation="relu")(ctx)
    h = tf.keras.layers.Dropout(0.25)(h)
    h = tf.keras.layers.Dense(64, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.20)(h)
    out = tf.keras.layers.Dense(1, name="delta_out")(h)

    return tf.keras.Model(inputs, out)

model = build_model(WINDOW, N_FEATURES)

# Use MSE (as per your prof preference)
opt = tf.keras.optimizers.Adam(1e-3, clipnorm=1.0)
model.compile(optimizer=opt, loss="mse")

print("\nMODEL PARAMS:", model.count_params())
model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(CKPT_NAME, monitor="val_loss", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-5),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
]

history = model.fit(
    X_train, y_train_delta,
    validation_data=(X_val, y_val_delta),
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# Test evaluation (reconstruct y)
# -----------------------------
delta_pred = model.predict(X_test, verbose=0).ravel().astype(np.float32)
y_pred = (y_last_test + delta_pred).astype(np.float32)
y_pred = np.clip(y_pred, CLIP_RANGE[0], CLIP_RANGE[1])

mse_scaled = mean_squared_error(y_test, y_pred)
rmse_scaled = float(np.sqrt(mse_scaled))
mae_scaled = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Byte metrics
y_test_log = inverse_minmax(y_test)
y_pred_log = inverse_minmax(y_pred)
y_test_bytes = log1p_to_bytes(y_test_log)
y_pred_bytes = log1p_to_bytes(y_pred_log)

mse_bytes = mean_squared_error(y_test_bytes, y_pred_bytes)
rmse_bytes = float(np.sqrt(mse_bytes))
mae_bytes = mean_absolute_error(y_test_bytes, y_pred_bytes)

print("\n📊 TEST RESULTS (Delta + Attention, reconstructed to y)")
print(f"R² (scaled target)  : {r2:.4f}")
print(f"MSE (scaled target) : {mse_scaled:.6f}")
print(f"RMSE (scaled target): {rmse_scaled:.4f}")
print(f"MAE (scaled target) : {mae_scaled:.4f}")
print(f"MAE (bytes)         : {mae_bytes:.2f}")
print(f"RMSE (bytes)        : {rmse_bytes:.2f}")

# -----------------------------
# Autoregressive forecast (delta version)
# -----------------------------
def forecast_future_delta(model, last_window, steps=5):
    current = last_window.copy()
    out = []

    lag_hist = []
    if lag_idxs:
        for idx in lag_idxs:
            lag_hist.append(float(current[-1, idx]))

    for _ in range(steps):
        delta = float(model.predict(current[np.newaxis], verbose=0)[0, 0])
        last_target = float(current[-1, target_idx])
        pred = last_target + delta
        pred = float(np.clip(pred, CLIP_RANGE[0], CLIP_RANGE[1]))
        out.append(pred)

        current[:-1] = current[1:]
        new_row = current[-2].copy()

        new_row[target_idx] = pred

        if lag_idxs:
            new_row[lag_idxs[0]] = pred
            for j in range(1, len(lag_idxs)):
                new_row[lag_idxs[j]] = lag_hist[j-1] if lag_hist else new_row[lag_idxs[j-1]]
            lag_hist = [pred] + (lag_hist[:-1] if lag_hist else [pred]*(len(lag_idxs)-1))

        current[-1] = new_row

    return np.array(out, dtype=np.float32)

PLOT_LEN = min(300, len(y_test))
start_idx = PLOT_LEN - 1
forecast_scaled = forecast_future_delta(model, X_test[start_idx], steps=FORECAST_HORIZON)

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

plt.title(f"Delta + {ATTENTION_TYPE.upper()} Attention | R²={r2:.3f} (scaled)", fontsize=14)
plt.xlabel("Time windows (250ms)")
plt.ylabel("Target (scaled)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=140)
plt.show()

print(f"\n✅ Saved plot to {PLOT_FILE}")
print(f"✅ Saved best model to {CKPT_NAME}")

