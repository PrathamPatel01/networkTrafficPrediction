# train_baseline_plus_additive_warmstart.py
# ✅ Baseline + Additive Attention (warm-start from .keras baseline checkpoint)
# ✅ Works with Keras .keras format (no by_name load)
# ✅ Attention is gated so it can only provide a small correction (helps prove attention helps)

import os, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -----------------------------
# CONFIG
# -----------------------------
BASELINE_CKPT = "best_baseline.keras"   # your existing baseline checkpoint (.keras)
OUT_CKPT      = "best_baseline_additive_warmstart.keras"
PLOT_FILE     = "baseline_additive_warmstart.png"

FORECAST_HORIZON = 20
CLIP_RANGE = (0.0, 1.0)

# Two-phase training
EPOCHS_PHASE1 = 25   # train ONLY attention/gate/proj first
EPOCHS_PHASE2 = 80   # fine-tune all
BATCH_SIZE = 32

LR_PHASE1 = 1e-3
LR_PHASE2 = 3e-4

# -----------------------------
# Reproducibility
# -----------------------------
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 85)
print("🚀 BASELINE + ADDITIVE ATTENTION (warm-start + gated residual) | MSE")
print("=" * 85)

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

# For inverse transform (scaled -> log1p(bytes))
y_scale  = float(data["y_scale"][0])
y_offset = float(data["y_offset"][0])

def inverse_minmax(y_scaled):
    return (y_scaled - y_offset) / y_scale

def log1p_to_bytes(z):
    return np.expm1(z)

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"WINDOW={WINDOW}, N_FEATURES={N_FEATURES}")

# -----------------------------
# Find indices for Target + Lag cols for forecasting updates
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
# Additive attention layer
# -----------------------------
class TemporalAdditiveAttention(tf.keras.layers.Layer):
    """
    Additive attention over time.
    Input:  (B, T, H)
    Output: (B, H) context
    """
    def __init__(self, attn_units=32, **kwargs):
        super().__init__(**kwargs)
        self.W = tf.keras.layers.Dense(attn_units, activation="tanh")
        self.v = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, x):
        score = self.v(self.W(x))          # (B,T,1)
        w = tf.nn.softmax(score, axis=1)   # (B,T,1)
        ctx = tf.reduce_sum(w * x, axis=1) # (B,H)
        return ctx

# -----------------------------
# Baseline model builder (must match your baseline exactly + layer names)
# -----------------------------
def build_baseline(window, n_features):
    inp = tf.keras.Input(shape=(window, n_features), name="inp")

    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu", name="conv1")(inp)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.MaxPooling1D(2, name="pool")(x)
    x = tf.keras.layers.Dropout(0.25, name="drop_cnn")(x)

    x = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2, name="lstm1")(x)
    x = tf.keras.layers.LSTM(32, dropout=0.1, name="lstm2")(x)

    x = tf.keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = tf.keras.layers.Dropout(0.25, name="drop_dense")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="dense2")(x)
    out = tf.keras.layers.Dense(1, name="out")(x)

    return tf.keras.Model(inp, out, name="baseline")

# -----------------------------
# Baseline + attention model (gated residual correction)
# -----------------------------
def build_baseline_plus_attention(window, n_features):
    inp = tf.keras.Input(shape=(window, n_features), name="inp")

    # Same CNN backbone
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu", name="conv1")(inp)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.MaxPooling1D(2, name="pool")(x)
    x = tf.keras.layers.Dropout(0.25, name="drop_cnn")(x)

    # LSTM1 gives sequences (needed for attention)
    seq = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2, name="lstm1")(x)

    # Attention context from seq
    seq_norm = tf.keras.layers.LayerNormalization(name="attn_ln")(seq)
    ctx = TemporalAdditiveAttention(attn_units=32, name="additive_attn")(seq_norm)  # (B,64)

    # Baseline summarizer: LSTM2 gives vector (B,32)
    last = tf.keras.layers.LSTM(32, dropout=0.1, name="lstm2")(seq)

    # Project ctx to match last dim
    ctx_proj = tf.keras.layers.Dense(32, activation="linear", name="ctx_proj")(ctx)

    # Gate so attention starts small (bias ~ -2 => sigmoid ~ 0.12)
    gate_inp = tf.keras.layers.Concatenate(name="gate_concat")([last, ctx_proj])
    gate = tf.keras.layers.Dense(
        1, activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(-2.0),
        name="gate"
    )(gate_inp)

    gated_ctx = tf.keras.layers.Multiply(name="gated_ctx")([ctx_proj, gate])
    fused = tf.keras.layers.Add(name="fused_last_plus_attn")([last, gated_ctx])

    # Same dense head as baseline (names match for weight copy)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense1")(fused)
    x = tf.keras.layers.Dropout(0.25, name="drop_dense")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="dense2")(x)
    out = tf.keras.layers.Dense(1, name="out")(x)

    return tf.keras.Model(inp, out, name="baseline_plus_additive")

# -----------------------------
# Build models
# -----------------------------
baseline_arch = build_baseline(WINDOW, N_FEATURES)
attn_model = build_baseline_plus_attention(WINDOW, N_FEATURES)

# -----------------------------
# Load baseline checkpoint (.keras) and copy weights layer-by-layer
# -----------------------------
try:
    baseline_loaded = tf.keras.models.load_model(BASELINE_CKPT, compile=False)
    print(f"✅ Loaded baseline model from {BASELINE_CKPT}")
except Exception as e:
    raise RuntimeError(f"❌ Could not load {BASELINE_CKPT}. Train baseline first or fix path.\n{e}")

loaded_map = {layer.name: layer for layer in baseline_loaded.layers}

copied, skipped = 0, 0
for layer in attn_model.layers:
    if layer.name in loaded_map:
        src = loaded_map[layer.name]
        try:
            layer.set_weights(src.get_weights())
            copied += 1
        except Exception:
            skipped += 1

print(f"✅ Warm-start complete (keras-safe). Copied: {copied}, Skipped: {skipped}")
attn_model.summary()

# -----------------------------
# Phase 1: Train only new layers (attention + gate)
# -----------------------------
TRAINABLE_NEW = {"attn_ln", "additive_attn", "ctx_proj", "gate_concat", "gate", "gated_ctx", "fused_last_plus_attn"}

for layer in attn_model.layers:
    layer.trainable = (layer.name in TRAINABLE_NEW)

attn_model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_PHASE1, clipnorm=1.0),
    loss="mse"
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(OUT_CKPT, monitor="val_loss", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-5),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    tf.keras.callbacks.TerminateOnNaN(),
]

print("\n🧊 Phase 1: training ONLY attention/gate/projection (backbone frozen)")
attn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_PHASE1,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# Phase 2: Fine-tune all layers
# -----------------------------
for layer in attn_model.layers:
    layer.trainable = True

attn_model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_PHASE2, clipnorm=1.0),
    loss="mse"
)

print("\n🔥 Phase 2: fine-tuning ALL layers (small LR)")
attn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_PHASE2,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# Test evaluation (NO clipping!)
# -----------------------------
y_pred = attn_model.predict(X_test, verbose=0).ravel()

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

print("\n📊 TEST RESULTS (Warm-start Baseline + Additive, gated)")
print(f"R² (scaled target)  : {r2:.4f}")
print(f"MSE (scaled target) : {mse_scaled:.6f}")
print(f"RMSE (scaled target): {rmse_scaled:.4f}")
print(f"MAE (scaled target) : {mae_scaled:.4f}")
print(f"MAE (bytes)         : {mae_bytes:.2f}")
print(f"RMSE (bytes)        : {rmse_bytes:.2f}")

# -----------------------------
# Forecast (clip only here)
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

PLOT_LEN = min(300, len(y_test))
start_idx = PLOT_LEN - 1
forecast_scaled = forecast_future(attn_model, X_test[start_idx], steps=FORECAST_HORIZON, clip_range=CLIP_RANGE)

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

plt.title(f"Warm-start Baseline + Additive | R²={r2:.3f} (scaled)", fontsize=14)
plt.xlabel("Time windows (250ms)")
plt.ylabel("Target (scaled)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=140)
plt.show()

print(f"\n✅ Saved plot to {PLOT_FILE}")
print(f"✅ Saved best model to {OUT_CKPT}")

