# train_baseline_plus_mha_warmstart.py
# ✅ Baseline + Multi-Head Self-Attention (warm-start from .keras baseline checkpoint)
# ✅ Normal training: load baseline -> add MHA -> train -> evaluate -> forecast -> plot
# ✅ Includes MHA attention logs (recent-mass, lookback, top key steps, gate stats)

import os, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -----------------------------
# CONFIG
# -----------------------------
BASELINE_CKPT = "best_baseline.keras"
OUT_CKPT      = "best_baseline_mha_warmstart.keras"
PLOT_FILE     = "baseline_mha_warmstart.png"

FORECAST_HORIZON = 20
CLIP_RANGE = (0.0, 1.0)

# Training
EPOCHS_PHASE1 = 25
EPOCHS_PHASE2 = 80
BATCH_SIZE = 32

# Safer phase-1 LR (prevents loss explosions you saw)
LR_PHASE1 = 3e-4
LR_PHASE2 = 2e-4

# MHA config
N_HEADS = 4
KEY_DIM = 16
ATTN_DROPOUT = 0.1

# Attention log settings (time interpretation)
FREQ_MS     = 250.0
POOL_FACTOR = 2  # MaxPooling1D(2) halves time => 1 step ~= 0.5s

# -----------------------------
# Reproducibility
# -----------------------------
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 90)
print("🚀 BASELINE + MHA (warm-start + gated residual) | MSE")
print("=" * 90)

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
# Indices for forecasting updates
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
# Baseline builder (must match your baseline EXACTLY)
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
# Baseline + MHA (gated residual correction)
# -----------------------------
def build_baseline_plus_mha(window, n_features):
    inp = tf.keras.Input(shape=(window, n_features), name="inp")

    # Same CNN backbone
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu", name="conv1")(inp)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.MaxPooling1D(2, name="pool")(x)
    x = tf.keras.layers.Dropout(0.25, name="drop_cnn")(x)

    # LSTM1 sequence (T shrinks after pool)
    seq = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2, name="lstm1")(x)

    # MHA over time (self-attention)
    seq_norm = tf.keras.layers.LayerNormalization(name="attn_ln")(seq)

    mha_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=N_HEADS,
        key_dim=KEY_DIM,
        dropout=ATTN_DROPOUT,
        name="mha"
    )
    attn_out = mha_layer(seq_norm, seq_norm)

    # Residual + norm
    attn_res = tf.keras.layers.Add(name="attn_residual")([seq_norm, attn_out])
    attn_res = tf.keras.layers.LayerNormalization(name="attn_post_ln")(attn_res)

    # Summarize attention output to a vector
    ctx64 = tf.keras.layers.GlobalAveragePooling1D(name="attn_pool")(attn_res)  # (B,64)

    # Baseline summarizer (keep it)
    last32 = tf.keras.layers.LSTM(32, dropout=0.1, name="lstm2")(seq)  # (B,32)

    # Project ctx to 32
    ctx32 = tf.keras.layers.Dense(32, activation="linear", name="ctx_proj")(ctx64)

    # Gate (start small)
    gate_inp = tf.keras.layers.Concatenate(name="gate_concat")([last32, ctx32])
    gate = tf.keras.layers.Dense(
        1, activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(-2.0),
        name="gate"
    )(gate_inp)

    gated_ctx = tf.keras.layers.Multiply(name="gated_ctx")([ctx32, gate])
    fused = tf.keras.layers.Add(name="fused_last_plus_mha")([last32, gated_ctx])

    # Same dense head as baseline
    x = tf.keras.layers.Dense(64, activation="relu", name="dense1")(fused)
    x = tf.keras.layers.Dropout(0.25, name="drop_dense")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="dense2")(x)
    out = tf.keras.layers.Dense(1, name="out")(x)

    return tf.keras.Model(inp, out, name="baseline_plus_mha")

# -----------------------------
# MHA Attention logging (like your additive logs)
# -----------------------------
def build_mha_probe(model: tf.keras.Model) -> tf.keras.Model:
    """
    Probe returns:
      scores: (B, heads, T, T) from the MHA layer (avg later)
      gate  : (B,1)
    """
    seq_norm = model.get_layer("attn_ln").output
    mha = model.get_layer("mha")
    gate = model.get_layer("gate").output

    _, scores = mha(seq_norm, seq_norm, return_attention_scores=True)  # (B, heads, T, T)
    return tf.keras.Model(model.input, [scores, gate], name="mha_probe")

def log_mha_attention(model: tf.keras.Model, X_probe, freq_ms=250.0, pool_factor=2):
    probe = build_mha_probe(model)
    scores, gate = probe(X_probe, training=False)
    scores = scores.numpy().astype(np.float64)          # (B, heads, T, T)
    gate = gate.numpy().ravel().astype(np.float64)      # (B,)

    # Average across batch and heads -> (T, T)
    S = scores.mean(axis=(0, 1))

    # Key importance distribution: average across queries -> (T,)
    key_imp = S.mean(axis=0)
    T = key_imp.shape[0]

    step_s = (freq_ms / 1000.0) * pool_factor
    span_s = (T - 1) * step_s

    def share_last(seconds):
        k = int(np.ceil(seconds / step_s))
        k = max(1, min(k, T))
        return float(key_imp[-k:].sum())

    share_5  = share_last(5.0)
    share_10 = share_last(10.0)
    share_15 = share_last(15.0)

    idx = np.arange(T, dtype=np.float64)
    com = float((key_imp * idx).sum() / (key_imp.sum() + 1e-12))  # 0=oldest
    lookback_s = float(span_s - com * step_s)

    top_steps = np.argsort(key_imp)[-5:][::-1].tolist()
    top_steps_sec_before = [float(span_s - s * step_s) for s in top_steps]

    print("\n[MHA ATTN QUICK LOG]")
    print(f"T={T} steps | step≈{step_s:.2f}s | span≈{span_s:.1f}s")
    print(f"Attention mass last  5s: {share_5:.3f}")
    print(f"Attention mass last 10s: {share_10:.3f}")
    print(f"Attention mass last 15s: {share_15:.3f}")
    print(f"Lookback (center-of-mass): {lookback_s:.2f}s before prediction")
    print(f"Top-5 key steps (idx): {top_steps}")
    print(f"Top-5 sec-before-pred: {[round(x,2) for x in top_steps_sec_before]}")
    print(f"Gate mean={float(gate.mean()):.3f} (p90={float(np.quantile(gate,0.9)):.3f})\n")

# -----------------------------
# Build models
# -----------------------------
baseline_arch = build_baseline(WINDOW, N_FEATURES)
mha_model = build_baseline_plus_mha(WINDOW, N_FEATURES)

# -----------------------------
# Warm-start from baseline checkpoint (.keras) and copy weights
# -----------------------------
baseline_loaded = tf.keras.models.load_model(BASELINE_CKPT, compile=False)
print(f"✅ Loaded baseline model from {BASELINE_CKPT}")

loaded_map = {layer.name: layer for layer in baseline_loaded.layers}

copied, skipped = 0, 0
for layer in mha_model.layers:
    if layer.name in loaded_map:
        src = loaded_map[layer.name]
        try:
            sw, tw = src.get_weights(), layer.get_weights()
            if len(sw) == len(tw) and all(a.shape == b.shape for a, b in zip(sw, tw)):
                layer.set_weights(sw)
                copied += 1
            else:
                skipped += 1
        except Exception:
            skipped += 1

print(f"✅ Warm-start complete. Copied: {copied}, Skipped: {skipped}")
mha_model.summary()

# -----------------------------
# Fresh callbacks per phase (avoid state leakage)
# -----------------------------
def make_callbacks():
    return [
        tf.keras.callbacks.ModelCheckpoint(OUT_CKPT, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

# -----------------------------
# Phase 1: Train only new layers
# -----------------------------
TRAINABLE_NEW = {
    "attn_ln", "mha", "attn_residual", "attn_post_ln",
    "attn_pool", "ctx_proj",
    "gate_concat", "gate", "gated_ctx", "fused_last_plus_mha",
}

for layer in mha_model.layers:
    layer.trainable = (layer.name in TRAINABLE_NEW)

mha_model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_PHASE1, clipnorm=1.0),
    loss="mse"
)

print("\n🧊 Phase 1: training ONLY MHA/gate/projection (backbone frozen)")
mha_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_PHASE1,
    batch_size=BATCH_SIZE,
    callbacks=make_callbacks(),
    verbose=1
)

# -----------------------------
# Phase 2: Fine-tune all layers
# -----------------------------
for layer in mha_model.layers:
    layer.trainable = True

mha_model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_PHASE2, clipnorm=1.0),
    loss="mse"
)

print("\n🔥 Phase 2: fine-tuning ALL layers (small LR)")
mha_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_PHASE2,
    batch_size=BATCH_SIZE,
    callbacks=make_callbacks(),
    verbose=1
)

# -----------------------------
# Test evaluation
# -----------------------------
y_pred = mha_model.predict(X_test, verbose=0).ravel()

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

print("\n📊 TEST RESULTS (Warm-start Baseline + MHA, gated)")
print(f"R² (scaled target)  : {r2:.4f}")
print(f"MSE (scaled target) : {mse_scaled:.6f}")
print(f"RMSE (scaled target): {rmse_scaled:.4f}")
print(f"MAE (scaled target) : {mae_scaled:.4f}")
print(f"MAE (bytes)         : {mae_bytes:.2f}")
print(f"RMSE (bytes)        : {rmse_bytes:.2f}")

# -----------------------------
# Attention logs (after training, on fixed probe batch)
# -----------------------------
X_probe = X_val[:256].astype(np.float32)
log_mha_attention(mha_model, X_probe, freq_ms=FREQ_MS, pool_factor=POOL_FACTOR)

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
forecast_scaled = forecast_future(mha_model, X_test[start_idx], steps=FORECAST_HORIZON, clip_range=CLIP_RANGE)

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

plt.title(f"Baseline + MHA | R²={r2:.3f} (scaled)", fontsize=14)
plt.xlabel("Time windows (250ms)")
plt.ylabel("Target (scaled)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=140)
plt.show()

print(f"\n✅ Saved plot to {PLOT_FILE}")
print(f"✅ Saved best model to {OUT_CKPT}")