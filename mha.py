# # train_enhanced_attention.py
import os, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -----------------------------
# CONFIG
# -----------------------------
BASELINE_CKPT = "best_baseline.keras"
OUT_CKPT = "best_enhanced_attention.keras"
PLOT_FILE = "enhanced_attention_result.png"

# Hyperparameters
FORECAST_HORIZON = 20
EPOCHS_PHASE1 = 30
EPOCHS_PHASE2 = 100
BATCH_SIZE = 32
LR_PHASE1 = 1e-3
LR_PHASE2 = 2e-4

# Multi-head attention config
N_HEADS = 4
KEY_DIM = 16
DROPOUT_RATE = 0.1

# -----------------------------
# Reproducibility
# -----------------------------
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 90)
print("🚀 ENHANCED ATTENTION: Multi-Head + Skip Connections | MSE")
print("=" * 90)

# Load data
data = np.load("data/dataset.npz", allow_pickle=True)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]
feature_names = data["feature_names"].tolist()

WINDOW = X_train.shape[1]
N_FEATURES = X_train.shape[2]

# For inverse transform
y_scale = float(data["y_scale"][0])
y_offset = float(data["y_offset"][0])

def inverse_minmax(y_scaled):
    return (y_scaled - y_offset) / y_scale

def log1p_to_bytes(z):
    return np.expm1(z)

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"WINDOW={WINDOW}, N_FEATURES={N_FEATURES}")

# -----------------------------
# Enhanced Model with Multi-Head Attention
# -----------------------------
def build_enhanced_attention_model(window, n_features, n_heads=4, key_dim=16):
    inputs = tf.keras.Input(shape=(window, n_features), name="input")
    
    # 1. Initial convolution with batch normalization
    x = tf.keras.layers.Conv1D(128, 7, padding="same", name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Conv1D(64, 5, padding="same", name="conv2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    # 2. Bidirectional LSTM for sequence encoding
    lstm_out = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2),
        name="bilstm"
    )(x)
    
    # 3. Layer normalization before attention
    lstm_norm = tf.keras.layers.LayerNormalization(name="pre_attention_norm")(lstm_out)
    
    # 4. Multi-Head Attention
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=n_heads,
        key_dim=key_dim,
        dropout=DROPOUT_RATE,
        name="multi_head_attention"
    )(lstm_norm, lstm_norm)
    
    # 5. Skip connection (residual) with attention
    attention_residual = tf.keras.layers.Add(name="attention_skip")([lstm_norm, attention_output])
    attention_residual = tf.keras.layers.LayerNormalization(name="post_attention_norm")(attention_residual)
    
    # 6. Global context pooling (alternative to LSTM pooling)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool")(attention_residual)
    max_pool = tf.keras.layers.GlobalMaxPooling1D(name="global_max_pool")(attention_residual)
    context = tf.keras.layers.Concatenate(name="context_concat")([avg_pool, max_pool])
    
    # 7. TimeDistributed attention on LSTM outputs for fine-grained context
    lstm_for_attention = tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.1)(attention_residual)
    
    # Attention weights calculation
    attention_weights = tf.keras.layers.Dense(1, activation="tanh", name="time_attention_dense")(lstm_for_attention)
    attention_weights = tf.keras.layers.Flatten(name="attention_flatten")(attention_weights)
    attention_weights = tf.keras.layers.Activation("softmax", name="time_attention_weights")(attention_weights)
    
    # Apply attention
    attention_weights_expanded = tf.keras.layers.RepeatVector(32)(attention_weights)
    attention_weights_expanded = tf.keras.layers.Permute([2, 1])(attention_weights_expanded)
    weighted_lstm = tf.keras.layers.Multiply(name="apply_time_attention")([lstm_for_attention, attention_weights_expanded])
    temporal_context = tf.keras.layers.GlobalAveragePooling1D(name="temporal_context")(weighted_lstm)
    
    # 8. Combine all contexts
    combined = tf.keras.layers.Concatenate(name="final_concat")([context, temporal_context])
    
    # 9. Dense layers with skip connections
    dense1 = tf.keras.layers.Dense(128, activation="relu", name="dense1")(combined)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense1 = tf.keras.layers.Dropout(0.3)(dense1)
    
    dense2 = tf.keras.layers.Dense(64, activation="relu", name="dense2")(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    dense2 = tf.keras.layers.Dropout(0.25)(dense2)
    
    # Skip connection in dense layers
    dense_skip = tf.keras.layers.Dense(64, activation="linear")(combined)
    dense2 = tf.keras.layers.Add()([dense2, dense_skip])
    dense2 = tf.keras.layers.Activation("relu")(dense2)
    
    outputs = tf.keras.layers.Dense(1, name="output")(dense2)
    
    return tf.keras.Model(inputs, outputs, name="enhanced_attention_model")

# -----------------------------
# Build and Load Baseline Weights
# -----------------------------
# First load baseline to get its structure
baseline_loaded = tf.keras.models.load_model(BASELINE_CKPT, compile=False)

# Build enhanced model
model = build_enhanced_attention_model(WINDOW, N_FEATURES, N_HEADS, KEY_DIM)

# Try to copy compatible weights (focus on CNN/LSTM parts)
print("\n🔄 Attempting weight transfer from baseline...")
copied = 0
for target_layer in model.layers:
    try:
        if target_layer.name in [l.name for l in baseline_loaded.layers]:
            source_layer = [l for l in baseline_loaded.layers if l.name == target_layer.name][0]
            if len(source_layer.get_weights()) == len(target_layer.get_weights()):
                target_layer.set_weights(source_layer.get_weights())
                copied += 1
    except:
        continue

print(f"✅ Transferred {copied} layers from baseline")

# -----------------------------
# Advanced Training Strategy
# -----------------------------
# Phase 1: Train only attention and new layers
new_layers = ["multi_head_attention", "attention_skip", "post_attention_norm", 
              "global_avg_pool", "global_max_pool", "context_concat",
              "time_attention_dense", "attention_flatten", "time_attention_weights",
              "apply_time_attention", "temporal_context", "final_concat"]

for layer in model.layers:
    layer.trainable = layer.name in new_layers

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_PHASE1, clipnorm=1.0),
    loss="mse",
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")]
)

# Custom learning rate scheduler
def lr_schedule(epoch, lr):
    if epoch < 10:
        return 1e-3
    elif epoch < 25:
        return 5e-4
    else:
        return 2e-4

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        OUT_CKPT,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=6,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.LearningRateScheduler(lr_schedule),
    tf.keras.callbacks.TensorBoard(log_dir="./logs_enhanced", update_freq="batch"),
]

print("\n🧊 Phase 1: Training attention mechanisms and new layers")
history1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_PHASE1,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Phase 2: Fine-tune all layers
for layer in model.layers:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_PHASE2, clipnorm=1.0),
    loss="mse",
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")]
)

print("\n🔥 Phase 2: Fine-tuning all layers")
history2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_PHASE2,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test, verbose=0).ravel()

# Scaled metrics
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

print("\n" + "="*60)
print("📊 ENHANCED ATTENTION TEST RESULTS")
print("="*60)
print(f"R² (scaled target)  : {r2:.4f}")
print(f"MSE (scaled target) : {mse_scaled:.6f}")
print(f"RMSE (scaled target): {rmse_scaled:.4f}")
print(f"MAE (scaled target) : {mae_scaled:.4f}")
print(f"MAE (bytes)         : {mae_bytes:.2f}")
print(f"RMSE (bytes)        : {rmse_bytes:.2f}")

# -----------------------------
# Comparative Analysis
# -----------------------------
# Load baseline predictions for comparison
baseline_model = tf.keras.models.load_model(BASELINE_CKPT, compile=False)
y_pred_baseline = baseline_model.predict(X_test, verbose=0).ravel()
r2_baseline = r2_score(y_test, y_pred_baseline)

print("\n" + "="*60)
print("📈 IMPROVEMENT ANALYSIS")
print("="*60)
print(f"Baseline R²        : {r2_baseline:.4f}")
print(f"Enhanced R²        : {r2:.4f}")
print(f"Absolute Improvement: {r2 - r2_baseline:.4f}")
print(f"Relative Improvement: {((r2 - r2_baseline) / r2_baseline * 100):.1f}%")

# -----------------------------
# Attention Visualization
# -----------------------------
# Extract attention weights for visualization
attention_layer = model.get_layer("multi_head_attention")
attention_model = tf.keras.Model(
    inputs=model.input,
    outputs=[attention_layer.output, model.output]
)

sample_input = X_test[:1]
attention_output, _ = attention_model.predict(sample_input, verbose=0)

# Visualize attention weights
plt.figure(figsize=(15, 10))

# Plot 1: Performance comparison
plt.subplot(2, 2, 1)
plt.plot(y_test[:100], label="Actual", alpha=0.7)
plt.plot(y_pred_baseline[:100], '--', label=f"Baseline (R²={r2_baseline:.3f})", alpha=0.7)
plt.plot(y_pred[:100], ':', label=f"Enhanced (R²={r2:.3f})", alpha=0.7, linewidth=2)
plt.title("Predictions Comparison (First 100 samples)")
plt.xlabel("Time windows")
plt.ylabel("Scaled Target")
plt.legend()
plt.grid(alpha=0.3)

# Plot 2: Residuals comparison
plt.subplot(2, 2, 2)
residuals_baseline = y_test - y_pred_baseline
residuals_enhanced = y_test - y_pred
plt.hist(residuals_baseline, bins=50, alpha=0.5, label=f"Baseline (σ={np.std(residuals_baseline):.3f})")
plt.hist(residuals_enhanced, bins=50, alpha=0.5, label=f"Enhanced (σ={np.std(residuals_enhanced):.3f})")
plt.title("Residual Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)

# Plot 3: Scatter plot
plt.subplot(2, 2, 3)
plt.scatter(y_test, y_pred_baseline, alpha=0.3, s=10, label="Baseline")
plt.scatter(y_test, y_pred, alpha=0.3, s=10, label="Enhanced")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label="Perfect")
plt.title("Predictions vs Actual")
plt.xlabel("Actual (scaled)")
plt.ylabel("Predicted (scaled)")
plt.legend()
plt.grid(alpha=0.3)

# Plot 4: Attention weights visualization
plt.subplot(2, 2, 4)
if attention_output.ndim == 3:
    # Average across heads
    attention_weights = np.mean(np.abs(attention_output[0]), axis=-1)
    plt.imshow(attention_weights.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.title("Attention Weights (averaged across heads)")
    plt.xlabel("Time Step")
    plt.ylabel("Feature Dimension")
else:
    plt.text(0.5, 0.5, "Attention visualization\nrequires 3D output", 
             ha='center', va='center', transform=plt.gca().transAxes)

plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight')
plt.show()

# -----------------------------
# Statistical Significance Test
# -----------------------------
from scipy import stats

# Paired t-test for residuals
t_stat, p_value = stats.ttest_rel(np.abs(residuals_baseline), np.abs(residuals_enhanced))

print("\n" + "="*60)
print("🔬 STATISTICAL SIGNIFICANCE TEST")
print("="*60)
print(f"Paired t-test p-value: {p_value:.6f}")
if p_value < 0.05:
    print("✅ Statistically significant improvement (p < 0.05)")
else:
    print("⚠️  Improvement not statistically significant")

# Save final model
model.save(OUT_CKPT)
print(f"\n✅ Saved enhanced model to {OUT_CKPT}")
print(f"✅ Saved comprehensive plot to {PLOT_FILE}")

# Print summary
print("\n" + "="*60)
print("🎯 SUMMARY")
print("="*60)
print(f"1. Baseline R²: {r2_baseline:.4f}")
print(f"2. Enhanced R²: {r2:.4f}")
print(f"3. Improvement: {r2 - r2_baseline:.4f} ({(r2/r2_baseline-1)*100:.1f}%)")
print(f"4. MAE Reduction: {mean_absolute_error(y_test, y_pred_baseline) - mae_scaled:.4f}")
print(f"5. Statistical significance: {'Yes' if p_value < 0.05 else 'No'} (p={p_value:.4f})")