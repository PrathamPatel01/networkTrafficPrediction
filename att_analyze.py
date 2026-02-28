

import argparse
import numpy as np
import tensorflow as tf


# ---- Custom layer class must match what you used in training ----
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


def build_attention_probe(model: tf.keras.Model) -> tf.keras.Model:
    """
    Keras-3-safe: compute attention weights using Keras layers on symbolic tensors.
    Outputs: attn_weights (B,T), gate (B,1)
    """
    seq_norm = model.get_layer("attn_ln").output
    attn = model.get_layer("additive_attn")

    # score = v(tanh(W(x))) : (B,T,1)
    score = attn.v(attn.W(seq_norm))

    # softmax over time using Keras layer (safe for KerasTensor)
    w = tf.keras.layers.Softmax(axis=1)(score)  # (B,T,1)
    w = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=-1))(w)  # (B,T)

    gate = model.get_layer("gate").output  # (B,1)

    return tf.keras.Model(model.input, [w, gate], name="attn_probe")


def attn_quick_log(model, X_probe, freq_ms=250.0, pool_factor=2):
    probe = build_attention_probe(model)
    W, G = probe(X_probe, training=False)
    W = W.numpy().astype(np.float64)
    G = G.numpy().ravel().astype(np.float64)

    B, T = W.shape
    step_s = (freq_ms / 1000.0) * pool_factor
    span_s = (T - 1) * step_s

    mean_w = W.mean(axis=0)

    def share_last(seconds):
        k = int(np.ceil(seconds / step_s))
        k = max(1, min(k, T))
        return float(mean_w[-k:].sum())

    share_5  = share_last(5.0)
    share_10 = share_last(10.0)
    share_15 = share_last(15.0)

    idx = np.arange(T, dtype=np.float64)
    com = (W * idx).sum(axis=1) / (W.sum(axis=1) + 1e-12)     # 0=oldest
    lookback_s = span_s - (com * step_s)                      # 0=most recent

    top_steps = np.argsort(mean_w)[-5:][::-1].tolist()
    top_steps_sec_before = [float(span_s - s * step_s) for s in top_steps]

    print("\n[ATTN QUICK LOG]")
    print(f"T={T} steps | step≈{step_s:.2f}s | span≈{span_s:.1f}s")
    print(f"Attention mass last  5s: {share_5:.3f}")
    print(f"Attention mass last 10s: {share_10:.3f}")
    print(f"Attention mass last 15s: {share_15:.3f}")
    print(f"Lookback median(s): {float(np.median(lookback_s)):.2f} | mean(s): {float(np.mean(lookback_s)):.2f}")
    print(f"Top-5 mean-attn steps: {top_steps}")
    print(f"Top-5 sec-before-pred: {[round(x,2) for x in top_steps_sec_before]}")
    print(f"Gate mean={float(G.mean()):.3f} (p90={float(np.quantile(G,0.9)):.3f})\n")


def build_feature_indices(feature_names):
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    target_idx = name_to_idx.get("Target", None)

    lag_idxs = []
    k = 1
    while f"Lag_{k}" in name_to_idx:
        lag_idxs.append(name_to_idx[f"Lag_{k}"])
        k += 1

    return target_idx, lag_idxs


def forecast_future(model, last_window, target_idx, lag_idxs, steps=20, clip_range=(0.0, 1.0)):
    """
    Same logic as your trainer: shift window, carry exogenous features,
    update Target and Lag_1..Lag_K.
    """
    current = last_window.copy()
    out = []

    lag_hist = []
    if lag_idxs:
        for idx in lag_idxs:
            lag_hist.append(float(current[-1, idx]))

    for _ in range(steps):
        pred = float(model.predict(current[np.newaxis], verbose=0)[0, 0])

        if clip_range is not None:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best_baseline_additive_warmstart.keras")
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--freq_ms", type=float, default=250.0)
    ap.add_argument("--pool_factor", type=int, default=2)
    ap.add_argument("--probe_n", type=int, default=256)
    ap.add_argument("--forecast_horizon", type=int, default=20)
    ap.add_argument("--clip_min", type=float, default=0.0)
    ap.add_argument("--clip_max", type=float, default=1.0)
    ap.add_argument("--plot_len", type=int, default=300)
    args = ap.parse_args()

    data = np.load(args.data, allow_pickle=True)
    X_val = data["X_val"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32).ravel()
    feature_names = data["feature_names"].tolist()

    model = tf.keras.models.load_model(
        args.model,
        custom_objects={"TemporalAdditiveAttention": TemporalAdditiveAttention},
        compile=False
    )
    print(f"✅ Loaded model: {args.model}")
    print(f"X_test={X_test.shape} y_test={y_test.shape}")

    # ---- Attention logs ----
    X_probe = X_val[:min(args.probe_n, len(X_val))].astype(np.float32)
    attn_quick_log(model, X_probe, freq_ms=args.freq_ms, pool_factor=args.pool_factor)

    # ---- Forecast sanity check ----
    target_idx, lag_idxs = build_feature_indices(feature_names)
    print(f"Target idx: {target_idx}")
    print(f"Lag idxs  : {lag_idxs} (count={len(lag_idxs)})")

    PLOT_LEN = min(args.plot_len, len(y_test))
    start_idx = PLOT_LEN - 1

    forecast_scaled = forecast_future(
        model,
        X_test[start_idx],
        target_idx=target_idx,
        lag_idxs=lag_idxs,
        steps=args.forecast_horizon,
        clip_range=(args.clip_min, args.clip_max),
    )

    print("\n🔎 Forecast sanity check")
    print("forecast head:", forecast_scaled[:5])
    print("forecast std :", float(np.std(forecast_scaled)))


if __name__ == "__main__":
    main()