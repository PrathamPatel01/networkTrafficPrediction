# preprocess.py  (COMPLETE, FIXED)
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

WINDOW = 120
LEAD = 1
MAX_LAG = 5

def one_hot_protocol(df, col="Main_Protocol", top_k=8):
    top = df[col].value_counts().head(top_k).index
    df[col] = df[col].where(df[col].isin(top), other="Other")
    oh = pd.get_dummies(df[col], prefix="Proto", dtype=np.int8)  # numeric dummies
    return df.drop(columns=[col]).join(oh)

def add_lags(d, col="Target", max_lag=5):
    d = d.copy()
    for lag in range(1, max_lag + 1):
        d[f"Lag_{lag}"] = d[col].shift(lag)
    return d.dropna().reset_index(drop=True)

def make_sequences(df_part, feature_cols, target_col, window, lead):
    X_vals = df_part.loc[:, feature_cols].to_numpy(dtype=np.float32)
    y_vals = df_part.loc[:, target_col].to_numpy(dtype=np.float32)

    X, y = [], []
    n = len(df_part)
    for i in range(n - window - lead):
        X.append(X_vals[i:i+window])
        y.append(y_vals[i+window+lead])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

def preprocess(
    input_csv="data/cleaned_data.csv",
    output_npz="data/dataset.npz",
    window=WINDOW,
    lead=LEAD
):
    print("🔄 Preprocessing...")

    df = pd.read_csv(input_csv)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    # Target in log space
    df["Target_raw"] = np.log1p(pd.to_numeric(df["Total_Length"], errors="coerce").fillna(0).astype(float))

    # One-hot protocol on full df (consistent columns across splits)
    df = one_hot_protocol(df, "Main_Protocol", top_k=8)
    proto_cols = [c for c in df.columns if c.startswith("Proto_")]

    base_num = [
        "Total_Length",
        "Packet_Count",
        "Mean_Length",
        "Std_Length",
        "Max_Length",
        "Unique_Sources",
        "Unique_Destinations",
    ]

    # Time split (no leakage)
    split_train = int(len(df) * 0.7)
    split_val = int(len(df) * 0.8)

    train_df = df.iloc[:split_train].copy()
    val_df   = df.iloc[split_train:split_val].copy()
    test_df  = df.iloc[split_val:].copy()

    # ✅ Ensure base numeric are float32 before assigning scaled floats (avoids pandas warnings)
    for c in base_num:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce").fillna(0).astype(np.float32)
        val_df[c]   = pd.to_numeric(val_df[c], errors="coerce").fillna(0).astype(np.float32)
        test_df[c]  = pd.to_numeric(test_df[c], errors="coerce").fillna(0).astype(np.float32)

    # Scale base numeric on TRAIN only
    scaler_X = MinMaxScaler()
    train_scaled = scaler_X.fit_transform(train_df.loc[:, base_num].to_numpy())
    val_scaled   = scaler_X.transform(val_df.loc[:, base_num].to_numpy())
    test_scaled  = scaler_X.transform(test_df.loc[:, base_num].to_numpy())

    train_df.loc[:, base_num] = pd.DataFrame(train_scaled, columns=base_num, index=train_df.index).astype(np.float32)
    val_df.loc[:, base_num]   = pd.DataFrame(val_scaled, columns=base_num, index=val_df.index).astype(np.float32)
    test_df.loc[:, base_num]  = pd.DataFrame(test_scaled, columns=base_num, index=test_df.index).astype(np.float32)

    # Scale Target on TRAIN only
    scaler_y = MinMaxScaler()
    train_df["Target"] = scaler_y.fit_transform(train_df[["Target_raw"]]).ravel().astype(np.float32)
    val_df["Target"]   = scaler_y.transform(val_df[["Target_raw"]]).ravel().astype(np.float32)
    test_df["Target"]  = scaler_y.transform(test_df[["Target_raw"]]).ravel().astype(np.float32)

    # Add lags inside each split (no boundary mixing)
    train_df = add_lags(train_df, col="Target", max_lag=MAX_LAG)
    val_df   = add_lags(val_df, col="Target", max_lag=MAX_LAG)
    test_df  = add_lags(test_df, col="Target", max_lag=MAX_LAG)

    lag_cols = [f"Lag_{k}" for k in range(1, MAX_LAG + 1)]

    # Features: base numeric + protocol + Target + lags
    feature_cols = base_num + proto_cols + ["Target"] + lag_cols
    feature_cols = list(dict.fromkeys(feature_cols))  # remove duplicates safely

    # Final numeric enforcement
    for c in feature_cols:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce").fillna(0).astype(np.float32)
        val_df[c]   = pd.to_numeric(val_df[c], errors="coerce").fillna(0).astype(np.float32)
        test_df[c]  = pd.to_numeric(test_df[c], errors="coerce").fillna(0).astype(np.float32)

    # Sequences
    X_train, y_train = make_sequences(train_df, feature_cols, "Target", window, lead)
    X_val, y_val     = make_sequences(val_df, feature_cols, "Target", window, lead)
    X_test, y_test   = make_sequences(test_df, feature_cols, "Target", window, lead)

    np.savez(
        output_npz,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=np.array(feature_cols, dtype=object),
        y_scale=scaler_y.scale_,
        y_offset=scaler_y.min_,
    )

    print(f"✅ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"   Features: {len(feature_cols)} | WINDOW={window} | LEAD={lead}")

if __name__ == "__main__":
    preprocess()


# preprocess.py  (MIMIC REPO)
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# import joblib

# SEQ_LEN = 5

# def create_sequences(df, seq_len):
#     xs, ys, ts = [], [], []
#     for i in range(len(df) - seq_len):
#         x = df["Length"].iloc[i:i+seq_len].astype(float).values
#         y = float(df["Length"].iloc[i+seq_len])
#         xs.append(x)
#         ys.append(y)
#         ts.append(df["Timestamp"].iloc[i+seq_len])
#     X = np.array(xs, dtype=np.float32)
#     y = np.array(ys, dtype=np.float32)
#     ts = np.array(ts, dtype="datetime64[ns]")
#     return X, y, ts

# def preprocess(
#     input_csv="data/cleaned_data.csv",
#     output_npz="data/dataset.npz",
#     scaler_file="data/scaler.gz",
#     seq_len=SEQ_LEN
# ):
#     print("🔄 Preprocessing (repo-style)...")

#     df = pd.read_csv(input_csv)
#     df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
#     df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

#     # Mimic repo: fit scaler on full data (optimistic)
#     scaler = MinMaxScaler()
#     df["Length"] = scaler.fit_transform(df[["Length"]]).astype(np.float32)

#     X, y, ts = create_sequences(df, seq_len)
#     X = X.reshape((X.shape[0], X.shape[1], 1)).astype(np.float32)

#     np.savez(
#         output_npz,
#         X=X,
#         y=y,
#         ts=ts
#     )
#     joblib.dump(scaler, scaler_file)

#     print(f"✅ Saved dataset: X={X.shape}, y={y.shape} to {output_npz}")
#     print(f"✅ Saved scaler to {scaler_file}")

# if __name__ == "__main__":
#     preprocess()
