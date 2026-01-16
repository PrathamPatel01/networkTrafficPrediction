import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

WINDOW = 50
LEAD = 5

def preprocess(input_csv="data/cleaned_data.csv", output_npz="data/dataset.npz"):
    print("🔄 Preprocessing...")

    df = pd.read_csv(input_csv)

    # Encode protocol
    le = LabelEncoder()
    df["Protocol"] = le.fit_transform(df["Main_Protocol"])

    features = [
        "Unique_Sources",
        "Unique_Destinations",
        "Protocol"
    ]

    target = "Total_Length_Smoothed"

    # Scale
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    df[features] = scaler_X.fit_transform(df[features])
    df[target] = scaler_y.fit_transform(df[[target]])

    # Add lag features
    max_lag = 5
    for lag in range(1, max_lag+1):
        df[f"Lag_{lag}"] = df[target].shift(lag)
        features.append(f"Lag_{lag}")

    df = df.dropna().reset_index(drop=True)

    # Create sequences
    X, y = [], []
    values = df[features + [target]].values

    for i in range(len(values) - WINDOW - LEAD):
        X.append(values[i:i+WINDOW, :-1])
        y.append(values[i+WINDOW+LEAD, -1])

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    np.savez(
        output_npz,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler_y=scaler_y
    )

    print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")


if __name__ == "__main__": preprocess()

