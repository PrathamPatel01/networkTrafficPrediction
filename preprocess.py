import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

WINDOW = 30
LEAD = 5

def preprocess():
    print("🔄 Preprocessing...")

    df = pd.read_csv("data/cleaned_data.csv")

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
    df["Lag_1"] = df[target].shift(1)
    df["Lag_5"] = df[target].shift(5)

    df = df.dropna().reset_index(drop=True)
    features += ["Lag_1", "Lag_5"]

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
        "data/dataset.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")

if __name__ == "__main__":
    preprocess()
