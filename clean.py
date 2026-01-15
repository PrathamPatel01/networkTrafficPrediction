import pandas as pd

def safe_mode(x):
    """Return mode if exists, else fallback"""
    m = x.mode()
    if len(m) > 0:
        return m.iloc[0]
    return "None"

def clean_data():
    print("🔄 Cleaning data...")

    df = pd.read_csv("data/input_data.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    cleaned = (
        df
        .groupby(pd.Grouper(key="Timestamp", freq="1s"))
        .agg(
            Total_Length=("Length", "sum"),
            Unique_Sources=("SourceIP", "nunique"),
            Unique_Destinations=("DestinationIP", "nunique"),
            Main_Protocol=("Protocol", safe_mode),
        )
        .reset_index()
    )

    # Smooth target
    cleaned["Total_Length_Smoothed"] = (
        cleaned["Total_Length"]
        .rolling(window=5, min_periods=1)
        .mean()
    )

    cleaned.to_csv("data/cleaned_data.csv", index=False)
    print(f"✅ Saved {len(cleaned)} seconds")

if __name__ == "__main__":
    clean_data()
