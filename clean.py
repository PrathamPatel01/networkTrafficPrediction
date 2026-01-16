# =========================
# 1️⃣ CLEANING DATA
# =========================
import pandas as pd

def safe_mode(x):
    """Return mode if exists, else fallback"""
    m = x.mode()
    if len(m) > 0:
        return m.iloc[0]
    return "Unknown"  # Changed from "None" for LabelEncoder compatibility

def clean_data(input_csv="data/input_data.csv", output_csv="data/cleaned_data.csv", freq="1s", smooth_window=5):
    print("🔄 Cleaning data...")

    df = pd.read_csv(input_csv)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Group by interval
    cleaned = (
        df.groupby(pd.Grouper(key="Timestamp", freq=freq))
        .agg(
            Total_Length=("Length", "sum"),
            Unique_Sources=("SourceIP", "nunique"),
            Unique_Destinations=("DestinationIP", "nunique"),
            Main_Protocol=("Protocol", safe_mode)
        )
        .reset_index()
    )

    # Smooth target
    cleaned["Total_Length_Smoothed"] = (
        cleaned["Total_Length"]
        .rolling(window=smooth_window, min_periods=1)
        .mean()
    )

    # Ensure continuous timeline
    full_range = pd.date_range(start=cleaned["Timestamp"].min(),
                               end=cleaned["Timestamp"].max(),
                               freq=freq)
    cleaned = cleaned.set_index("Timestamp").reindex(full_range).fillna(0).rename_axis("Timestamp").reset_index()

    cleaned.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(cleaned)} seconds to {output_csv}")

if __name__ == "__main__": clean_data()


