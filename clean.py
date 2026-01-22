# clean_data.py
import pandas as pd

def safe_mode(x):
    m = x.mode()
    return m.iloc[0] if len(m) > 0 else "Unknown"

def clean_data(
    input_csv="data/input_data.csv",
    output_csv="data/cleaned_data.csv",
    freq="250ms",
):
    print("🔄 Cleaning data...")

    df = pd.read_csv(input_csv)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])

    cleaned = (
        df.groupby(pd.Grouper(key="Timestamp", freq=freq))
        .agg(
            Total_Length=("Length", "sum"),
            Packet_Count=("Length", "size"),
            Mean_Length=("Length", "mean"),
            Std_Length=("Length", "std"),
            Max_Length=("Length", "max"),
            Unique_Sources=("SourceIP", "nunique"),
            Unique_Destinations=("DestinationIP", "nunique"),
            Main_Protocol=("Protocol", safe_mode),
        )
        .reset_index()
    )

    cleaned["Std_Length"] = cleaned["Std_Length"].fillna(0)

    # Ensure continuous timeline
    full_range = pd.date_range(
        start=cleaned["Timestamp"].min(),
        end=cleaned["Timestamp"].max(),
        freq=freq
    )
    cleaned = cleaned.set_index("Timestamp").reindex(full_range)
    cleaned.index.name = "Timestamp"
    cleaned = cleaned.reset_index()

    numeric_cols = [
        "Total_Length", "Packet_Count", "Mean_Length", "Std_Length", "Max_Length",
        "Unique_Sources", "Unique_Destinations"
    ]
    for c in numeric_cols:
        cleaned[c] = pd.to_numeric(cleaned[c], errors="coerce").fillna(0)

    cleaned["Main_Protocol"] = cleaned["Main_Protocol"].fillna("Unknown")

    cleaned.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(cleaned)} windows to {output_csv}")

if __name__ == "__main__":
    clean_data()



    
# # clean_data.py  (MIMIC REPO)
# import pandas as pd

# def clean_data(
#     input_csv="data/input_data.csv",
#     output_csv="data/cleaned_data.csv",
#     freq="10s"
# ):
#     print("🔄 Cleaning (repo-style)...")

#     df = pd.read_csv(input_csv)
#     df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
#     df = df.dropna(subset=["Timestamp"])

#     # Sum packet Length over fixed 10s windows
#     out = (
#         df.groupby(pd.Grouper(key="Timestamp", freq=freq))
#           .agg(Length=("Length", "sum"))
#           .reset_index()
#     )

#     # Ensure continuous timeline (fill missing windows with 0)
#     full_range = pd.date_range(out["Timestamp"].min(), out["Timestamp"].max(), freq=freq)
#     out = (out.set_index("Timestamp")
#               .reindex(full_range)
#               .fillna(0)
#               .rename_axis("Timestamp")
#               .reset_index())

#     out.to_csv(output_csv, index=False)
#     print(f"✅ Saved {len(out)} rows to {output_csv}")

# if __name__ == "__main__":
#     clean_data()
