
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def preprocess_data():
    """Preprocess data"""
    print("🔄 Preprocessing data...")
    
    df = pd.read_csv('data/cleaned_data.csv')
    
    # Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Encode
    encoder = LabelEncoder()
    train_df['Main_Protocol'] = encoder.fit_transform(train_df['Main_Protocol'])
    test_df['Main_Protocol'] = encoder.transform(test_df['Main_Protocol'])
    
    # Scale
    scaler = MinMaxScaler()
    cols = ['Total_Length', 'Unique_Sources', 'Unique_Destinations']
    
    train_df[cols] = scaler.fit_transform(train_df[cols])
    test_df[cols] = scaler.transform(test_df[cols])
    
    # Save
    train_df.to_csv('data/train_data.csv', index=False)
    test_df.to_csv('data/test_data.csv', index=False)
    
    print(f"✅ Saved: {len(train_df)} train, {len(test_df)} test")
    return train_df, test_df

if __name__ == "__main__":
    preprocess_data()