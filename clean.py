
import pandas as pd

def clean_data():
    """Clean and aggregate data"""
    print("🔄 Cleaning data...")
    
    df = pd.read_csv('data/input_data.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # 1-second aggregation
    cleaned = df.groupby(pd.Grouper(key='Timestamp', freq='1S')).agg({
        'Length': 'sum',
        'SourceIP': 'nunique',
        'DestinationIP': 'nunique',
        'Protocol': lambda x: x.mode().iloc[0] if not x.empty else 'None'
    }).reset_index()
    
    cleaned.columns = ['Timestamp', 'Total_Length', 'Unique_Sources', 
                      'Unique_Destinations', 'Main_Protocol']
    
    # Save
    cleaned.to_csv('data/cleaned_data.csv', index=False)
    print(f"✅ Cleaned data saved: {len(cleaned)} seconds")
    return cleaned

if __name__ == "__main__":
    clean_data()




    