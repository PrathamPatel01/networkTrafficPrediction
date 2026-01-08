import pandas as pd

# 1. Load data
df = pd.read_csv('data/input_data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# 2. Aggregation with multiple columns
# We use 'nunique' to count distinct IPs and a custom function to find the most common protocol.
cleaned_data = df.groupby(pd.Grouper(key='Timestamp', freq='1S')).agg({
    'Length': 'sum',
    'SourceIP': 'nunique',
    'DestinationIP': 'nunique',
    'Protocol': lambda x: x.mode().iloc[0] if not x.empty else 'None'
}).reset_index()

# Rename for clarity
cleaned_data.columns = ['Timestamp', 'Total_Length', 'Unique_Sources', 'Unique_Destinations', 'Main_Protocol']

# 3. Save
cleaned_data.to_csv('data/cleaned_data.csv', index=False)
print(cleaned_data.head())