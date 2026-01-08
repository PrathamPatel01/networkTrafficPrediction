# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# # 1. Load the cleaned data you just made
# df = pd.read_csv('data/cleaned_data.csv')

# # 2. STEP 2: SCALING & ENCODING
# # Turn 'TLS', 'QUIC' into numbers 0, 1, 2...
# encoder = LabelEncoder()
# df['Protocol_ID'] = encoder.fit_transform(df['Main_Protocol'])

# # Shrink big numbers to be between 0 and 1
# scaler = MinMaxScaler()
# cols_to_scale = ['Total_Length', 'Unique_Sources', 'Unique_Destinations']
# df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# # 3. STEP 3: SEQUENCING (The "Sliding Window")
# def create_sequences(data, window_size=10):
#     X, y = [], []
#     # Features: Length, Sources, Dests, Protocol
#     feature_cols = ['Total_Length', 'Unique_Sources', 'Unique_Destinations', 'Protocol_ID']
#     data_values = data[feature_cols].values
#     target_values = data['Total_Length'].values

#     for i in range(len(data) - window_size):
#         X.append(data_values[i : i + window_size]) # Last 10 seconds
#         y.append(target_values[i + window_size])   # The next 1 second (target)
#     return np.array(X), np.array(y)

# X, y = create_sequences(df)

# print(f"Prepared {len(X)} sequences!")
# print("Now we are ready for the CNN-LSTM.")


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 1. Load the data you just saved
df = pd.read_csv('data/cleaned_data.csv')

# 2. Encode Protocol (Text to Numbers)
encoder = LabelEncoder()
df['Main_Protocol'] = encoder.fit_transform(df['Main_Protocol'])

# 3. Scale the numbers (0 to 1) 
# Very important for LSTMs to work!
scaler = MinMaxScaler()
cols = ['Total_Length', 'Unique_Sources', 'Unique_Destinations']
df[cols] = scaler.fit_transform(df[cols])

# 4. Create the 50-second window with a 5-second lead prediction
def create_sequences(data, window=50, lead=5):
    X, y = [], []
    # Features: [Length, Sources, Dests, Protocol]
    feature_data = data[['Total_Length', 'Unique_Sources', 'Unique_Destinations', 'Main_Protocol']].values
    
    for i in range(len(data) - window - lead):
        X.append(feature_data[i : i + window])       # 50 sec history
        y.append(feature_data[i + window + lead, 0]) # 5th sec in future (Length)
    return np.array(X), np.array(y)

X, y = create_sequences(df)

print(f"Input Shape (X): {X.shape}") # Should be (Samples, 50, 4)
print(f"Target Shape (y): {y.shape}") # Should be (Samples,)