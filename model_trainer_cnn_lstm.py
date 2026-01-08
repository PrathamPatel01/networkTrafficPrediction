import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# --- 1. PREPARE DATA ---
df = pd.read_csv('data/cleaned_data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Group packets into 1-second snapshots
data = df.groupby(pd.Grouper(key='Timestamp', freq='1S')).agg({
    'Length': 'sum',
    'SourceIP': 'nunique',
    'DestinationIP': 'nunique',
    'Protocol': lambda x: x.mode().iloc[0] if not x.empty else 'None'
}).reset_index()

# Encode Protocol names to numbers & Scale values to 0-1
encoder = LabelEncoder()
data['Protocol_ID'] = encoder.fit_transform(data['Protocol'])
scaler = MinMaxScaler()
data[['Length', 'SourceIP', 'DestinationIP']] = scaler.fit_transform(data[['Length', 'SourceIP', 'DestinationIP']])

# --- 2. CREATE SEQUENCES (10-second history) ---
def create_sequences(df, window=10):
    X, y = [], []
    # Using 4 features: Length, Unique Sources, Unique Dests, Protocol ID
    vals = df[['Length', 'SourceIP', 'DestinationIP', 'Protocol_ID']].values
    for i in range(len(vals) - window):
        X.append(vals[i : i + window])
        y.append(vals[i + window, 0]) # Target is just the 'Length'
    return np.array(X), np.array(y)

X, y = create_sequences(data)
split = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# --- 3. BUILD CNN-LSTM-ATTENTION MODEL ---
inputs = layers.Input(shape=(10, 4))

# CNN Layer: 32 filters, size 3
cnn = layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
cnn = layers.MaxPooling1D(2)(cnn)

# LSTM Layer: 50 units
lstm = layers.LSTM(50, return_sequences=True)(cnn)

# Attention Mechanism
att = layers.Dense(1, activation='tanh')(lstm)
att = layers.Flatten()(att)
att = layers.Activation('softmax')(att)
att = layers.RepeatVector(50)(att)
att = layers.Permute([2, 1])(att)
multiplied = layers.Multiply()([lstm, att])
repr = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(multiplied)

# Output
outputs = layers.Dense(1)(repr)
model = models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')

# --- 4. TRAIN AND PREDICT ---
print("Training AI... please wait.")
model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)

# Make a prediction and visualize
preds = model.predict(X_test)
plt.plot(y_test[:100], label='Real Traffic')
plt.plot(preds[:100], label='AI Prediction')
plt.legend()
plt.title("Is the AI correct?")
plt.show()