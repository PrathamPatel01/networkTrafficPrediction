# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# import matplotlib.pyplot as plt

# # --- 1. DATA LOADING & SEQUENCING ---
# df = pd.read_csv('data/cleaned_data.csv')

# # Encode & Scale
# encoder = LabelEncoder()
# df['Main_Protocol'] = encoder.fit_transform(df['Main_Protocol'])
# scaler = MinMaxScaler()
# df[['Total_Length', 'Unique_Sources', 'Unique_Destinations']] = scaler.fit_transform(
#     df[['Total_Length', 'Unique_Sources', 'Unique_Destinations']]
# )

# def create_sequences(data, window=50, lead=5):
#     X, y = [], []
#     feature_data = data[['Total_Length', 'Unique_Sources', 'Unique_Destinations', 'Main_Protocol']].values
#     for i in range(len(data) - window - lead):
#         X.append(feature_data[i : i + window])
#         y.append(feature_data[i + window + lead, 0])
#     return np.array(X).astype('float32'), np.array(y).astype('float32') # float32 is better for M4

# X, y = create_sequences(df)
# split = int(len(X) * 0.8)
# X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# # --- 2. BUILD THE CNN-LSTM ---
# model = models.Sequential([
#     # CNN Layer: Slides over 50 seconds to find 'shapes'
#     layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(50, 4)),
#     layers.MaxPooling1D(pool_size=2),
    
#     # LSTM Layer: Remembers the flow of those shapes
#     layers.LSTM(100, return_sequences=False),
    
#     # Dense Layers: Interprets the memory into a prediction
#     layers.Dense(50, activation='relu'),
#     layers.Dense(1) 
# ])

# model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# # --- 3. TRAIN ---
# print("🚀 Starting Baseline Training on system...")
# history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
# # --- 4. SAVE & PLOT LOSS ---
# model.save('baseline_cnn_lstm.keras')

# plt.figure(figsize=(10, 5))
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Baseline Training Performance')
# plt.legend()
# plt.savefig('training_loss.png') # Saves automatically
# print("Training loss plot saved as training_loss.png")
# plt.close() # Closes the window so the script continues!

# # --- 5. PREDICTION & VISUALIZATION ---
# print("Generating predictions...")
# y_pred = model.predict(X_test)

# plt.figure(figsize=(15, 7))
# plt.plot(y_test[:150], label='Actual Traffic (Real Data)', color='royalblue', linewidth=2)
# plt.plot(y_pred[:150], label='Predicted Traffic (CNN-LSTM)', color='crimson', linestyle='--', linewidth=2)

# plt.title('Baseline CNN-LSTM Performance: 5-Second Lead Prediction', fontsize=14)
# plt.xlabel('Time (Seconds)', fontsize=12)
# plt.ylabel('Traffic Volume (Scaled)', fontsize=12)
# plt.legend(loc='upper right')
# plt.grid(True, alpha=0.3)

# plt.savefig('baseline_prediction_results.png') # Saves automatically
# print("Prediction plot saved as baseline_prediction_results.png")
# plt.close() # Finish script


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt

# --- 1. DATA LOADING & PREPROCESSING ---
df = pd.read_csv("data/cleaned_data.csv")
encoder = LabelEncoder()
df["Main_Protocol"] = encoder.fit_transform(df["Main_Protocol"])

scaler = MinMaxScaler()
df[["Total_Length", "Unique_Sources", "Unique_Destinations"]] = scaler.fit_transform(
    df[["Total_Length", "Unique_Sources", "Unique_Destinations"]]
)

def create_sequences(data, window=50, lead=5):
    X, y = [], []
    # Using float32 for better performance on most systems
    features = data[["Total_Length", "Unique_Sources", "Unique_Destinations", "Main_Protocol"]].values.astype("float32")
    for i in range(len(data) - window - lead):
        X.append(features[i : i + window])
        y.append(features[i + window + lead, 0]) 
    return np.array(X), np.array(y)

WINDOW, LEAD = 50, 5
X, y = create_sequences(df, WINDOW, LEAD)
split = int(len(X) * 0.8)
X_test, y_test = X[split:], y[split:]

# --- 2. BUILD MODEL ---
model = models.Sequential([
    layers.Conv1D(64, 3, padding='same', activation="relu", input_shape=(WINDOW, 4)),
    layers.MaxPooling1D(2),
    layers.LSTM(100),
    layers.Dense(50, activation="relu"),
    layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse")

# --- 3. TRAIN ---
print("🚀 Training Model...")
model.fit(X[:split], y[:split], epochs=20, batch_size=32, verbose=0)

# --- 4. PREDICTIONS ---
y_pred = model.predict(X_test, verbose=0).flatten()

# Recursive forecast function to get the "Green Line"
def get_extended_forecast(model, start_window, steps=5):
    forecast = []
    curr_window = start_window.copy()
    for _ in range(steps):
        p = model.predict(curr_window[np.newaxis, :, :], verbose=0)[0, 0]
        forecast.append(p)
        # Shift the window: remove first row, add prediction as new row
        new_row = curr_window[-1].copy()
        new_row[0] = p # Update the Total_Length feature
        curr_window = np.vstack([curr_window[1:], new_row])
    return np.array(forecast)

extended_5s = get_extended_forecast(model, X_test[-1], steps=5)

# --- 5. VISUALIZATION (LIGHT STYLE + AXIS LABELS) ---
PAST_VIEW = 100
plt.style.use('ggplot') # Light, clean professional style
plt.figure(figsize=(15, 7))

# X-axis indices
x_history = np.arange(PAST_VIEW)
x_extended = np.arange(PAST_VIEW - 1, PAST_VIEW + 5) 

# 1. Actual Traffic (Blue)
plt.plot(x_history, y_test[-PAST_VIEW:], 
         label="Actual Traffic (Real Data)", color="royalblue", linewidth=2.5)

# 2. Predicted Traffic (Crimson)
plt.plot(x_history, y_pred[-PAST_VIEW:], 
         label="Predicted Traffic (CNN-LSTM)", color="crimson", linestyle=":", linewidth=2)

# 3. Extended Prediction (Green)
# Connect the last 'Actual' point to the future forecast
green_line_y = np.concatenate([[y_test[-1]], extended_5s])
plt.plot(x_extended, green_line_y, 
         label="5s Extended Forecast", color="#2ca02c", linewidth=4, marker='s', markersize=7)

# Visual separator for "Now"
plt.axvline(PAST_VIEW - 1, color="black", linestyle="--", alpha=0.3)
plt.text(PAST_VIEW - 1.5, plt.ylim()[1]*0.8, "NOW", rotation=90, fontweight='bold', color='black')

# LABELS AND TITLES
plt.title("Network Traffic Forecast: CNN-LSTM Baseline Performance", fontsize=16, pad=15)
plt.xlabel("Time (Relative Seconds)", fontsize=12)
plt.ylabel("Traffic Volume (Min-Max Scaled)", fontsize=12)

plt.legend(loc="upper left", frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()