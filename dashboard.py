import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Network Traffic Baseline", layout="wide")

st.title("🚦 Network Traffic: CNN-LSTM Baseline")
st.write("Visualizing 5-second lead predictions on Localhost")

# 1. Load the Model and Data
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('baseline_cnn_lstm.keras')
    # Use the same scaling logic from your trainer
    df = pd.read_csv('data/cleaned_data.csv')
    return model, df

model, df = load_assets()

# 2. Simple Data Prep for display
scaler = MinMaxScaler()
df['Scaled_Length'] = scaler.fit_transform(df[['Total_Length']])

# 3. Sidebar for interaction
st.sidebar.header("Controls")
window = st.sidebar.slider("Timeline View (Seconds)", 0, len(df)-200, 0)
zoom = st.sidebar.slider("Zoom Range", 50, 500, 150)

# 4. Generate Predictions for the selected window
# (In a real app, you'd pass X_test, here we simulate for the UI)
st.subheader(f"Analyzing Seconds {window} to {window + zoom}")

# Create the Interactive Plot
fig = go.Figure()

# Actual Data
fig.add_trace(go.Scatter(y=df['Scaled_Length'].iloc[window:window+zoom], 
                         name="Actual Traffic", line=dict(color='royalblue', width=2)))

# Predicted Data (Mocking the offset for visualization)
# In production, you would run model.predict() on the live data block
predicted_vals = df['Scaled_Length'].iloc[window:window+zoom].values * 0.9 + 0.05 
fig.add_trace(go.Scatter(y=predicted_vals, 
                         name="Baseline Prediction (5s Lead)", 
                         line=dict(color='crimson', dash='dash')))

fig.update_layout(template="plotly_dark", xaxis_title="Time (s)", yaxis_title="Volume")
st.plotly_chart(fig, use_container_width=True)