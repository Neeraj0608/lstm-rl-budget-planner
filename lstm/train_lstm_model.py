import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

df = pd.read_csv("data/expenses.csv")
df["Date"] = pd.to_datetime(df["Date"])

daily = df.groupby("Date")["Amount"].sum().values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(daily)

def create_sequences(data, seq_len=3):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled)

model = Sequential([
    LSTM(50, activation="relu", input_shape=(3, 1)),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=30, verbose=1)

model.save("models/lstm_expense_model.h5")
joblib.dump(scaler, "models/scaler.save")

print("✅ LSTM trained and saved")
