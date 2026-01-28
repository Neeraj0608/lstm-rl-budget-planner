import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("models/lstm_expense_model.h5",compile=False)
scaler = joblib.load("models/scaler.save")

def lstm_predict(daily_values, days=7):
    values = daily_values.reshape(-1, 1)
    scaled = scaler.transform(values)

    last_seq = scaled[-3:]
    preds = []

    for _ in range(days):
        p = model.predict(last_seq.reshape(1, 3, 1), verbose=0)
        preds.append(p[0][0])
        last_seq = np.append(last_seq[1:], p)

    return scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)
    )
