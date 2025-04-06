import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Load data
data = pd.read_csv("bbca_data.csv")[['Close']].values

# 2. Normalisasi (0-1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 3. Siapkan dataset
X, y = [], []
for i in range(60, len(data)):
    X.append(data_scaled[i-60:i, 0])
    y.append(data_scaled[i, 0])
X, y = np.array(X), np.array(y)

# 4. Bangun model
model = Sequential([
    LSTM(50, input_shape=(X.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 5. Latih model
model.fit(X, y, epochs=20, batch_size=32)

# 6. Simpan model
model.save("lstm_model.h5")
print("Model selesai dilatih!")