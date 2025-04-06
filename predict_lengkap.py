# --------------------------
# 1. IMPORT LIBRARY
# --------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# --------------------------
# 2. LOAD & PREPARE DATA
# --------------------------
# Baca file CSV
df = pd.read_csv("data_saham.csv")

# Pastikan kolom 'Date' dan 'Close' ada
assert 'Date' in df.columns, "Kolom 'Date' tidak ditemukan"
assert 'Close' in df.columns, "Kolom 'Close' tidak ditemukan"

# Konversi tanggal dan urutkan
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Ambil harga penutupan
close_prices = df['Close'].values.reshape(-1, 1)

# Normalisasi data (0-1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_prices)

# --------------------------
# 3. BUAT DATASET LSTM
# --------------------------
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i+time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data)

# Bagi data train-test (80-20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape untuk LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# --------------------------
# 4. BANGUN MODEL
# --------------------------
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(100),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# --------------------------
# 5. TRAINING MODEL
# --------------------------
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    verbose=1
)

# --------------------------
# 6. PREDIKSI
# --------------------------
# Prediksi data test
test_predict = model.predict(X_test)
test_predict = scaler.inverse_transform(test_predict)

# Prediksi 7 hari ke depan
future_days = 7
future_predictions = []
last_sequence = scaled_data[-time_step:]

for _ in range(future_days):
    next_pred = model.predict(last_sequence.reshape(1, time_step, 1))
    future_predictions.append(next_pred[0,0])
    last_sequence = np.append(last_sequence[1:], next_pred)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# --------------------------
# 7. VISUALISASI
# --------------------------
plt.figure(figsize=(15, 6))
plt.plot(df['Date'][train_size+time_step+1:], scaler.inverse_transform(y_test.reshape(-1, 1)), label='Harga Aktual')
plt.plot(df['Date'][train_size+time_step+1:], test_predict, label='Prediksi Test', alpha=0.7)

# Plot prediksi 7 hari ke depan
future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=future_days+1)[1:]
plt.plot(future_dates, future_predictions, 'ro--', label='Prediksi Masa Depan')

plt.title('Prediksi Harga Saham')
plt.xlabel('Tanggal')
plt.ylabel('Harga')
plt.legend()
plt.grid()
plt.show()

# --------------------------
# 8. SIMPAN HASIL
# --------------------------
# Simpan prediksi ke CSV
result_df = pd.DataFrame({
    'Tanggal': future_dates,
    'Prediksi': future_predictions.flatten()
})
result_df.to_csv('prediksi_7_hari.csv', index=False)
print("\nHasil prediksi disimpan di 'prediksi_7_hari.csv'")

# Simpan model
model.save("model_lstm.h5")
print("Model disimpan sebagai 'model_lstm.h5'")