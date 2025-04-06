# --------------------------
# 1. IMPORT LIBRARY
# --------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf  # Untuk download data saham otomatis
import os

# --------------------------
# 2. DOWNLOAD/LOAD DATA
# --------------------------
def download_data(stock_code, start_date, end_date):
    """
    Download data saham otomatis dari Yahoo Finance
    Contoh: download_data('BBCA.JK', '2020-01-01', '2023-12-31')
    """
    print(f"Mendownload data {stock_code}...")
    data = yf.download(stock_code, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data.to_csv('data_saham.csv', index=False)
    return data

# Jika file CSV belum ada, download data
if not os.path.exists('data_saham.csv'):
    df = download_data('BBCA.JK', '2020-01-01', '2023-12-31')
else:
    df = pd.read_csv('data_saham.csv')
    df['Date'] = pd.to_datetime(df['Date'])

print("\n5 Data Teratas:")
print(df.head())

# --------------------------
# 3. PREPROCESSING DATA
# --------------------------
# Pilih kolom 'Close' dan normalisasi
close_prices = df['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# --------------------------
# 4. SPLIT DATA (80% Train, 20% Test)
# --------------------------
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# --------------------------
# 5. BUAT DATASET LSTM
# --------------------------
def create_dataset(data, time_step=60):
    """
    Membuat dataset untuk LSTM
    time_step: Jumlah hari yang digunakan untuk prediksi 1 hari berikutnya
    """
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape data untuk LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# --------------------------
# 6. BANGUN MODEL LSTM
# --------------------------
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),  # Mencegah overfitting
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Callback untuk stop training jika tidak ada improvement
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# --------------------------
# 7. TRAINING MODEL
# --------------------------
print("\nMemulai training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# --------------------------
# 8. EVALUASI MODEL
# --------------------------
# Prediksi untuk data test
test_predict = model.predict(X_test)
test_predict = scaler.inverse_transform(test_predict)

# Hitung RMSE (Root Mean Squared Error)
rmse = np.sqrt(np.mean(((test_predict - scaler.inverse_transform(y_test.reshape(-1,1))) ** 2)))
print(f"\nRMSE: {rmse:.2f}")

# --------------------------
# 9. PREDIKSI HARI DEPAN
# --------------------------
def predict_future_days(model, last_sequence, days=7):
    """Prediksi beberapa hari ke depan"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        next_pred = model.predict(current_sequence.reshape(1, time_step, 1))
        predictions.append(next_pred[0,0])
        current_sequence = np.append(current_sequence[1:], next_pred)
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Ambil 60 hari terakhir untuk prediksi
last_60_days = scaled_data[-time_step:]
future_predictions = predict_future_days(model, last_60_days, days=7)

print("\nPrediksi 7 Hari ke Depan:")
for i, price in enumerate(future_predictions, 1):
    print(f"Hari ke-{i}: Rp {price[0]:.2f}")

# --------------------------
# 10. VISUALISASI
# --------------------------
plt.figure(figsize=(16,8))
plt.title('Perbandingan Harga Aktual vs Prediksi')
plt.plot(df['Date'][train_size+time_step+1:train_size+time_step+1+len(y_test)], 
         scaler.inverse_transform(y_test.reshape(-1,1)), 
         label='Harga Aktual')
plt.plot(df['Date'][train_size+time_step+1:train_size+time_step+1+len(y_test)], 
         test_predict, 
         label='Prediksi', 
         alpha=0.7)
plt.legend()
plt.grid()
plt.show()

# --------------------------
# 11. SIMPAN MODEL & HASIL
# --------------------------
model.save('model_lstm_final.h5')
print("\nModel disimpan sebagai 'model_lstm_final.h5'")

# Simpan prediksi ke CSV
future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=8)[1:]  # 7 hari setelah tanggal terakhir
pd.DataFrame({
    'Tanggal': future_dates,
    'Prediksi': future_predictions.flatten()
}).to_csv('prediksi_7hari.csv', index=False)
print("Hasil prediksi disimpan di 'prediksi_7hari.csv'")