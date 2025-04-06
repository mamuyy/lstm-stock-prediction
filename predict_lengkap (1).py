
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('data_saham.csv')
data_close = data['Close'].values.reshape(-1, 1)

# Normalisasi
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_close)

# Load model (.keras format)
model = load_model('model_lstm.keras')

# Ambil 60 data terakhir untuk prediksi
last_60 = scaled_data[-60:].reshape(1, 60, 1)

# Prediksi 7 hari ke depan
predictions = []
for _ in range(7):
    pred = model.predict(last_60, verbose=0)
    predictions.append(float(scaler.inverse_transform(pred)[0][0]))
    last_60 = np.append(last_60[:, 1:, :], [[[pred[0][0]]]], axis=1)

# Simulasi tanggal prediksi
start_date = datetime.today() + timedelta(days=1)
dates = [start_date + timedelta(days=i) for i in range(len(predictions))]

# Simpan ke CSV
df = pd.DataFrame({
    'Tanggal': [d.strftime('%Y-%m-%d') for d in dates],
    'Prediksi Harga': predictions
})
df.to_csv('prediksi_7hari.csv', index=False)

print("✅ Hasil prediksi disimpan sebagai prediksi_7hari.csv")
