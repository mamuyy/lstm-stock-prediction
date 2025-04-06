import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os

print("Membuat folder data...")
os.makedirs('data', exist_ok=True)

print("Mengunduh data BBCA.JK dari Yahoo Finance...")
try:
    # Download data 5 tahun terakhir
    data = yf.download(
        "BBCA.JK",
        start="2020-01-01",
        end="2024-12-31",
        progress=False
    )
    
    # Simpan ke CSV
    data.to_csv('data/bbca_data.csv')
    print("\nData berhasil disimpan di: data/bbca_data.csv")
    
    # Tampilkan preview
    print("\n5 data terakhir:")
    print(data.tail())
    
    # Visualisasi
    data['Close'].plot(figsize=(10, 6), title='Harga Saham BBCA.JK')
    plt.savefig('data/harga_bbcajk.png')
    print("Grafik disimpan di: data/harga_bbcajk.png")
    
except Exception as e:
    print(f"Error: {e}")