import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from load_data import load_train_data

# Buat folder reports jika belum ada
os.makedirs('reports', exist_ok=True)

df = load_train_data()

# 1. Distribusi target (histogram)
plt.figure(figsize=(8,5))
sns.histplot(df['Price'], bins=50, kde=True)
plt.title('Distribusi Harga Mobil Bekas (dalam Lakh)')
plt.xlabel('Harga (Lakh)')
plt.ylabel('Frekuensi')
plt.savefig('reports/eda_target_distribution.png', dpi=150)
plt.close()

# 2. Missing values per kolom
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    plt.figure(figsize=(8,4))
    missing.plot(kind='bar')
    plt.title('Jumlah Nilai Hilang per Kolom')
    plt.ylabel('Jumlah')
    plt.savefig('reports/eda_missing_values.png', dpi=150)
    plt.close()
else:
    # buat placeholder kalau tidak ada missing
    plt.figure()
    plt.text(0.5,0.5,'Tidak ada missing values', ha='center', va='center')
    plt.savefig('reports/eda_missing_values.png')
    plt.close()

# 3. Hubungan fitur numerik dengan target (scatter plot: Year vs Price)
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='Year', y='Price', alpha=0.3)
plt.title('Tahun vs Harga')
plt.savefig('reports/eda_year_vs_price.png', dpi=150)
plt.close()

# 4. Boxplot harga per jenis bahan bakar
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Fuel_Type', y='Price')
plt.title('Harga berdasarkan Jenis Bahan Bakar')
plt.xticks(rotation=45)
plt.savefig('reports/eda_fuel_vs_price.png', dpi=150)
plt.close()

# 5. Heatmap korelasi antar fitur numerik (tambahan)
numeric_cols = ['Year', 'Kilometers_Driven', 'Price']
# tambahkan Mileage, Engine, Power setelah dibersihkan nanti, tapi sekarang ambil yang ada
# kita ekstrak dulu angka dari Mileage, Engine, Power secara sederhana untuk korelasi
def extract_number(s):
    if isinstance(s, str):
        import re
        nums = re.findall(r'[\d.]+', s)
        return float(nums[0]) if nums else np.nan
    return np.nan

df_temp = df.copy()
for col in ['Mileage', 'Engine', 'Power']:
    df_temp[col] = df_temp[col].apply(extract_number)

numeric_cols = ['Year', 'Kilometers_Driven', 'Price', 'Mileage', 'Engine', 'Power']
corr = df_temp[numeric_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap Korelasi Fitur Numerik')
plt.savefig('reports/eda_correlation_heatmap.png', dpi=150)
plt.close()

print("Grafik EDA tersimpan di folder reports/")