import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Muat pipeline
pipeline = joblib.load('models/model_regresi.joblib')

# Muat data test (dari split yang disimpan di train.py, tapi kita load ulang dari CSV)
# Sebaiknya kita simpan X_test, y_test setelah split, tetapi untuk demonstrasi kita split ulang
# dengan random_state yang sama agar konsisten.
df = pd.read_csv('data/train-data.csv', index_col=0)
# Lakukan pembersihan yang sama
import re
def extract_number(s):
    if isinstance(s, str):
        nums = re.findall(r'[\d.]+', s)
        return float(nums[0]) if nums else np.nan
    return np.nan
for col in ['Mileage', 'Engine', 'Power']:
    df[col] = df[col].apply(extract_number)
    if col == 'Mileage':
        df[col] = df[col].replace(0, np.nan)
df['Brand'] = df['Name'].str.split().str[0]
df.drop(['Name', 'New_Price'], axis=1, inplace=True)
X = df.drop('Price', axis=1)
y = df['Price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prediksi
y_pred = pipeline.predict(X_test)

# Metrik
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test MAE: {mae:.3f} Lakh")
print(f"Test RMSE: {rmse:.3f} Lakh")
print(f"Test R²: {r2:.3f}")

# Plot prediksi vs aktual
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Harga Aktual (Lakh)')
plt.ylabel('Harga Prediksi (Lakh)')
plt.title('Prediksi vs Aktual')
plt.savefig('reports/evaluation_scatter.png', dpi=150)
plt.close()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Prediksi')
plt.ylabel('Residual')
plt.title('Residual Plot')
plt.savefig('reports/evaluation_residuals.png', dpi=150)
plt.close()

# Simpan metrik ke file
with open('reports/evaluation_metrics.txt', 'w') as f:
    f.write(f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}")