import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

os.makedirs('models', exist_ok=True)

# --- Load data ---
df = pd.read_csv('data/train-data.csv', index_col=0)

# --- Fungsi pembersihan ---
def extract_number(s):
    if isinstance(s, str):
        nums = re.findall(r'[\d.]+', s)
        return float(nums[0]) if nums else np.nan
    return np.nan

# Bersihkan kolom Mileage, Engine, Power
for col in ['Mileage', 'Engine', 'Power']:
    df[col] = df[col].apply(extract_number)
    # Nilai 0 pada Mileage dianggap missing
    if col == 'Mileage':
        df[col] = df[col].replace(0, np.nan)

# Ambil brand dari Name (kata pertama)
df['Brand'] = df['Name'].str.split().str[0]

# Drop kolom yang tidak dipakai
df.drop(['Name', 'New_Price'], axis=1, inplace=True)

# Pisahkan fitur dan target
X = df.drop('Price', axis=1)
y = df['Price']

# --- Split data (sebelum preprocessing) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Definisikan kolom berdasarkan tipe ---
cat_cols = ['Fuel_Type', 'Transmission', 'Owner_Type', 'Location', 'Brand']
num_cols = ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats']

# --- Preprocessing untuk numerik ---
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# --- Preprocessing untuk kategorikal ---
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# --- ColumnTransformer ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# --- Bandingkan 3 model dengan 5-fold CV ---
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'RandomForest': RandomForestRegressor(random_state=42)
}

cv_scores = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_scores[name] = (-scores.mean(), scores.std())
    print(f"{name}: MAE = {-scores.mean():.3f} ± {scores.std():.3f}")

# Pilih model terbaik (RandomForest biasanya unggul)
best_model_name = min(cv_scores, key=lambda k: cv_scores[k][0])
print(f"\nModel terbaik berdasarkan CV: {best_model_name}")

# --- Tuning untuk RandomForest (jika terpilih) ---
if best_model_name == 'RandomForest':
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5]
    }
    base_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('regressor', RandomForestRegressor(random_state=42))])
    grid_search = GridSearchCV(base_pipeline, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    final_pipeline = grid_search.best_estimator_
    print("Best params:", grid_search.best_params_)
else:
    final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', models[best_model_name])])
    final_pipeline.fit(X_train, y_train)

# --- Simpan pipeline ---
joblib.dump(final_pipeline, 'models/model_regresi.joblib')

# Simpan metadata
import json
metadata = {
    'model_type': best_model_name,
    'features': list(X.columns),
    'target': 'Price',
    'cv_mae_mean': cv_scores[best_model_name][0],
    'cv_mae_std': cv_scores[best_model_name][1]
}
with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Model dan metadata tersimpan.")