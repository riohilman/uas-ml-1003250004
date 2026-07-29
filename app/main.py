import os
import re
import json
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
from typing import Optional
import joblib

# ------------------------------------------------------------------
# 1. Tentukan path model (absolut agar selalu ditemukan)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_regresi.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.json")
LOG_PATH = os.path.join(BASE_DIR, "reports", "prediction_log.jsonl")

# Global variable untuk menyimpan model
model = None
model_cv_mae = None  # dipakai sebagai fallback margin ketidakpastian


def log_prediction(request_data: dict, predicted_price: float):
    """Mencatat setiap prediksi (fitur input + hasil + timestamp) untuk pemantauan drift."""
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        entry = {
            "timestamp": time.time(),
            "input": request_data,
            "predicted_price": predicted_price,
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        # Logging tidak boleh menggagalkan request prediksi
        print(f"⚠️ Gagal mencatat log prediksi: {e}")

# ------------------------------------------------------------------
# 2. Lifespan event untuk memuat model saat startup
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_cv_mae
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded from {MODEL_PATH}")
        else:
            print(f"❌ Model file not found: {MODEL_PATH}")
            model = None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

    try:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH) as f:
                meta = json.load(f)
            model_cv_mae = meta.get("cv_mae_mean")
    except Exception as e:
        print(f"⚠️ Gagal memuat metadata: {e}")
        model_cv_mae = None

    yield
    # Cleanup jika diperlukan

app = FastAPI(
    title="Prediksi Harga Mobil Bekas",
    version="1.0",
    lifespan=lifespan
)

# ------------------------------------------------------------------
# 3. Skema Request & Response dengan validasi
# ------------------------------------------------------------------
class PredictRequest(BaseModel):
    Year: int = Field(..., ge=1990, le=2025, description="Tahun pembuatan")
    Kilometers_Driven: int = Field(..., ge=0, description="Jarak tempuh (km)")
    Fuel_Type: str = Field(..., pattern="^(Petrol|Diesel|CNG|LPG|Electric)$")
    Transmission: str = Field(..., pattern="^(Manual|Automatic)$")
    Owner_Type: str = Field(..., pattern="^(First|Second|Third|Fourth & Above)$")
    Mileage: Optional[float] = Field(None, description="Konsumsi bahan bakar (dalam angka, misal 19.67)")
    Engine: Optional[float] = Field(None, description="Kapasitas mesin (dalam CC, misal 1582)")
    Power: Optional[float] = Field(None, description="Tenaga (dalam bhp, misal 126.2)")
    Seats: Optional[float] = Field(None, ge=2, le=10)
    Location: Optional[str] = Field(None, description="Kota lokasi penjualan")
    Brand: Optional[str] = Field(None, description="Merek kendaraan")

    @field_validator('Mileage', mode='before')
    @classmethod
    def parse_mileage(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            nums = re.findall(r'[\d.]+', v)
            return float(nums[0]) if nums else None
        return v

    @field_validator('Engine', mode='before')
    @classmethod
    def parse_engine(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            nums = re.findall(r'[\d.]+', v)
            return float(nums[0]) if nums else None
        return v

    @field_validator('Power', mode='before')
    @classmethod
    def parse_power(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            nums = re.findall(r'[\d.]+', v)
            return float(nums[0]) if nums else None
        return v

class PredictResponse(BaseModel):
    predicted_price: float
    confidence_interval: list[float] = Field(
        description="Estimasi rentang harga (Lakh), diturunkan dari sebaran prediksi antar pohon Random Forest atau MAE cross-validation"
    )
    message: str = "Success"

# ------------------------------------------------------------------
# 4. Endpoints
# ------------------------------------------------------------------
@app.get("/")
def root():
    return {"service": "Car Price Predictor", "status": "running"}

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict-harga", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Siapkan data dalam DataFrame
    data = {
        'Year': [request.Year],
        'Kilometers_Driven': [request.Kilometers_Driven],
        'Fuel_Type': [request.Fuel_Type],
        'Transmission': [request.Transmission],
        'Owner_Type': [request.Owner_Type],
        'Mileage': [request.Mileage],
        'Engine': [request.Engine],
        'Power': [request.Power],
        'Seats': [request.Seats],
        'Location': [request.Location or "Unknown"],
        'Brand': [request.Brand or "Unknown"]
    }
    df = pd.DataFrame(data)

    # Pastikan urutan kolom sesuai dengan training
    expected_cols = ['Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission',
                     'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats',
                     'Location', 'Brand']
    df = df[expected_cols]

    pred = model.predict(df)[0]

    # Estimasi ketidakpastian: kalau modelnya Random Forest, pakai sebaran
    # prediksi antar pohon (lebih akurat); kalau tidak, fallback ke ±MAE CV.
    regressor = model.named_steps.get("regressor")
    if hasattr(regressor, "estimators_"):
        preprocessor = model.named_steps["preprocessor"]
        Xt = preprocessor.transform(df)
        tree_preds = np.array([t.predict(Xt) for t in regressor.estimators_])
        margin = float(tree_preds.std())
    else:
        margin = float(model_cv_mae) if model_cv_mae is not None else 0.0

    lower = round(max(0.0, pred - margin), 2)
    upper = round(pred + margin, 2)
    predicted_price = round(float(pred), 2)

    log_prediction(request.model_dump(), predicted_price)

    return PredictResponse(
        predicted_price=predicted_price,
        confidence_interval=[lower, upper],
    )