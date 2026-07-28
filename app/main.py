from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import pandas as pd
from typing import Optional
import re

app = FastAPI(title="Prediksi Harga Mobil Bekas", version="1.0")

# Muat model saat startup
model = None

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load('models/model_regresi.joblib')

# Definisikan skema input
class PredictRequest(BaseModel):
    Year: int = Field(..., ge=1990, le=2025, description="Tahun pembuatan")
    Kilometers_Driven: int = Field(..., ge=0, description="Jarak tempuh (km)")
    Fuel_Type: str = Field(..., regex="^(Petrol|Diesel|CNG|LPG|Electric)$")
    Transmission: str = Field(..., regex="^(Manual|Automatic)$")
    Owner_Type: str = Field(..., regex="^(First|Second|Third|Fourth & Above)$")
    Mileage: Optional[str] = Field(None, description="Konsumsi bahan bakar (misal '19.67 kmpl')")
    Engine: Optional[str] = Field(None, description="Kapasitas mesin (misal '1582 CC')")
    Power: Optional[str] = Field(None, description="Tenaga (misal '126.2 bhp')")
    Seats: Optional[float] = Field(None, ge=2, le=10)
    Location: Optional[str] = Field(None)
    Brand: Optional[str] = Field(None)

    @validator('Mileage', pre=True, always=True)
    def validate_mileage(cls, v):
        if v is None:
            return np.nan
        if isinstance(v, str):
            nums = re.findall(r'[\d.]+', v)
            return float(nums[0]) if nums else np.nan
        return v

    @validator('Engine', pre=True, always=True)
    def validate_engine(cls, v):
        if v is None:
            return np.nan
        if isinstance(v, str):
            nums = re.findall(r'[\d.]+', v)
            return float(nums[0]) if nums else np.nan
        return v

    @validator('Power', pre=True, always=True)
    def validate_power(cls, v):
        if v is None:
            return np.nan
        if isinstance(v, str):
            nums = re.findall(r'[\d.]+', v)
            return float(nums[0]) if nums else np.nan
        return v

class PredictResponse(BaseModel):
    predicted_price: float
    message: str = "Success"

# Endpoint root
@app.get("/")
def root():
    return {"service": "Car Price Predictor", "status": "running"}

# Endpoint health
@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

# Endpoint prediksi
@app.post("/predict-harga", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # Buat dataframe satu baris
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
        'Location': [request.Location],
        'Brand': [request.Brand]
    }
    df = pd.DataFrame(data)
    # Kolom Brand jika None, isi dengan 'Unknown'
    df['Brand'] = df['Brand'].fillna('Unknown')
    df['Location'] = df['Location'].fillna('Unknown')
    # Pastikan urutan kolom sama dengan saat training
    expected_cols = ['Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission',
                     'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats',
                     'Location', 'Brand']
    df = df[expected_cols]
    pred = model.predict(df)[0]
    return PredictResponse(predicted_price=round(pred, 2))