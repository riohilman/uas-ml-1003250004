import pytest
import os
import joblib
from fastapi.testclient import TestClient
from app.main import app, MODEL_PATH

# ------------------------------------------------------------------
# Fixture untuk memuat model sebelum semua test (otomatis)
# ------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def load_model():
    """Memastikan model termuat sebelum test dijalankan."""
    if not os.path.exists(MODEL_PATH):
        pytest.fail(f"Model tidak ditemukan di {MODEL_PATH}. Jalankan 'python src/train.py' terlebih dahulu.")
    # Muat model dan set ke global variable di module main
    import app.main
    app.main.model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded for testing from {MODEL_PATH}")
    yield

# ------------------------------------------------------------------
# 1. Test mekanis (4 test)
# ------------------------------------------------------------------
client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Car Price Predictor"

def test_predict_valid():
    payload = {
        "Year": 2015,
        "Kilometers_Driven": 41000,
        "Fuel_Type": "Diesel",
        "Transmission": "Manual",
        "Owner_Type": "First",
        "Mileage": "19.67",
        "Engine": "1582",
        "Power": "126.2",
        "Seats": 5.0,
        "Location": "Pune",
        "Brand": "Hyundai"
    }
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)

def test_predict_invalid_value():
    # Kirim Year di bawah 1990 (melanggar validasi) -> 422
    payload = {
        "Year": 1800,
        "Kilometers_Driven": 41000,
        "Fuel_Type": "Diesel",
        "Transmission": "Manual",
        "Owner_Type": "First"
    }
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 422

def test_predict_missing_field():
    # Field wajib Fuel_Type tidak disertakan -> 422
    payload = {
        "Year": 2015,
        "Kilometers_Driven": 41000,
        "Transmission": "Manual",
        "Owner_Type": "First"
    }
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 422

def test_predict_invalid_enum():
    payload = {
        "Year": 2015,
        "Kilometers_Driven": 41000,
        "Fuel_Type": "Batubara",  # tidak valid
        "Transmission": "Manual",
        "Owner_Type": "First",
        "Mileage": "19.67",
        "Engine": "1582",
        "Power": "126.2",
        "Seats": 5.0
    }
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 422

# ------------------------------------------------------------------
# 2. Behavioral test
# ------------------------------------------------------------------
def test_older_car_cheaper():
    base = {
        "Year": 2015,
        "Kilometers_Driven": 50000,
        "Fuel_Type": "Petrol",
        "Transmission": "Manual",
        "Owner_Type": "First",
        "Mileage": "18.2",
        "Engine": "1199",
        "Power": "88.7",
        "Seats": 5.0,
        "Location": "Mumbai",
        "Brand": "Honda"
    }
    newer = base.copy()
    newer["Year"] = 2018
    older = base.copy()
    older["Year"] = 2010

    resp_new = client.post("/predict-harga", json=newer)
    resp_old = client.post("/predict-harga", json=older)
    assert resp_new.status_code == 200
    assert resp_old.status_code == 200
    assert resp_new.json()["predicted_price"] > resp_old.json()["predicted_price"]

def test_higher_mileage_cheaper():
    base = {
        "Year": 2015,
        "Kilometers_Driven": 50000,
        "Fuel_Type": "Petrol",
        "Transmission": "Manual",
        "Owner_Type": "First",
        "Mileage": "18.2",
        "Engine": "1199",
        "Power": "88.7",
        "Seats": 5.0,
        "Location": "Mumbai",
        "Brand": "Honda"
    }
    low_mile = base.copy()
    low_mile["Kilometers_Driven"] = 20000
    high_mile = base.copy()
    high_mile["Kilometers_Driven"] = 100000

    resp_low = client.post("/predict-harga", json=low_mile)
    resp_high = client.post("/predict-harga", json=high_mile)
    assert resp_low.status_code == 200
    assert resp_high.status_code == 200
    assert resp_low.json()["predicted_price"] > resp_high.json()["predicted_price"]