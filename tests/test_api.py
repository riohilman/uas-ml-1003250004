import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# 1. Test health endpoint
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

# 2. Test root
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Car Price Predictor" in response.json()["service"]

# 3. Test prediksi valid
def test_predict_valid():
    payload = {
        "Year": 2015,
        "Kilometers_Driven": 41000,
        "Fuel_Type": "Diesel",
        "Transmission": "Manual",
        "Owner_Type": "First",
        "Mileage": "19.67 kmpl",
        "Engine": "1582 CC",
        "Power": "126.2 bhp",
        "Seats": 5.0,
        "Location": "Pune",
        "Brand": "Hyundai"
    }
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)

# 4. Test missing field -> 422
def test_predict_missing_field():
    payload = {
        "Year": 2015,
        "Kilometers_Driven": 41000,
        "Fuel_Type": "Diesel",
        "Transmission": "Manual",
        "Owner_Type": "First"
        # Mileage dll hilang
    }
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 422

# 5. Test invalid enum -> 422
def test_predict_invalid_enum():
    payload = {
        "Year": 2015,
        "Kilometers_Driven": 41000,
        "Fuel_Type": "InvalidFuel",
        "Transmission": "Manual",
        "Owner_Type": "First",
        "Mileage": "19.67 kmpl",
        "Engine": "1582 CC",
        "Power": "126.2 bhp",
        "Seats": 5.0
    }
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 422

# --- Behavioral tests ---

# 6. Test: kendaraan lebih tua dengan spesifikasi sama harus diprediksi lebih murah
def test_older_car_cheaper():
    base = {
        "Year": 2015,
        "Kilometers_Driven": 50000,
        "Fuel_Type": "Petrol",
        "Transmission": "Manual",
        "Owner_Type": "First",
        "Mileage": "18.2 kmpl",
        "Engine": "1199 CC",
        "Power": "88.7 bhp",
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

# 7. Test: jarak tempuh lebih tinggi dengan spesifikasi sama harus diprediksi lebih murah
def test_higher_mileage_cheaper():
    base = {
        "Year": 2015,
        "Kilometers_Driven": 50000,
        "Fuel_Type": "Petrol",
        "Transmission": "Manual",
        "Owner_Type": "First",
        "Mileage": "18.2 kmpl",
        "Engine": "1199 CC",
        "Power": "88.7 bhp",
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