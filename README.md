# UAS Machine Learning - Kasus B (Regresi Harga Kendaraan Bekas)

**NIM** : 1003250004  
**Nama** : Rio Hilman 
**Kasus** : B - Estimasi Harga Kendaraan Bekas  
**Semester** : Ganjil 2026/2027

---

## 📌 Deskripsi Masalah

Marketplace otomotif ingin membantu penjual menentukan **harga jual wajar** untuk kendaraan bekas. Tantangan utama adalah hubungan non-linear antara **umur kendaraan** (tahun pembuatan) dengan **harga**, serta keberadaan **outlier harga ekstrem** (mobil mewah). Sistem yang dibangun berupa model regresi yang memprediksi harga dalam satuan **Lakh Rupee India**, dan disajikan sebagai REST API yang siap diintegrasikan dengan aplikasi lain.

---

## 📊 Dataset

| Item | Detail |
|------|--------|
| **Nama** | Used Cars Price Prediction |
| **Sumber** | [Kaggle - Avi Kasliwal](https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction) |
| **Lisensi** | CC0 (Public Domain) - bebas digunakan untuk pendidikan dan penelitian |
| **Jumlah Data Train** | 6.019 baris |
| **Jumlah Data Test** | 1.234 baris |
| **Target** | `Price` (harga dalam Lakh) |

Dataset ini berisi informasi kendaraan bekas dari berbagai merek, lokasi, tahun, jarak tempuh, jenis bahan bakar, transmisi, dan spesifikasi mesin.

---

## 📁 Struktur Proyek
```text
uas-ml-1003250004/
├── src/                 # Kode training & EDA
│   ├── load_data.py     # Download dataset dari Kaggle + load
│   ├── eda.py           # Eksplorasi data & generate grafik
│   ├── train.py         # Training & evaluasi dengan CV
│   └── evaluate.py      # Evaluasi final pada test set
├── app/                 # Kode serving (FastAPI)
│   └── main.py          # API endpoint prediksi
├── tests/               # Test otomatis (pytest)
│   └── test_api.py      # 7+ test (mekanis & behavioral)
├── data/                # Dataset (dihasilkan, masuk .gitignore)
├── models/              # Artefak model (dihasilkan, masuk .gitignore)
├── reports/             # Grafik EDA & evaluasi
├── requirements.txt     # Dependensi training (versi fleksibel)
├── requirements-api.txt # Dependensi serving (versi di-pin persis)
├── .gitignore           # Abaikan data/, models/, dll.
└── README.md            # Dokumentasi proyek (file ini)
```

---

## ⚙️ Prasyarat (Sebelum Menjalankan)

1. **Python** ≥ 3.10 terinstal.
2. **Git** terinstal.
3. **Kaggle API Token** (untuk mendownload dataset otomatis):
   - Login ke [Kaggle](https://www.kaggle.com/) → Akun → **"Create Legacy API Key"**.
   - Download `kaggle.json` dan simpan di:
     - **Mac/Linux**: `~/.kaggle/kaggle.json`
     - **Windows**: `C:\Users\<username>\.kaggle\kaggle.json`
   - (Mac/Linux) Set permission: `chmod 600 ~/.kaggle/kaggle.json`

---

## 🚀 Langkah Menjalankan Proyek dari Nol

Ikuti langkah-langkah berikut - penguji dapat mereproduksi seluruh sistem tanpa bertanya kepada Anda.

1. Clone repositori

   ```bash
   git clone https://github.com/riohilman/uas-ml-1003250004.git
   cd uas-ml-1003250004
   ```

2. Buat dan aktifkan virtual environment

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

3. Install dependensi training

   ```bash
   pip install -r requirements.txt
   ```

4. Download dataset secara otomatis dari Kaggle

   ```bash
   python src/load_data.py
   ```

   Skrip ini akan mendownload dan mengekstrak train-data.csv dan test-data.csv ke folder data/. Jika file sudah ada, download dilewati.

5. Jalankan EDA untuk menghasilkan grafik

   ```bash
   python src/eda.py
   ```

   Grafik akan tersimpan di folder reports/ sebagai PNG.

6. Latih model dan simpan artefak

   ```bash
   python src/train.py
   ```

   Model pipeline utuh akan tersimpan di models/model_regresi.joblib dan metadata di models/metadata.json.

7. Evaluasi model pada test set

   ```bash
   python src/evaluate.py
   ```

   Metrik dan grafik evaluasi akan tersimpan di reports/.

8. Jalankan server API untuk serving

   Pastikan dependensi serving terinstal.

   ```bash
   pip install -r requirements-api.txt   # Jika belum terinstal
   uvicorn app.main:app --reload --port 8100
   ```

   Server akan berjalan di http://localhost:8100. Buka Swagger UI di http://localhost:8100/docs.

9. Jalankan test otomatis

    Buka terminal baru (atau hentikan server dengan Ctrl+C), lalu:

    ```bash
    python -m pytest tests/ -v
    ```

    Pastikan semua test berwarna hijau (PASSED).

### Contoh Pemanggilan API (curl)

#### Prediksi Berhasil (HTTP 200)

```bash
curl -X POST http://localhost:8100/predict-harga \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

Response yang diharapkan:

```json
{
  "predicted_price": 12.35,
  "message": "Success"
}
```

#### Request Tidak Valid (HTTP 422)

Input dengan Fuel_Type tidak dikenal akan ditolak oleh validasi Pydantic.

```bash
curl -X POST http://localhost:8100/predict-harga \
  -H "Content-Type: application/json" \
  -d '{
    "Year": 2015,
    "Kilometers_Driven": 41000,
    "Fuel_Type": "Batubara",
    "Transmission": "Manual",
    "Owner_Type": "First"
  }'
```

Response yang diharapkan (422 Unprocessable Entity):

```json
{
  "detail": [
    {
      "type": "string_pattern_mismatch",
      "loc": ["body", "Fuel_Type"],
      "msg": "String should match pattern '^(Petrol|Diesel|CNG|LPG|Electric)$'",
      "input": "Batubara"
    }
  ]
}
```

### Test Otomatis (pytest)

Terdapat minimal 7 test yang mencakup:

4 test mekanis (health check, root, prediksi valid, error handling 422).

3 behavioral test (kendaraan lebih tua harus lebih murah, jarak tempuh lebih tinggi harus lebih murah).

Jalankan dengan:

```bash
python -m pytest tests/ -v
```

Hasil yang diharapkan:

```text
tests/test_api.py::test_health PASSED
tests/test_api.py::test_root PASSED
tests/test_api.py::test_predict_valid PASSED
tests/test_api.py::test_predict_missing_field PASSED
tests/test_api.py::test_predict_invalid_enum PASSED
tests/test_api.py::test_older_car_cheaper PASSED
tests/test_api.py::test_higher_mileage_cheaper PASSED
==================== 7 passed in X.XXs ====================
```
📌 Catatan Penting 
1. Mengapa data/ dan models/ Dimasukkan ke .gitignore?
Folder data/ dan models/ berisi artefak yang ukurannya besar (dataset dan file model) dan dapat dihasilkan ulang kapan saja oleh kode. Jika artefak ini ikut dikomit ke Git:

Repositori menjadi sangat berat (500 MB+) dan lambat di-clone.

Risiko konflik versi (misal dataset diupdate tapi kode tidak).

Melanggar prinsip reproducible research.

Penguji tetap bisa memproduksi ulang isinya dengan menjalankan src/load_data.py (untuk data) dan src/train.py (untuk model) - sebagaimana dijelaskan di langkah di atas.

2. Mengapa requirements-api.txt Versinya Di-pin Persis, Sedangkan requirements.txt Tidak?
Lingkungan Training (requirements.txt):

Menggunakan versi yang lebih fleksibel (misal >= atau tanpa batas bawah).

Tujuannya untuk memudahkan eksperimen dan pembaruan library saat pengembangan.

Lingkungan Serving (requirements-api.txt):

Versi di-pin persis (contoh: scikit-learn==1.7.2).

Tujuannya untuk menghindari version skew. Model yang dilatih dengan joblib (pickle) menyimpan struktur internal objek scikit-learn. Jika library di server (saat serving) memiliki versi berbeda (misal 1.9.0), proses joblib.load() bisa gagal atau menghasilkan prediksi yang salah/tidak stabil. Dengan mem-pin versi, lingkungan serving selalu identik dengan lingkungan training, memastikan model dapat dimuat dan berjalan dengan benar.

