import os
import pandas as pd
import shutil
import kagglehub

DATA_DIR = "data"
TRAIN_FILE = "train-data.csv"
TEST_FILE = "test-data.csv"
KAGGLE_DATASET = "avikasliwal/used-cars-price-prediction"

def download_kaggle_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(os.path.join(DATA_DIR, TRAIN_FILE)):
        print(f"Dataset sudah ada di {DATA_DIR}/")
        return
    
    print(f"Downloading dataset {KAGGLE_DATASET} using kagglehub...")
    
    # Download dataset ke cache folder
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    
    # Copy file dari cache ke folder data/
    for file in os.listdir(path):
        if file.endswith('.csv'):
            src = os.path.join(path, file)
            dst = os.path.join(DATA_DIR, file)
            shutil.copy2(src, dst)
            print(f"  Copied {file}")
    
    print(f"Dataset berhasil disalin ke {DATA_DIR}/")

def print_dataset_info(df, name):
    print(f"\n=== Info dataset: {name} ===")
    print(f"Jumlah baris  : {df.shape[0]}")
    print(f"Jumlah kolom  : {df.shape[1]}")
    print("\nTipe tiap kolom:")
    print(df.dtypes)
    print("\nJumlah nilai hilang per kolom:")
    missing = df.isna().sum()
    print(missing[missing >= 0])  # tampilkan semua kolom, termasuk yang 0 missing


def load_train_data():
    download_kaggle_dataset()
    df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE), index_col=0)
    print_dataset_info(df, TRAIN_FILE)
    return df

def load_test_data():
    download_kaggle_dataset()
    df = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE), index_col=0)
    print_dataset_info(df, TEST_FILE)
    return df

if __name__ == "__main__":
    download_kaggle_dataset()
    load_train_data()
    load_test_data()