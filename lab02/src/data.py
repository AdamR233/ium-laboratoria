from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "wine.csv"
TEST_SIZE = 0.2
RANDOM_SEED = 42
N_SPLITS = 5


# Zadanie 1: wspólny zbiór danych i podział dla obu modeli
def load_data():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        load_wine(as_frame=True).frame.to_csv(DATA_PATH, index=False)
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns="target").to_numpy(dtype=np.float32)
    y = df["target"].to_numpy()
    return X, y


def split_data(X, y):
    # Ten sam random_state i stratify gwarantują identyczny podział w obu modelach
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Cały zbiór:        {X.shape}, liczność klas: {np.bincount(y).tolist()}")
    print(f"Zbiór treningowy:  {X_train.shape}, liczność klas: {np.bincount(y_train).tolist()}")
    print(f"Zbiór testowy:     {X_test.shape}, liczność klas: {np.bincount(y_test).tolist()}")
