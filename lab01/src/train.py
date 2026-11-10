import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Ścieżki liczone względem katalogu lab01, niezależnie od miejsca uruchomienia skryptu
BASE_DIR = Path(__file__).resolve().parent.parent

# Zadanie 6: konfiguracja wczytywana z pliku .env (drugi argument getenv to wartość domyślna)
load_dotenv(BASE_DIR / ".env")

DATA_PATH = BASE_DIR / os.getenv("DATA_PATH", "data/wine.csv")
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

print("=== Konfiguracja z .env ===")
print(f"DATA_PATH={DATA_PATH.relative_to(BASE_DIR)}, TEST_SIZE={TEST_SIZE}, RANDOM_SEED={RANDOM_SEED}\n")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# Zadanie 3: wczytanie i eksploracja danych
def load_data(path):
    # Przy pierwszym uruchomieniu zbiór z scikit-learn jest zapisywany do CSV w data/
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        load_wine(as_frame=True).frame.to_csv(path, index=False)
    return pd.read_csv(path)


df = load_data(DATA_PATH)
class_names = load_wine().target_names

print("=== Pierwsze wiersze (head) ===")
print(df.head())
print("\n=== Rozmiar zbioru (shape) ===")
print(df.shape)
print("\n=== Statystyki opisowe (describe) ===")
print(df.describe().T.round(2))
print("\n=== Liczność klas ===")
print(df["target"].value_counts().sort_index())


# Zadanie 4: podział danych 80/20 i trening modelu
X = df.drop(columns="target")
y = df["target"]

# stratify=y zachowuje proporcje klas w obu zbiorach
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
print(f"\nZbiór treningowy: {X_train.shape}, testowy: {X_test.shape}")

# Cechy mają bardzo różne skale (np. proline ~1000, hue ~1), stąd większy limit iteracji
model = LogisticRegression(max_iter=10000, random_state=RANDOM_SEED)
model.fit(X_train, y_train)

model_path = BASE_DIR / "models" / "logistic_regression.joblib"
joblib.dump(model, model_path)
print(f"Model zapisany do: {model_path.relative_to(BASE_DIR)}")


# Zadanie 5: ocena modelu na zbiorze testowym i zapis wyników
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=class_names, digits=4)

results = (
    "Zbiór danych: wine (scikit-learn)\n"
    "Model: LogisticRegression\n"
    f"Podział: {1 - TEST_SIZE:.0%} trening / {TEST_SIZE:.0%} test, random_state={RANDOM_SEED}\n"
    f"Liczba próbek testowych: {len(y_test)}\n\n"
    f"Accuracy: {accuracy:.4f}\n\n"
    f"Classification report:\n{report}"
)

print("\n=== Wyniki na zbiorze testowym ===")
print(results)

results_path = BASE_DIR / "wyniki.txt"
results_path.write_text(results, encoding="utf-8")
print(f"Wyniki zapisane do: {results_path.relative_to(BASE_DIR)}")
