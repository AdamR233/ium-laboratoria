import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import N_SPLITS, RANDOM_SEED, load_data, split_data


# Zadanie 2: Pipeline = skalowanie cech + klasyfikator
def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ])


# Zadanie 4: 5-krotna walidacja krzyżowa
def cross_validate_sklearn(X, y):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    start = time.perf_counter()
    acc = cross_val_score(build_pipeline(), X, y, cv=kf, scoring="accuracy")
    elapsed = time.perf_counter() - start
    f1 = cross_val_score(build_pipeline(), X, y, cv=kf, scoring="f1_macro")
    return acc, f1, elapsed


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("=== Zadanie 2: Pipeline (StandardScaler + LogisticRegression) ===")
    model = build_pipeline()
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    y_pred = model.predict(X_test)
    print(f"Czas treningu: {train_time:.4f} s")
    print(f"Accuracy na zbiorze testowym: {accuracy_score(y_test, y_pred):.4f}")

    print(f"\n=== Zadanie 4: walidacja krzyżowa ({N_SPLITS}-fold) ===")
    acc, f1, cv_time = cross_validate_sklearn(X, y)
    for i, score in enumerate(acc, start=1):
        print(f"Fold {i}: accuracy = {score:.4f}")
    print(f"Średnia accuracy: {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"Średnie F1 (macro): {f1.mean():.4f} ± {f1.std():.4f}")
    print(f"Czas walidacji krzyżowej: {cv_time:.4f} s")
