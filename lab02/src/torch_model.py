import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import N_SPLITS, RANDOM_SEED, load_data, split_data

EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.01
N_CLASSES = 3


# Zadanie 3: sieć w pełni połączona z dwiema warstwami ukrytymi
class WineNet(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x):
        return self.layers(x)


def train_and_evaluate(X_train, y_train, X_test, y_test, verbose=False):
    torch.manual_seed(RANDOM_SEED)

    # Skaler dopasowany tylko do danych treningowych, tak jak w Pipeline sklearn
    scaler = StandardScaler().fit(X_train)
    X_train_t = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    X_test_t = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)

    # Nowa instancja sieci przy każdym wywołaniu, czyli świeże wagi w każdym foldzie
    model = WineNet(X_train.shape[1], N_CLASSES)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        if verbose and epoch % 10 == 0:
            print(f"Epoka {epoch:2d}/{EPOCHS}, strata: {epoch_loss / len(X_train_t):.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).argmax(dim=1).numpy()
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="macro")


# Zadanie 5: ręczna walidacja krzyżowa z tym samym podziałem KFold co w sklearn
def cross_validate_torch(X, y):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    acc, f1 = [], []
    start = time.perf_counter()
    for train_idx, test_idx in kf.split(X):
        fold_acc, fold_f1 = train_and_evaluate(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        acc.append(fold_acc)
        f1.append(fold_f1)
    elapsed = time.perf_counter() - start
    return np.array(acc), np.array(f1), elapsed


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("=== Zadanie 3: sieć w PyTorch ===")
    print(WineNet(X.shape[1], N_CLASSES))
    start = time.perf_counter()
    acc, _ = train_and_evaluate(X_train, y_train, X_test, y_test, verbose=True)
    train_time = time.perf_counter() - start
    print(f"Czas treningu: {train_time:.4f} s")
    print(f"Accuracy na zbiorze testowym: {acc:.4f}")

    print(f"\n=== Zadanie 5: walidacja krzyżowa ({N_SPLITS}-fold) ===")
    acc, f1, cv_time = cross_validate_torch(X, y)
    for i, score in enumerate(acc, start=1):
        print(f"Fold {i}: accuracy = {score:.4f}")
    print(f"Średnia accuracy: {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"Średnie F1 (macro): {f1.mean():.4f} ± {f1.std():.4f}")
    print(f"Czas walidacji krzyżowej: {cv_time:.4f} s")
