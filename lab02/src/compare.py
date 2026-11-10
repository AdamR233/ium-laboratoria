import inspect
import time

import sklearn_model
import torch_model
from data import BASE_DIR, N_SPLITS, load_data, split_data


def code_lines(*objects):
    # Liczba niepustych linii kodu (bez komentarzy) potrzebnych do zbudowania i wytrenowania modelu
    lines = []
    for obj in objects:
        lines += inspect.getsource(obj).splitlines()
    return sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))


def timed_fit_sklearn(X_train, y_train):
    start = time.perf_counter()
    sklearn_model.build_pipeline().fit(X_train, y_train)
    return time.perf_counter() - start


def timed_fit_torch(X_train, y_train, X_test, y_test):
    start = time.perf_counter()
    torch_model.train_and_evaluate(X_train, y_train, X_test, y_test)
    return time.perf_counter() - start


# Zadanie 6: zestawienie wyników obu modeli
X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)

sk_acc, sk_f1, sk_cv_time = sklearn_model.cross_validate_sklearn(X, y)
pt_acc, pt_f1, pt_cv_time = torch_model.cross_validate_torch(X, y)
sk_fit_time = timed_fit_sklearn(X_train, y_train)
pt_fit_time = timed_fit_torch(X_train, y_train, X_test, y_test)

sk_lines = code_lines(sklearn_model.build_pipeline, sklearn_model.cross_validate_sklearn)
pt_lines = code_lines(torch_model.WineNet, torch_model.train_and_evaluate, torch_model.cross_validate_torch)

rows = [
    ("Metryka", "scikit-learn", "PyTorch"),
    (f"Accuracy CV ({N_SPLITS}-fold)", f"{sk_acc.mean():.4f} ± {sk_acc.std():.4f}", f"{pt_acc.mean():.4f} ± {pt_acc.std():.4f}"),
    (f"F1 macro CV ({N_SPLITS}-fold)", f"{sk_f1.mean():.4f} ± {sk_f1.std():.4f}", f"{pt_f1.mean():.4f} ± {pt_f1.std():.4f}"),
    ("Czas treningu (1 model)", f"{sk_fit_time:.4f} s", f"{pt_fit_time:.4f} s"),
    (f"Czas walidacji ({N_SPLITS} modeli)", f"{sk_cv_time:.4f} s", f"{pt_cv_time:.4f} s"),
    ("Linie kodu modelu", str(sk_lines), str(pt_lines)),
]
widths = [max(len(r[i]) for r in rows) for i in range(3)]
table = "\n".join("  ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in rows)
folds = "\n".join(
    f"Fold {i}: sklearn = {a:.4f}, PyTorch = {b:.4f}" for i, (a, b) in enumerate(zip(sk_acc, pt_acc), start=1)
)

if abs(sk_acc.mean() - pt_acc.mean()) < max(sk_acc.std(), pt_acc.std()):
    verdict = "Różnica średniej dokładności jest mniejsza niż odchylenie standardowe, więc oba modele radzą sobie porównywalnie."
elif sk_acc.mean() > pt_acc.mean():
    verdict = "Lepszą średnią dokładność osiągnął model scikit-learn."
else:
    verdict = "Lepszą średnią dokładność osiągnęła sieć w PyTorch."
speed = pt_fit_time / sk_fit_time

results = (
    "Porównanie modeli na zbiorze wine (walidacja krzyżowa)\n\n"
    f"{table}\n\n"
    f"Dokładność w poszczególnych foldach:\n{folds}\n\n"
    f"Wniosek: {verdict} Trening sieci w PyTorch trwał około {speed:.0f} razy dłużej "
    f"i wymagał {pt_lines} linii kodu wobec {sk_lines} w scikit-learn (własna klasa sieci, "
    "pętla treningowa i ręczna walidacja krzyżowa). Dla małego zbioru tabelarycznego prostszy "
    "Pipeline scikit-learn jest wystarczający i wygodniejszy.\n"
)

print(results)
(BASE_DIR / "wyniki.txt").write_text(results, encoding="utf-8")
print("Wyniki zapisane do: wyniki.txt")
