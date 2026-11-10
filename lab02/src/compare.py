import inspect
import time

import sklearn_model
import torch_model
from data import BASE_DIR, N_SPLITS, load_data, split_data


def count_code_lines(*functions):
    # Liczy niepuste linie kodu bez komentarzy
    count = 0
    for function in functions:
        for line in inspect.getsource(function).splitlines():
            if line.strip() and not line.strip().startswith("#"):
                count += 1
    return count


# Zadanie 6: porównanie obu modeli
X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)

sk_acc, sk_f1, sk_cv_time = sklearn_model.cross_validate_sklearn(X, y)
pt_acc, pt_f1, pt_cv_time = torch_model.cross_validate_torch(X, y)

start = time.perf_counter()
sklearn_model.build_pipeline().fit(X_train, y_train)
sk_fit_time = time.perf_counter() - start

start = time.perf_counter()
torch_model.train_and_evaluate(X_train, y_train, X_test, y_test)
pt_fit_time = time.perf_counter() - start

sk_lines = count_code_lines(sklearn_model.build_pipeline, sklearn_model.cross_validate_sklearn)
pt_lines = count_code_lines(torch_model.WineNet, torch_model.train_and_evaluate, torch_model.cross_validate_torch)

rows = [
    ("Metryka", "scikit-learn", "PyTorch"),
    (f"Accuracy CV ({N_SPLITS}-fold)", f"{sk_acc.mean():.4f} ± {sk_acc.std():.4f}", f"{pt_acc.mean():.4f} ± {pt_acc.std():.4f}"),
    (f"F1 macro CV ({N_SPLITS}-fold)", f"{sk_f1.mean():.4f} ± {sk_f1.std():.4f}", f"{pt_f1.mean():.4f} ± {pt_f1.std():.4f}"),
    ("Czas treningu (1 model)", f"{sk_fit_time:.4f} s", f"{pt_fit_time:.4f} s"),
    (f"Czas walidacji ({N_SPLITS} modeli)", f"{sk_cv_time:.4f} s", f"{pt_cv_time:.4f} s"),
    ("Linie kodu modelu", str(sk_lines), str(pt_lines)),
]
table = "\n".join(f"{name:<25}  {sk:<15}  {pt:<15}" for name, sk, pt in rows)

folds = []
for i in range(N_SPLITS):
    folds.append(f"Fold {i + 1}: sklearn = {sk_acc[i]:.4f}, PyTorch = {pt_acc[i]:.4f}")

diff = sk_acc.mean() - pt_acc.mean()
if abs(diff) < max(sk_acc.std(), pt_acc.std()):
    verdict = "Różnica średniej dokładności jest mniejsza niż odchylenie standardowe, więc oba modele radzą sobie porównywalnie."
elif diff > 0:
    verdict = "Lepszą średnią dokładność osiągnął model scikit-learn."
else:
    verdict = "Lepszą średnią dokładność osiągnęła sieć w PyTorch."

results = (
    "Porównanie modeli na zbiorze wine (walidacja krzyżowa)\n\n"
    f"{table}\n\n"
    "Dokładność w poszczególnych foldach:\n" + "\n".join(folds) + "\n\n"
    f"Wniosek: {verdict} Trening sieci w PyTorch trwał około {pt_fit_time / sk_fit_time:.0f} razy dłużej "
    f"i wymagał {pt_lines} linii kodu wobec {sk_lines} w scikit-learn (własna klasa sieci, "
    "pętla treningowa i ręczna walidacja krzyżowa). Dla małego zbioru tabelarycznego prostszy "
    "Pipeline scikit-learn jest wystarczający i wygodniejszy.\n"
)

print(results)
(BASE_DIR / "wyniki.txt").write_text(results, encoding="utf-8")
print("Wyniki zapisane do: wyniki.txt")
