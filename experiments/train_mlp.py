"""Навчання класифікатора типу файлу і звіт.

    python experiments/train_mlp.py   # навчити, зберегти sarc/mlp_model.json, звіт у docs/MLP_RESULTS.md

Протокол (методичка, 10.42): train/ — навчання; valid/ — вибір моделі за macro-F1;
holdout/ є лише у викладача. Тому число на valid — робочий орієнтир, а не підсумкова оцінка:
модель обиралася саме за ним, тож воно трохи завищене.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sarc.analysis import TYPE_NAMES  # noqa: E402
from sarc.features import FEATURE_NAMES, extract  # noqa: E402
from sarc.mlp import MLP, confusion_matrix, f1_per_class, macro_f1  # noqa: E402

K = 7
SEEDS = (1, 2, 3)            # три незалежні ініціалізації
HIDDEN = 16
EPOCHS = 300
MAGIC = [FEATURE_NAMES.index("compressed_magic"), FEATURE_NAMES.index("exec_magic")]


def load(part: str) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(open(ROOT / "data" / part / "labels.csv", encoding="utf-8")))
    X = np.array([extract((ROOT / "data" / part / r["file"]).read_bytes()) for r in rows])
    y = np.array([int(r["type"]) for r in rows])
    return X, y


def train_runs(Xtr, ytr, Xva, yva, cols=None):
    """Три ініціалізації; повертає список результатів і найкращу модель за macro-F1 на valid."""
    cols = list(range(Xtr.shape[1])) if cols is None else cols
    runs, best = [], None
    for seed in SEEDS:
        model = MLP(len(cols), HIDDEN, K, seed=seed)
        t0 = time.perf_counter()
        hist = model.fit(Xtr[:, cols], ytr, epochs=EPOCHS, seed=seed)
        run = {
            "seed": seed,
            "train_f1": macro_f1(ytr, model.predict(Xtr[:, cols]), K),
            "valid_f1": macro_f1(yva, model.predict(Xva[:, cols]), K),
            "loss_first": hist[0], "loss_last": hist[-1],
            "loss_curve": hist[::10],
            "seconds": time.perf_counter() - t0,
        }
        runs.append(run)
        if best is None or run["valid_f1"] > best[0]["valid_f1"]:
            best = (run, model)
    return runs, best


def baselines(ytr, yva) -> dict:
    """Мажоритарний клас і випадковий вибір (середнє за 1000 спроб)."""
    majority = np.bincount(ytr, minlength=K).argmax()
    rng = np.random.default_rng(0)
    rand = np.mean([macro_f1(yva, rng.integers(0, K, len(yva)), K) for _ in range(1000)])
    return {"majority": macro_f1(yva, np.full(len(yva), majority), K), "random": float(rand)}


def table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "sarc" / "mlp_model.json"))
    ap.add_argument("--report", default=str(ROOT / "docs" / "MLP_RESULTS.md"))
    ap.add_argument("--raw", default=str(ROOT / "experiments" / "results" / "mlp_runs.json"))
    args = ap.parse_args()

    Xtr, ytr = load("train")
    Xva, yva = load("valid")
    base = baselines(ytr, yva)
    runs, (best_run, model) = train_runs(Xtr, ytr, Xva, yva)
    no_magic_cols = [i for i in range(len(FEATURE_NAMES)) if i not in MAGIC]
    runs_nm, (best_nm, _) = train_runs(Xtr, ytr, Xva, yva, no_magic_cols)

    Xco, yco = load("corpus")
    corpus_f1 = macro_f1(yco, model.predict(Xco), K)
    pred = model.predict(Xva)
    cm = confusion_matrix(yva, pred, K)
    f1 = f1_per_class(cm)
    model.save(args.model, {"features": FEATURE_NAMES, "hidden": HIDDEN, "seed": best_run["seed"],
                            "valid_macro_f1": best_run["valid_f1"]})

    Path(args.raw).parent.mkdir(parents=True, exist_ok=True)
    raw = {"corpus_macro_f1": corpus_f1, "baselines": base, "runs": runs, "runs_without_magic": runs_nm}
    Path(args.raw).write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")

    short = [TYPE_NAMES[i].split()[0][:8] for i in range(K)]
    means = [[TYPE_NAMES[c]] + [f"{Xtr[ytr == c, j].mean():.3f}" for j in range(len(FEATURE_NAMES))]
             for c in range(K)]
    lines = [
        "# Класифікатор типу файлу (згенеровано experiments/train_mlp.py)\n",
        f"Train: {len(ytr)} файлів, valid: {len(yva)}. MLP: {len(FEATURE_NAMES)} ознак → {HIDDEN} (ReLU) "
        f"→ {K} (softmax), {EPOCHS} епох, SGD міні-батчами по 32, lr = 0.05, L2 = 1e-4.\n",
        "Valid — робочий орієнтир: модель обиралася за ним. "
        "Підсумковий macro-F1 рахує викладач на holdout.\n",
        "## 1. Ознаки: середнє по класах на train\n",
        "Перевірка, що ознаки мають сенс, до навчання (методичка, 10.41).\n",
        table(["Клас"] + FEATURE_NAMES, means), "",
        "## 2. Baseline і три ініціалізації\n",
        table(["Модель", "macro-F1 train", "macro-F1 valid", "loss: перша → остання епоха", "час, с"],
              [["випадковий вибір", "—", f"{base['random']:.3f}", "—", "—"],
               ["мажоритарний клас", "—", f"{base['majority']:.3f}", "—", "—"]]
              + [[f"MLP, seed {r['seed']}", f"{r['train_f1']:.3f}", f"{r['valid_f1']:.3f}",
                  f"{r['loss_first']:.3f} → {r['loss_last']:.3f}", f"{r['seconds']:.1f}"] for r in runs]),
        f"\nОбрано seed {best_run['seed']} (найкращий macro-F1 на valid = {best_run['valid_f1']:.3f}). "
        f"Поріг з team_config: F1_min = 0.61.\n",
        "## 3. Confusion matrix обраної моделі на valid\n",
        "Рядок — справжній клас, стовпець — передбачений.\n",
        table(["справжній \\ передбачений"] + short + ["F1"],
              [[TYPE_NAMES[i]] + list(cm[i]) + [f"{f1[i]:.3f}"] for i in range(K)]), "",
        "## 4. Внесок сигнатур форматів\n",
        "Ті самі три ініціалізації без ознак compressed_magic і exec_magic:\n",
        table(["Варіант", "найкращий macro-F1 valid"],
              [["усі ознаки", f"{best_run['valid_f1']:.3f}"],
               ["без сигнатур", f"{best_nm['valid_f1']:.3f}"]]),
        "",
        "## 5. Додаткова перевірка на corpus/\n",
        f"macro-F1 обраної моделі на corpus/ (21 файл, у навчанні і виборі моделі не брав участі): "
        f"**{corpus_f1:.3f}**.\n",
        "Чесне застереження: першу версію ознак перевіряли на corpus/ і через одну помилку (періодичний "
        "шаблон з байтом «;», прийнятий за CSV) додали ознаку autocorrelation. Тож corpus/ — не зовсім "
        "незалежна перевірка.\n",
        "Сирі результати (криві loss, усі запуски): `experiments/results/mlp_runs.json`.\n",
    ]
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
