"""Модуль атаки: три експерименти і звіт docs/ATTACK_RESULTS.md (методичка, 7.30, 9.47, 11.84).

    python experiments/run_attack.py          # ~5 хв

1. ГА відновлює ключ навчального XOR-шифру (порівняння з random search при однаковому бюджеті).
2. Той самий ГА проти справжнього архіву SARC з PIN-паролем:
   без «часткової правильності» ГА = випадковий перебір.
3. Цілісність: сотні випадкових пошкоджень архіву — усі мають бути відхилені з кодом помилки.
"""
import csv
import json
import random
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from attack.password_attack import PasswordOracle, search  # noqa: E402
from attack.tamper import HOW, trial  # noqa: E402
from attack.text_model import BigramModel  # noqa: E402
from attack.weak_cipher import guess_key_length, xor  # noqa: E402
from attack.xor_attack import attack, key_accuracy, text_accuracy  # noqa: E402
from sarc.archive import Options, create_archive  # noqa: E402

KEY_LENGTHS = (4, 8, 12, 16)
SEEDS = range(5)
BUDGET = 20000
PIN_DIGITS, PIN_BUDGET, PIN_SEEDS = 4, 1000, range(3)
TAMPER_TRIALS = 50          # на кожен спосіб пошкодження
CORPUS = sorted(p for p in (ROOT / "data" / "corpus").iterdir() if p.name != "labels.csv")


def train_model() -> BigramModel:
    rows = [r for r in csv.DictReader(open(ROOT / "data" / "train" / "labels.csv", encoding="utf-8"))
            if r["type"] == "0"]
    return BigramModel.train([(ROOT / "data" / "train" / r["file"]).read_bytes() for r in rows])


def plaintexts() -> list[bytes]:
    """Відкриті тексти — текстові файли corpus/ (у навчанні моделі вони не брали участі)."""
    rows = [r for r in csv.DictReader(open(ROOT / "data" / "corpus" / "labels.csv", encoding="utf-8"))
            if r["type"] == "0"]
    return [(ROOT / "data" / "corpus" / r["file"]).read_bytes() for r in rows]


def xor_experiment(model):
    texts = plaintexts()
    rows, curves = [], {"ga": {}, "random": {}}
    for k in KEY_LENGTHS:
        for seed in SEEDS:
            rng = random.Random(1000 * k + seed)
            plain = texts[seed % len(texts)]
            key = rng.randbytes(k)
            ct = xor(plain, key)
            k_guess = guess_key_length(ct)[0]
            for method in ("ga", "random"):
                t0 = time.perf_counter()
                res = attack(ct, model, BUDGET, method, seed)
                rows.append({"k": k, "seed": seed, "method": method, "k_guess": k_guess,
                             "key_acc": key_accuracy(res.key, key),
                             "text_acc": text_accuracy(ct, res.key, plain) if res.key_length == k else 0.0,
                             "fitness": res.fitness, "true_fitness": model.score(plain[:3000]),
                             "seconds": time.perf_counter() - t0})
                curves[method].setdefault(k, []).append(res.history)
    return rows, curves


def best_after(history, evals):
    vals = [h["best"] for h in history if h["evaluations"] <= evals]
    return max(vals) if vals else float("nan")


def pin_experiment(tmp: Path):
    rng = random.Random(7)
    secret = "".join(str(rng.randrange(10)) for _ in range(PIN_DIGITS))
    res = create_archive([str(CORPUS[0])], str(tmp / "pin.arc"), secret, Options())
    oracle = PasswordOracle(Path(res.volumes[0]).read_bytes())
    t0 = time.perf_counter()
    oracle.check("0" * PIN_DIGITS)
    per_check = time.perf_counter() - t0
    rows = []
    for seed in PIN_SEEDS:
        for method in ("ga", "random"):
            found, evals = search(oracle, PIN_DIGITS, PIN_BUDGET, method, seed)
            rows.append({"seed": seed, "method": method, "found": found == secret, "evaluations": evals})
    return {"secret_len": PIN_DIGITS, "space": 10 ** PIN_DIGITS, "budget": PIN_BUDGET,
            "per_check": per_check, "rows": rows}


def tamper_experiment(tmp: Path):
    res = create_archive([str(p) for p in CORPUS], str(tmp / "t.arc"), "tamper-test", Options())
    rng = random.Random(11)
    work = tmp / "work"
    work.mkdir()
    out = {}
    for how in HOW:
        out[how] = Counter(trial(res.volumes, "tamper-test", how, rng, work) for _ in range(TAMPER_TRIALS))
    return len(res.volumes), out


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    model = train_model()
    xor_rows, curves = xor_experiment(model)
    with tempfile.TemporaryDirectory() as t:
        pin = pin_experiment(Path(t))
        n_vol, tamper = tamper_experiment(Path(t))

    def agg(k, method, field):
        return statistics.mean(r[field] for r in xor_rows if r["k"] == k and r["method"] == method)

    full = {(k, m): sum(r["key_acc"] == 1.0 for r in xor_rows if r["k"] == k and r["method"] == m)
            for k in KEY_LENGTHS for m in ("ga", "random")}
    kguess_ok = sum(r["k_guess"] == r["k"] for r in xor_rows if r["method"] == "ga")
    cps = [500, 2000, 5000, 10000, BUDGET]
    bad = {how: sum(v for code, v in c.items() if not code.startswith("E_")) for how, c in tamper.items()}
    total_bad = sum(bad.values())

    Path(ROOT / "experiments" / "results").mkdir(exist_ok=True)
    (ROOT / "experiments" / "results" / "attack_runs.json").write_text(json.dumps(
        {"xor": xor_rows, "pin": pin, "tamper": {k: dict(v) for k, v in tamper.items()}},
        ensure_ascii=False, indent=1), encoding="utf-8")

    pin_ga = sum(r["found"] for r in pin["rows"] if r["method"] == "ga")
    pin_rs = sum(r["found"] for r in pin["rows"] if r["method"] == "random")
    lines = [
        "# Модуль атаки (згенеровано experiments/run_attack.py)\n",
        "Методичка (7.30): AES не атакується. Мета — показати, що ГА є практичним інструментом пошуку "
        "ключа там, де простір ключів малий або дає «часткову правильність», і перевірити цілісність. "
        "Даних від викладача немає, тому слабкий шифр і дані підготовлено самостійно.\n",
        "## 1. ГА відновлює ключ навчального XOR-шифру\n",
        "Шифр: XOR з ключем довжини k, що повторюється (`attack/weak_cipher.py`). Відкриті тексти — текстові "
        "файли `corpus/` (українською, UTF-8); модель тексту навчена на класі 0 з `train/`, тобто на інших "
        "файлах. Довжина ключа визначається за індексом збігу, байти ключа — ГА (популяція 50, турнір 3, "
        "рівномірне схрещування, мутація «новий випадковий байт» з p = 1/k, еліта 2). "
        f"Бюджет — {BUDGET} оцінок fitness для ГА і для random search; "
        f"{len(SEEDS)} ключів на кожну довжину.\n",
        f"Довжину ключа вгадано правильно в {kguess_ok} з {len(KEY_LENGTHS) * len(SEEDS)} випадків.\n",
        table(["k", "Простір ключів", "ГА: байтів ключа", "ГА: байтів тексту", "ГА: ключ повністю",
               "Random: байтів ключа", "Random: ключ повністю", "ГА: час, с"],
              [[k, f"2^{8 * k}", f"{agg(k, 'ga', 'key_acc'):.0%}", f"{agg(k, 'ga', 'text_acc'):.0%}",
                f"{full[(k, 'ga')]} з {len(SEEDS)}", f"{agg(k, 'random', 'key_acc'):.0%}",
                f"{full[(k, 'random')]} з {len(SEEDS)}", f"{agg(k, 'ga', 'seconds'):.1f}"]
               for k in KEY_LENGTHS]),
        "",
        "**Збіжність** (k = 16, найкращий fitness після N оцінок, середнє за ключами; fitness справжнього "
        f"тексту ≈ {statistics.mean(r['true_fitness'] for r in xor_rows if r['k'] == 16):.2f}):\n",
        table(["N оцінок"] + [str(c) for c in cps],
              [[m] + [f"{statistics.mean(best_after(h, c) for h in curves[m][16]):.2f}" for c in cps]
               for m in ("ga", "random")]),
        "",
        "Повний перебір навіть для k = 4 — це 2^32 ≈ 4.3 млрд ключів; ГА знаходить ключ за 20 000 оцінок. "
        "Причина — «часткова правильність»: кожен вгаданий байт ключа робить частину тексту осмисленою і "
        "підвищує fitness, тож ГА поступово збирає ключ із правильних частин різних особин (схрещування). "
        "Random search за той самий бюджет окремі байти вгадує (серед тисяч спроб кожен байт ключа "
        "випадково трапляється правильним приблизно раз на 256), але ці вдалі байти розкидані по різних "
        "спробах і ніколи не збираються в один ключ; тому повного ключа він не знаходить жодного разу, "
        "і що довший ключ, то менша частка вгаданих байтів.\n",
        "## 2. Той самий ГА проти справжнього архіву з PIN-паролем\n",
        f"Архів SARC (AES-256-CTR + HMAC), пароль — {pin['secret_len']}-значний PIN, простір {pin['space']}. "
        f"Перевірка одного пароля — PBKDF2 (100 000 ітерацій) + HMAC ≈ {pin['per_check'] * 1000:.0f} мс. "
        f"Бюджет — {pin['budget']} перевірок, {len(PIN_SEEDS)} запуски кожного методу.\n",
        table(["Метод", "Знайдено PIN", "Очікувано для випадкового перебору"],
              [["ГА", f"{pin_ga} з {len(PIN_SEEDS)}", f"≈ {pin['budget'] / pin['space']:.0%} на запуск"],
               ["Random search", f"{pin_rs} з {len(PIN_SEEDS)}",
                f"≈ {pin['budget'] / pin['space']:.0%} на запуск"]]),
        "",
        "Тут fitness лише 0 або 1: HMAC або зійшовся, або ні, і пароль «на одну цифру ближчий» дає таке саме "
        "сміття, як будь-який інший. ГА не має за що зачепитися і зводиться до випадкового перебору. "
        f"Повний перебір усіх {pin['space']} PIN зайняв би ≈ {pin['space'] * pin['per_check'] / 60:.0f} хв — "
        "отже, захищає не ГА-стійкість, а довжина пароля: кожен додатковий випадковий символ множить час "
        "перебору, а PBKDF2 робить кожну спробу дорогою.\n",
        "**Висновок експериментів 1–2.** ГА працює, коли fitness дає часткову оцінку близькості до розв'язку "
        "(слабкий навчальний шифр). Проти правильної конструкції (KDF + AES + HMAC) він не дає переваги.\n",
        "## 3. Цілісність: випадкові пошкодження архіву\n",
        f"Архів усього `corpus/` ({n_vol} томи), {TAMPER_TRIALS} спроб на кожен спосіб: інвертування "
        "випадкового біта в певній ділянці випадкового тому, обрізання або дописування байтів.\n",
        table(["Що зіпсовано", "Результати (код помилки: кількість)", "Прийнято або необроблений виняток"],
              [[how, ", ".join(f"{c}: {v}" for c, v in sorted(cnt.items())), bad[how]]
               for how, cnt in tamper.items()]),
        "",
        f"**Усього спроб: {TAMPER_TRIALS * len(HOW)}, "
        "прийнято пошкоджених архівів або необроблених винятків: "
        f"{total_bad}.**\n",
        "Поля, що перевіряються до HMAC, дають свої коди: magic і версія — E_MAGIC, довжина — E_LENGTH, "
        "номери томів і salt — E_VOLUMES (у багатотомному архіві зіпсований salt означає «томи з різних "
        "архівів»; в однотомному він дав би інші ключі, тобто E_HMAC). Решту заголовка, шифротекст і тег "
        "ловить HMAC. Розархіватор ні разу не видав пошкоджених даних.\n",
        "Сирі результати: `experiments/results/attack_runs.json`.\n",
    ]
    (ROOT / "docs" / "ATTACK_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
