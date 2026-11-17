"""ГА проти random search на корпусі (методичка, 10.56–10.60) і звіт docs/GA_RESULTS.md.

    python experiments/run_ga.py               # ~15–20 хв; кеш оцінок дозволяє перервати і продовжити
    python experiments/run_ga.py --budget 40   # швидка перевірка

Fitness хромосоми = Q на corpus/ (experiments/measure.py): повний цикл архівація -> розпакування,
побайтова перевірка (не відновився файл -> Q = 0, найгірший fitness), t — медіана з FIT_RUNS прогонів.
Fitness однієї хромосоми рахується один раз і кешується (experiments/results/ga_cache.json):
ГА і random search беруть значення з одного кешу, тож порівнюються на однакових числах.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from measure import CORPUS, Q_CFG, _one_run, measure_q  # noqa: E402

from sarc.archive import Options  # noqa: E402
from sarc.ga import Evaluator, GAConfig, genetic_algorithm, random_search  # noqa: E402
from sarc.lz77 import LZParams  # noqa: E402

# Допустимі значення генів. Вікно не більше блоку (16384): LZ77-посилання не виходять за блок,
# тож 32768 стискало б так само, як 16384.
WINDOWS = [256, 512, 1024, 2048, 4096, 8192, 16384]
MIN_MATCHES = [3, 4, 5, 6, 7, 8]
THRESHOLDS = [7.0, 7.5, 7.8, 7.9, 7.95, 7.97, 7.99, 8.0]   # ентропія блоку, вище — STORED; 8.0 = вимкнено
CHAINS = [4, 8, 16, 32, 64, 128, 256]                      # глибина пошуку збігу в хеш-ланцюжку
GENES = [("window", WINDOWS), ("min_match", MIN_MATCHES), ("threshold", THRESHOLDS), ("max_chain", CHAINS)]
SIZES = [len(v) for _, v in GENES]
# Параметри, що були типовими ДО ГА (після ГА типовими стали параметри переможця, див. lz77.py)
BASELINE = (WINDOWS.index(16384), MIN_MATCHES.index(3), THRESHOLDS.index(8.0), CHAINS.index(64))

FIT_RUNS = 3      # медіана часу для fitness
FINAL_RUNS = 15   # перевимір переможця і baseline: почергово, щоб фоновий шум діяв на обох однаково
SEEDS = (1, 2, 3)
CACHE = ROOT / "experiments" / "results" / "ga_cache.json"


def decode(ch) -> dict:
    return {name: values[i] for (name, values), i in zip(GENES, ch, strict=True)}


def options(ch) -> Options:
    p = decode(ch)
    return Options(lz=LZParams(window=p["window"], min_match=p["min_match"], max_chain=p["max_chain"]),
                   entropy_threshold=p["threshold"])


def fitness(ch) -> float:
    return measure_q(options(ch), runs=FIT_RUNS)["Q"]


def load_cache() -> dict:
    if CACHE.exists():
        return {tuple(json.loads(k)): v for k, v in json.loads(CACHE.read_text()).items()}
    return {}


def save_cache(cache: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({json.dumps(list(k)): v for k, v in sorted(cache.items())}))


def measure_interleaved(chromosomes, runs: int) -> dict:
    """Q кожної хромосоми за медіаною часу з runs прогонів; прогони чергуються (A, B, A, B, ...),
    тож повільніший момент машини припадає на обидві конфігурації, а не на одну."""
    times = {c: [] for c in chromosomes}
    out = {}
    for _ in range(runs):
        for c in chromosomes:
            res, t, ok = _one_run(options(c), CORPUS)
            times[c].append(t)
            out[c] = (res, ok and out.get(c, (None, True))[1])
    result = {}
    for c, (res, ok) in out.items():
        t = statistics.median(times[c])
        b_orig, b_arc = res.original_bytes, res.archive_bytes
        s = b_orig / (b_orig + b_arc)
        tt = Q_CFG["t_ref_секунд"] / (Q_CFG["t_ref_секунд"] + t)
        q = Q_CFG["alpha"] * s + Q_CFG["beta"] * tt if ok else 0.0
        result[c] = dict(orig=b_orig, arc=b_arc, ratio=b_orig / b_arc, S=s, T=tt, t=t,
                         t_min=min(times[c]), t_max=max(times[c]), Q=q, ok=ok)
    return result


def best_at(history: list[dict], evals: int) -> float:
    """Найкращий fitness, відомий після evals оцінок (для кривих збіжності)."""
    vals = [h["best"] for h in history if h["evaluations"] <= evals]
    return max(vals) if vals else float("nan")


def fmt(ch) -> str:
    p = decode(ch)
    return f"{p['window']} / {p['min_match']} / {p['threshold']} / {p['max_chain']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--report", default=str(ROOT / "docs" / "GA_RESULTS.md"))
    ap.add_argument("--raw", default=str(ROOT / "experiments" / "results" / "ga_runs.json"))
    args = ap.parse_args()

    cache = load_cache()
    cfg = GAConfig()
    runs = {"ga": [], "random": []}
    for seed in SEEDS:
        for method in ("ga", "random"):
            ev = Evaluator(fitness, args.budget, cache)
            if method == "ga":
                res = genetic_algorithm(SIZES, ev, cfg, seed=seed)
            else:
                res = random_search(SIZES, ev, seed=seed)
            runs[method].append({"seed": seed, "best": list(res.best), "best_fitness": res.best_fitness,
                                 "evaluations": res.evaluations, "history": res.history})
            save_cache(cache)
            print(f"{method} seed {seed}: Q = {res.best_fitness:.5f} {fmt(res.best)} "
                  f"({res.evaluations} оцінок, у кеші {len(cache)})", flush=True)

    # Переможець за кешованим fitness, потім чесний перевимір разом із baseline
    winner = max((tuple(r["best"]) for m in runs.values() for r in m), key=lambda c: cache[c])
    final = measure_interleaved([BASELINE, winner], FINAL_RUNS)
    base_m, best_m = final[BASELINE], final[winner]

    # Елітизм: найкращий fitness ГА не спадає між поколіннями
    elitism_ok = all(all(a["best"] <= b["best"] for a, b in zip(r["history"], r["history"][1:], strict=False))
                     for r in runs["ga"])

    checkpoints = [c for c in (10, 25, 50, 100, 150, 200, args.budget) if c <= args.budget]
    checkpoints = sorted(set(checkpoints))
    curve = {m: [statistics.mean(best_at(r["history"], c) for r in runs[m]) for c in checkpoints]
             for m in runs}
    top = sorted(cache.items(), key=lambda kv: kv[1], reverse=True)[:5]

    raw = {"budget": args.budget, "ga_config": cfg.__dict__, "genes": {n: v for n, v in GENES},
           "baseline": {"chromosome": list(BASELINE), **base_m},
           "winner": {"chromosome": list(winner), **best_m}, "elitism_ok": elitism_ok, "runs": runs}
    Path(args.raw).write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")

    def row(name, m):
        return (f"| {name} | {m['arc']} | {m['ratio']:.3f} | {m['t']:.2f} "
                f"({m['t_min']:.2f}–{m['t_max']:.2f}) | {m['S']:.5f} | {m['T']:.5f} | **{m['Q']:.5f}** |")

    lines = [
        "# Генетичний алгоритм проти random search (згенеровано experiments/run_ga.py)\n",
        "Хромосома: вікно LZ77 / мінімальна довжина збігу / поріг ентропії для NO_COMPRESSION / "
        "глибина пошуку в хеш-ланцюжку. "
        f"Простір: {' × '.join(str(s) for s in SIZES)} = {math.prod(SIZES)} конфігурацій.\n",
        f"Fitness = Q на corpus/ (повний цикл, побайтова перевірка, медіана часу з {FIT_RUNS} прогонів). "
        f"Бюджет кожного запуску — {args.budget} різних оцінок fitness, для ГА і random search однаковий. "
        f"ГА: популяція {cfg.population}, турнір {cfg.tournament}, рівномірне схрещування "
        f"(p = {cfg.p_crossover}), мутація кожного гена p = {cfg.p_mutation} (зсув на ±1..2), "
        f"еліта {cfg.elite}.\n",
        "## 1. Baseline і найкраща знайдена конфігурація\n",
        f"Обидві перевиміряні {FINAL_RUNS} прогонами почергово (медіана часу).\n",
        "| Конфігурація | Архів, Б | Коефіцієнт | t, с (мін–макс) | S | T | Q |",
        "|---|---:|---:|---:|---:|---:|---:|",
        row(f"baseline до ГА {fmt(BASELINE)}", base_m),
        row(f"найкраща {fmt(winner)}", best_m),
        "",
        f"Приріст Q: {best_m['Q'] - base_m['Q']:+.5f}. Під час пошуку ця конфігурація мала Q = "
        f"{cache[winner]:.5f}: значення з пошуку завищене, бо серед сотень зашумлених вимірів "
        f"найкращий — це завжди трохи «щасливий» прогін за часом. Тому порівнювати з baseline можна "
        f"лише після перевиміру.\n",
        "**Чому приріст малий.** На корпусі t ≈ 0.5 с при t_ref = 30 с, тож T ≈ 0.98 майже не залежить "
        "від параметрів, і Q визначає переважно стиснення S. А S між розумними конфігураціями "
        "відрізняється на десяті частки відсотка. Шум часу (±0.05 с) змінює Q приблизно на ±0.0005, "
        "тобто на величину того ж порядку, що й різниця між сусідніми конфігураціями.\n",
        "## 2. ГА проти random search\n",
        "| Seed | ГА: найкращий Q | ГА: конфігурація | Random: найкращий Q | Random: конфігурація |",
        "|---|---:|---|---:|---|",
    ]
    for g, r in zip(runs["ga"], runs["random"], strict=True):
        lines.append(f"| {g['seed']} | {g['best_fitness']:.5f} | {fmt(g['best'])} | "
                     f"{r['best_fitness']:.5f} | {fmt(r['best'])} |")
    lines += [
        f"| **середнє** | **{statistics.mean(r['best_fitness'] for r in runs['ga']):.5f}** | | "
        f"**{statistics.mean(r['best_fitness'] for r in runs['random']):.5f}** | |",
        "",
        "**Збіжність** — найкращий Q після N оцінок (середнє за трьома seed):\n",
        "| N оцінок | " + " | ".join(str(c) for c in checkpoints) + " |",
        "|---|" + "---:|" * len(checkpoints),
        "| ГА | " + " | ".join(f"{v:.5f}" for v in curve["ga"]) + " |",
        "| Random | " + " | ".join(f"{v:.5f}" for v in curve["random"]) + " |",
        "",
        "Значення в цьому розділі — fitness із пошуку (з кешу), однакові для обох методів. "
        "Різниця між ГА і random search того ж порядку, що й шум fitness, тож перевага ГА тут — "
        "у стабільності (три seed — один результат) і швидшій збіжності, а не у великому прирості Q.\n",
        f"**Елітизм:** найкращий fitness ГА не спадає між поколіннями — "
        f"{'так, в усіх запусках' if elitism_ok else 'НІ, помилка'}.\n",
        "## 3. П'ять найкращих конфігурацій серед усіх оцінених\n",
        f"Оцінено різних конфігурацій: {len(cache)} з {math.prod(SIZES)}.\n",
        "| Конфігурація | Q |", "|---|---:|",
    ] + [f"| {fmt(c)} | {q:.5f} |" for c, q in top] + [
        "",
        "Сирі результати (історія кожного покоління, усі оцінки): `experiments/results/ga_runs.json`, "
        "`experiments/results/ga_cache.json`.\n",
    ]
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
