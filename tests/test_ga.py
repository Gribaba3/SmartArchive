"""Генетичний алгоритм: тести по частинах (методичка, 6.14–6.15).

Тут fitness штучний і швидкий: так перевіряються самі оператори ГА, без запуску архіватора.
"""
import random
from collections import Counter

import pytest

from sarc.ga import (
    Evaluator,
    GAConfig,
    genetic_algorithm,
    mutate,
    random_chromosome,
    random_search,
    tournament_select,
    uniform_crossover,
)

SIZES = [7, 6, 8, 7]


def peak(ch):
    """Штучний fitness з одним максимумом у точці (5, 1, 7, 2): чим ближче, тим більше."""
    target = (5, 1, 7, 2)
    return -sum(abs(a - b) for a, b in zip(ch, target, strict=True))


# ---------- генерація популяції ----------

def test_random_chromosomes_within_bounds():
    rng = random.Random(0)
    pop = [random_chromosome(SIZES, rng) for _ in range(500)]
    assert len(pop) == 500
    assert all(len(c) == len(SIZES) for c in pop)
    assert all(0 <= g < n for c in pop for g, n in zip(c, SIZES, strict=True))
    # кожне значення кожного гена хоч раз трапилося — генерація покриває весь простір
    for i, n in enumerate(SIZES):
        assert {c[i] for c in pop} == set(range(n))


# ---------- selection ----------

def test_tournament_prefers_better():
    """Особини з fitness 0..9: у турнірі з 3 найкраща (9) має вигравати частіше за найгіршу (0)."""
    pop = [(i,) for i in range(10)]
    fit = {c: c[0] for c in pop}
    rng = random.Random(1)
    wins = Counter(tournament_select(pop, fit, 3, rng) for _ in range(5000))
    assert wins[(9,)] > wins[(5,)] > wins[(0,)]
    assert wins[(0,)] == 0 and wins[(1,)] == 0    # дві найгірші не можуть перемогти в турнірі з 3


# ---------- crossover ----------

def test_crossover_takes_genes_from_parents():
    a, b = (0, 0, 0, 0), (6, 5, 7, 6)
    rng = random.Random(2)
    children = [uniform_crossover(a, b, rng) for _ in range(200)]
    assert all(g in (x, y) for c in children for g, x, y in zip(c, a, b, strict=True))
    # рівномірне схрещування: трапляються діти, змішані з обох батьків
    assert any(c not in (a, b) for c in children)


# ---------- mutation ----------

def test_mutation_sometimes_happens_and_stays_in_bounds():
    rng = random.Random(3)
    edge = (0, 5, 0, 6)                       # гени на межах — найнебезпечніший випадок
    mutated = [mutate(edge, SIZES, 0.5, rng) for _ in range(1000)]
    assert any(m != edge for m in mutated)
    assert all(0 <= g < n for m in mutated for g, n in zip(m, SIZES, strict=True))


def test_mutation_probability_zero_and_one():
    rng = random.Random(4)
    ch = (3, 3, 3, 3)
    assert mutate(ch, SIZES, 0.0, rng) == ch
    assert all(g != 3 for g in mutate(ch, SIZES, 1.0, rng))   # зсув на ±1..2 завжди змінює ген


# ---------- бюджет ----------

def test_evaluator_counts_only_new_chromosomes():
    calls = []
    ev = Evaluator(lambda c: calls.append(c) or 1.0, budget=3)
    for ch in [(0,), (0,), (1,), (0,), (2,)]:
        ev(ch)
    assert ev.evaluations == 3 and len(calls) == 3 and ev.exhausted()


def test_shared_cache_does_not_share_budget():
    """Спільний кеш: fitness не рахується вдруге, але кожен запуск лічить свої оцінки."""
    calls, cache = [], {}
    a = Evaluator(lambda c: calls.append(c) or 1.0, 5, cache)
    b = Evaluator(lambda c: calls.append(c) or 1.0, 5, cache)
    a((1,))
    b((1,))
    assert len(calls) == 1 and a.evaluations == b.evaluations == 1


@pytest.mark.parametrize("method", ["ga", "random"])
def test_budget_is_respected(method):
    ev = Evaluator(peak, budget=40)
    if method == "ga":
        res = genetic_algorithm(SIZES, ev, GAConfig(), seed=5)
    else:
        res = random_search(SIZES, ev, seed=5)
    assert res.evaluations == 40


# ---------- elitism і результат ----------

@pytest.mark.parametrize("seed", range(5))
def test_elitism_best_never_decreases(seed):
    """Методичка, 6.14: best_final >= best_initial; у нас — між БУДЬ-ЯКИМИ сусідніми поколіннями."""
    res = genetic_algorithm(SIZES, Evaluator(peak, 120), GAConfig(), seed=seed)
    bests = [h["best"] for h in res.history]
    assert all(a <= b for a, b in zip(bests, bests[1:], strict=False))
    assert bests[-1] >= bests[0]


def test_without_elitism_best_can_be_lost():
    """Контрольний тест: без еліти і з сильною мутацією найкраща особина поколінь іноді губиться.
    Показує, що тест вище справді перевіряє елітизм, а не проходить сам собою."""
    lost = False
    for seed in range(20):
        cfg = GAConfig(elite=0, p_mutation=0.9)
        res = genetic_algorithm(SIZES, Evaluator(peak, 150), cfg, seed=seed)
        bests = [h["best"] for h in res.history]
        lost = lost or any(b < a for a, b in zip(bests, bests[1:], strict=False))
    assert lost


def test_ga_finds_peak_and_beats_random_on_average():
    """На гладкому fitness ГА з бюджетом 60 частіше доходить до максимуму, ніж random search."""
    ga = [genetic_algorithm(SIZES, Evaluator(peak, 60), GAConfig(), seed=s).best_fitness for s in range(10)]
    rs = [random_search(SIZES, Evaluator(peak, 60), seed=s).best_fitness for s in range(10)]
    assert max(ga) == 0                       # максимум (точка (5, 1, 7, 2)) знайдено
    assert sum(ga) / 10 > sum(rs) / 10


def test_same_seed_same_result():
    a = genetic_algorithm(SIZES, Evaluator(peak, 50), GAConfig(), seed=9)
    b = genetic_algorithm(SIZES, Evaluator(peak, 50), GAConfig(), seed=9)
    assert a.best == b.best and a.history == b.history


def test_failed_restore_is_worst_fitness(monkeypatch):
    """Методичка, 10.58: хромосома, що ламає відновлення, має найгірший fitness (Q = 0)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    import measure

    from sarc.archive import Options

    real = measure.extract

    def broken_extract(volumes, password, out):   # «відновлює» файли з пошкодженим останнім байтом
        names = real(volumes, password, out)
        for n in names:
            p = Path(out, n)
            p.write_bytes(p.read_bytes()[:-1] + b"?")
        return names

    monkeypatch.setattr(measure, "extract", broken_extract)
    files = [p for p in measure.CORPUS if p.stat().st_size > 0][:2]
    assert measure.measure_q(Options(), files=files, runs=1)["Q"] == 0.0
