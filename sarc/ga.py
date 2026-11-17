"""Генетичний алгоритм і random search для підбору параметрів (методичка, 4.14, 10.57–10.60).

Хромосома — кортеж ІНДЕКСІВ у списках допустимих значень генів (GENES у experiments/run_ga.py):
    гени   = [window, min_match, поріг NO_COMPRESSION, max_chain]
    хромосома (2, 0, 7, 3)  ->  window = WINDOWS[2], min_match = MIN_MATCHES[0], ...
Через індекси мутація і схрещування не можуть дати значення поза допустимими межами.

Fitness максимізується (у нас fitness = Q). Обидва методи отримують однаковий БЮДЖЕТ —
кількість різних хромосом, для яких реально рахувався fitness. Повтор уже оціненої хромосоми
бере значення з кешу і бюджет не витрачає: так порівняння ГА з random search чесне (10.59).

Один крок ГА:
    1. selection  — турнір: беремо TOURNAMENT випадкових особин, перемагає найкраща;
    2. crossover  — рівномірний: кожен ген дитини від випадкового з двох батьків;
    3. mutation   — кожен ген з імовірністю p_mut зсувається на ±1 або ±2 позиції в списку значень;
    4. elitism    — ELITE найкращих переходять у нове покоління без змін,
                    тому найкращий fitness між поколіннями не зменшується.
"""
import random
from dataclasses import dataclass, field
from typing import Callable

Chromosome = tuple[int, ...]


@dataclass
class GAConfig:
    population: int = 10
    tournament: int = 3
    p_crossover: float = 0.9
    p_mutation: float = 0.25   # імовірність мутації КОЖНОГО гена
    elite: int = 2
    # "shift" — зсув на ±1..2 позиції (значення генів упорядковані: вікна, пороги);
    # "reset" — нове випадкове значення (значення не впорядковані: байти ключа, де 0x41 не «ближче» до 0x42)
    mutation: str = "shift"
    max_stalled: int = 50   # стільки поколінь поспіль без жодної нової хромосоми -> зупинка


@dataclass
class SearchResult:
    best: Chromosome
    best_fitness: float
    evaluations: int                                   # скільки разів реально рахувався fitness
    history: list[dict] = field(default_factory=list)  # по поколіннях (ГА) або по оцінках (random)


class Evaluator:
    """Обгортка над fitness: кеш + лічильник бюджету.

    Кеш можна передати спільний для кількох запусків (наприклад, ГА і random search):
    fitness однієї хромосоми тоді рахується один раз, а кожен запуск окремо лічить СВОЇ оцінки.
    """

    def __init__(self, fitness: Callable[[Chromosome], float], budget: int, cache: dict = None):
        self.fitness = fitness
        self.budget = budget
        self.cache = {} if cache is None else cache
        self.seen: set = set()          # хромосоми, оцінені в ЦЬОМУ запуску
        self.evaluations = 0

    def exhausted(self) -> bool:
        return self.evaluations >= self.budget

    def __call__(self, ch: Chromosome) -> float:
        if ch not in self.seen:
            self.seen.add(ch)
            self.evaluations += 1
        if ch not in self.cache:
            self.cache[ch] = self.fitness(ch)
        return self.cache[ch]


# ---------- оператори ----------

def random_chromosome(sizes: list[int], rng: random.Random) -> Chromosome:
    """sizes[i] — кількість допустимих значень i-го гена."""
    return tuple(rng.randrange(n) for n in sizes)


def tournament_select(pop: list[Chromosome], fit: dict, k: int, rng: random.Random) -> Chromosome:
    """k випадкових особин, перемагає з найбільшим fitness."""
    return max(rng.sample(pop, min(k, len(pop))), key=lambda c: fit[c])


def uniform_crossover(a: Chromosome, b: Chromosome, rng: random.Random) -> Chromosome:
    """Кожен ген дитини береться або від a, або від b з імовірністю 1/2."""
    return tuple(x if rng.random() < 0.5 else y for x, y in zip(a, b, strict=True))


def mutate(ch: Chromosome, sizes: list[int], p: float, rng: random.Random,
           mode: str = "shift") -> Chromosome:
    """Кожен ген з імовірністю p змінюється.

    mode="shift": зсув на ±1 або ±2 позиції з обрізанням до меж. Підходить, коли значення генів
    упорядковані (вікно 4096 «ближче» до 8192, ніж до 512): малий крок досліджує околицю рішення.
    mode="reset": нове випадкове значення, відмінне від поточного. Підходить, коли порядку немає.
    """
    out = []
    for g, n in zip(ch, sizes, strict=True):
        if rng.random() < p:
            if mode == "reset" and n > 1:
                g = (g + rng.randrange(1, n)) % n     # будь-яке інше значення, рівноймовірно
            elif mode == "shift":
                g = min(n - 1, max(0, g + rng.choice((-2, -1, 1, 2))))
        out.append(g)
    return tuple(out)


# ---------- алгоритми ----------

def genetic_algorithm(sizes: list[int], evaluate: Evaluator, cfg: GAConfig = None,
                      seed: int = 0) -> SearchResult:
    """ГА до вичерпання бюджету evaluate.budget. Повертає найкращу хромосому та історію поколінь."""
    cfg = cfg or GAConfig()
    rng = random.Random(seed)
    pop = [random_chromosome(sizes, rng) for _ in range(cfg.population)]
    fit = {c: evaluate(c) for c in pop}
    history = []
    generation = 0
    stalled = 0
    while True:
        ranked = sorted(pop, key=lambda c: fit[c], reverse=True)
        history.append({"generation": generation, "evaluations": evaluate.evaluations,
                        "best": fit[ranked[0]], "mean": sum(fit[c] for c in pop) / len(pop),
                        "best_chromosome": list(ranked[0])})
        if evaluate.exhausted() or stalled > cfg.max_stalled:
            break   # stalled: популяція зійшлася і нових хромосом не з'являється — далі марно
        before = evaluate.evaluations
        children = ranked[:cfg.elite]                     # елітизм
        while len(children) < cfg.population and not evaluate.exhausted():
            a = tournament_select(pop, fit, cfg.tournament, rng)
            b = tournament_select(pop, fit, cfg.tournament, rng)
            child = uniform_crossover(a, b, rng) if rng.random() < cfg.p_crossover else a
            child = mutate(child, sizes, cfg.p_mutation, rng, cfg.mutation)
            fit[child] = evaluate(child)
            children.append(child)
        pop = children
        generation += 1
        stalled = stalled + 1 if evaluate.evaluations == before else 0
    best = max(fit, key=fit.get)
    return SearchResult(best, fit[best], evaluate.evaluations, history)


def random_search(sizes: list[int], evaluate: Evaluator, seed: int = 0) -> SearchResult:
    """Випадкові хромосоми до вичерпання того самого бюджету.

    Генератор зсунутий відносно ГА, щоб з однаковим seed методи не стартували з тих самих хромосом.
    """
    rng = random.Random(seed + 1_000_003)
    best, best_fit, history = None, float("-inf"), []
    total = 1
    for n in sizes:
        total *= n
    while not evaluate.exhausted() and len(evaluate.seen) < total:
        ch = random_chromosome(sizes, rng)
        f = evaluate(ch)
        if f > best_fit:
            best, best_fit = ch, f
        history.append({"evaluations": evaluate.evaluations, "best": best_fit})
    return SearchResult(best, best_fit, evaluate.evaluations, history)
