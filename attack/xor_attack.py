"""Атака на навчальний XOR-шифр: довжина ключа за індексом збігу, байти ключа — генетичним алгоритмом.

Хромосома = ключ: k генів, кожен — байт 0..255. Fitness = BigramModel.score(xor(шифротекст, ключ)):
наскільки результат розшифрування схожий на текст. Мутація "reset" (новий випадковий байт), бо
байти ключа не впорядковані: 0x41 не «ближче» до правильного 0x42, ніж 0xF0.
"""
from dataclasses import dataclass

from sarc.ga import Evaluator, GAConfig, genetic_algorithm, random_search

from .text_model import BigramModel
from .weak_cipher import guess_key_length, xor

SAMPLE = 3000   # для fitness досить перших 3000 байтів шифротексту: так оцінка в рази швидша


@dataclass
class AttackResult:
    key_length: int          # знайдена довжина ключа
    key: bytes               # знайдений ключ
    fitness: float
    evaluations: int
    history: list


def ga_config(k: int) -> GAConfig:
    """Популяція 50, у середньому один мутований байт на дитину (p = 1/k)."""
    return GAConfig(population=50, tournament=3, p_crossover=0.9, p_mutation=1 / k, elite=2,
                    mutation="reset", max_stalled=1000)


def attack(ciphertext: bytes, model: BigramModel, budget: int, method: str = "ga", seed: int = 0,
           key_length: int = None) -> AttackResult:
    """Відновити ключ. key_length=None — визначити довжину за індексом збігу."""
    k = key_length or guess_key_length(ciphertext)[0]
    sample = ciphertext[:SAMPLE]
    ev = Evaluator(lambda ch: model.score(xor(sample, bytes(ch))), budget)
    if method == "ga":
        res = genetic_algorithm([256] * k, ev, ga_config(k), seed=seed)
    else:
        res = random_search([256] * k, ev, seed=seed)
    return AttackResult(k, bytes(res.best), res.best_fitness, res.evaluations, res.history)


def key_accuracy(found: bytes, true: bytes) -> float:
    """Частка правильно відновлених байтів ключа (0, якщо довжина не збіглася)."""
    if len(found) != len(true):
        return 0.0
    return sum(a == b for a, b in zip(found, true, strict=True)) / len(true)


def text_accuracy(ciphertext: bytes, found: bytes, plaintext: bytes) -> float:
    """Частка байтів відкритого тексту, відновлених правильно знайденим ключем."""
    got = xor(ciphertext, found)
    return sum(a == b for a, b in zip(got, plaintext, strict=True)) / len(plaintext)
