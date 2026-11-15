"""Канонічні коди Гаффмана: з довжин кодів відновлюються самі коди.

Побудова довжин за частотами (черга з пріоритетами) — build_lengths; декодування — Decoder.
"""
import heapq

from .bitio import BitReader
from .errors import E_PAYLOAD, ArchiveError

MAX_CODE_LEN = 15      # ліміт для НАШОГО кодера (⚠ припущення щодо специфікації)
DECODE_MAX_LEN = 32    # розархіватор приймає довші коди — для сумісності з іншими командами
TABLE_MAX_LEN = 15     # до цієї довжини декодуємо через таблицю, довші — по бітах


def canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    """{символ: довжина} -> {символ: (код, довжина)}.

    Правило: сортуємо за (довжина, символ); перший код 0; кожен наступний = попередній + 1,
    зсунутий вліво на різницю довжин. Тому в архів досить записати лише довжини.
    """
    codes = {}
    code = 0
    prev_len = 0
    for sym, ln in sorted(lengths.items(), key=lambda kv: (kv[1], kv[0])):
        code <<= ln - prev_len
        codes[sym] = (code, ln)
        code += 1
        prev_len = ln
    return codes


class Decoder:
    """Табличне декодування: беремо max_len бітів наперед, таблиця одразу дає (символ, довжина).

    Код довжини l займає в таблиці 2^(max_len - l) комірок — усі продовження цього префікса.
    Для дуже довгих кодів (> TABLE_MAX_LEN) таблиця завелика, тоді декодуємо по одному біту.
    """

    def __init__(self, lengths: dict[int, int]):
        self.max_len = 0
        self.table = None
        if not lengths:
            return
        if any(not 1 <= ln <= DECODE_MAX_LEN for ln in lengths.values()):
            raise ArchiveError(E_PAYLOAD, "некоректна довжина коду")
        # Нерівність Крафта: сума 2^-l <= 1, інакше коди перетинаються
        if sum(1 << (DECODE_MAX_LEN - ln) for ln in lengths.values()) > (1 << DECODE_MAX_LEN):
            raise ArchiveError(E_PAYLOAD, "довжини кодів порушують нерівність Крафта")
        self.max_len = max(lengths.values())
        codes = canonical_codes(lengths)
        if self.max_len <= TABLE_MAX_LEN:
            self.table = [None] * (1 << self.max_len)
            for sym, (code, ln) in codes.items():
                shift = self.max_len - ln
                start = code << shift
                self.table[start:start + (1 << shift)] = [(sym, ln)] * (1 << shift)
        else:
            self.by_code = {(ln, code): sym for sym, (code, ln) in codes.items()}

    def read(self, br: BitReader) -> int:
        if self.table is not None:
            entry = self.table[br.peek(self.max_len)]
            if entry is None:
                raise ArchiveError(E_PAYLOAD, "невідомий код Гаффмана")
            br.skip(entry[1])
            return entry[0]
        code = 0
        for ln in range(1, self.max_len + 1):
            code = (code << 1) | br.bit()
            sym = self.by_code.get((ln, code))
            if sym is not None:
                return sym
        raise ArchiveError(E_PAYLOAD, "невідомий код Гаффмана")


# ---------- побудова кодів (архіватор) ----------

def _huffman_lengths(freqs: dict[int, int]) -> dict[int, int]:
    """Класичний Гаффман: двічі беремо з черги два найрідші вузли і зливаємо їх.

    Жадібний вибір оптимальний: два найрідші символи в оптимальному дереві можна зробити
    братами на найглибшому рівні (обмін не збільшує сумарну довжину), після злиття задача
    зводиться до такої самої з алфавітом на один символ менше.
    Вузол у черзі: (частота, порядковий номер для стабільності, список символів піддерева).
    """
    heap = [(f, i, [s]) for i, (s, f) in enumerate(sorted(freqs.items()))]
    heapq.heapify(heap)
    depth = dict.fromkeys(freqs, 0)
    counter = len(heap)
    while len(heap) > 1:
        f1, _, syms1 = heapq.heappop(heap)
        f2, _, syms2 = heapq.heappop(heap)
        for s in syms1 + syms2:          # кожне злиття додає всім символам піддерева 1 біт
            depth[s] += 1
        heapq.heappush(heap, (f1 + f2, counter, syms1 + syms2))
        counter += 1
    return depth


def build_lengths(freqs: dict[int, int], max_len: int = MAX_CODE_LEN) -> dict[int, int]:
    """{символ: частота} -> {символ: довжина коду} з обмеженням довжини max_len.

    Один символ -> довжина 1 (код "0"), бо код нульової довжини записати неможливо.
    Якщо найдовший код перевищує ліміт, частоти «сплющуються» (f -> f//2 + 1) і дерево
    будується заново: рідкісні символи наближаються до частих, дерево стає нижчим.
    Це не оптимально в точності (точний метод — package-merge), але втрата мізерна.
    """
    freqs = {s: f for s, f in freqs.items() if f > 0}
    if not freqs:
        return {}
    if len(freqs) == 1:
        return {next(iter(freqs)): 1}
    if len(freqs) > (1 << max_len):
        raise ValueError("алфавіт завеликий для такого ліміту довжини")
    while True:
        lengths = _huffman_lengths(freqs)
        if max(lengths.values()) <= max_len:
            return lengths
        freqs = {s: f // 2 + 1 for s, f in freqs.items()}
