"""Канонічні коди Гаффмана: з довжин кодів відновлюються самі коди.

Тут — спільна частина для архіватора і розархіватора. Побудова довжин за частотами
(черга з пріоритетами) — зона «Стиснення», додається окремим модулем.
"""
from .bitio import BitReader
from .errors import E_PAYLOAD, ArchiveError

MAX_CODE_LEN = 15      # ліміт для НАШОГО кодера (⚠ припущення щодо специфікації)
DECODE_MAX_LEN = 32    # розархіватор приймає довші коди — для сумісності з іншими командами


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
    """Канонічне декодування без дерева: для кожної довжини знаємо перший код і кількість кодів."""

    def __init__(self, lengths: dict[int, int]):
        if not lengths:
            self.max_len = 0
            return
        if any(not 1 <= ln <= DECODE_MAX_LEN for ln in lengths.values()):
            raise ArchiveError(E_PAYLOAD, "некоректна довжина коду")
        # Нерівність Крафта: сума 2^-l <= 1, інакше коди перетинаються
        if sum(1 << (DECODE_MAX_LEN - ln) for ln in lengths.values()) > (1 << DECODE_MAX_LEN):
            raise ArchiveError(E_PAYLOAD, "довжини кодів порушують нерівність Крафта")
        self.max_len = max(lengths.values())
        ordered = sorted(lengths.items(), key=lambda kv: (kv[1], kv[0]))
        self.symbols = [s for s, _ in ordered]
        self.count = [0] * (self.max_len + 1)
        for _, ln in ordered:
            self.count[ln] += 1
        self.first_code = [0] * (self.max_len + 1)
        self.first_index = [0] * (self.max_len + 1)
        code = index = 0
        for ln in range(1, self.max_len + 1):
            code <<= 1
            self.first_code[ln] = code
            self.first_index[ln] = index
            code += self.count[ln]
            index += self.count[ln]

    def read(self, br: BitReader) -> int:
        code = 0
        for ln in range(1, self.max_len + 1):
            code = (code << 1) | br.bit()
            offset = code - self.first_code[ln]
            if offset < self.count[ln]:
                return self.symbols[self.first_index[ln] + offset]
        raise ArchiveError(E_PAYLOAD, "невідомий код Гаффмана")
