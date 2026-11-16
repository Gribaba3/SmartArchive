"""LZ77: потік байтів -> токени (літерал або посилання «повернись на d, візьми l байтів»).

Пошук збігів — хеш-ланцюжки: для кожної позиції рахуємо хеш трьох байтів;
head[h] — остання позиція з таким хешем, prev[i] — попередня позиція з тим самим хешем.
Від поточної позиції йдемо ланцюжком назад не далі за max_chain кандидатів і не далі
за вікно. Отже, пошук у позиції коштує O(max_chain · довжина збігу), а не O(n).

Токен: int 0..255 — літерал; (довжина, відстань) — посилання.
Посилання не виходять за межі блоку: кожен блок кодується незалежно (так у еталоні).
"""
from dataclasses import dataclass

from .codec import MAX_MATCH, MIN_MATCH


@dataclass(frozen=True)
class LZParams:
    window: int = 16384      # максимальна відстань посилання
    min_match: int = 3       # коротші збіги не беремо (формат дозволяє від 3)
    max_chain: int = 64      # глибина пошуку в хеш-ланцюжку
    lazy_gain: int = 1       # лінивий розбір: наскільки довшим має бути збіг у i+1


class MatchFinder:
    def __init__(self, data: bytes, params: LZParams):
        self.data = data
        self.p = params
        self.head: dict[int, int] = {}
        self.prev = [-1] * len(data)
        self.inserted = 0  # позиції < inserted уже додані в ланцюжки

    def _hash(self, i: int) -> int:
        d = self.data
        return (d[i] << 16) | (d[i + 1] << 8) | d[i + 2]   # самі три байти — без колізій

    def insert_until(self, end: int) -> None:
        """Додати в ланцюжки всі позиції до end (не включно)."""
        last = len(self.data) - 2
        i = self.inserted
        while i < end:
            if i < last:
                h = self._hash(i)
                self.prev[i] = self.head.get(h, -1)
                self.head[h] = i
            i += 1
        self.inserted = max(self.inserted, end)

    def longest(self, i: int) -> tuple[int, int]:
        """Найдовший збіг для позиції i серед попередніх позицій: (довжина, відстань)."""
        d = self.data
        n = len(d)
        if i + MIN_MATCH > n:
            return 0, 0
        self.insert_until(i)
        limit = min(MAX_MATCH, n - i)
        best_len, best_dist = 0, 0
        cand = self.head.get(self._hash(i), -1)
        chain = self.p.max_chain
        while cand >= 0 and chain > 0 and i - cand <= self.p.window:
            # швидка перевірка: кандидат може дати довший збіг, лише якщо в нього збігається
            # і байт на позиції best_len (інакше його збіг точно не довший за поточний)
            could_be_longer = best_len < limit and d[cand + best_len] == d[i + best_len]
            if could_be_longer:
                k = 0
                while k < limit and d[cand + k] == d[i + k]:
                    k += 1
                if k > best_len:
                    best_len, best_dist = k, i - cand
                    if k == limit:
                        break
            cand = self.prev[cand]
            chain -= 1
        if best_len < max(self.p.min_match, MIN_MATCH):
            return 0, 0
        return best_len, best_dist


def greedy_parse(data: bytes, params: LZParams | None = None) -> list:
    """Жадібний розбір: у кожній позиції беремо найдовший знайдений збіг."""
    params = params or LZParams()
    mf = MatchFinder(data, params)
    tokens = []
    i, n = 0, len(data)
    while i < n:
        length, dist = mf.longest(i)
        if length:
            tokens.append((length, dist))
            i += length
        else:
            tokens.append(data[i])
            i += 1
    return tokens


def lazy_parse(data: bytes, params: LZParams | None = None) -> list:
    """Лінива жадібність: якщо в позиції i+1 є суттєво довший збіг (на lazy_gain і більше),
    у позиції i видаємо літерал і переходимо до i+1. Інакше беремо збіг у i, як жадібний."""
    params = params or LZParams()
    mf = MatchFinder(data, params)
    tokens = []
    i, n = 0, len(data)
    cur = mf.longest(0) if n else (0, 0)
    while i < n:
        length, dist = cur
        if not length:
            tokens.append(data[i])
            i += 1
            cur = mf.longest(i) if i < n else (0, 0)
            continue
        nxt = mf.longest(i + 1) if i + 1 < n else (0, 0)
        if nxt[0] >= length + params.lazy_gain:
            tokens.append(data[i])     # відкладаємо: кращий збіг починається на байт пізніше
            i += 1
            cur = nxt
        else:
            tokens.append((length, dist))
            i += length
            cur = mf.longest(i) if i < n else (0, 0)
    return tokens


def tokens_to_bytes(tokens: list) -> bytes:
    """Зворотне перетворення — для тестів."""
    out = bytearray()
    for t in tokens:
        if isinstance(t, int):
            out.append(t)
        else:
            n, d = t
            start = len(out) - d
            for k in range(n):
                out.append(out[start + k])
    return bytes(out)
