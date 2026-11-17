"""LZ77: потік байтів -> токени (літерал або посилання «повернись на d, візьми l байтів»).

Пошук збігів — хеш-ланцюжки: для кожної позиції рахуємо хеш трьох байтів;
head[h] — остання позиція з таким хешем, prev[i] — попередня позиція з тим самим хешем.
Від поточної позиції йдемо ланцюжком назад не далі за max_chain кандидатів і не далі
за вікно. Отже, пошук у позиції коштує O(max_chain · довжина збігу), а не O(n).

Токен: int 0..255 — літерал; (довжина, відстань) — посилання.
Посилання не виходять за межі блоку: кожен блок кодується незалежно (так у еталоні).
"""
from dataclasses import dataclass

from .codec import MAX_MATCH, MIN_MATCH, WINDOW


@dataclass(frozen=True)
class LZParams:
    # window і min_match підібрав ГА (docs/GA_RESULTS.md, DECISIONS Р17); до ГА було 16384 і 3
    window: int = 4096       # максимальна відстань посилання
    min_match: int = 5       # коротші збіги не беремо (формат дозволяє від 3)
    max_chain: int = 64      # глибина пошуку в хеш-ланцюжку
    lazy_gain: int = 1       # лінивий розбір: наскільки довшим має бути збіг у i+1

    def __post_init__(self):
        # Межі формату: інакше архіватор створить архів, який жоден розархіватор не відкриє.
        # Важливо для ГА: мутація не повинна виводити параметри за ці межі.
        if not 1 <= self.window <= WINDOW:
            raise ValueError(f"window має бути від 1 до {WINDOW}")
        if not MIN_MATCH <= self.min_match <= MAX_MATCH:
            raise ValueError(f"min_match має бути від {MIN_MATCH} до {MAX_MATCH}")


class MatchFinder:
    def __init__(self, data: bytes, params: LZParams):
        self.data = data
        self.p = params
        self.head: dict[int, int] = {}
        self.prev = [-1] * len(data)
        self.inserted = 0  # позиції < inserted уже додані в ланцюжки

    def longest(self, i: int) -> tuple[int, int]:
        """Найдовший збіг для позиції i серед попередніх позицій: (довжина, відстань).

        Функція викликається для кожної позиції (на корпусі ~130 тис. разів), тому тут важить не
        алгоритм, а накладні витрати Python на виклики: додавання позицій у ланцюжки і хеш
        вбудовані сюди, а атрибути об'єкта скопійовані в локальні змінні (вони читаються швидше).
        """
        d = self.data
        n = len(d)
        if i + MIN_MATCH > n:
            return 0, 0
        head, prev = self.head, self.prev
        # додаємо в ланцюжки всі позиції до i (не включно); хеш = самі три байти, без колізій
        if self.inserted < i:
            for j in range(self.inserted, min(i, n - 2)):
                h = (d[j] << 16) | (d[j + 1] << 8) | d[j + 2]
                prev[j] = head.get(h, -1)
                head[h] = j
            self.inserted = i
        limit = min(MAX_MATCH, n - i)
        best_len, best_dist = 0, 0
        cand = head.get((d[i] << 16) | (d[i + 1] << 8) | d[i + 2], -1)
        chain = self.p.max_chain
        window = self.p.window
        while cand >= 0 and chain > 0 and i - cand <= window:
            # швидка перевірка: кандидат може дати довший збіг, лише якщо в нього збігається
            # і байт на позиції best_len (інакше його збіг точно не довший за поточний)
            if best_len < limit and d[cand + best_len] == d[i + best_len]:
                k = 0
                while k < limit and d[cand + k] == d[i + k]:
                    k += 1
                if k > best_len:
                    best_len, best_dist = k, i - cand
                    if k == limit:
                        break
            cand = prev[cand]
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
