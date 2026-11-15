"""Розкладання закодованих блоків по томах: рюкзак 0/1 (ДП) або жадібний за v/w.

Для кожного тому:
  1. кандидати — усі ще не розміщені блоки; таблиця файлів тому = усі їхні файли,
     вона резервується ДО розв'язання рюкзака, тому місткість W відома заздалегідь;
  2. W = max_volume - 41 (заголовок) - 32 (HMAC) - 6 (лічильники payload) - таблиця файлів;
  3. вага блоку = coded_len + 15 (запис у таблиці блоків), цінність — пріоритет типу файлу;
  4. рюкзак обирає блоки максимальної сумарної цінності, решта йде в наступний том.

Формула з team_config віднімає ще 15·k від W — це те саме, що +15 у вазі кожного блоку;
рахуємо один раз, у вазі.
"""
from dataclasses import dataclass

from .container import HEADER_LEN, FileEntry, file_table_size
from .crypto import TAG_LEN
from .errors import E_CAPACITY, ArchiveError

PAYLOAD_COUNTERS = 6   # u16 file_count + u16 reserved + u16 block_count
BLOCK_ENTRY_LEN = 15


@dataclass
class Item:
    key: object       # будь-який ідентифікатор блоку
    weight: int
    value: int


def knapsack_dp(items: list[Item], capacity: int) -> list[Item]:
    """Точний рюкзак 0/1, O(n·W) за часом.

    best[w] — найбільша цінність, яку можна набрати вагою не більше w з уже розглянутих
    предметів. Для кожного предмета йдемо по w згори вниз, щоб він не взявся двічі:
        best[w] = max(best[w], best[w - weight] + value).
    take[i] запам'ятовує, при яких w предмет i покращив відповідь — за ним відновлюємо вибір.
    Пам'ять O(n·W) біт: по bytearray на предмет.
    """
    best = [0] * (capacity + 1)
    take = []
    for it in items:
        row = bytearray(capacity + 1)
        wt, val = it.weight, it.value
        for w in range(capacity, wt - 1, -1):
            cand = best[w - wt] + val
            if cand > best[w]:
                best[w] = cand
                row[w] = 1
        take.append(row)
    chosen = []
    w = capacity
    for i in range(len(items) - 1, -1, -1):   # відновлення: ідемо від останнього предмета
        if take[i][w]:
            chosen.append(items[i])
            w -= items[i].weight
    chosen.reverse()
    return chosen


def knapsack_greedy(items: list[Item], capacity: int) -> list[Item]:
    """Жадібний: за спаданням v/w кладемо все, що влазить. Швидко, але не завжди оптимально."""
    chosen, left = [], capacity
    for it in sorted(items, key=lambda x: x.value / x.weight, reverse=True):
        if it.weight <= left:
            chosen.append(it)
            left -= it.weight
    return chosen


@dataclass
class PlannedBlock:
    file_name: str
    block_index: int
    coded_len: int
    value: int


def volume_capacity(max_volume: int, files: list[FileEntry]) -> int:
    return max_volume - HEADER_LEN - TAG_LEN - PAYLOAD_COUNTERS - file_table_size(files)


def plan_volumes(blocks: list[PlannedBlock], files: dict[str, FileEntry], max_volume: int,
                 solver=knapsack_dp) -> list[list[PlannedBlock]]:
    """Ділить блоки на томи. Повертає список томів, кожен — список блоків."""
    remaining = list(blocks)
    volumes = []
    while remaining:
        cand_files = list(dict.fromkeys(b.file_name for b in remaining))
        capacity = volume_capacity(max_volume, [files[n] for n in cand_files])
        if capacity <= 0:
            raise ArchiveError(E_CAPACITY, "таблиця файлів не вміщується в порожній том: "
                                           "збільште розмір тому або архівуйте менше файлів")
        items = [Item(b, b.coded_len + BLOCK_ENTRY_LEN, b.value) for b in remaining]
        too_big = [it for it in items if it.weight > capacity]
        if too_big:
            raise ArchiveError(E_CAPACITY, f"блок {too_big[0].key.file_name}#{too_big[0].key.block_index} "
                                           "більший за місткість тому: зменште block_size "
                                           "або збільште розмір тому")
        chosen = {id(it.key) for it in solver(items, capacity)}
        volumes.append([b for b in remaining if id(b) in chosen])
        remaining = [b for b in remaining if id(b) not in chosen]
    return volumes
