"""Розкладання закодованих блоків по томах: рюкзак 0/1 (ДП) або жадібний за v/w.

Для кожного тому:
  1. кандидати — усі ще не розміщені блоки; таблиця файлів тому = усі їхні файли,
     вона резервується ДО розв'язання рюкзака, тому місткість W відома заздалегідь;
  2. W = max_volume - 41 (заголовок) - 32 (HMAC) - 6 (лічильники payload) - таблиця файлів;
  3. вага блоку = coded_len + 15 (запис у таблиці блоків), цінність — пріоритет типу файлу
     (team_config: "пріоритети_типів");
  4. рюкзак обирає блоки максимальної сумарної цінності, решта йде в наступний том.

Формула з team_config віднімає ще 15·k від W — це те саме, що +15 у вазі кожного блоку;
рахуємо один раз, у вазі.

Три розв'язувачі (усі повертають індекси вибраних предметів):
  knapsack_dp_weight — класичне ДП за вагою, O(n·W). Точне, але W ≈ 65 000, і для тисяч
                       блоків це десятки секунд і n·W байтів пам'яті на кожен том.
  knapsack_dp_value  — ДП за цінністю, O(n·V), V = сума цінностей ≤ 3n. Теж точне:
                       для кожної досяжної цінності v шукаємо мінімальну вагу. Цінності
                       в нас малі (1..3), тому це в сотні разів швидше. Використовується за замовчуванням.
  knapsack_greedy    — жадібний за спаданням v/w, O(n log n), не завжди оптимальний.
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


def knapsack_dp_weight(items: list[Item], capacity: int) -> list[int]:
    """Класичне ДП за вагою.

    best[w] — найбільша цінність, яку можна набрати вагою не більше w з уже розглянутих
    предметів. Для кожного предмета йдемо по w згори вниз, щоб він не взявся двічі:
        best[w] = max(best[w], best[w - weight] + value).
    take[i][w] = 1, якщо предмет i покращив best[w] — за цим відновлюємо вибір з кінця.
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
    for i in range(len(items) - 1, -1, -1):
        if take[i][w]:
            chosen.append(i)
            w -= items[i].weight
    return chosen[::-1]


def knapsack_dp_value(items: list[Item], capacity: int) -> list[int]:
    """ДП за цінністю (двоїсте до ДП за вагою).

    minw[v] — найменша вага, з якою можна набрати рівно цінність v з уже розглянутих
    предметів (INF — неможливо). Переходи згори вниз по v, щоб предмет не взявся двічі:
        minw[v] = min(minw[v], minw[v - value] + weight).
    Відповідь — найбільше v з minw[v] <= capacity. Серед рівних цінностей автоматично
    обирається найлегший набір — у томі лишається більше вільного місця.
    """
    total = sum(it.value for it in items)
    inf = capacity + 1
    minw = [0] + [inf] * total
    take = []
    for it in items:
        row = bytearray(total + 1)
        wt, val = it.weight, it.value
        for v in range(total, val - 1, -1):
            cand = minw[v - val] + wt
            if cand < minw[v]:
                minw[v] = cand
                row[v] = 1
        take.append(row)
    v = max(v for v in range(total + 1) if minw[v] <= capacity)
    chosen = []
    for i in range(len(items) - 1, -1, -1):
        if take[i][v]:
            chosen.append(i)
            v -= items[i].value
    return chosen[::-1]


def knapsack_greedy(items: list[Item], capacity: int) -> list[int]:
    """Жадібний: за спаданням v/w кладемо все, що влазить. Швидко, але не завжди оптимально."""
    chosen, left = [], capacity
    for i in sorted(range(len(items)), key=lambda i: items[i].value / items[i].weight, reverse=True):
        if items[i].weight <= left:
            chosen.append(i)
            left -= items[i].weight
    return sorted(chosen)


SOLVERS = {"dp": knapsack_dp_value, "dp-weight": knapsack_dp_weight, "greedy": knapsack_greedy}


@dataclass
class PlannedBlock:
    file_name: str
    block_index: int
    coded_len: int
    value: int


@dataclass
class PlannedVolume:
    files: list[FileEntry]       # таблиця файлів тому (усі кандидати на момент пакування)
    blocks: list[PlannedBlock]   # вибрані рюкзаком блоки


def volume_capacity(max_volume: int, files: list[FileEntry]) -> int:
    return max_volume - HEADER_LEN - TAG_LEN - PAYLOAD_COUNTERS - file_table_size(files)


def plan_volumes(blocks: list[PlannedBlock], files: dict[str, FileEntry], max_volume: int,
                 solver=knapsack_dp_value) -> list[PlannedVolume]:
    """Ділить блоки на томи. Таблиця файлів кожного тому визначається тут і лише тут —
    архіватор використовує саме її, тож розмір тому гарантовано збігається з розрахунком."""
    remaining = list(blocks)
    volumes = []
    while remaining:
        table = [files[n] for n in dict.fromkeys(b.file_name for b in remaining)]
        capacity = volume_capacity(max_volume, table)
        if capacity <= 0:
            raise ArchiveError(E_CAPACITY, "таблиця файлів не вміщується в порожній том: "
                                           "збільште розмір тому або архівуйте менше файлів")
        items = [Item(i, b.coded_len + BLOCK_ENTRY_LEN, b.value) for i, b in enumerate(remaining)]
        too_big = next((b for b, it in zip(remaining, items, strict=True) if it.weight > capacity), None)
        if too_big is not None:
            raise ArchiveError(E_CAPACITY, f"блок {too_big.file_name}#{too_big.block_index} "
                                           "більший за місткість тому: зменште block_size "
                                           "або збільште розмір тому")
        chosen = set(solver(items, capacity))
        volumes.append(PlannedVolume(table, [b for i, b in enumerate(remaining) if i in chosen]))
        remaining = [b for i, b in enumerate(remaining) if i not in chosen]
    if not volumes:
        # Порожній вхід допустимий (методичка, 10.39): 1 том, file_count = 0, block_count = 0
        volumes.append(PlannedVolume([], []))
    return volumes
