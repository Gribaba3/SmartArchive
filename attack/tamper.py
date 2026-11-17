"""Перевірка цілісності: випадково псуємо байти архіву і переконуємося, що розархіватор його відхиляє.

Кожна спроба: беремо справжній архів, псуємо один том одним способом, пробуємо розпакувати.
Добрий результат — ArchiveError з кодом. Погані — архів «розпакувався» (тиха підміна даних)
або вилетів не-ArchiveError (необроблений виняток). Таких не повинно бути жодного.
"""
import random
from pathlib import Path

from sarc.container import HEADER_LEN, TAG_LEN
from sarc.errors import ArchiveError
from sarc.extract import read_archive

# Ділянки тому: (назва, функція (довжина тому) -> (початок, кінець))
REGIONS = {
    "magic і версія (0–4)": lambda n: (0, 5),
    "номер і кількість томів (5–8)": lambda n: (5, 9),
    "salt (9–24)": lambda n: (9, 25),
    "nonce (25–36)": lambda n: (25, 37),
    "ciphertext_len (37–40)": lambda n: (37, 41),
    "шифротекст": lambda n: (HEADER_LEN, n - TAG_LEN),
    "тег HMAC": lambda n: (n - TAG_LEN, n),
}


def corrupt(data: bytes, how: str, rng: random.Random) -> bytes:
    """how: назва ділянки з REGIONS (інвертуємо випадковий біт), «обрізати» або «дописати»."""
    b = bytearray(data)
    if how == "обрізати хвіст":
        return bytes(b[:len(b) - rng.randint(1, 64)])
    if how == "дописати байти":
        return bytes(b + rng.randbytes(rng.randint(1, 64)))
    lo, hi = REGIONS[how](len(b))
    pos = rng.randrange(lo, hi)
    b[pos] ^= 1 << rng.randrange(8)
    return bytes(b)


HOW = list(REGIONS) + ["обрізати хвіст", "дописати байти"]


def trial(volumes: list[str], password: str, how: str, rng: random.Random, workdir: Path) -> str:
    """Одна спроба. Повертає код помилки, 'ПРИЙНЯТО' або 'ВИНЯТОК: <тип>'."""
    victim = rng.randrange(len(volumes))
    paths = []
    for i, v in enumerate(volumes):
        data = Path(v).read_bytes()
        if i == victim:
            data = corrupt(data, how, rng)
        p = workdir / f"t{i}.arc"
        p.write_bytes(data)
        paths.append(str(p))
    try:
        read_archive(paths, password)
        return "ПРИЙНЯТО"
    except ArchiveError as e:
        return e.code
    except Exception as e:   # noqa: BLE001 — саме це ми й шукаємо: будь-який необроблений виняток
        return f"ВИНЯТОК: {type(e).__name__}"
