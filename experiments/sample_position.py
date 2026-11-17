"""Чи залежить тип файлу від того, з якої частини файлу взято зразок (issue #13, DECISIONS Р20).

    python experiments/sample_position.py

Для кожного файлу corpus/, більшого за SAMPLE_SIZE, класифікуємо три зразки по 16 КБ:
початок, середину і кінець, і порівнюємо зі справжнім класом.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sarc.analysis import TYPE_NAMES, classify  # noqa: E402
from sarc.features import SAMPLE_SIZE  # noqa: E402


def main():
    rows = list(csv.DictReader(open(ROOT / "data" / "corpus" / "labels.csv", encoding="utf-8")))
    print("| Файл | Розмір | Справжній клас | Початок | Середина | Кінець |")
    print("|---|---:|---|---|---|---|")
    for r in rows:
        d = (ROOT / "data" / "corpus" / r["file"]).read_bytes()
        if len(d) <= SAMPLE_SIZE:
            continue
        mid = len(d) // 2 - SAMPLE_SIZE // 2
        parts = [d[:SAMPLE_SIZE], d[mid:mid + SAMPLE_SIZE], d[-SAMPLE_SIZE:]]
        got = [TYPE_NAMES[classify("x", p)] for p in parts]
        print(f"| {r['file']} | {len(d)} | {TYPE_NAMES[int(r['type'])]} | " + " | ".join(got) + " |")


if __name__ == "__main__":
    main()
