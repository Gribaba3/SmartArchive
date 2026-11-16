"""Аналіз файлу: тип і стратегія стиснення.

⚠ ТИМЧАСОВО: тип визначається за розширенням з запасним варіантом за ентропією.
Зона «Аналіз файлу» замінює classify() перцептроном за ознаками вмісту
(ентропія, гістограма, частка ASCII, автокореляція, ентропія біграм).
Інтерфейс лишається тим самим: classify(name, sample) -> тип 0..6.
"""
import math
from collections import Counter

TYPE_NAMES = {
    0: "природний текст", 1: "вихідний код", 2: "структуровані дані",
    3: "растр без стиснення", 4: "виконуваний файл", 5: "вже стиснуте", 6: "штучні дані",
}
ALREADY_COMPRESSED = 5

_BY_EXTENSION = {
    ".txt": 0, ".md": 0,
    ".py": 1, ".c": 1, ".h": 1, ".java": 1, ".js": 1, ".cs": 1, ".cpp": 1,
    ".csv": 2, ".json": 2, ".log": 2, ".xml": 2,
    ".bmp": 3, ".pgm": 3, ".ppm": 3,
    ".exe": 4, ".so": 4, ".dll": 4, ".bin": 4,
    ".z": 5, ".zip": 5, ".gz": 5, ".jpg": 5, ".jpeg": 5, ".png": 5, ".mp3": 5, ".7z": 5,
    ".dat": 6,
}


def entropy(data: bytes) -> float:
    """Ентропія Шеннона в бітах на байт: 0 — один символ, 8 — рівномірний шум."""
    if not data:
        return 0.0
    n = len(data)
    return -sum(c / n * math.log2(c / n) for c in Counter(data).values())


def classify(name: str, sample: bytes) -> int:
    dot = name.rfind(".")
    ext = name[dot:].lower() if dot >= 0 else ""
    if ext in _BY_EXTENSION:
        return _BY_EXTENSION[ext]
    h = entropy(sample)
    if h > 7.5:
        return ALREADY_COMPRESSED
    if h < 1.0:
        return 6
    printable = sum(32 <= b < 127 or b in (9, 10, 13) for b in sample) / max(1, len(sample))
    return 0 if printable > 0.95 else 4
