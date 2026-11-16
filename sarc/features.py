"""Ознаки файлу для класифікатора: лише з байтів, ім'я та розширення не використовуються.

Ідея: кожна ознака — одне число, що описує якусь властивість даних. Разом вони утворюють
вектор, за яким перцептрон вирішує, до якого з 7 класів належить файл.
Рахуються за першими SAMPLE_SIZE байтами (один блок), бо тип потрібен ще до кодування блоків.
"""
import numpy as np

SAMPLE_SIZE = 16384
MAX_LAG = 16

# Сигнатури (перші байти) поширених форматів. Без них вже стиснутий файл і випадковий шум
# статистично не розрізнити: в обох ентропія ≈ 8 біт/байт і рівномірна гістограма.
COMPRESSED_MAGIC = (
    b"\x78\x01", b"\x78\x5e", b"\x78\x9c", b"\x78\xda",   # zlib
    b"\x1f\x8b",                                          # gzip
    b"PK\x03\x04",                                        # zip
    b"\x89PNG", b"\xff\xd8\xff",                          # png, jpeg
    b"BZh", b"\xfd7zXZ", b"7z\xbc\xaf",                   # bzip2, xz, 7z
    b"ID3", b"\xff\xfb",                                  # mp3
)
EXEC_MAGIC = (b"\x7fELF", b"MZ", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf")  # ELF, PE, Mach-O

FEATURE_NAMES = [
    "entropy",          # ентропія байтів / 8: 0 — один символ, 1 — шум
    "bigram_entropy",   # додаткова ентропія пари байтів понад одиночний, / 8
    "printable",        # частка друкованих ASCII і \t \n \r
    "utf8",             # 1, якщо зразок — коректний UTF-8 (текст кирилицею теж текст)
    "letters",          # частка латинських літер
    "digits",           # частка цифр
    "separators",       # частка , ; | \t — роздільники таблиць
    "newlines",         # частка \n
    "indented_lines",   # частка рядків, що починаються з пробілу чи табуляції (код)
    "zeros",            # частка нульових байтів
    "high_bytes",       # частка байтів >= 128
    "neighbor_diff",    # середня |x[i] - x[i-1]| / 255: мала для плавного растру
    "repeats",          # частка позицій, де байт дорівнює попередньому
    "autocorrelation",  # max по k = 1..MAX_LAG частки x[i] == x[i-k]: ~1 для періодичних шаблонів
    "chi_square",       # log10(1 + χ²/255): наскільки гістограма далека від рівномірної
    "distinct",         # кількість різних байтів / 256
    "compressed_magic", # 1, якщо починається сигнатурою стиснутого формату
    "exec_magic",       # 1, якщо починається сигнатурою виконуваного файлу
]
N_FEATURES = len(FEATURE_NAMES)


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def extract(data: bytes) -> np.ndarray:
    """Байти -> вектор із N_FEATURES чисел (порядок як у FEATURE_NAMES)."""
    d = data[:SAMPLE_SIZE]
    n = len(d)
    if n == 0:
        return np.zeros(N_FEATURES)
    a = np.frombuffer(d, dtype=np.uint8).astype(np.int64)
    counts = np.bincount(a, minlength=256)

    h1 = _entropy(counts)
    if n > 1:
        pairs = np.bincount(a[:-1] * 256 + a[1:], minlength=65536)
        h2 = _entropy(pairs) - h1          # скільки нової інформації дає другий байт пари
        diffs = np.diff(a)
        neighbor_diff = np.abs(diffs).mean() / 255
        repeats = (diffs == 0).mean()
        autocorr = max((a[k:] == a[:-k]).mean() for k in range(1, min(MAX_LAG, n - 1) + 1))
    else:
        h2 = neighbor_diff = repeats = autocorr = 0.0

    printable = (counts[32:127].sum() + counts[9] + counts[10] + counts[13]) / n
    try:
        d.decode("utf-8")
        utf8 = 1.0
    except UnicodeDecodeError:
        # останній символ міг обрізатися на межі зразка — пробуємо без кількох останніх байтів
        try:
            d[:-3].decode("utf-8")
            utf8 = 1.0
        except UnicodeDecodeError:
            utf8 = 0.0
    letters = (counts[65:91].sum() + counts[97:123].sum()) / n
    digits = counts[48:58].sum() / n
    separators = (counts[ord(",")] + counts[ord(";")] + counts[ord("|")] + counts[9]) / n
    newlines = counts[10] / n
    lines = d.split(b"\n")
    indented = sum(1 for ln in lines if ln[:1] in (b" ", b"\t")) / len(lines)
    expected = n / 256
    chi = ((counts - expected) ** 2 / expected).sum() / 255

    return np.array([
        h1 / 8,
        h2 / 8,
        printable,
        utf8,
        letters,
        digits,
        separators,
        newlines,
        indented,
        counts[0] / n,
        counts[128:].sum() / n,
        neighbor_diff,
        repeats,
        autocorr,
        np.log10(1 + chi),
        np.count_nonzero(counts) / 256,
        float(d.startswith(COMPRESSED_MAGIC)),
        float(d.startswith(EXEC_MAGIC)),
    ], dtype=np.float64)
