"""Вимірювання для звіту: метрика Q і таблиці 1, 2, 3, 4 (розділ 12 умови).

    python tools/measure.py                 # друкує Markdown
    python tools/measure.py -o docs/RESULTS.md

Q = α·S + β·T,  S = B_orig / (B_orig + B_arc),  T = t_ref / (t_ref + t),
t — сумарний час архівації і розпакування; Q = 0, якщо хоч один файл не відновлено.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sarc.analysis import TYPE_NAMES, classify  # noqa: E402
from sarc.archive import Options, create_archive, encode_block  # noqa: E402
from sarc.codec import encode_tokens  # noqa: E402
from sarc.crypto import aes_ctr  # noqa: E402
from sarc.extract import extract  # noqa: E402
from sarc.lz77 import greedy_parse, lazy_parse  # noqa: E402
from sarc.packing import BLOCK_ENTRY_LEN, Item, knapsack_dp, knapsack_greedy, volume_capacity  # noqa: E402

CONFIG = json.loads((ROOT / "data" / "team_config.json").read_text(encoding="utf-8"))
Q_CFG = CONFIG["метрика_Q"]
CORPUS = sorted(p for p in (ROOT / "data" / "corpus").iterdir() if p.name != "labels.csv")
PASSWORD = "measure"


def measure_q(opts: Options, files=CORPUS) -> dict:
    """Повний цикл архівація -> розпакування, перевірка побайтово, обчислення Q."""
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        res = create_archive([str(p) for p in files], os.path.join(tmp, "a.arc"), PASSWORD, opts)
        extract(res.volumes, PASSWORD, os.path.join(tmp, "out"))
        t = time.perf_counter() - t0
        ok = all((Path(tmp, "out", p.name).read_bytes() == p.read_bytes()) for p in files)
    b_orig, b_arc = res.original_bytes, res.archive_bytes
    s = b_orig / (b_orig + b_arc)
    tt = Q_CFG["t_ref_секунд"] / (Q_CFG["t_ref_секунд"] + t)
    q = Q_CFG["alpha"] * s + Q_CFG["beta"] * tt if ok else 0.0
    return dict(orig=b_orig, arc=b_arc, ratio=b_orig / b_arc, S=s, T=tt, t=t, Q=q, ok=ok,
                volumes=len(res.volumes))


def table_parsers() -> str:
    """Таблиця 1: розмір і час жадібного та лінивого розбору за групами корпусу."""
    groups = {}
    for p in CORPUS:
        data = p.read_bytes()
        g = groups.setdefault(classify(p.name, data[:16384]), {"raw": 0})
        g["raw"] += len(data)
        for name, parse in (("greedy", greedy_parse), ("lazy", lazy_parse)):
            t0 = time.perf_counter()
            size = sum(len(encode_tokens(parse(data[k:k + 16384]))) for k in range(0, len(data), 16384))
            g[name] = g.get(name, 0) + size
            g[name + "_t"] = g.get(name + "_t", 0) + time.perf_counter() - t0
    rows = ["| Група | Сирий, Б | Жадібний, Б | Лінивий, Б | Різниця | Час жадібн., с | Час лінив., с |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for gt in sorted(groups):
        g = groups[gt]
        diff = (g["lazy"] - g["greedy"]) / g["greedy"] * 100
        rows.append(f"| {TYPE_NAMES[gt]} | {g['raw']} | {g['greedy']} | {g['lazy']} | {diff:+.1f} % "
                    f"| {g['greedy_t']:.3f} | {g['lazy_t']:.3f} |")
    note = ("Розмір — сума закодованих блоків LZ77 + Гаффман "
            "(без відмови від стиснення і без накладних витрат тому).")
    return "\n".join(rows) + "\n\n" + note


def table_blocks() -> str:
    """Таблиця 2: Q повної системи і з вимкненим по черзі блоком."""
    variants = [
        ("Повна система (лінивий LZ77 + Гаффман + відмова від стиснення + рюкзак ДП)", Options()),
        ("Жадібний розбір замість лінивого", Options(parser="greedy")),
        ("Без LZ77 (лише Гаффман по байтах)", Options(parser="none")),
        ("Без стиснення взагалі (лише шифрування)", Options(parser="stored")),
        ("Жадібний рюкзак замість ДП", Options(packer="greedy")),
    ]
    rows = ["| Варіант | Архів, Б | Коеф. | S | t, с | T | Q |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, opts in variants:
        m = measure_q(opts)
        rows.append(f"| {name} | {m['arc']} | {m['ratio']:.3f} | {m['S']:.4f} | {m['t']:.2f} "
                    f"| {m['T']:.4f} | {m['Q']:.4f} |")
    return "\n".join(rows)


def table_knapsack() -> str:
    """Таблиця 3: ДП проти жадібного на першому томі корпусу."""
    opts = Options()
    items, files = [], []
    for p in CORPUS:
        data = p.read_bytes()
        ftype = classify(p.name, data[:16384])
        files.append(p.name)
        for k in range(0, len(data), opts.block_size):
            eb = encode_block(data[k:k + opts.block_size], ftype, opts)
            items.append(Item((p.name, k), len(eb.data) + BLOCK_ENTRY_LEN, opts.priorities[ftype]))
    from sarc.container import FileEntry
    cap = volume_capacity(opts.max_volume, [FileEntry(n, 0, 0, 0) for n in files])
    rows = ["| Метод | Цінність | Вага, Б | Блоків | Час, мс |", "|---|---:|---:|---:|---:|"]
    res = {}
    for name, solver in (("ДП (рюкзак 0/1)", knapsack_dp), ("Жадібний за v/w", knapsack_greedy)):
        t0 = time.perf_counter()
        chosen = solver(items, cap)
        dt = (time.perf_counter() - t0) * 1000
        res[name] = sum(i.value for i in chosen)
        rows.append(f"| {name} | {res[name]} | {sum(i.weight for i in chosen)} | {len(chosen)} | {dt:.1f} |")
    return "\n".join(rows) + f"\n\nМісткість першого тому W = {cap} Б, кандидатів {len(items)} блоків."


def table_order() -> str:
    """Таблиця 4: стиснути -> зашифрувати проти зашифрувати -> стиснути."""
    key, nonce = os.urandom(32), os.urandom(12)
    raw = comp_first = enc_first = 0
    for p in CORPUS:
        data = p.read_bytes()
        raw += len(data)
        for k in range(0, len(data), 16384):
            blk = data[k:k + 16384]
            comp_first += min(len(blk), len(encode_tokens(lazy_parse(blk))))
            enc = aes_ctr(key, nonce, blk)
            enc_first += len(encode_tokens(lazy_parse(enc)))
    return ("| Порядок | Розмір, Б | Коефіцієнт |\n|---|---:|---:|\n"
            f"| стиснути → зашифрувати | {comp_first} | {raw / comp_first:.3f} |\n"
            f"| зашифрувати → стиснути | {enc_first} | {raw / enc_first:.3f} |\n\n"
            "Шифротекст AES-CTR статистично не відрізняється від шуму: повторів для LZ77 немає, "
            "частоти байтів рівномірні, тож Гаффман теж нічого не виграє (і навіть додає таблиці).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out")
    args = ap.parse_args()
    full = measure_q(Options())
    parts = [
        "# Результати вимірювань (згенеровано tools/measure.py)\n",
        f"Корпус: {len(CORPUS)} файлів, {full['orig']} Б. α = {Q_CFG['alpha']}, β = {Q_CFG['beta']}, "
        f"t_ref = {Q_CFG['t_ref_секунд']} с. Опорний пакувальник викладача: "
        f"{CONFIG['довідково']['розмір_архіву_опорного_пакувальника']} Б "
        f"(коефіцієнт {CONFIG['довідково']['коефіцієнт']}).\n",
        f"**Повна система:** архів {full['arc']} Б у {full['volumes']} томах, "
        f"коефіцієнт {full['ratio']:.3f}, "
        f"t = {full['t']:.2f} с, S = {full['S']:.4f}, T = {full['T']:.4f}, **Q = {full['Q']:.4f}**, "
        f"відновлено побайтово: {'так' if full['ok'] else 'НІ'}.\n",
        "## Таблиця 1. Розбір потоку: жадібний і лінивий\n", table_parsers(),
        "\n## Таблиця 2. Вплив блоків на Q\n", table_blocks(),
        "\n\nПерцептрон і ГА ще не реалізовані, тому в таблиці відсутні.\n",
        "## Таблиця 3. Рюкзак першого тому: ДП проти жадібного\n", table_knapsack(),
        "\n## Таблиця 4. Порядок операцій\n", table_order(), "",
    ]
    text = "\n".join(parts)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
