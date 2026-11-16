"""Кодек блоку method = 0: токени LZ77, закодовані трьома канонічними деревами Гаффмана.

Формат блоку (docs/FORMAT_SPEC.md, розділ 4):
    дерево LIT, дерево LEN, дерево DIST, u32 token_count, u32 bitstream_len, bitstream
Дерево: u16 n, далі n × (u16 символ, u8 довжина коду).
"""
import struct
from collections import Counter

from .bitio import BitReader, BitWriter
from .errors import E_CODE, E_PAYLOAD, ArchiveError
from .huffman import Decoder, build_lengths, canonical_codes

MATCH_SYMBOL = 256   # у дереві LIT означає «далі посилання»
MIN_MATCH = 3
MAX_MATCH = 258
WINDOW = 32768       # максимальна відстань посилання за форматом (методичка, 10.19)

# Розміри алфавітів (методичка, 10.22): символ поза алфавітом -> E_CODE
LIT_ALPHABET = 257   # 0..255 байти, 256 = «далі посилання»
LEN_ALPHABET = 256   # довжина збігу - 3, тобто 3..258
DIST_ALPHABET = 16   # bit_length(dist - 1) для dist 1..32768 -> 0..15


def dist_symbol(dist: int) -> tuple[int, int, int]:
    """dist -> (символ b, кількість додаткових бітів, значення додаткових бітів)."""
    b = (dist - 1).bit_length()
    if b <= 1:
        return b, 0, 0
    return b, b - 1, (dist - 1) - (1 << (b - 1))


def _read_tree(buf: bytes, pos: int, alphabet: int) -> tuple[dict[int, int], int]:
    if pos + 2 > len(buf):
        raise ArchiveError(E_PAYLOAD, "блок обрізаний (дерево)")
    (n,) = struct.unpack_from(">H", buf, pos)
    pos += 2
    if pos + 3 * n > len(buf):
        raise ArchiveError(E_PAYLOAD, "блок обрізаний (дерево)")
    lengths = {}
    for _ in range(n):
        sym, ln = struct.unpack_from(">HB", buf, pos)
        pos += 3
        if sym >= alphabet:
            raise ArchiveError(E_CODE, f"символ {sym} поза алфавітом 0..{alphabet - 1}")
        if sym in lengths:
            raise ArchiveError(E_CODE, f"символ {sym} у дереві двічі")
        lengths[sym] = ln
    return lengths, pos


def decode_block(buf: bytes, raw_len: int) -> bytes:
    lit_l, pos = _read_tree(buf, 0, LIT_ALPHABET)
    len_l, pos = _read_tree(buf, pos, LEN_ALPHABET)
    dist_l, pos = _read_tree(buf, pos, DIST_ALPHABET)
    if pos + 8 > len(buf):
        raise ArchiveError(E_PAYLOAD, "блок обрізаний (лічильники)")
    token_count, bits_len = struct.unpack_from(">II", buf, pos)
    pos += 8
    if pos + bits_len != len(buf):
        raise ArchiveError(E_PAYLOAD, "bitstream_len не збігається з розміром блоку")
    if token_count > raw_len:  # кожен токен дає щонайменше 1 байт
        raise ArchiveError(E_PAYLOAD, "token_count більший за raw_len")

    lit, length, dist = Decoder(lit_l), Decoder(len_l), Decoder(dist_l)
    br = BitReader(buf[pos:])
    out = bytearray()
    for _ in range(token_count):
        sym = lit.read(br)
        if sym < 256:
            out.append(sym)
            continue
        # символи поза алфавітами вже відсіяні в _read_tree, тому sym == 256, n <= 258, b <= 15
        n = length.read(br) + MIN_MATCH
        b = dist.read(br)
        d = 1 if b == 0 else ((1 << (b - 1)) | br.read(b - 1)) + 1
        if d > len(out):  # d <= 32768 гарантує b <= 15
            raise ArchiveError(E_CODE, "посилання за межі вже декодованих даних")
        start = len(out) - d
        if d >= n:
            out += out[start:start + n]
        else:
            # збіг перекриває сам себе (наприклад, d=1 — повтор одного байта):
            # це повторення шматка довжини d, тому множимо його і обрізаємо до n
            chunk = out[start:]
            out += (chunk * (n // d + 1))[:n]
        if len(out) > raw_len:
            break
    if len(out) != raw_len:
        raise ArchiveError(E_PAYLOAD, "розмір розкодованого блоку != raw_len")
    return bytes(out)


# ---------- кодування (архіватор) ----------

def _write_tree(lengths: dict[int, int]) -> bytes:
    out = bytearray(struct.pack(">H", len(lengths)))
    for sym in sorted(lengths):
        out += struct.pack(">HB", sym, lengths[sym])
    return bytes(out)


def encode_tokens(tokens: list) -> bytes:
    """Токени -> байти блоку method 0.

    Токен — або int (літерал 0..255), або кортеж (довжина, відстань) для посилання.
    Частоти рахуються окремо для трьох алфавітів, і для кожного будується свій код Гаффмана.
    """
    lit_f, len_f, dist_f = Counter(), Counter(), Counter()
    for t in tokens:
        if isinstance(t, int):
            lit_f[t] += 1
        else:
            n, d = t
            lit_f[MATCH_SYMBOL] += 1
            len_f[n - MIN_MATCH] += 1
            dist_f[dist_symbol(d)[0]] += 1
    lit_l, len_l, dist_l = build_lengths(lit_f), build_lengths(len_f), build_lengths(dist_f)
    lit_c, len_c, dist_c = canonical_codes(lit_l), canonical_codes(len_l), canonical_codes(dist_l)

    bw = BitWriter()
    for t in tokens:
        if isinstance(t, int):
            bw.write(*lit_c[t])
        else:
            n, d = t
            bw.write(*lit_c[MATCH_SYMBOL])
            bw.write(*len_c[n - MIN_MATCH])
            b, nextra, extra = dist_symbol(d)
            bw.write(*dist_c[b])
            if nextra:
                bw.write(extra, nextra)
    bits = bw.getvalue()
    return (_write_tree(lit_l) + _write_tree(len_l) + _write_tree(dist_l)
            + struct.pack(">II", len(tokens), len(bits)) + bits)


def token_cost_bits(tokens: list) -> int:
    """Вартість за фіксованою таблицею з team_config (для порівняння розборів і ДП)."""
    total = 0
    for t in tokens:
        if isinstance(t, int):
            total += 9
        else:
            total += 17 + max(0, (t[1] - 1).bit_length() - 1)
    return total
