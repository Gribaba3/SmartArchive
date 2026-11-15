"""Кодек блоку method = 0: токени LZ77, закодовані трьома канонічними деревами Гаффмана.

Формат блоку (docs/FORMAT_SPEC.md, розділ 4):
    дерево LIT, дерево LEN, дерево DIST, u32 token_count, u32 bitstream_len, bitstream
Дерево: u16 n, далі n × (u16 символ, u8 довжина коду).
"""
import struct

from .bitio import BitReader
from .errors import E_PAYLOAD, ArchiveError
from .huffman import Decoder

MATCH_SYMBOL = 256   # у дереві LIT означає «далі посилання»
MIN_MATCH = 3
MAX_MATCH = 258


def dist_symbol(dist: int) -> tuple[int, int, int]:
    """dist -> (символ b, кількість додаткових бітів, значення додаткових бітів)."""
    b = (dist - 1).bit_length()
    if b <= 1:
        return b, 0, 0
    return b, b - 1, (dist - 1) - (1 << (b - 1))


def _read_tree(buf: bytes, pos: int) -> tuple[dict[int, int], int]:
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
        lengths[sym] = ln
    return lengths, pos


def decode_block(buf: bytes, raw_len: int) -> bytes:
    lit_l, pos = _read_tree(buf, 0)
    len_l, pos = _read_tree(buf, pos)
    dist_l, pos = _read_tree(buf, pos)
    if pos + 8 > len(buf):
        raise ArchiveError(E_PAYLOAD, "блок обрізаний (лічильники)")
    token_count, bits_len = struct.unpack_from(">II", buf, pos)
    pos += 8
    if pos + bits_len != len(buf):
        raise ArchiveError(E_PAYLOAD, "bitstream_len не збігається з розміром блоку")

    lit, length, dist = Decoder(lit_l), Decoder(len_l), Decoder(dist_l)
    br = BitReader(buf[pos:])
    out = bytearray()
    for _ in range(token_count):
        sym = lit.read(br)
        if sym < 256:
            out.append(sym)
            continue
        if sym != MATCH_SYMBOL:
            raise ArchiveError(E_PAYLOAD, "символ LIT > 256")
        n = length.read(br) + MIN_MATCH
        b = dist.read(br)
        d = 1 if b == 0 else ((1 << (b - 1)) | br.read(b - 1)) + 1
        if d > len(out) or n > MAX_MATCH:
            raise ArchiveError(E_PAYLOAD, "посилання за межі вже декодованих даних")
        start = len(out) - d
        for i in range(n):          # побайтово: збіг може перекривати сам себе (d < n)
            out.append(out[start + i])
        if len(out) > raw_len:
            break
    if len(out) != raw_len:
        raise ArchiveError(E_PAYLOAD, "розмір розкодованого блоку != raw_len")
    return bytes(out)
