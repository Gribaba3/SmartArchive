import random

import pytest

from sarc.bitio import BitReader, BitWriter
from sarc.codec import dist_symbol
from sarc.errors import ArchiveError
from sarc.huffman import Decoder, canonical_codes


def test_bits_roundtrip():
    rng = random.Random(1)
    items = [(rng.getrandbits(n), n) for n in (rng.randint(1, 17) for _ in range(500))]
    w = BitWriter()
    for v, n in items:
        w.write(v, n)
    r = BitReader(w.getvalue())
    assert [r.read(n) for _, n in items] == [v for v, _ in items]


def test_msb_first():
    w = BitWriter()
    w.write(0b101, 3)
    assert w.getvalue() == bytes([0b10100000])


def test_canonical_example():
    # класичний приклад: A=2, B=1, C=3, D=3  ->  B=0, A=10, C=110, D=111
    codes = canonical_codes({ord("A"): 2, ord("B"): 1, ord("C"): 3, ord("D"): 3})
    assert codes[ord("B")] == (0b0, 1)
    assert codes[ord("A")] == (0b10, 2)
    assert codes[ord("C")] == (0b110, 3)
    assert codes[ord("D")] == (0b111, 3)


def test_decoder_reads_canonical_codes():
    lengths = {0: 2, 1: 2, 2: 3, 3: 3, 4: 3, 5: 4, 6: 4}
    codes = canonical_codes(lengths)
    msg = [random.Random(2).choice(list(lengths)) for _ in range(300)]
    w = BitWriter()
    for s in msg:
        w.write(*codes[s])
    r, d = BitReader(w.getvalue()), Decoder(lengths)
    assert [d.read(r) for _ in msg] == msg


def test_single_symbol_alphabet():
    assert canonical_codes({65: 1}) == {65: (0, 1)}
    r = BitReader(bytes([0b00000000]))
    assert Decoder({65: 1}).read(r) == 65


def test_kraft_violation_rejected():
    with pytest.raises(ArchiveError):
        Decoder({0: 1, 1: 1, 2: 1})


@pytest.mark.parametrize("dist,expected", [(1, (0, 0, 0)), (2, (1, 0, 0)), (3, (2, 1, 0)),
                                           (4, (2, 1, 1)), (12, (4, 3, 3)), (16384, (14, 13, 8191))])
def test_dist_symbol(dist, expected):
    assert dist_symbol(dist) == expected


def test_kraft_equality_required():
    """Для більш ніж одного символу код має бути повним: сума 2^-l = 1 (методичка, 10.25)."""
    with pytest.raises(ArchiveError) as e:
        Decoder({0: 2, 1: 2, 2: 2})          # 3/4 < 1: рядок "11" нічого не означає
    assert e.value.code == "E_CODE"
    Decoder({0: 2, 1: 2, 2: 2, 3: 2})        # 4/4 = 1: коректно


def test_max_code_length_is_15():
    """Найдовший допустимий код — 15 біт (методичка, 10.24)."""
    ok = {i: i for i in range(1, 15)}
    ok[15], ok[16] = 15, 15                  # 1/2 + 1/4 + ... + 2·(1/2^15) = 1
    Decoder(ok)
    bad = {i: i for i in range(1, 16)}
    bad[16], bad[17] = 16, 16
    with pytest.raises(ArchiveError) as e:
        Decoder(bad)
    assert e.value.code == "E_CODE"


def test_empty_tree_read_is_controlled():
    with pytest.raises(ArchiveError) as e:
        Decoder({}).read(BitReader(b"\x00"))
    assert e.value.code == "E_CODE"


@pytest.mark.parametrize("dist,ok", [(32767, True), (32768, True), (32769, False)])
def test_distance_limit(dist, ok):
    """Межа вікна 32768 (методичка, 10.20): 32768 ще кодується символом DIST 15, 32769 — уже 16."""
    assert (dist_symbol(dist)[0] <= 15) == ok


def test_decode_reference_at_distance_32768():
    """Посилання на відстань 32768 декодується; блок з DIST-символом 16 відхиляється."""
    from sarc.codec import decode_block, encode_tokens
    data = bytes(range(256)) * 128           # 32768 байтів літералами
    tokens = list(data) + [(4, 32768)]       # посилання рівно на початок буфера
    assert decode_block(encode_tokens(tokens), len(data) + 4) == data + data[:4]


def test_lz_params_respect_format_limits():
    from sarc.lz77 import LZParams
    LZParams(window=32768, min_match=3)
    for bad in ({"window": 32769}, {"window": 0}, {"min_match": 2}, {"min_match": 259}):
        with pytest.raises(ValueError):
            LZParams(**bad)


def test_length_limit_on_fibonacci_worst_case():
    """Частоти Фібоначчі дають найглибше дерево Гаффмана: без ліміту код сягає 24 біт.
    З лімітом 15 — усі коди <= 15, Крафт виконується, а середня довжина зростає мало."""
    from sarc.huffman import _huffman_lengths, build_lengths
    fib = [1, 1]
    while len(fib) < 25:
        fib.append(fib[-1] + fib[-2])
    freqs = dict(enumerate(fib))
    free = _huffman_lengths(freqs)
    limited = build_lengths(freqs, 15)
    assert max(free.values()) == 24
    assert max(limited.values()) <= 15
    assert sum(2 ** -ln for ln in limited.values()) <= 1
    cost = lambda lens: sum(freqs[s] * lens[s] for s in freqs)  # noqa: E731
    assert cost(limited) / cost(free) < 1.01        # втрата менше 1 %
