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


def test_long_codes_fallback_decoder():
    """Коди довші за TABLE_MAX_LEN декодуються по бітах (сумісність з іншими командами)."""
    lengths = {i: i for i in range(1, 17)}
    lengths[17] = 16                       # тепер сума 2^-l рівно 1
    codes = canonical_codes(lengths)
    msg = list(lengths) * 3
    w = BitWriter()
    for s in msg:
        w.write(*codes[s])
    d = Decoder(lengths)
    assert d.table is None
    r = BitReader(w.getvalue())
    assert [d.read(r) for _ in msg] == msg
