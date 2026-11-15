"""12 приймальних кейсів C1 (п. 9.1 умови). Назва тесту = номер кейсу."""
import os
import random
import zlib
from pathlib import Path

import pytest

from sarc.archive import Options, create_archive
from sarc.codec import decode_block, encode_tokens
from sarc.container import METHOD_HUFFMAN, METHOD_STORED, unpack_volume
from sarc.crypto import derive_keys
from sarc.errors import ArchiveError
from sarc.extract import read_archive
from sarc.huffman import Decoder, build_lengths, canonical_codes
from sarc.lz77 import greedy_parse, lazy_parse, tokens_to_bytes
from sarc.packing import Item, knapsack_dp, knapsack_greedy

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
REF = ROOT / "data" / "reference"
PW = "test-password"


def roundtrip(tmp_path, files: dict[str, bytes], opts: Options = None):
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    paths = []
    for name, data in files.items():
        (src / name).write_bytes(data)
        paths.append(str(src / name))
    res = create_archive(paths, str(tmp_path / "a.arc"), PW, opts)
    assert read_archive(res.volumes, PW) == files
    return res


def blocks_of(res):
    salt = Path(res.volumes[0]).read_bytes()[9:25]
    keys = derive_keys(PW, salt)
    return [b for v in res.volumes for b in unpack_volume(Path(v).read_bytes(), *keys).blocks]


def test_c01_text_file(tmp_path):
    roundtrip(tmp_path, {"book.txt": (CORPUS / "f0007.txt").read_bytes()})


def test_c02_empty_and_one_byte(tmp_path):
    roundtrip(tmp_path, {"empty.dat": b"", "one.dat": b"x"})


def test_c03_all_seven_groups(tmp_path):
    files = {p.name: p.read_bytes() for p in sorted(CORPUS.glob("f*"))}
    res = roundtrip(tmp_path, files)
    assert len(res.volumes) >= 2      # корпус більший за том 64 КБ


def test_c04_same_bytes_compress_a_lot(tmp_path):
    res = roundtrip(tmp_path, {"zeros.dat": bytes(50_000)})
    assert res.original_bytes / res.archive_bytes > 5


def test_c05_noise_is_stored(tmp_path):
    noise = random.Random(5).randbytes(40_000)
    res = roundtrip(tmp_path, {"noise.bin": noise})
    for b in blocks_of(res):
        assert b.method == METHOD_STORED and len(b.data) == b.raw_len   # coded_len = raw_len
    overhead = res.archive_bytes - res.original_bytes                    # службові байти окремо
    assert 0 < overhead < 300


def test_c06_compressed_group_stored(tmp_path):
    files = {p.name: p.read_bytes() for p in CORPUS.glob("*.z")}
    res = roundtrip(tmp_path, files)
    assert all(b.method == METHOD_STORED for b in blocks_of(res))


def test_c07_huffman_from_canonical_lengths():
    freqs = {s: random.Random(s).randint(1, 1000) for s in range(60)}
    lengths = build_lengths(freqs)
    codes = canonical_codes(lengths)
    assert len({c for c in codes.values()}) == len(codes)          # коди різні
    assert sum(2 ** -ln for ln in lengths.values()) <= 1           # Крафт
    assert max(lengths.values()) <= 15
    data = (CORPUS / "f0001.py").read_bytes()[:16384]
    enc = encode_tokens(lazy_parse(data))
    assert decode_block(enc, len(data)) == data                   # дерево відновлено з довжин
    assert isinstance(Decoder(lengths), Decoder)


def test_c08_degenerate_alphabet(tmp_path):
    assert build_lengths({65: 10}) == {65: 1}
    res = roundtrip(tmp_path, {"same.dat": b"A" * 5000, "one_sym.txt": b"B"})
    assert any(b.method == METHOD_HUFFMAN for b in blocks_of(res))


def test_c09_greedy_and_lazy_both_work_and_are_measured():
    sizes = {}
    for name, parse in (("greedy", greedy_parse), ("lazy", lazy_parse)):
        total = 0
        for p in sorted(CORPUS.glob("*.py")) + sorted(CORPUS.glob("*.txt")):
            data = p.read_bytes()
            for k in range(0, len(data), 16384):
                blk = data[k:k + 16384]
                tokens = parse(blk)
                assert tokens_to_bytes(tokens) == blk
                total += len(encode_tokens(tokens))
        sizes[name] = total
    assert sizes["lazy"] <= sizes["greedy"]            # на тексті й коді лінивий не гірший


@pytest.mark.parametrize("seed", range(30))
def test_c10_knapsack_dp_not_worse_than_greedy(seed):
    rng = random.Random(seed)
    items = [Item(i, rng.randint(15, 3000), rng.randint(1, 3)) for i in range(rng.randint(1, 25))]
    cap = rng.randint(100, 20000)
    dp, gr = knapsack_dp(items, cap), knapsack_greedy(items, cap)
    assert sum(i.weight for i in dp) <= cap
    assert sum(i.value for i in dp) >= sum(i.value for i in gr)


def test_c10_knapsack_dp_beats_greedy_on_classic_case():
    # жадібний за v/w бере легкий «вигідний» предмет і вже не влазить важкий
    items = [Item("a", 1, 2), Item("b", 10, 10)]
    assert sum(i.value for i in knapsack_greedy(items, 10)) == 2
    assert sum(i.value for i in knapsack_dp(items, 10)) == 10


def test_c11_tamper_wrong_password_truncated(tmp_path):
    res = roundtrip(tmp_path, {"a.txt": b"hello world " * 100})
    vol = Path(res.volumes[0])
    data = vol.read_bytes()
    for pos in (0, 10, 50, len(data) - 1):
        bad = bytearray(data)
        bad[pos] ^= 1
        vol.write_bytes(bytes(bad))
        with pytest.raises(ArchiveError):
            read_archive([str(vol)], PW)
    vol.write_bytes(data)
    with pytest.raises(ArchiveError) as e:
        read_archive([str(vol)], "wrong")
    assert e.value.code == "E_HMAC"
    vol.write_bytes(data[:-5])
    with pytest.raises(ArchiveError) as e:
        read_archive([str(vol)], PW)
    assert e.value.code == "E_LENGTH"


def test_c12_volumes(tmp_path):
    big = random.Random(1).randbytes(30_000) + bytes(40_000)        # більший за том
    opts = Options(max_volume=8192, block_size=4096)
    res = roundtrip(tmp_path, {"big.bin": big, "small.txt": b"abc" * 1000}, opts)
    assert len(res.volumes) > 3
    assert all(os.path.getsize(v) <= 8192 for v in res.volumes)
    vols = list(res.volumes)
    random.Random(2).shuffle(vols)
    assert read_archive(vols, PW)["big.bin"] == big                  # довільний порядок
    with pytest.raises(ArchiveError) as e:
        read_archive(vols[1:], PW)
    assert e.value.code == "E_VOLUME"                                # відсутній том
    with pytest.raises(ArchiveError) as e:
        read_archive(vols + vols[:1], PW)
    assert e.value.code == "E_VOLUME"                                # дубльований том
    ref = read_archive(sorted(str(p) for p in REF.glob("ref_06*.arc")), "algo2026")
    assert len(ref) == 14                                            # еталонні архіви


# ---------- крайні випадки C2 ----------

def test_c2_input_validation(tmp_path):
    with pytest.raises(ArchiveError) as e:
        create_archive([str(tmp_path / "missing")], str(tmp_path / "a.arc"), PW)
    assert e.value.code == "E_INPUT"
    with pytest.raises(ArchiveError) as e:
        create_archive([str(tmp_path)], str(tmp_path / "a.arc"), PW)
    assert e.value.code == "E_INPUT"
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "f.txt").write_bytes(b"1")
    (tmp_path / "f.txt").write_bytes(b"2")
    with pytest.raises(ArchiveError) as e:
        create_archive([str(tmp_path / "x" / "f.txt"), str(tmp_path / "f.txt")], str(tmp_path / "a.arc"), PW)
    assert e.value.code == "E_INPUT"


def test_c2_volume_smaller_than_header(tmp_path):
    (tmp_path / "f.txt").write_bytes(b"data")
    with pytest.raises(ArchiveError) as e:
        create_archive([str(tmp_path / "f.txt")], str(tmp_path / "a.arc"), PW, Options(max_volume=60))
    assert e.value.code == "E_CAPACITY"


def test_c2_block_bigger_than_volume(tmp_path):
    (tmp_path / "f.bin").write_bytes(random.Random(0).randbytes(5000))
    with pytest.raises(ArchiveError) as e:
        create_archive([str(tmp_path / "f.bin")], str(tmp_path / "a.arc"), PW,
                       Options(max_volume=2000, block_size=4096))
    assert e.value.code == "E_CAPACITY"


def test_c2_file_shorter_than_min_match(tmp_path):
    roundtrip(tmp_path, {"ab.txt": b"ab", "abc.txt": b"abcabc"})


def test_archive_does_not_overwrite(tmp_path):
    (tmp_path / "f.txt").write_bytes(b"1")
    create_archive([str(tmp_path / "f.txt")], str(tmp_path / "a.arc"), PW)
    with pytest.raises(ArchiveError) as e:
        create_archive([str(tmp_path / "f.txt")], str(tmp_path / "a.arc"), PW)
    assert e.value.code == "E_EXISTS"


def test_salt_and_nonce_are_random(tmp_path):
    (tmp_path / "f.txt").write_bytes(b"same content")
    a = create_archive([str(tmp_path / "f.txt")], str(tmp_path / "a.arc"), PW).volumes[0]
    b = create_archive([str(tmp_path / "f.txt")], str(tmp_path / "b.arc"), PW).volumes[0]
    ha, hb = Path(a).read_bytes()[:41], Path(b).read_bytes()[:41]
    assert ha[9:25] != hb[9:25] and ha[25:37] != hb[25:37]


def test_crc_in_table(tmp_path):
    data = b"crc check " * 50
    res = roundtrip(tmp_path, {"c.txt": data})
    salt = Path(res.volumes[0]).read_bytes()[9:25]
    vol = unpack_volume(Path(res.volumes[0]).read_bytes(), *derive_keys(PW, salt))
    assert vol.files[0].crc32 == zlib.crc32(data)
