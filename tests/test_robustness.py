"""Зіпсовані блоки, небезпечні імена, повтор nonce, перезапис — без необроблених винятків."""
import struct

import pytest

from sarc.errors import ArchiveError
from sarc.extract import extract, read_archive
from tests.helpers import PASSWORD, huffman_block_file, stored_file, tree, write_volume


def code_of(fn, *args):
    with pytest.raises(ArchiveError) as e:
        fn(*args)
    return e.value.code


# ---------- блоки Гаффмана ----------

def blk(lit, ln, dist, tokens, bits: bytes) -> bytes:
    return tree(lit) + tree(ln) + tree(dist) + struct.pack(">II", tokens, len(bits)) + bits


# Коректний блок "AAAAA": LIT {A:1, 256:1}, LEN {1:1} (довжина 4 = 1+3), DIST {0:1} (dist 1).
# Токени: літерал A (біт 0), посилання (1) + LEN (0) + DIST (0) -> біти 0100 -> 0x40.
AM = [(65, 1), (256, 1)]
GOOD = blk(AM, [(1, 1)], [(0, 1)], 2, b"\x40")


def test_handmade_block_ok(tmp_path):
    assert read_archive([huffman_block_file(tmp_path, GOOD, b"AAAAA")], PASSWORD) == {"x.bin": b"AAAAA"}


@pytest.mark.parametrize("block", [
    blk([(65, 1), (65, 1)], [], [], 1, b"\x00"),             # символ у дереві двічі
    blk([(65, 1), (66, 1), (67, 1)], [], [], 1, b"\x00"),    # порушено нерівність Крафта
    blk(AM, [(1, 1)], [(0, 1)], 9, b"\x40"),                  # token_count > raw_len
    blk([(65, 2), (66, 2)], [], [], 5, b"\x00"),             # бітовий потік закінчився
    blk([(65, 2), (66, 2)], [], [], 1, b"\xff"),             # невідомий код
    blk(AM, [(1, 1)], [(0, 1)], 1, b"\x80"),                  # посилання на порожній вихід
    GOOD[:-3],                                                # блок обрізаний
], ids=["dup", "kraft", "tokens", "eof", "badcode", "dist", "cut"])
def test_corrupted_blocks(tmp_path, block):
    assert code_of(read_archive, [huffman_block_file(tmp_path, block, b"AAAAA")], PASSWORD) == "E_PAYLOAD"


# ---------- імена ----------

@pytest.mark.parametrize("bad", ["CON", "nul.txt", "LPT1.log", "name.", "name ", "tab\tname", "x" * 256])
def test_windows_unsafe_names(tmp_path, bad):
    f, b = stored_file(bad, b"abc")
    assert code_of(read_archive, [write_volume(tmp_path / "v.arc", [f], [b])], PASSWORD) == "E_PAYLOAD"


def test_duplicate_name_in_table(tmp_path):
    f1, b1 = stored_file("a.txt", b"1")
    f2, _ = stored_file("a.txt", b"2", 1)
    assert code_of(read_archive, [write_volume(tmp_path / "v.arc", [f1, f2], [b1])], PASSWORD) == "E_PAYLOAD"


def test_names_differing_only_by_case(tmp_path):
    f1, b1 = stored_file("a.txt", b"1")
    f2, b2 = stored_file("A.txt", b"2", 1)
    vol = write_volume(tmp_path / "v.arc", [f1, f2], [b1, b2])
    assert code_of(read_archive, [vol], PASSWORD) == "E_PAYLOAD"


# ---------- томи ----------

def test_repeated_nonce_rejected(tmp_path):
    f1, b1 = stored_file("a.txt", b"one")
    f2, b2 = stored_file("b.txt", b"two")
    v0 = write_volume(tmp_path / "0.arc", [f1], [b1], index=0, count=2, nonce=bytes(12))
    v1 = write_volume(tmp_path / "1.arc", [f2], [b2], index=1, count=2, nonce=bytes(12))
    assert code_of(read_archive, [v0, v1], PASSWORD) == "E_VOLUME"


def test_missing_input_file(tmp_path):
    assert code_of(read_archive, [str(tmp_path / "nope.arc")], PASSWORD) == "E_INPUT"


# ---------- перезапис ----------

def test_no_overwrite_without_force(tmp_path):
    f, b = stored_file("a.txt", b"new")
    vol = write_volume(tmp_path / "v.arc", [f], [b])
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.txt").write_bytes(b"old")
    assert code_of(extract, [vol], PASSWORD, str(out)) == "E_EXISTS"
    assert (out / "a.txt").read_bytes() == b"old"
    assert sorted(p.name for p in out.iterdir()) == ["a.txt"]   # тимчасова папка прибрана
    extract([vol], PASSWORD, str(out), overwrite=True)
    assert (out / "a.txt").read_bytes() == b"new"
