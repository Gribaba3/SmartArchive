"""Розархіватор проти еталонних архівів викладача (data/reference/)."""
import json
import random
import zlib
from pathlib import Path

import pytest

from sarc.container import METHOD_STORED, BlockEntry, FileEntry, Volume, pack_volume
from sarc.crypto import derive_keys
from sarc.errors import ArchiveError
from sarc.extract import extract, read_archive

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
PASSWORD = "algo2026"
REF_SETS = sorted(p.stem for p in REF.glob("ref_*.json"))


def volumes_of(name: str) -> list[str]:
    return sorted(str(p) for p in REF.glob(f"{name}*.arc"))


@pytest.mark.parametrize("name", REF_SETS)
def test_reference_archive(name):
    expected = json.loads((REF / f"{name}.json").read_text(encoding="utf-8"))["вміст"]
    files = read_archive(volumes_of(name), PASSWORD)
    assert set(files) == set(expected)
    for fname, meta in expected.items():
        assert len(files[fname]) == meta["розмір"]
        assert f"{zlib.crc32(files[fname]):08X}" == meta["crc32"].upper()


def test_volumes_any_order():
    vols = volumes_of("ref_06_two_volumes")
    reference = read_archive(vols, PASSWORD)
    for seed in range(3):
        random.Random(seed).shuffle(vols)
        assert read_archive(vols, PASSWORD) == reference


EXPECTED_FAILURES = json.loads((REF / "expected_failures.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("fname", sorted(EXPECTED_FAILURES))
def test_bad_archives(fname):
    with pytest.raises(ArchiveError) as e:
        read_archive([str(REF / fname)], PASSWORD)
    assert e.value.code == EXPECTED_FAILURES[fname]["очікуваний_код"]


def test_bad_version(tmp_path):
    """Версія ≠ 1 -> E_MAGIC ще до перевірки HMAC (методичка, 10.64)."""
    data = bytearray((REF / "ref_01_single_file.arc").read_bytes())
    data[4] = 2
    (tmp_path / "v.arc").write_bytes(data)
    with pytest.raises(ArchiveError) as e:
        read_archive([str(tmp_path / "v.arc")], PASSWORD)
    assert e.value.code == "E_MAGIC"


def test_wrong_ciphertext_len(tmp_path):
    """Поле ciphertext_len не відповідає розміру тому -> E_LENGTH (методичка, 10.64)."""
    data = bytearray((REF / "ref_01_single_file.arc").read_bytes())
    data[40] ^= 1
    (tmp_path / "v.arc").write_bytes(data)
    with pytest.raises(ArchiveError) as e:
        read_archive([str(tmp_path / "v.arc")], PASSWORD)
    assert e.value.code == "E_LENGTH"


@pytest.mark.parametrize("index,count", [(1, 1), (5, 2), (0, 0)])
def test_volume_index_out_of_range(tmp_path, index, count):
    """volume_index ≥ volume_count (або count = 0) -> E_VOLUMES (методичка, 10.53)."""
    data = bytearray((REF / "ref_01_single_file.arc").read_bytes())
    data[5:7] = index.to_bytes(2, "big")
    data[7:9] = count.to_bytes(2, "big")
    (tmp_path / "v.arc").write_bytes(data)
    with pytest.raises(ArchiveError) as e:
        read_archive([str(tmp_path / "v.arc")], PASSWORD)
    assert e.value.code == "E_VOLUMES"


def test_wrong_password():
    with pytest.raises(ArchiveError) as e:
        read_archive(volumes_of("ref_01_single_file"), "wrong")
    assert e.value.code == "E_HMAC"


def test_missing_volume():
    vols = volumes_of("ref_06_two_volumes")
    with pytest.raises(ArchiveError) as e:
        read_archive(vols[:2], PASSWORD)
    assert e.value.code == "E_VOLUMES"


def test_duplicate_volume():
    vols = volumes_of("ref_06_two_volumes")
    with pytest.raises(ArchiveError) as e:
        read_archive(vols + vols[:1], PASSWORD)
    assert e.value.code == "E_VOLUMES"


@pytest.mark.parametrize("cut", [0, 3, 20, 41, 100])
def test_truncated_never_crashes(tmp_path, cut):
    data = (REF / "ref_04_seven_groups.arc").read_bytes()[:cut]
    p = tmp_path / "cut.arc"
    p.write_bytes(data)
    with pytest.raises(ArchiveError):
        read_archive([str(p)], PASSWORD)


def _single_file_volume(tmp_path, name: str) -> str:
    salt, nonce = bytes(16), bytes(12)
    k_enc, k_mac = derive_keys(PASSWORD, salt)
    vol = Volume(0, 1, salt, nonce,
                 [FileEntry(name, 3, zlib.crc32(b"abc"), 6)],
                 [BlockEntry(0, 0, 3, METHOD_STORED, b"abc")])
    p = tmp_path / "x.arc"
    p.write_bytes(pack_volume(vol, k_enc, k_mac))
    return str(p)


@pytest.mark.parametrize("bad", ["../evil.txt", "/etc/passwd", "a\\b", "C:x", "..", "a\x00b"])
def test_unsafe_names_rejected(tmp_path, bad):
    with pytest.raises(ArchiveError) as e:
        read_archive([_single_file_volume(tmp_path, bad)], PASSWORD)
    assert e.value.code == "E_PAYLOAD"


def test_extract_writes_files(tmp_path):
    out = tmp_path / "out"
    names = extract(volumes_of("ref_04_seven_groups"), PASSWORD, str(out))
    assert sorted(p.name for p in out.iterdir()) == names


def test_extract_is_atomic(tmp_path):
    """Невдале розпакування не чіпає вже наявні файли і нічого не лишає після себе."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_bytes(b"old")
    with pytest.raises(ArchiveError):
        extract([str(REF / "bad_06_bad_crc.arc")], PASSWORD, str(out))
    assert [p.name for p in out.iterdir()] == ["keep.txt"]
    assert (out / "keep.txt").read_bytes() == b"old"
