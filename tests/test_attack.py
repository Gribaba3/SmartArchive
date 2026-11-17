"""Модуль атаки: слабкий шифр, модель тексту, пошук ключа ГА, перевірка пароля, пошкодження архіву."""
import csv
import random
from pathlib import Path

import pytest

from attack.password_attack import PasswordOracle, search
from attack.tamper import HOW, trial
from attack.text_model import BigramModel
from attack.weak_cipher import guess_key_length, index_of_coincidence, xor
from attack.xor_attack import attack, key_accuracy, text_accuracy
from sarc.archive import Options, create_archive
from sarc.ga import mutate

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def model():
    rows = [r for r in csv.DictReader(open(ROOT / "data" / "train" / "labels.csv", encoding="utf-8"))
            if r["type"] == "0"]
    return BigramModel.train([(ROOT / "data" / "train" / r["file"]).read_bytes() for r in rows[:20]])


@pytest.fixture(scope="module")
def text():
    return (ROOT / "data" / "corpus" / "f0007.txt").read_bytes()


# ---------- шифр ----------

def test_xor_is_its_own_inverse():
    data = random.Random(0).randbytes(1000)
    key = b"\x01\x02\xff"
    assert xor(xor(data, key), key) == data
    assert xor(b"AB", b"\x01") == b"@C"


def test_empty_key_rejected():
    with pytest.raises(ValueError):
        xor(b"abc", b"")


def test_index_of_coincidence_text_vs_noise(text):
    import numpy as np
    assert index_of_coincidence(np.frombuffer(text, dtype=np.uint8)) > 0.03
    assert index_of_coincidence(np.frombuffer(random.Random(1).randbytes(20000), dtype=np.uint8)) < 0.006


@pytest.mark.parametrize("k", [3, 7, 12])
def test_key_length_guess(text, k):
    assert guess_key_length(xor(text, random.Random(k).randbytes(k)))[0] == k


# ---------- модель тексту (fitness) ----------

def test_text_scores_higher_than_garbage(model, text):
    assert model.score(text) > model.score(xor(text, b"\x5a\xc3\x11\x80"))


def test_more_correct_key_bytes_higher_score(model, text):
    """Часткова правильність: кожен вгаданий байт ключа підвищує fitness — на цьому тримається ГА."""
    key = bytes(range(10, 18))
    ct = xor(text, key)
    wrong = bytearray(b"\x00" * 8)
    scores = []
    for i in range(9):
        guess = bytes(key[:i]) + bytes(wrong[i:])
        scores.append(model.score(xor(ct, guess)))
    assert all(a < b for a, b in zip(scores, scores[1:], strict=False))


# ---------- пошук ключа ----------

def test_ga_recovers_short_key(model, text):
    key = b"k3Y!"
    ct = xor(text, key)
    res = attack(ct, model, budget=20000, method="ga", seed=1)
    assert res.key_length == 4 and res.key == key
    assert text_accuracy(ct, res.key, text) == 1.0


def test_ga_beats_random_search(model, text):
    key = random.Random(5).randbytes(8)
    ct = xor(text, key)
    ga = attack(ct, model, budget=15000, method="ga", seed=2)
    rs = attack(ct, model, budget=15000, method="random", seed=2)
    assert key_accuracy(ga.key, key) > key_accuracy(rs.key, key)
    assert ga.evaluations == rs.evaluations == 15000


def test_accuracy_helpers():
    assert key_accuracy(b"abcd", b"abXd") == 0.75
    assert key_accuracy(b"abc", b"abcd") == 0.0
    assert text_accuracy(xor(b"hello", b"k"), b"k", b"hello") == 1.0


def test_reset_mutation_changes_gene_and_stays_in_range():
    rng = random.Random(3)
    ch = (0, 255, 17)
    for _ in range(200):
        m = mutate(ch, [256, 256, 256], 1.0, rng, mode="reset")
        assert all(0 <= g < 256 for g in m)
        assert all(a != b for a, b in zip(m, ch, strict=True))


# ---------- пароль справжнього архіву ----------

@pytest.fixture(scope="module")
def pin_archive(tmp_path_factory):
    d = tmp_path_factory.mktemp("pin")
    (d / "f.txt").write_bytes(b"secret data\n" * 50)
    res = create_archive([str(d / "f.txt")], str(d / "a.arc"), "37", Options())
    return Path(res.volumes[0]).read_bytes()


def test_password_oracle(pin_archive):
    oracle = PasswordOracle(pin_archive)
    assert oracle.check("37") and not oracle.check("38") and oracle.calls == 2


def test_pin_search_finds_with_full_budget(pin_archive):
    """2-значний PIN: бюджет 100 = весь простір, тож обидва методи знаходять пароль."""
    assert search(PasswordOracle(pin_archive), 2, 100, "random", 0)[0] == "37"
    assert search(PasswordOracle(pin_archive), 2, 100, "ga", 0)[0] == "37"


# ---------- цілісність ----------

@pytest.fixture(scope="module")
def two_volume_archive(tmp_path_factory):
    d = tmp_path_factory.mktemp("tamper")
    (d / "a.bin").write_bytes(random.Random(1).randbytes(30000))
    opts = Options(max_volume=16384, block_size=4096)
    res = create_archive([str(d / "a.bin")], str(d / "a.arc"), "pw", opts)
    assert len(res.volumes) >= 2
    return res.volumes, d


@pytest.mark.parametrize("how", HOW)
def test_every_corruption_rejected_with_code(two_volume_archive, how):
    volumes, d = two_volume_archive
    rng = random.Random(HOW.index(how))
    work = d / f"w{HOW.index(how)}"
    work.mkdir()
    for _ in range(5):
        assert trial(volumes, "pw", how, rng, work).startswith("E_")
