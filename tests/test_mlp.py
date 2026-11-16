"""Перцептрон і ознаки: тести за методичкою (6.11, 10.41, чек-лист MLP)."""
import math
import random
import zlib

import numpy as np
import pytest

from sarc.features import FEATURE_NAMES, N_FEATURES, extract
from sarc.mlp import MLP, confusion_matrix, cross_entropy, f1_per_class, macro_f1, softmax


def feat(data: bytes, name: str) -> float:
    return extract(data)[FEATURE_NAMES.index(name)]


# ---------- ознаки мають сенс (10.41) ----------

def test_text_has_high_printable_fraction():
    assert feat(b"The quick brown fox jumps over the lazy dog.\n" * 50, "printable") == 1.0


def test_noise_has_high_entropy_and_zeros_low():
    noise = random.Random(1).randbytes(16384)
    assert feat(noise, "entropy") > 0.99
    assert feat(bytes(16384), "entropy") == 0.0


def test_periodic_pattern_has_high_autocorrelation():
    assert feat(b"\xf8l\xa3;" * 1000, "autocorrelation") == 1.0
    assert feat(random.Random(2).randbytes(4000), "autocorrelation") < 0.05


def test_signatures():
    assert feat(zlib.compress(b"abc" * 100), "compressed_magic") == 1.0
    assert feat(b"\x7fELF" + bytes(100), "exec_magic") == 1.0
    assert feat(b"plain text", "compressed_magic") == 0.0


def test_cyrillic_text_is_utf8():
    assert feat("Привіт, світе!\n".encode() * 20, "utf8") == 1.0


@pytest.mark.parametrize("data", [b"", b"x", b"ab"])
def test_tiny_inputs_do_not_crash(data):
    v = extract(data)
    assert v.shape == (N_FEATURES,) and np.isfinite(v).all()


# ---------- forward і loss на ручних вагах (6.11) ----------

def tiny_model() -> MLP:
    """2 входи -> 2 приховані -> 2 класи, ваги задані вручну, без нормалізації."""
    m = MLP(2, 2, 2)
    m.W1 = np.array([[1.0, -1.0], [0.0, 2.0]])
    m.b1 = np.array([0.0, -1.0])
    m.W2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    m.b2 = np.array([0.0, 0.0])
    return m


def test_forward_by_hand():
    # x = [1, 1]: z1 = [1·1 + 1·0 + 0, 1·(-1) + 1·2 - 1] = [1, 0]; ReLU -> [1, 0]
    # логіти = [1, 0]; softmax = [e/(e+1), 1/(e+1)]
    p = tiny_model().predict_proba(np.array([1.0, 1.0]))[0]
    e = math.e
    assert p == pytest.approx([e / (e + 1), 1 / (e + 1)])


def test_softmax_sums_to_one_and_survives_large_values():
    p = softmax(np.array([[1000.0, 1000.0, 0.0]]))
    assert p.sum() == pytest.approx(1.0)
    assert p[0, :2] == pytest.approx([0.5, 0.5])


def test_cross_entropy_known_values():
    uniform = np.full((1, 7), 1 / 7)
    assert cross_entropy(uniform, np.array([3])) == pytest.approx(math.log(7))
    assert cross_entropy(np.array([[0.0, 1.0]]), np.array([1])) == pytest.approx(0.0, abs=1e-9)


# ---------- backpropagation: числова перевірка градієнта (6.11) ----------

def test_gradient_check():
    """Градієнт backprop порівнюємо з числовим: (loss(w + ε) - loss(w - ε)) / 2ε."""
    rng = np.random.default_rng(0)
    m = MLP(5, 4, 3, seed=1)
    X = rng.normal(size=(8, 5))
    y = rng.integers(0, 3, 8)
    l2 = 0.01
    grads = m.backward(X, y, l2)
    eps = 1e-6
    for name in ("W1", "b1", "W2", "b2"):
        param = getattr(m, name)
        numeric = np.zeros_like(param)
        for idx in np.ndindex(param.shape):
            old = param[idx]
            param[idx] = old + eps
            up = m.loss(X, y, l2)
            param[idx] = old - eps
            down = m.loss(X, y, l2)
            param[idx] = old
            numeric[idx] = (up - down) / (2 * eps)
        assert np.allclose(grads[name], numeric, atol=1e-6), name


# ---------- навчання ----------

def test_training_reduces_loss_and_learns_separable_data():
    """Синтетичні дані: три хмари точок. Loss має впасти, точність — бути високою."""
    rng = np.random.default_rng(3)
    centers = np.array([[0, 0], [5, 5], [0, 5]])
    y = rng.integers(0, 3, 300)
    X = centers[y] + rng.normal(scale=0.7, size=(300, 2))
    m = MLP(2, 8, 3, seed=0)
    hist = m.fit(X, y, epochs=50, lr=0.05)
    assert hist[-1] < hist[0] / 3
    assert (m.predict(X) == y).mean() > 0.95


def test_same_seed_same_model():
    X = np.random.default_rng(0).normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    a, b = MLP(3, 4, 2, seed=5), MLP(3, 4, 2, seed=5)
    a.fit(X, y, epochs=5, seed=5)
    b.fit(X, y, epochs=5, seed=5)
    assert np.array_equal(a.W1, b.W1)


def test_constant_feature_does_not_divide_by_zero():
    X = np.column_stack([np.ones(20), np.arange(20.0)])
    y = (np.arange(20) >= 10).astype(int)
    m = MLP(2, 4, 2)
    m.fit(X, y, epochs=3)
    assert np.isfinite(m.predict_proba(X)).all()


def test_save_load_roundtrip(tmp_path):
    m = MLP(3, 4, 2, seed=1)
    m.mean, m.std = np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.5, 2.0])
    m.save(str(tmp_path / "m.json"))
    m2 = MLP.load(str(tmp_path / "m.json"))
    x = np.array([[0.3, -1.0, 2.0]])
    assert np.allclose(m.predict_proba(x), m2.predict_proba(x))


# ---------- метрики ----------

def test_confusion_matrix_and_macro_f1_by_hand():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 2])
    assert confusion_matrix(y_true, y_pred, 3).tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    # клас 0: P = 1/2, R = 1/2 -> F1 = 0.5; клас 1: P = 2/3, R = 1 -> 0.8; клас 2: P = 1, R = 1/2 -> 2/3
    assert macro_f1(y_true, y_pred, 3) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_never_predicted_class_has_f1_zero():
    """Методичка, 4.13: якщо P + R = 0, F1 = 0 — так macro-F1 карає за ігнорування класу."""
    cm = confusion_matrix(np.array([0, 0, 1]), np.array([0, 0, 0]), 2)
    assert f1_per_class(cm)[1] == 0.0


# ---------- підключення до архіватора ----------

def test_classify_ignores_extension():
    """Тип визначається за вмістом: зіпсоване розширення не обманює класифікатор."""
    from sarc.analysis import classify
    from tests.test_acceptance import CORPUS
    zdata = zlib.compress(b"".join(p.read_bytes() for p in sorted(CORPUS.glob("*.txt"))))
    assert classify("archive.txt", zdata[:16384]) == 5
    text = (CORPUS / "f0000.txt").read_bytes()
    assert classify("f0000.bin", text[:16384]) == classify("f0000.txt", text[:16384]) == 0


def test_model_meets_f1_min_on_valid():
    """Збережена модель на valid не гірша за F1_min з team_config (0.61)."""
    import csv
    import json

    from sarc.analysis import classify
    from tests.test_acceptance import ROOT
    cfg = json.loads((ROOT / "data" / "team_config.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(ROOT / "data" / "valid" / "labels.csv", encoding="utf-8")))
    y = np.array([int(r["type"]) for r in rows])
    pred = np.array([classify("x", (ROOT / "data" / "valid" / r["file"]).read_bytes()[:16384]) for r in rows])
    assert macro_f1(y, pred, 7) >= cfg["класифікатор"]["F1_min"]


def test_fixed_strategy_still_roundtrips(tmp_path):
    """use_class=False (фіксована стратегія для порівняння) теж дає коректний архів."""
    from sarc.archive import Options
    from tests.test_acceptance import roundtrip
    files = {"a.z": zlib.compress(b"hello world " * 500), "b.txt": b"some text\n" * 300}
    roundtrip(tmp_path, files, Options(use_class=False))


# ---------- виправлення за рев'ю #5 ----------

@pytest.mark.parametrize("size", [0, 1, 4095, 4096, 4097, 16384, 20000, 40000])
@pytest.mark.parametrize("block_size", [4096, 16384, 65536])
def test_blocks_split_stream_exactly(size, block_size):
    """_blocks дає ті самі блоки, що й просте різання по block_size (порожній файл — один блок)."""
    import io

    from sarc.archive import _blocks
    from sarc.features import SAMPLE_SIZE
    data = random.Random(size).randbytes(size)
    fh = io.BytesIO(data)
    first = fh.read(max(SAMPLE_SIZE, block_size))
    blocks = list(_blocks(fh, first, block_size))
    expected = [data[i:i + block_size] for i in range(0, len(data), block_size)] or [b""]
    assert blocks == expected


def test_classifier_sees_full_sample_with_small_blocks(tmp_path, monkeypatch):
    """З block_size = 4096 класифікатор однаково отримує SAMPLE_SIZE байтів."""
    import sarc.archive as archive_mod
    from sarc.archive import Options
    from sarc.features import SAMPLE_SIZE
    from tests.test_acceptance import roundtrip
    seen = []
    real = archive_mod.classify
    monkeypatch.setattr(archive_mod, "classify", lambda name, s: seen.append(len(s)) or real(name, s))
    data = random.Random(7).randbytes(40000)
    roundtrip(tmp_path, {"x.bin": data}, Options(block_size=4096, max_volume=8192))
    assert seen == [SAMPLE_SIZE]


@pytest.mark.parametrize("content", [None, b"not json", b'{"model": {"W1": [[1.0]]}}',
                                     b'{"model": {"W1": [[1.0]], "b1": [0], "W2": [[1.0]], "b2": [0],'
                                     b' "mean": [0], "std": [1]}}'],
                         ids=["missing", "broken_json", "missing_fields", "wrong_shape"])
def test_bad_model_file_gives_e_model(tmp_path, monkeypatch, content):
    import sarc.analysis as analysis
    from sarc.archive import create_archive
    from sarc.errors import ArchiveError
    path = tmp_path / "model.json"
    if content is not None:
        path.write_bytes(content)
    monkeypatch.setattr(analysis, "MODEL_PATH", path)
    monkeypatch.setattr(analysis, "_model", None)
    (tmp_path / "f.txt").write_bytes(b"hello")
    with pytest.raises(ArchiveError) as e:
        create_archive([str(tmp_path / "f.txt")], str(tmp_path / "a.arc"), "pw")
    assert e.value.code == "E_MODEL"
