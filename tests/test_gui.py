"""GUI: логіку перевіряємо без дисплея; саме вікно — лише якщо є tkinter і дисплей."""
import os
import random

import pytest

from sarc.gui import archive_action, extract_action, run_safely


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    (d / "a.txt").write_bytes(b"hello world\n" * 500)
    (d / "b.bin").write_bytes(random.Random(1).randbytes(30000))
    return d


def test_archive_then_extract_roundtrip(folder, tmp_path):
    ok, text = run_safely(archive_action, [str(folder)], str(tmp_path / "a.arc"), "pw", "pw", "20000", False)
    assert ok and "Готово: 2 файлів" in text
    volumes = sorted(str(p) for p in tmp_path.glob("a*.arc"))
    assert len(volumes) >= 2                                   # том 20000 Б -> кілька томів
    ok, text = run_safely(extract_action, volumes, "pw", str(tmp_path / "out"), False)
    assert ok and "відновлено 2 файлів" in text
    for name in ("a.txt", "b.bin"):
        assert (tmp_path / "out" / name).read_bytes() == (folder / name).read_bytes()


@pytest.mark.parametrize("args,msg", [
    (([], "x.arc", "pw", "pw", "65536", False), "не вибрано"),
    ((["f"], "", "pw", "pw", "65536", False), "куди зберегти"),
    ((["f"], "x.arc", "", "", "65536", False), "порожній"),
    ((["f"], "x.arc", "pw", "pW", "65536", False), "не збігаються"),
    ((["f"], "x.arc", "pw", "pw", "64 КБ", False), "цілим числом"),
])
def test_archive_input_errors(args, msg):
    ok, text = run_safely(archive_action, *args)
    assert not ok and text.startswith("E_INPUT") and msg in text


def test_wrong_password_shows_code(folder, tmp_path):
    run_safely(archive_action, [str(folder)], str(tmp_path / "a.arc"), "pw", "pw", "65536", False)
    ok, text = run_safely(extract_action, [str(tmp_path / "a.arc")], "nope", str(tmp_path / "out"), False)
    assert not ok and text.startswith("E_HMAC")


def test_extract_input_errors(tmp_path):
    assert run_safely(extract_action, [], "pw", "out", False)[1].startswith("E_INPUT")
    assert run_safely(extract_action, ["a.arc"], "pw", "", False)[1].startswith("E_INPUT")


def test_missing_file_is_controlled_error(tmp_path):
    ok, text = run_safely(archive_action, [str(tmp_path / "nope.txt")], str(tmp_path / "a.arc"),
                          "pw", "pw", "65536", False)
    assert not ok and text.startswith("E_")


def test_window_builds():
    """Вікно створюється і закривається (лише якщо є tkinter і дисплей)."""
    tk = pytest.importorskip("tkinter")
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("немає дисплея")
    import sarc.gui as gui
    created = []
    real_tk = tk.Tk
    orig_mainloop = real_tk.mainloop
    real_tk.mainloop = lambda self, n=0: (created.append(self.title()), self.update(), self.destroy())
    try:
        assert gui.main() == 0
    finally:
        real_tk.mainloop = orig_mainloop
    assert created and created[0].startswith("SARC")
