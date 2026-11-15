import random
from pathlib import Path

from sarc.cli import main

REF = Path(__file__).resolve().parent.parent / "data" / "reference"


def test_cli_extract_with_env_password(tmp_path, monkeypatch):
    monkeypatch.setenv("SARC_PASSWORD", "algo2026")
    rc = main(["extract", "-o", str(tmp_path), str(REF / "ref_01_single_file.arc")])
    assert rc == 0
    assert (tmp_path / "hello.txt").stat().st_size == 480


def test_cli_password_flag_warns(tmp_path, capsys):
    rc = main(["extract", "-p", "algo2026", "-o", str(tmp_path), str(REF / "ref_01_single_file.arc")])
    assert rc == 0
    assert "SARC_PASSWORD" in capsys.readouterr().err


def test_cli_expands_glob_itself(tmp_path, monkeypatch):
    """Windows cmd/PowerShell не розгортають *, тому це робить програма."""
    monkeypatch.setenv("SARC_PASSWORD", "algo2026")
    rc = main(["extract", "-o", str(tmp_path), str(REF / "ref_06_two_volumes.part*.arc")])
    assert rc == 0
    assert len(list(tmp_path.iterdir())) == 14


def test_cli_accepts_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("SARC_PASSWORD", "algo2026")
    vols = tmp_path / "vols"
    vols.mkdir()
    for p in REF.glob("ref_06_two_volumes.part*.arc"):
        (vols / p.name).write_bytes(p.read_bytes())
    rc = main(["extract", "-o", str(tmp_path / "out"), str(vols)])
    assert rc == 0


def test_cli_reports_error_code_without_password(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SARC_PASSWORD", "secret-pw")
    rc = main(["extract", "-o", str(tmp_path), str(REF / "bad_01_tampered_byte.arc")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "E_HMAC" in err
    assert "secret-pw" not in err


def test_cli_no_volumes_found(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SARC_PASSWORD", "x")
    assert main(["extract", "-o", str(tmp_path), str(tmp_path / "*.arc")]) == 2
    assert "E_INPUT" in capsys.readouterr().err


def test_cli_archive_then_extract(tmp_path, monkeypatch):
    monkeypatch.setenv("SARC_PASSWORD", "pw")
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_bytes(b"hello " * 1000)
    (src / "b.bin").write_bytes(random.Random(0).randbytes(20_000))   # не стискається
    arc = tmp_path / "x.arc"
    assert main(["archive", "-o", str(arc), str(src), "--volume-size", "8192", "--block-size", "4096",
                 "--parser", "greedy"]) == 0
    assert len(list(tmp_path.glob("x.part*.arc"))) >= 3
    assert main(["extract", "-o", str(tmp_path / "out"), str(tmp_path / "x.part*.arc")]) == 0
    for f in src.iterdir():
        assert (tmp_path / "out" / f.name).read_bytes() == f.read_bytes()


def test_cli_archive_missing_input(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SARC_PASSWORD", "pw")
    assert main(["archive", "-o", str(tmp_path / "x.arc"), str(tmp_path / "nope.txt")]) == 2
    assert "E_INPUT" in capsys.readouterr().err
