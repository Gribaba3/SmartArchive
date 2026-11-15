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
