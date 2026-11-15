from pathlib import Path

from sarc.cli import main

REF = Path(__file__).resolve().parent.parent / "data" / "reference"


def test_cli_extract_ok(tmp_path, capsys):
    rc = main(["extract", "-p", "algo2026", "-o", str(tmp_path), str(REF / "ref_01_single_file.arc")])
    assert rc == 0
    assert (tmp_path / "hello.txt").stat().st_size == 480


def test_cli_reports_error_code_without_password(tmp_path, capsys):
    rc = main(["extract", "-p", "secret-pw", "-o", str(tmp_path), str(REF / "bad_01_tampered_byte.arc")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "E_HMAC" in err
    assert "secret-pw" not in err
