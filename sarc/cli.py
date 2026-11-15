"""Командний рядок.

    python -m sarc extract -p ПАРОЛЬ -o ПАПКА том0.arc [том1.arc ...]

Код повернення: 0 — успіх, 2 — контрольована відмова (код друкується в stderr).
"""
import argparse
import getpass
import sys

from .errors import ArchiveError
from .extract import extract


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sarc", description="Розумний захищений архіватор")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="розпакувати архів")
    ex.add_argument("volumes", nargs="+", help="усі томи архіву, у будь-якому порядку")
    ex.add_argument("-o", "--out", default=".", help="папка для файлів")
    ex.add_argument("-p", "--password", help="пароль (якщо не задано — буде запит)")

    args = ap.parse_args(argv)
    password = args.password if args.password is not None else getpass.getpass("Пароль: ")
    try:
        if args.cmd == "extract":
            for name in extract(args.volumes, password, args.out):
                print(name)
    except ArchiveError as e:
        print(e, file=sys.stderr)  # пароль і ключі в повідомлення не потрапляють
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
