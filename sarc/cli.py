"""Командний рядок.

    python -m sarc extract -o ПАПКА том0.arc [том1.arc ...]
    python -m sarc extract -o ПАПКА "архів.part*.arc"     # шаблон розгортає сама програма
    python -m sarc extract -o ПАПКА папка_з_томами/        # усі *.arc у папці

Пароль (у порядку пріоритету):
    змінна середовища SARC_PASSWORD -> інтерактивний запит.
    -p/--password лише для тестів і демонстрації: так пароль потрапляє в історію shell
    і видимий у списку процесів.

Код повернення: 0 — успіх, 2 — контрольована відмова (код друкується в stderr).
"""
import argparse
import getpass
import glob
import os
import sys

from .errors import E_INPUT, ArchiveError
from .extract import extract

PASSWORD_ENV = "SARC_PASSWORD"


def resolve_volumes(args: list[str]) -> list[str]:
    """Розгортає шаблони і папки самостійно: Windows cmd/PowerShell не розгортають `*`."""
    result = []
    for a in args:
        if os.path.isdir(a):
            result += sorted(glob.glob(os.path.join(glob.escape(a), "*.arc")))
        elif glob.has_magic(a):
            result += sorted(glob.glob(a))
        else:
            result.append(a)
    if not result:
        raise ArchiveError(E_INPUT, "не знайдено жодного тому")
    return list(dict.fromkeys(result))  # той самий шлях двічі — це не дубльований том


def resolve_password(cli_value: str | None) -> str:
    if cli_value is not None:
        print("Увага: пароль у командному рядку видно в історії shell; "
              f"краще {PASSWORD_ENV} або запит.", file=sys.stderr)
        return cli_value
    env = os.environ.get(PASSWORD_ENV)
    if env is not None:
        return env
    return getpass.getpass("Пароль: ")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sarc", description="Розумний захищений архіватор")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="розпакувати архів")
    ex.add_argument("volumes", nargs="+", help="томи, шаблон (*.arc) або папка; порядок довільний")
    ex.add_argument("-o", "--out", default=".", help="папка для файлів")
    ex.add_argument("-p", "--password", help=f"лише для тестів; інакше {PASSWORD_ENV} або запит")
    ex.add_argument("-f", "--force", action="store_true", help="перезаписувати наявні файли")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "extract":
            volumes = resolve_volumes(args.volumes)
            password = resolve_password(args.password)
            for name in extract(volumes, password, args.out, overwrite=args.force):
                print(name)
    except ArchiveError as e:
        print(e, file=sys.stderr)  # пароль і ключі в повідомлення не потрапляють
        return 2
    except OSError as e:
        print(f"{E_INPUT}: {e.strerror}: {e.filename}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
