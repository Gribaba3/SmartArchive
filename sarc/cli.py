"""Командний рядок.

    python -m sarc archive -o архів.arc файл1 файл2 папка/ [--volume-size 65536] [--parser lazy]

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

from .archive import Options, create_archive
from .errors import E_INPUT, ArchiveError
from .extract import extract
from .lz77 import LZParams

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


def resolve_inputs(args: list[str]) -> list[str]:
    """Файли, шаблони і папки (файли з папки — без вкладених папок)."""
    result = []
    for a in args:
        if os.path.isdir(a):
            result += sorted(p for p in glob.glob(os.path.join(glob.escape(a), "*")) if os.path.isfile(p))
        elif glob.has_magic(a):
            result += sorted(glob.glob(a))
        else:
            result.append(a)
    return list(dict.fromkeys(result))


def resolve_password(cli_value: str | None, confirm: bool = False) -> str:
    if cli_value is not None:
        print("Увага: пароль у командному рядку видно в історії shell; "
              f"краще {PASSWORD_ENV} або запит.", file=sys.stderr)
        return cli_value
    env = os.environ.get(PASSWORD_ENV)
    if env is not None:
        return env
    password = getpass.getpass("Пароль: ")
    if confirm and getpass.getpass("Ще раз: ") != password:
        raise ArchiveError(E_INPUT, "паролі не збігаються")
    return password


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sarc", description="Розумний захищений архіватор")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ar = sub.add_parser("archive", help="створити архів")
    ar.add_argument("inputs", nargs="+", help="файли, шаблони або папки")
    ar.add_argument("-o", "--out", required=True, help="ім'я архіву; кілька томів -> .partN.arc")
    ar.add_argument("-p", "--password", help=f"лише для тестів; інакше {PASSWORD_ENV} або запит")
    ar.add_argument("-f", "--force", action="store_true", help="перезаписувати наявні томи")
    ar.add_argument("--volume-size", type=int, default=65536, help="максимальний розмір тому, байт")
    ar.add_argument("--block-size", type=int, default=16384, help="розмір блоку, байт")
    ar.add_argument("--parser", choices=["lazy", "greedy"], default="lazy", help="розбір LZ77")
    ar.add_argument("--packer", choices=["dp", "greedy"], default="dp", help="рюкзак для томів")
    ar.add_argument("--window", type=int, default=16384, help="вікно пошуку збігів")
    ar.add_argument("--min-match", type=int, default=3, help="мінімальна довжина збігу (>= 3)")
    ar.add_argument("--max-chain", type=int, default=64, help="глибина пошуку збігів")

    ex = sub.add_parser("extract", help="розпакувати архів")
    ex.add_argument("volumes", nargs="+", help="томи, шаблон (*.arc) або папка; порядок довільний")
    ex.add_argument("-o", "--out", default=".", help="папка для файлів")
    ex.add_argument("-p", "--password", help=f"лише для тестів; інакше {PASSWORD_ENV} або запит")
    ex.add_argument("-f", "--force", action="store_true", help="перезаписувати наявні файли")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "archive":
            opts = Options(block_size=args.block_size, max_volume=args.volume_size, parser=args.parser,
                           packer=args.packer, lz=LZParams(window=args.window, min_match=args.min_match,
                                                           max_chain=args.max_chain))
            res = create_archive(resolve_inputs(args.inputs), args.out,
                                 resolve_password(args.password, confirm=True), opts, overwrite=args.force)
            for v in res.volumes:
                print(v)
            ratio = res.original_bytes / res.archive_bytes if res.archive_bytes else 0
            print(f"{res.original_bytes} -> {res.archive_bytes} байт, коефіцієнт {ratio:.3f}",
                  file=sys.stderr)
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
