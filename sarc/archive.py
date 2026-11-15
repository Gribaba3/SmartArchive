"""Архіватор: файли -> блоки -> LZ77 + Гаффман (або без стиснення) -> рюкзак по томах -> AES + HMAC.

Порядок операцій — стиснути, потім зашифрувати, потім HMAC (Encrypt-then-MAC):
зашифровані дані схожі на шум, і стиснути їх після шифрування вже неможливо.
"""
import os
import struct
import tempfile
import zlib
from dataclasses import dataclass, field

from .analysis import ALREADY_COMPRESSED, classify, entropy
from .codec import encode_tokens
from .container import METHOD_HUFFMAN, METHOD_STORED, BlockEntry, FileEntry, Volume, check_name, pack_volume
from .crypto import NONCE_LEN, SALT_LEN, derive_keys
from .errors import E_CAPACITY, E_EXISTS, E_INPUT, ArchiveError
from .lz77 import LZParams, greedy_parse, lazy_parse
from .packing import PlannedBlock, knapsack_dp, knapsack_greedy, plan_volumes

# Значення за замовчуванням — з data/team_config.json (команда T01)
DEFAULT_PRIORITIES = {0: 3, 1: 3, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1}


@dataclass
class Options:
    block_size: int = 16384
    max_volume: int = 65536
    parser: str = "lazy"                 # "greedy" | "lazy"; для звіту ще "none" (без LZ77), "stored"
    packer: str = "dp"                   # "dp" | "greedy"
    lz: LZParams = field(default_factory=LZParams)
    # Вище цього порогу блок навіть не пробуємо стискати. За замовчуванням вимкнено (8.0 = максимум):
    # растр PGM має ентропію байтів 7.97 — як шум, — але завдяки кореляції сусідніх пікселів
    # стискається в ~8 разів. Ентропія гістограми не бачить порядку байтів. Параметр для ГА.
    entropy_threshold: float = 8.0
    priorities: dict = field(default_factory=lambda: dict(DEFAULT_PRIORITIES))


@dataclass
class EncodedBlock:
    raw_len: int
    method: int
    data: bytes


@dataclass
class ArchiveResult:
    volumes: list[str]
    original_bytes: int
    archive_bytes: int
    files: dict = field(default_factory=dict)   # ім'я -> (тип, кількість блоків, стиснутих)


def encode_block(raw: bytes, ftype: int, opts: Options) -> EncodedBlock:
    """Стратегія для одного блоку. Відмова від стиснення (NO_COMPRESSION), якщо:
    файл уже стиснутий; ентропія блоку вища за поріг; або стиснення не дало виграшу."""
    if not raw or ftype == ALREADY_COMPRESSED or entropy(raw) > opts.entropy_threshold:
        return EncodedBlock(len(raw), METHOD_STORED, raw)
    if opts.parser == "stored":                      # експеримент «вимкнено все стиснення»
        return EncodedBlock(len(raw), METHOD_STORED, raw)
    if opts.parser == "none":                        # експеримент «лише Гаффман, без LZ77»
        tokens = list(raw)
    else:
        parse = lazy_parse if opts.parser == "lazy" else greedy_parse
        tokens = parse(raw, opts.lz)
    coded = encode_tokens(tokens)
    if len(coded) >= len(raw):
        return EncodedBlock(len(raw), METHOD_STORED, raw)
    return EncodedBlock(len(raw), METHOD_HUFFMAN, coded)


def _validate_inputs(paths: list[str]) -> list[tuple[str, str]]:
    """-> [(шлях, ім'я в архіві)]. Перевіряє існування, права, тип і унікальність імен."""
    result, seen = [], {}
    if not paths:
        raise ArchiveError(E_INPUT, "не задано жодного файлу")
    for p in paths:
        if not os.path.exists(p):
            raise ArchiveError(E_INPUT, f"файл не існує: {p}")
        if not os.path.isfile(p):
            raise ArchiveError(E_INPUT, f"не звичайний файл: {p}")
        if not os.access(p, os.R_OK):
            raise ArchiveError(E_INPUT, f"немає прав на читання: {p}")
        if os.path.getsize(p) >= 1 << 32:
            raise ArchiveError(E_INPUT, f"файл ≥ 4 ГіБ, формат зберігає розмір у u32: {p}")
        name = os.path.basename(p)
        try:
            check_name(name)
        except ArchiveError:
            raise ArchiveError(E_INPUT, f"ім'я не можна зберегти в архіві: {name!r}") from None
        if name.casefold() in seen:
            other = seen[name.casefold()]
            raise ArchiveError(E_INPUT, f"два файли з однаковим ім'ям {name!r}: {other} і {p}")
        seen[name.casefold()] = p
        result.append((p, name))
    return result


def _volume_paths(out_path: str, count: int) -> list[str]:
    if count == 1:
        return [out_path]
    base = out_path[:-4] if out_path.lower().endswith(".arc") else out_path
    return [f"{base}.part{i}.arc" for i in range(count)]


def create_archive(paths: list[str], out_path: str, password: str, opts: Options = None,
                   overwrite: bool = False, salt: bytes = None, nonce_prefix: bytes = None) -> ArchiveResult:
    """salt і nonce_prefix задаються лише в тестах; у звичайному режимі — випадкові."""
    opts = opts or Options()
    if opts.block_size <= 0 or opts.max_volume <= 0:
        raise ArchiveError(E_CAPACITY, "block_size і max_volume мають бути додатними")
    inputs = _validate_inputs(paths)

    files: dict[str, FileEntry] = {}
    encoded: dict[tuple[str, int], EncodedBlock] = {}
    planned: list[PlannedBlock] = []
    result = ArchiveResult([], 0, 0)
    for path, name in inputs:
        try:
            fh = open(path, "rb")
        except OSError as e:
            raise ArchiveError(E_INPUT, f"не вдалося відкрити {path}: {e.strerror}") from None
        with fh:
            first = fh.read(opts.block_size)
            ftype = classify(name, first)
            crc, size, index, chunk = 0, 0, 0, first
            while True:                       # потокове читання: у пам'яті один сирий блок
                crc = zlib.crc32(chunk, crc)
                size += len(chunk)
                blk = encode_block(chunk, ftype, opts)
                encoded[(name, index)] = blk
                planned.append(PlannedBlock(name, index, len(blk.data), opts.priorities.get(ftype, 1)))
                index += 1
                chunk = fh.read(opts.block_size)
                if not chunk:
                    break
        files[name] = FileEntry(name, size, crc, ftype, 1)
        result.original_bytes += size
        result.files[name] = (ftype, index, sum(encoded[(name, i)].method == METHOD_HUFFMAN
                                                 for i in range(index)))

    solver = knapsack_dp if opts.packer == "dp" else knapsack_greedy
    layout = plan_volumes(planned, files, opts.max_volume, solver)
    if len(layout) > 0xFFFF:
        raise ArchiveError(E_CAPACITY, "забагато томів (> 65535)")

    targets = _volume_paths(out_path, len(layout))
    if not overwrite:
        existing = [t for t in targets if os.path.exists(t)]
        if existing:
            raise ArchiveError(E_EXISTS, f"файли вже існують (додайте --force): {existing}")

    salt = salt if salt is not None else os.urandom(SALT_LEN)
    nonce_prefix = nonce_prefix if nonce_prefix is not None else os.urandom(NONCE_LEN - 4)
    k_enc, k_mac = derive_keys(password, salt)        # PBKDF2 один раз на архів

    out_dir = os.path.dirname(os.path.abspath(out_path))
    tmp_files = []
    try:
        for vol_index, vol_blocks in enumerate(layout):
            remaining_before = [b for later in layout[vol_index:] for b in later]
            table = [files[n] for n in dict.fromkeys(b.file_name for b in remaining_before)]
            idx = {f.name: i for i, f in enumerate(table)}
            blocks = []
            for pb in vol_blocks:
                eb = encoded[(pb.file_name, pb.block_index)]
                blocks.append(BlockEntry(idx[pb.file_name], pb.block_index, eb.raw_len, eb.method, eb.data))
            # nonce = випадковий префікс + номер тому: у межах архіву гарантовано різні
            nonce = nonce_prefix + struct.pack(">I", vol_index)
            data = pack_volume(Volume(vol_index, len(layout), salt, nonce, table, blocks), k_enc, k_mac)
            fd, tmp = tempfile.mkstemp(prefix=".sarc-", suffix=".tmp", dir=out_dir)
            tmp_files.append(tmp)
            with os.fdopen(fd, "wb") as out:
                out.write(data)
            result.archive_bytes += len(data)
        for tmp, target in zip(tmp_files, targets, strict=True):    # атомарно: лише після запису всіх томів
            os.replace(tmp, target)
        tmp_files = []
    finally:
        for tmp in tmp_files:
            try:
                os.remove(tmp)
            except OSError:
                pass
    result.volumes = targets
    return result
