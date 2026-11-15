"""Розархіватор: читає томи в будь-якому порядку, перевіряє все і лише потім пише файли.

Порядок перевірок кожного тому: magic/версія -> довжина -> HMAC -> payload.
Потім: набір томів повний і без дублів -> зібрати блоки -> CRC32 кожного файлу.
"""
import os
import shutil
import tempfile
import zlib
from dataclasses import dataclass

from .codec import decode_block
from .container import METHOD_STORED, FileEntry, read_header, unpack_volume
from .crypto import derive_keys
from .errors import E_CRC, E_INPUT, E_PAYLOAD, E_VOLUME, ArchiveError


@dataclass
class _Collected:
    meta: FileEntry
    parts: dict  # block_index -> bytes


def _read_volumes(paths: list[str], password: str):
    if not paths:
        raise ArchiveError(E_INPUT, "не задано жодного тому")
    raw = []
    for p in paths:
        try:
            with open(p, "rb") as fh:
                raw.append((p, fh.read()))
        except OSError as e:
            raise ArchiveError(E_INPUT, f"не вдалося прочитати {p}: {e.strerror}") from None

    headers = [read_header(data) for _, data in raw]  # спершу дешеві перевірки всіх томів
    salts = {h[2] for h in headers}
    counts = {h[1] for h in headers}
    if len(salts) != 1 or len(counts) != 1:
        raise ArchiveError(E_VOLUME, "томи з різних архівів")
    count = counts.pop()
    indices = [h[0] for h in headers]
    if len(set(indices)) != len(indices):
        raise ArchiveError(E_VOLUME, "дубльований том")
    if len(indices) != count:
        missing = sorted(set(range(count)) - set(indices))
        raise ArchiveError(E_VOLUME, f"відсутні томи: {missing}")

    k_enc, k_mac = derive_keys(password, salts.pop())  # PBKDF2 один раз на архів
    return [unpack_volume(data, k_enc, k_mac) for _, data in raw]


def read_archive(paths: list[str], password: str) -> dict[str, bytes]:
    """Повертає {ім'я: вміст}. Будь-яка проблема -> ArchiveError з кодом."""
    volumes = _read_volumes(paths, password)
    files: dict[str, _Collected] = {}
    for vol in volumes:
        for blk in vol.blocks:
            meta = vol.files[blk.file_index]
            entry = files.setdefault(meta.name, _Collected(meta, {}))
            if (entry.meta.size, entry.meta.crc32) != (meta.size, meta.crc32):
                raise ArchiveError(E_PAYLOAD, f"{meta.name}: різні метадані в томах")
            if blk.block_index in entry.parts:
                raise ArchiveError(E_PAYLOAD, f"{meta.name}: блок {blk.block_index} повторюється")
            data = blk.data if blk.method == METHOD_STORED else decode_block(blk.data, blk.raw_len)
            entry.parts[blk.block_index] = data
        # файл може бути в таблиці тому без жодного блоку — його блоки в інших томах
        for meta in vol.files:
            files.setdefault(meta.name, _Collected(meta, {}))

    result = {}
    for name, entry in files.items():
        if sorted(entry.parts) != list(range(len(entry.parts))):
            raise ArchiveError(E_PAYLOAD, f"{name}: пропущені блоки")
        content = b"".join(entry.parts[i] for i in range(len(entry.parts)))
        if len(content) != entry.meta.size:
            raise ArchiveError(E_PAYLOAD, f"{name}: розмір не збігається")
        if zlib.crc32(content) != entry.meta.crc32:
            raise ArchiveError(E_CRC, f"{name}: CRC32 не збігається")
        result[name] = content
    return result


def extract(paths: list[str], password: str, out_dir: str) -> list[str]:
    """Атомарне розпакування: спершу все перевіряється в пам'яті й пишеться в тимчасову
    папку; лише після повного успіху файли переносяться в out_dir."""
    contents = read_archive(paths, password)
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".sarc-", dir=out_dir)
    try:
        for name, data in contents.items():
            with open(os.path.join(tmp, name), "wb") as fh:
                fh.write(data)
        for name in contents:
            os.replace(os.path.join(tmp, name), os.path.join(out_dir, name))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return sorted(contents)
