"""Розархіватор: читає томи в будь-якому порядку, перевіряє все і лише потім пише файли.

Порядок перевірок:
  1. заголовки ВСІХ томів (magic/версія -> довжина) і набір томів: повний, без дублів,
     з одного архіву, nonce не повторюється;
  2. кожен том окремо: HMAC -> payload -> декодування блоків;
  3. кожен файл: блоки без пропусків, розмір, CRC32.

Потокова обробка: у пам'яті одночасно лише один том. Розкодовані блоки одразу пишуться
в тимчасову папку, файли збираються з них шматками, CRC32 рахується на льоту.
"""
import os
import shutil
import tempfile
import zlib
from dataclasses import dataclass, field

from .codec import decode_block
from .container import HEADER_LEN, METHOD_STORED, FileEntry, read_header, unpack_volume
from .crypto import derive_keys
from .errors import E_CRC, E_EXISTS, E_INPUT, E_PAYLOAD, E_VOLUME, ArchiveError

COPY_CHUNK = 1 << 20


@dataclass
class _VolumeInfo:
    path: str
    index: int
    count: int
    salt: bytes
    nonce: bytes


@dataclass
class _FileState:
    file_id: int
    meta: FileEntry
    blocks: set = field(default_factory=set)


def _scan_volumes(paths: list[str]) -> list[_VolumeInfo]:
    """Читаємо лише заголовки — це дешево і ловить більшість проблем до розшифрування."""
    if not paths:
        raise ArchiveError(E_INPUT, "не задано жодного тому")
    infos = []
    for p in paths:
        try:
            with open(p, "rb") as fh:
                head = fh.read(HEADER_LEN)
                size = os.fstat(fh.fileno()).st_size
        except OSError as e:
            raise ArchiveError(E_INPUT, f"не вдалося прочитати {p}: {e.strerror}") from None
        index, count, salt, nonce, _ = read_header(head, size)
        infos.append(_VolumeInfo(p, index, count, salt, nonce))

    if len({v.salt for v in infos}) != 1 or len({v.count for v in infos}) != 1:
        raise ArchiveError(E_VOLUME, "томи з різних архівів")
    indices = [v.index for v in infos]
    if len(set(indices)) != len(indices):
        raise ArchiveError(E_VOLUME, "дубльований том")
    if len(indices) != infos[0].count:
        missing = sorted(set(range(infos[0].count)) - set(indices))
        raise ArchiveError(E_VOLUME, f"відсутні томи: {missing}")
    # Ключ один на весь архів, тому повтор nonce у CTR = однаковий гамма-потік у двох томах:
    # XOR двох шифротекстів дав би XOR відкритих текстів. Такий архів не приймаємо.
    if len({v.nonce for v in infos}) != len(infos):
        raise ArchiveError(E_VOLUME, "nonce повторюється в різних томах")
    return sorted(infos, key=lambda v: v.index)


def _register(files: dict, meta: FileEntry) -> "_FileState":
    state = files.get(meta.name)
    if state is None:
        folded = meta.name.casefold()
        if any(name.casefold() == folded for name in files):
            # на Windows/macOS A.txt і a.txt — той самий файл, один перезаписав би інший
            raise ArchiveError(E_PAYLOAD, f"{meta.name}: ім'я відрізняється лише регістром")
        state = files[meta.name] = _FileState(len(files), meta)
    elif (state.meta.size, state.meta.crc32) != (meta.size, meta.crc32):
        raise ArchiveError(E_PAYLOAD, f"{meta.name}: різні метадані в томах")
    return state


def _decode_to(paths: list[str], password: str, workdir: str) -> dict[str, str]:
    """Розкодовує архів у workdir. Повертає {ім'я файлу: шлях до відновленого файлу}."""
    volumes = _scan_volumes(paths)
    k_enc, k_mac = derive_keys(password, volumes[0].salt)  # PBKDF2 один раз на архів
    files: dict[str, _FileState] = {}

    for info in volumes:
        with open(info.path, "rb") as fh:
            vol = unpack_volume(fh.read(), k_enc, k_mac)
        for meta in vol.files:
            _register(files, meta)
        for blk in vol.blocks:
            state = files[vol.files[blk.file_index].name]
            if blk.block_index in state.blocks:
                raise ArchiveError(E_PAYLOAD, f"{state.meta.name}: блок {blk.block_index} повторюється")
            data = blk.data if blk.method == METHOD_STORED else decode_block(blk.data, blk.raw_len)
            with open(os.path.join(workdir, f"{state.file_id}.{blk.block_index}"), "wb") as out:
                out.write(data)
            state.blocks.add(blk.block_index)

    result = {}
    for name, st in files.items():
        if not st.blocks or sorted(st.blocks) != list(range(len(st.blocks))):
            raise ArchiveError(E_PAYLOAD, f"{name}: пропущені блоки")
        target = os.path.join(workdir, f"{st.file_id}.out")
        size = crc = 0
        with open(target, "wb") as out:
            for i in range(len(st.blocks)):
                part = os.path.join(workdir, f"{st.file_id}.{i}")
                with open(part, "rb") as src:
                    while chunk := src.read(COPY_CHUNK):
                        crc = zlib.crc32(chunk, crc)
                        size += len(chunk)
                        out.write(chunk)
                os.remove(part)
        if size != st.meta.size:
            raise ArchiveError(E_PAYLOAD, f"{name}: розмір не збігається")
        if crc != st.meta.crc32:
            raise ArchiveError(E_CRC, f"{name}: CRC32 не збігається")
        result[name] = target
    return result


def read_archive(paths: list[str], password: str) -> dict[str, bytes]:
    """{ім'я: вміст} — зручно для тестів і невеликих архівів."""
    with tempfile.TemporaryDirectory(prefix="sarc-") as tmp:
        restored = _decode_to(paths, password, tmp)
        result = {}
        for name, path in restored.items():
            with open(path, "rb") as fh:
                result[name] = fh.read()
        return result


def extract(paths: list[str], password: str, out_dir: str, overwrite: bool = False) -> list[str]:
    """Атомарне розпакування: усе розкодовується й перевіряється в тимчасовій папці всередині
    out_dir; лише після повного успіху файли переносяться на місце. Наявні файли без
    overwrite=True не перезаписуються."""
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".sarc-", dir=out_dir)
    try:
        restored = _decode_to(paths, password, tmp)
        if not overwrite:
            existing = [n for n in restored if os.path.exists(os.path.join(out_dir, n))]
            if existing:
                raise ArchiveError(E_EXISTS, f"файли вже існують (додайте --force): {existing}")
        for name, path in restored.items():
            os.replace(path, os.path.join(out_dir, name))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return sorted(restored)
