"""Структура тому: 41-байтовий заголовок, payload (таблиця файлів + таблиця блоків + дані), HMAC.

Усі числа big-endian. Опис полів: docs/FORMAT_SPEC.md, розділи 1 і 3.
"""
import struct
from dataclasses import dataclass, field

from .crypto import TAG_LEN, aes_ctr, mac, mac_ok
from .errors import E_HMAC, E_LENGTH, E_MAGIC, E_PAYLOAD, ArchiveError

MAGIC = b"SARC"
VERSION = 1
HEADER_LEN = 41
HEADER = struct.Struct(">4sBHH16s12sI")  # magic, версія, № тому, к-сть томів, salt, nonce, ct_len
assert HEADER.size == HEADER_LEN

BLOCK_ENTRY = struct.Struct(">HIIIB")  # file_index, block_index, raw_len, coded_len, method
assert BLOCK_ENTRY.size == 15

METHOD_HUFFMAN = 0
METHOD_STORED = 1

FORBIDDEN_NAME_CHARS = set("/\\:\x00")


@dataclass
class FileEntry:
    name: str
    size: int
    crc32: int
    ftype: int
    strategy: int = 1


@dataclass
class BlockEntry:
    file_index: int
    block_index: int
    raw_len: int
    method: int
    data: bytes  # закодовані байти; coded_len = len(data)


@dataclass
class Volume:
    index: int
    count: int
    salt: bytes
    nonce: bytes
    files: list[FileEntry] = field(default_factory=list)
    blocks: list[BlockEntry] = field(default_factory=list)


# Імена, які Windows не дозволяє створити (незалежно від розширення: "nul.txt" теж не можна)
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} \
    | {f"LPT{i}" for i in range(1, 10)}
MAX_NAME_BYTES = 255


def check_name(name: str) -> None:
    """Захист від path traversal і від імен, які зламають розпакування на Windows."""
    bad = (
        not name
        or name in (".", "..")
        or any(ch in FORBIDDEN_NAME_CHARS or ord(ch) < 32 for ch in name)
        or name[-1] in " ."                                   # Windows мовчки обрізає
        or name.split(".")[0].upper() in WINDOWS_RESERVED
        or len(name.encode("utf-8")) > MAX_NAME_BYTES
    )
    if bad:
        raise ArchiveError(E_PAYLOAD, f"небезпечне ім'я файлу: {name!r}")


# ---------- payload ----------

def file_table_size(files: list[FileEntry]) -> int:
    """Скільки байтів займає таблиця файлів (для формули місткості рюкзака)."""
    return sum(2 + len(f.name.encode("utf-8")) + 4 + 4 + 1 + 1 for f in files)


def pack_payload(files: list[FileEntry], blocks: list[BlockEntry]) -> bytes:
    out = bytearray(struct.pack(">HH", len(files), 0))
    for f in files:
        name = f.name.encode("utf-8")
        out += struct.pack(">H", len(name)) + name
        out += struct.pack(">IIBB", f.size, f.crc32, f.ftype, f.strategy)
    out += struct.pack(">H", len(blocks))
    for b in blocks:
        out += BLOCK_ENTRY.pack(b.file_index, b.block_index, b.raw_len, len(b.data), b.method)
    for b in blocks:
        out += b.data
    return bytes(out)


class _Reader:
    """Читання з перевіркою меж: вихід за кінець -> E_PAYLOAD, а не IndexError."""

    def __init__(self, buf: bytes):
        self.buf, self.pos = buf, 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            raise ArchiveError(E_PAYLOAD, "payload обрізаний")
        chunk = self.buf[self.pos:self.pos + n]
        self.pos += n
        return chunk

    def unpack(self, fmt: str):
        s = struct.Struct(fmt)
        return s.unpack(self.take(s.size))


def unpack_payload(pt: bytes) -> tuple[list[FileEntry], list[BlockEntry]]:
    r = _Reader(pt)
    n_files, reserved = r.unpack(">HH")
    if reserved != 0:
        raise ArchiveError(E_PAYLOAD, "reserved != 0")
    files = []
    for _ in range(n_files):
        (name_len,) = r.unpack(">H")
        try:
            name = r.take(name_len).decode("utf-8")
        except UnicodeDecodeError:
            raise ArchiveError(E_PAYLOAD, "ім'я не в UTF-8") from None
        check_name(name)
        if any(f.name == name for f in files):
            raise ArchiveError(E_PAYLOAD, f"ім'я {name!r} двічі в таблиці файлів")
        size, crc, ftype, strategy = r.unpack(">IIBB")
        files.append(FileEntry(name, size, crc, ftype, strategy))
    (n_blocks,) = r.unpack(">H")
    entries = [r.unpack(BLOCK_ENTRY.format) for _ in range(n_blocks)]
    blocks = []
    for file_index, block_index, raw_len, coded_len, method in entries:
        if file_index >= n_files:
            raise ArchiveError(E_PAYLOAD, "file_index поза таблицею файлів")
        if method not in (METHOD_HUFFMAN, METHOD_STORED):
            raise ArchiveError(E_PAYLOAD, f"невідомий method {method}")
        if method == METHOD_STORED and coded_len != raw_len:
            raise ArchiveError(E_PAYLOAD, "STORED: coded_len != raw_len")
        blocks.append(BlockEntry(file_index, block_index, raw_len, method, r.take(coded_len)))
    if r.pos != len(pt):
        raise ArchiveError(E_PAYLOAD, "зайві байти в кінці payload")
    return files, blocks


# ---------- том цілком ----------

def pack_volume(vol: Volume, k_enc: bytes, k_mac: bytes) -> bytes:
    ct = aes_ctr(k_enc, vol.nonce, pack_payload(vol.files, vol.blocks))
    header = HEADER.pack(MAGIC, VERSION, vol.index, vol.count, vol.salt, vol.nonce, len(ct))
    body = header + ct
    return body + mac(k_mac, body)  # Encrypt-then-MAC: тег над заголовком і шифротекстом


def read_header(head: bytes, total_size: int) -> tuple[int, int, bytes, bytes, int]:
    """Перевірки ДО розшифрування: magic/версія, потім довжини.

    head — перші байти тому (досить HEADER_LEN), total_size — повний розмір файлу тому.
    Так заголовки всіх томів перевіряються без читання томів цілком.
    """
    if len(head) < 5 or head[:4] != MAGIC or head[4] != VERSION:
        raise ArchiveError(E_MAGIC, "не архів SARC v1")
    if len(head) < HEADER_LEN:
        raise ArchiveError(E_LENGTH, "том коротший за заголовок")
    _, _, index, count, salt, nonce, ct_len = HEADER.unpack(head[:HEADER_LEN])
    if total_size != HEADER_LEN + ct_len + TAG_LEN:
        raise ArchiveError(E_LENGTH, "розмір тому не відповідає ciphertext_len")
    if count == 0 or index >= count:
        raise ArchiveError(E_PAYLOAD, "некоректний номер тому")
    return index, count, salt, nonce, ct_len


def unpack_volume(data: bytes, k_enc: bytes, k_mac: bytes) -> Volume:
    index, count, salt, nonce, ct_len = read_header(data[:HEADER_LEN], len(data))
    body_end = HEADER_LEN + ct_len
    if not mac_ok(k_mac, data[:body_end], data[body_end:]):
        raise ArchiveError(E_HMAC, "HMAC не збігся (змінені дані або невірний пароль)")
    files, blocks = unpack_payload(aes_ctr(k_enc, nonce, data[HEADER_LEN:body_end]))
    return Volume(index, count, salt, nonce, files, blocks)
