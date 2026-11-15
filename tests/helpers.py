"""Збирання власних (зокрема навмисно зіпсованих) томів для тестів."""
import struct
import zlib

from sarc.container import METHOD_HUFFMAN, METHOD_STORED, BlockEntry, FileEntry, Volume, pack_volume
from sarc.crypto import derive_keys

PASSWORD = "algo2026"
SALT = bytes(16)


def stored_file(name: str, data: bytes, file_index: int = 0) -> tuple[FileEntry, BlockEntry]:
    return (FileEntry(name, len(data), zlib.crc32(data), 6),
            BlockEntry(file_index, 0, len(data), METHOD_STORED, data))


def write_volume(path, files, blocks, index=0, count=1, nonce=None) -> str:
    k_enc, k_mac = derive_keys(PASSWORD, SALT)
    nonce = nonce if nonce is not None else bytes(8) + struct.pack(">I", index)
    path.write_bytes(pack_volume(Volume(index, count, SALT, nonce, files, blocks), k_enc, k_mac))
    return str(path)


def tree(pairs) -> bytes:
    return struct.pack(">H", len(pairs)) + b"".join(struct.pack(">HB", s, ln) for s, ln in pairs)


def huffman_block_file(tmp_path, block: bytes, raw: bytes) -> str:
    """Том з одним файлом, чий блок закодовано вручну (method 0)."""
    meta = FileEntry("x.bin", len(raw), zlib.crc32(raw), 6)
    return write_volume(tmp_path / "x.arc", [meta], [BlockEntry(0, 0, len(raw), METHOD_HUFFMAN, block)])
