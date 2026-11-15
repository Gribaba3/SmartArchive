"""KDF, AES-256-CTR і HMAC-SHA256 за специфікацією.

Бібліотеку дозволено використовувати лише для AES і SHA-256 (умова проєкту, п. 11.4).
"""
import hashlib
import hmac
import struct

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

KDF_ITERATIONS = 100_000
SALT_LEN = 16
NONCE_LEN = 12
TAG_LEN = 32


def derive_keys(password: str, salt: bytes) -> tuple[bytes, bytes]:
    """PBKDF2 один раз на архів -> (K_enc, K_mac). Ключі різні: перші і останні 32 байти."""
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, KDF_ITERATIONS, 64)
    return key[:32], key[32:]


def aes_ctr(k_enc: bytes, nonce: bytes, data: bytes) -> bytes:
    """Шифрування і розшифрування однакові. Лічильник: nonce(12) || uint32_BE(i), i від 1."""
    counter_block = nonce + struct.pack(">I", 1)
    c = Cipher(algorithms.AES(k_enc), modes.CTR(counter_block)).encryptor()
    return c.update(data) + c.finalize()


def mac(k_mac: bytes, data: bytes) -> bytes:
    return hmac.new(k_mac, data, hashlib.sha256).digest()


def mac_ok(k_mac: bytes, data: bytes, tag: bytes) -> bool:
    # compare_digest: порівняння за сталий час, без витоку через таймінг
    return hmac.compare_digest(mac(k_mac, data), tag)
