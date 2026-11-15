"""Коди відмов розархіватора (див. docs/FORMAT_SPEC.md, розділ 5)."""

E_MAGIC = "E_MAGIC"      # не той magic або версія
E_LENGTH = "E_LENGTH"    # розмір тому не збігається з ciphertext_len (обрізаний)
E_HMAC = "E_HMAC"        # тег не збігся: змінені байти або невірний пароль
E_PAYLOAD = "E_PAYLOAD"  # структурно некоректний payload, небезпечне ім'я
E_CRC = "E_CRC"          # CRC32 відновленого файлу не збігся
E_VOLUME = "E_VOLUME"    # відсутній / дубльований / чужий том (код не заданий викладачем)
E_INPUT = "E_INPUT"      # проблема з вхідними файлами (немає, немає прав, ...)
E_EXISTS = "E_EXISTS"    # файл уже є в папці розпакування, а --force не задано
E_CAPACITY = "E_CAPACITY"  # конфігурація: блок або таблиця файлів не вміщується в том


class ArchiveError(Exception):
    """Контрольована помилка з кодом відмови."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
