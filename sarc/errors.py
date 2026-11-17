"""Коди відмов розархіватора (див. docs/FORMAT_SPEC.md, розділ 5)."""

E_MAGIC = "E_MAGIC"      # не той magic або версія
E_LENGTH = "E_LENGTH"    # розмір тому не збігається з ciphertext_len (обрізаний)
E_HMAC = "E_HMAC"        # тег не збігся: змінені байти або невірний пароль
E_PAYLOAD = "E_PAYLOAD"  # структурно некоректний payload, небезпечне ім'я
E_CODE = "E_CODE"        # некоректна таблиця кодів Гаффмана або посилання LZ77 (методичка, 6.24)
E_CRC = "E_CRC"          # CRC32 відновленого файлу не збігся
E_VOLUMES = "E_VOLUMES"  # набір томів некоректний (методичка, 10.53): відсутній, дубльований, чужий том
E_INPUT = "E_INPUT"      # проблема з вхідними файлами (немає, немає прав, ...)
E_OUTPUT = "E_OUTPUT"    # не вдалося записати архів (немає місця, немає прав)
E_EXISTS = "E_EXISTS"    # файл уже є в папці розпакування, а --force не задано
E_MODEL = "E_MODEL"      # не вдалося завантажити модель класифікатора (sarc/mlp_model.json)
E_CAPACITY = "E_CAPACITY"  # конфігурація: блок або таблиця файлів не вміщується в том


class ArchiveError(Exception):
    """Контрольована помилка з кодом відмови."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
