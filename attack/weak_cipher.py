"""Навчальний слабкий шифр для криптоаналітичної вправи (методичка, 7.30): XOR з повторюваним ключем.

    шифротекст[i] = відкритий_текст[i] XOR ключ[i mod k]

Шифрування і розшифрування — та сама операція. Чому шифр слабкий:
  - кожен байт ключа впливає лише на свої позиції (i mod k = j), тож ключ можна відновлювати
    частинами — і будь-яке наближення дає «частково правильний» текст;
  - статистика відкритого тексту (частоти літер, пар літер) просвічує крізь XOR.
У нашому архіваторі використовується AES-256-CTR; цей шифр існує ЛИШЕ для вправи.
"""
import numpy as np


def xor(data: bytes, key: bytes) -> bytes:
    """XOR з ключем, що повторюється. Та сама функція шифрує і розшифровує."""
    if not key:
        raise ValueError("ключ не може бути порожнім")
    a = np.frombuffer(data, dtype=np.uint8)
    stream = np.resize(np.frombuffer(key, dtype=np.uint8), len(a))   # ключ, повторений до довжини даних
    return (a ^ stream).tobytes()


def index_of_coincidence(column: np.ndarray) -> float:
    """Імовірність, що два випадкові байти стовпця однакові.

    Для тексту ~0.06 (деякі символи — пробіл, «e», «о» — дуже часті), для рівномірних байтів 1/256 ≈ 0.004.
    """
    n = len(column)
    if n < 2:
        return 0.0
    counts = np.bincount(column, minlength=256)
    return float((counts * (counts - 1)).sum() / (n * (n - 1)))


def guess_key_length(ciphertext: bytes, max_len: int = 32) -> tuple[int, list[float]]:
    """Довжина ключа за індексом збігу.

    Якщо k вгадано, кожен стовпець (позиції i mod k = j) зашифрований ОДНИМ байтом ключа, тобто це
    просто перейменований текст — його індекс збігу високий, як у тексту. Якщо k невірне, стовпець
    змішує кілька байтів ключа і індекс падає. Кратні справжнього k теж дають високий індекс,
    тому беремо найменше k, чий індекс не нижчий за 90 % найкращого.
    Повертає (k, середній індекс збігу для кожного k = 1..max_len).
    """
    a = np.frombuffer(ciphertext, dtype=np.uint8)
    scores = [float(np.mean([index_of_coincidence(a[j::k]) for j in range(k)]))
              for k in range(1, max_len + 1)]
    best = max(scores)
    k = next(i + 1 for i, s in enumerate(scores) if s >= 0.9 * best)
    return k, scores
