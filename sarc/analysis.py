"""Аналіз файлу: тип за вмістом (перцептрон) і ентропія.

classify(name, sample) -> тип 0..6. Тип визначає перцептрон (sarc/mlp.py) за ознаками
байтів (sarc/features.py); ім'я та розширення НЕ використовуються: на прихованій вибірці
викладача вони можуть не збігатися з вмістом. Модель навчає experiments/train_mlp.py
і зберігає в sarc/mlp_model.json.
"""
import math
from collections import Counter
from pathlib import Path

from . import features
from .errors import E_MODEL, ArchiveError
from .mlp import MLP

TYPE_NAMES = {
    0: "природний текст", 1: "вихідний код", 2: "структуровані дані",
    3: "растр без стиснення", 4: "виконуваний файл", 5: "вже стиснуте", 6: "штучні дані",
}
ALREADY_COMPRESSED = 5

MODEL_PATH = Path(__file__).with_name("mlp_model.json")
_model = None


def _get_model() -> MLP:
    """Модель читається з диска один раз і далі береться з пам'яті.

    Немає файлу, немає прав, битий JSON або не ті поля -> контрольована помилка E_MODEL,
    а не необроблений виняток (і не E_INPUT: вхідний файл користувача тут ні до чого).
    """
    global _model
    if _model is None:
        try:
            model = MLP.load(str(MODEL_PATH))
        except OSError as e:
            raise ArchiveError(E_MODEL, f"не вдалося прочитати модель {MODEL_PATH}: {e.strerror}") from None
        except (ValueError, KeyError, TypeError, IndexError):
            raise ArchiveError(E_MODEL, f"файл моделі {MODEL_PATH} пошкоджений") from None
        if model.W1.shape[0] != features.N_FEATURES:
            raise ArchiveError(E_MODEL, f"модель розрахована на {model.W1.shape[0]} ознак, "
                                        f"а features.py дає {features.N_FEATURES}: перенавчіть модель")
        _model = model
    return _model


def entropy(data: bytes) -> float:
    """Ентропія Шеннона в бітах на байт: 0 — один символ, 8 — рівномірний шум."""
    if not data:
        return 0.0
    n = len(data)
    return -sum(c / n * math.log2(c / n) for c in Counter(data).values())


def classify(name: str, sample: bytes) -> int:
    """name лишився в інтерфейсі для сумісності, але не використовується."""
    return int(_get_model().predict(features.extract(sample))[0])
