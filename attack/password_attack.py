"""Контрольний експеримент: той самий ГА проти СПРАВЖНЬОГО архіву SARC зі слабким паролем (PIN із цифр).

AES не атакується: перебираються лише паролі. Fitness кандидата = 1, якщо HMAC першого тому зійшовся
(пароль правильний), інакше 0. Проміжних значень немає: пароль «на одну цифру ближчий» до правильного
дає ті самі ключі-сміття, що й будь-який інший (PBKDF2 + HMAC не мають «часткової правильності»).
Тому ГА тут не має за що зачепитися і працює не краще за випадковий перебір — це і показує експеримент.
Захист від такої атаки — довгий пароль (простір 10^4 PIN-кодів перебирається за хвилини).
"""
from sarc.container import HEADER_LEN, TAG_LEN, read_header
from sarc.crypto import derive_keys, mac_ok
from sarc.ga import Evaluator, GAConfig, genetic_algorithm, random_search


class PasswordOracle:
    """Перевіряє пароль за HMAC одного тому, не розшифровуючи дані."""

    def __init__(self, volume_bytes: bytes):
        self.data = volume_bytes
        _, _, self.salt, _, ct_len = read_header(volume_bytes[:HEADER_LEN], len(volume_bytes))
        self.body = volume_bytes[:HEADER_LEN + ct_len]
        self.tag = volume_bytes[HEADER_LEN + ct_len:HEADER_LEN + ct_len + TAG_LEN]
        self.calls = 0

    def check(self, password: str) -> bool:
        self.calls += 1
        _, k_mac = derive_keys(password, self.salt)   # PBKDF2, 100 000 ітерацій — ~30 мс
        return mac_ok(k_mac, self.body, self.tag)


def pin(ch) -> str:
    return "".join(str(d) for d in ch)


def search(oracle: PasswordOracle, digits: int, budget: int, method: str, seed: int):
    """ГА або random search по PIN-кодах. Повертає (знайдений PIN або None, кількість оцінок)."""
    ev = Evaluator(lambda ch: 1.0 if oracle.check(pin(ch)) else 0.0, budget)
    sizes = [10] * digits
    if method == "ga":
        res = genetic_algorithm(sizes, ev, GAConfig(population=20, mutation="reset", p_mutation=1 / digits,
                                                    max_stalled=1000), seed=seed)
    else:
        res = random_search(sizes, ev, seed=seed)
    return (pin(res.best) if res.best_fitness == 1.0 else None), res.evaluations
