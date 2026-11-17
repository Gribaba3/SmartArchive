"""Модель «наскільки байти схожі на текст»: імовірності пар сусідніх байтів (біграм).

Навчається на текстах класу 0 з data/train/. Оцінка тексту — середній логарифм імовірності
його пар байтів: для справжнього тексту вона висока, для сміття після XOR з неправильним ключем — низька.
Це і є fitness для ГА: частково правильний ключ дає частково осмислений текст і проміжну оцінку.
"""
import numpy as np


class BigramModel:
    def __init__(self, logp: np.ndarray, log_uni: np.ndarray):
        self.logp = logp            # 256 × 256: log P(наступний байт b | поточний байт a)
        self.log_uni = log_uni      # 256: log P(байт)

    @classmethod
    def train(cls, texts: list[bytes], smoothing: float = 0.5) -> "BigramModel":
        """Рахуємо пари байтів; згладжування Лапласа дає ненульову імовірність невиданим парам."""
        counts = np.full((256, 256), smoothing)
        uni = np.full(256, smoothing)
        for t in texts:
            a = np.frombuffer(t, dtype=np.uint8)
            np.add.at(counts, (a[:-1], a[1:]), 1)
            uni += np.bincount(a, minlength=256)
        return cls(np.log(counts / counts.sum(axis=1, keepdims=True)), np.log(uni / uni.sum()))

    def score(self, data: bytes) -> float:
        """Середній log-імовірність окремих байтів + пар байтів (чим ближче до 0, тим більше схоже на текст).

        Частина за окремими байтами важлива для ГА: пара байтів залежить від ДВОХ сусідніх байтів ключа,
        і один правильно вгаданий байт ключа без сусіда пару не «виправляє». А окремий байт залежить
        лише від свого байта ключа, тож кожен вгаданий байт ключа одразу підвищує оцінку.
        """
        a = np.frombuffer(data, dtype=np.uint8)
        if len(a) < 2:
            return 0.0
        return float(self.log_uni[a].mean() + self.logp[a[:-1], a[1:]].mean())
