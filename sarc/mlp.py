"""Перцептрон з одним прихованим шаром (MLP), написаний вручну на numpy.

Будова:   x (ознаки) -> [W1, b1] -> ReLU -> h -> [W2, b2] -> softmax -> p (ймовірності 7 класів)

Навчання (міні-батчі, градієнтний спуск):
    forward   — рахуємо p для батча;
    loss      — cross-entropy: середнє -log p[правильний клас] (+ L2-штраф на ваги);
    backward  — backpropagation: похідні loss по всіх вагах (правило ланцюжка);
    update    — W -= lr * dW.

Ознаки нормалізуються (z-score) середнім і відхиленням, порахованими ЛИШЕ на train:
інакше одна ознака в діапазоні 0..1, а інша 0..0.01 — і друга майже не впливає на навчання.
"""
import json

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """Рядок оцінок -> рядок ймовірностей (сума 1). Віднімаємо максимум, щоб exp не переповнився."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(p: np.ndarray, y: np.ndarray) -> float:
    """Середнє -log(ймовірність правильного класу). 1e-12 — щоб не було log(0)."""
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


class MLP:
    def __init__(self, n_in: int, n_hidden: int, n_out: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # ініціалізація He: для ReLU масштаб sqrt(2 / n_in) зберігає розкид сигналу між шарами
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)
        self.mean = np.zeros(n_in)   # параметри нормалізації ознак (задаються в fit)
        self.std = np.ones(n_in)

    # ---------- прямий прохід ----------

    def normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def forward(self, Xn: np.ndarray):
        """Xn — уже нормалізовані ознаки. Повертає ймовірності і проміжні значення для backward."""
        z1 = Xn @ self.W1 + self.b1
        h = np.maximum(z1, 0)                 # ReLU
        p = softmax(h @ self.W2 + self.b2)
        return p, (Xn, z1, h)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(self.normalize(np.atleast_2d(X)))[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # ---------- зворотний прохід ----------

    def loss(self, Xn: np.ndarray, y: np.ndarray, l2: float = 0.0) -> float:
        p, _ = self.forward(Xn)
        return cross_entropy(p, y) + l2 / 2 * ((self.W1 ** 2).sum() + (self.W2 ** 2).sum())

    def backward(self, Xn: np.ndarray, y: np.ndarray, l2: float = 0.0) -> dict:
        """Градієнти loss по W1, b1, W2, b2 для батча (правило ланцюжка, від виходу до входу)."""
        m = len(y)
        p, (x, z1, h) = self.forward(Xn)
        # Для softmax + cross-entropy похідна по логітах проста: p - one_hot(y), поділене на m
        dz2 = p.copy()
        dz2[np.arange(m), y] -= 1
        dz2 /= m
        dW2 = h.T @ dz2 + l2 * self.W2
        db2 = dz2.sum(axis=0)
        dh = dz2 @ self.W2.T
        dz1 = dh * (z1 > 0)                   # похідна ReLU: 1, де z1 > 0, інакше 0
        dW1 = x.T @ dz1 + l2 * self.W1
        db1 = dz1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # ---------- навчання ----------

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 0.05,
            batch: int = 32, l2: float = 1e-4, seed: int = 0) -> list[float]:
        """Навчання міні-батчами. Повертає loss на train після кожної епохи."""
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0         # стала ознака: не ділимо на нуль
        Xn = self.normalize(X)
        rng = np.random.default_rng(seed)
        history = []
        for _ in range(epochs):
            order = rng.permutation(len(y))   # щоепохи новий порядок прикладів
            for start in range(0, len(y), batch):
                idx = order[start:start + batch]
                g = self.backward(Xn[idx], y[idx], l2)
                for name, grad in g.items():
                    setattr(self, name, getattr(self, name) - lr * grad)
            history.append(self.loss(Xn, y, l2))
        return history

    # ---------- збереження ----------

    def to_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("W1", "b1", "W2", "b2", "mean", "std")}

    @classmethod
    def from_dict(cls, d: dict) -> "MLP":
        W1 = np.array(d["W1"])
        model = cls(W1.shape[0], W1.shape[1], len(d["b2"]))
        for k in ("W1", "b1", "W2", "b2", "mean", "std"):
            setattr(model, k, np.array(d[k], dtype=np.float64))
        return model

    def save(self, path: str, extra: dict = None) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({**(extra or {}), "model": self.to_dict()}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "MLP":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f)["model"])


# ---------- метрики ----------

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """cm[i, j] — скільки файлів класу i модель віднесла до класу j."""
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t, p] += 1
    return cm


def f1_per_class(cm: np.ndarray) -> np.ndarray:
    """F1 = 2·P·R / (P + R); якщо P + R = 0 (клас не передбачено і не вгадано) — F1 = 0."""
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0)                # стовпці: скільки разів модель назвала цей клас
    actual = cm.sum(axis=1)                   # рядки: скільки файлів цього класу насправді
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    s = precision + recall
    return np.divide(2 * precision * recall, s, out=np.zeros_like(tp), where=s > 0)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Середнє F1 по всіх k класах: кожен клас важить однаково, незалежно від розміру."""
    return float(f1_per_class(confusion_matrix(y_true, y_pred, k)).mean())
