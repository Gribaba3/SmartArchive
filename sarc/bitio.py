"""Бітовий потік: біти від старшого до молодшого (MSB-first), останній байт доповнюється нулями."""
from .errors import E_PAYLOAD, ArchiveError


class BitWriter:
    def __init__(self):
        self.out = bytearray()
        self.acc = 0    # накопичені біти
        self.nbits = 0  # скільки їх у acc

    def write(self, value: int, nbits: int) -> None:
        self.acc = (self.acc << nbits) | value
        self.nbits += nbits
        while self.nbits >= 8:
            self.nbits -= 8
            self.out.append((self.acc >> self.nbits) & 0xFF)
        self.acc &= (1 << self.nbits) - 1

    def getvalue(self) -> bytes:
        if self.nbits:
            return bytes(self.out) + bytes([(self.acc << (8 - self.nbits)) & 0xFF])
        return bytes(self.out)


class BitReader:
    """Читає через буфер acc: байти докладаються, поки в буфері менше потрібної кількості бітів.
    peek(n) дивиться наперед без зсуву позиції — це потрібно табличному декодеру Гаффмана."""

    def __init__(self, data: bytes):
        self.data = data
        self.byte_pos = 0
        self.acc = 0
        self.nbits = 0
        self.remaining = len(data) * 8  # непрочитані біти потоку

    def _fill(self, n: int) -> None:
        while self.nbits < n and self.byte_pos < len(self.data):
            self.acc = (self.acc << 8) | self.data[self.byte_pos]
            self.byte_pos += 1
            self.nbits += 8

    def peek(self, n: int) -> int:
        """n наступних бітів; за кінцем потоку доповнюється нулями (вихід за кінець ловить skip)."""
        self._fill(n)
        if self.nbits >= n:
            return (self.acc >> (self.nbits - n)) & ((1 << n) - 1)
        return (self.acc << (n - self.nbits)) & ((1 << n) - 1)

    def skip(self, n: int) -> None:
        if n > self.remaining:
            raise ArchiveError(E_PAYLOAD, "бітовий потік закінчився")
        self._fill(n)
        self.nbits -= n
        self.remaining -= n
        self.acc &= (1 << self.nbits) - 1

    def read(self, n: int) -> int:
        if n == 0:
            return 0
        v = self.peek(n)
        self.skip(n)
        return v

    def bit(self) -> int:
        return self.read(1)
