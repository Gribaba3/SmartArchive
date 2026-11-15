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
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0  # номер біта
        self.total = len(data) * 8

    def bit(self) -> int:
        if self.pos >= self.total:
            raise ArchiveError(E_PAYLOAD, "бітовий потік закінчився")
        b = (self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return b

    def read(self, nbits: int) -> int:
        v = 0
        for _ in range(nbits):
            v = (v << 1) | self.bit()
        return v
