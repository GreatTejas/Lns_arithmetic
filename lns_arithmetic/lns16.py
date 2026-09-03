"""
lns16.py
========
LNS16 format: 16 bits total = 1 sign bit + 1 zero-flag bit + 14-bit L-code
(Q6.8 fixed point: 6 integer bits, 8 fractional bits).

Dynamic range:  |x| in [2^-32, 2^31.99...]  (~ 4.3e-10 .. 4.3e9)
L resolution:   1/256 -> relative quantization step ~ ln(2)/256 ~ 0.27%
"""

import numpy as np
from .core import LNSFormat

FORMAT = LNSFormat("LNS16", total_bits=16, int_bits=6, frac_bits=8)


def fp32_to_lns16(x):
    """float / FP32 -> packed 16-bit LNS integer."""
    return FORMAT.encode(np.float32(x))


def fp16_to_lns16(x):
    """FP16 -> packed 16-bit LNS integer."""
    return FORMAT.encode(np.float16(x))


def lns16_to_fp32(bits):
    return np.float32(FORMAT.decode(bits))


def lns16_to_fp16(bits):
    return np.float16(FORMAT.decode(bits))


def lns16_add(a_bits, b_bits):
    return FORMAT.add(a_bits, b_bits)


def lns16_mul(a_bits, b_bits):
    return FORMAT.mul(a_bits, b_bits)


def lns16_mac(a_bits, b_bits, c_bits):
    """c + a*b in LNS16."""
    return FORMAT.mac(a_bits, b_bits, c_bits)


class LNS16:
    """Thin object wrapper around a packed LNS16 word with operator overloading."""

    __slots__ = ("bits",)
    fmt = FORMAT

    def __init__(self, value=0.0, *, bits=None):
        self.bits = bits if bits is not None else FORMAT.encode(value)

    @classmethod
    def from_bits(cls, bits):
        return cls(bits=bits)

    def to_float(self):
        return FORMAT.decode(self.bits)

    def __add__(self, other):
        return LNS16(bits=FORMAT.add(self.bits, other.bits))

    def __mul__(self, other):
        return LNS16(bits=FORMAT.mul(self.bits, other.bits))

    def mac(self, a, b):
        """returns self + a*b"""
        return LNS16(bits=FORMAT.mac(a.bits, b.bits, self.bits))

    def __float__(self):
        return self.to_float()

    def __repr__(self):
        return f"LNS16({self.to_float():.6g})"
