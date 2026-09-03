"""
lns8.py
=======
LNS8 format: 8 bits total = 1 sign bit + 1 zero-flag bit + 6-bit L-code
(Q4.2 fixed point: 4 integer bits, 2 fractional bits).

Dynamic range:  |x| in [2^-8, 2^7.75]  (~ 0.0039 .. ~215)
L resolution:   1/4 -> relative quantization step ~ ln(2)/4 ~ 17%

LNS8 trades most of its precision for a still-usable dynamic range in
an 8-bit budget; see the README / error_analysis report for measured
error statistics.
"""

import numpy as np
from .core import LNSFormat

FORMAT = LNSFormat("LNS8", total_bits=8, int_bits=4, frac_bits=2)


def fp32_to_lns8(x):
    return FORMAT.encode(np.float32(x))


def fp16_to_lns8(x):
    return FORMAT.encode(np.float16(x))


def lns8_to_fp32(bits):
    return np.float32(FORMAT.decode(bits))


def lns8_to_fp16(bits):
    return np.float16(FORMAT.decode(bits))


def lns8_add(a_bits, b_bits):
    return FORMAT.add(a_bits, b_bits)


def lns8_mul(a_bits, b_bits):
    return FORMAT.mul(a_bits, b_bits)


def lns8_mac(a_bits, b_bits, c_bits):
    """c + a*b in LNS8."""
    return FORMAT.mac(a_bits, b_bits, c_bits)


class LNS8:
    """Thin object wrapper around a packed LNS8 word with operator overloading."""

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
        return LNS8(bits=FORMAT.add(self.bits, other.bits))

    def __mul__(self, other):
        return LNS8(bits=FORMAT.mul(self.bits, other.bits))

    def mac(self, a, b):
        """returns self + a*b"""
        return LNS8(bits=FORMAT.mac(a.bits, b.bits, self.bits))

    def __float__(self):
        return self.to_float()

    def __repr__(self):
        return f"LNS8({self.to_float():.6g})"
