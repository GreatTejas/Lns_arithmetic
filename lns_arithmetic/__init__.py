"""
lns_arithmetic
==============
A small importable Python library implementing the Logarithmic Number
System (LNS) in two fixed-width formats, LNS16 and LNS8, with FP32/FP16
conversion and log-domain addition, multiplication and multiply-
accumulate (MAC).

Quick start
-----------
>>> from lns_arithmetic import fp32_to_lns16, lns16_to_fp32, lns16_add, lns16_mul
>>> a = fp32_to_lns16(3.5)
>>> b = fp32_to_lns16(-1.25)
>>> lns16_to_fp32(lns16_add(a, b))
2.25
>>> lns16_to_fp32(lns16_mul(a, b))
-4.375

Or with the object-oriented wrappers:

>>> from lns_arithmetic import LNS16
>>> x = LNS16(3.5)
>>> y = LNS16(-1.25)
>>> float(x + y)
2.25
"""

from .core import LNSFormat
from .lns16 import (
    FORMAT as LNS16_FORMAT,
    LNS16,
    fp32_to_lns16,
    fp16_to_lns16,
    lns16_to_fp32,
    lns16_to_fp16,
    lns16_add,
    lns16_mul,
    lns16_mac,
)
from .lns8 import (
    FORMAT as LNS8_FORMAT,
    LNS8,
    fp32_to_lns8,
    fp16_to_lns8,
    lns8_to_fp32,
    lns8_to_fp16,
    lns8_add,
    lns8_mul,
    lns8_mac,
)

__version__ = "0.1.0"

__all__ = [
    "LNSFormat",
    "LNS16_FORMAT", "LNS16",
    "fp32_to_lns16", "fp16_to_lns16", "lns16_to_fp32", "lns16_to_fp16",
    "lns16_add", "lns16_mul", "lns16_mac",
    "LNS8_FORMAT", "LNS8",
    "fp32_to_lns8", "fp16_to_lns8", "lns8_to_fp32", "lns8_to_fp16",
    "lns8_add", "lns8_mul", "lns8_mac",
]
