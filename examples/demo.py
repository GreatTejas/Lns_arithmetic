"""
demo.py
=======
Minimal runnable demonstration of the lns_arithmetic package.
Run with:  python examples/demo.py
"""
from lns_arithmetic import (
    LNS16, LNS8,
    fp32_to_lns16, lns16_to_fp32, lns16_add, lns16_mul, lns16_mac,
    fp32_to_lns8, lns8_to_fp32, lns8_add, lns8_mul, lns8_mac,
)

print("== Functional API (LNS16) ==")
a = fp32_to_lns16(3.5)
b = fp32_to_lns16(-1.25)
c = fp32_to_lns16(0.5)
print("a =", lns16_to_fp32(a), " b =", lns16_to_fp32(b))
print("a + b =", lns16_to_fp32(lns16_add(a, b)), " (ref:", 3.5 + (-1.25), ")")
print("a * b =", lns16_to_fp32(lns16_mul(a, b)), " (ref:", 3.5 * (-1.25), ")")
print("c + a*b =", lns16_to_fp32(lns16_mac(a, b, c)), " (ref:", 0.5 + 3.5 * (-1.25), ")")

print("\n== Object-oriented API (LNS8) ==")
x = LNS8(6.0)
y = LNS8(-2.0)
z = LNS8(1.0)
print("x =", x, " y =", y)
print("x + y =", x + y, " (ref: 4.0)")
print("x * y =", x * y, " (ref: -12.0)")
print("z + x*y =", z.mac(x, y), " (ref: -11.0)")

print("\n== Overflow / underflow behaviour (LNS8, tiny range) ==")
print("encode(1e6) ->", lns8_to_fp32(fp32_to_lns8(1e6)), " (saturates to LNS8 max)")
print("encode(1e-9) ->", lns8_to_fp32(fp32_to_lns8(1e-9)), " (flushes to 0, below LNS8 min)")
