"""
test_correctness.py
====================
Unit tests covering round-trip conversion, add/mul/mac correctness
(within format tolerance), sign handling, zero handling, and
overflow/underflow behaviour, for both LNS16 and LNS8.

Run with:  pytest -q
"""
import math
import warnings
import pytest
import numpy as np

from lns_arithmetic import (
    LNS16_FORMAT, fp32_to_lns16, lns16_to_fp32, lns16_add, lns16_mul, lns16_mac,
    LNS8_FORMAT, fp32_to_lns8, lns8_to_fp32, lns8_add, lns8_mul, lns8_mac,
    LNS16, LNS8,
)

FORMATS = [
    ("LNS16", fp32_to_lns16, lns16_to_fp32, lns16_add, lns16_mul, lns16_mac, LNS16_FORMAT, 0.01),
    ("LNS8", fp32_to_lns8, lns8_to_fp32, lns8_add, lns8_mul, lns8_mac, LNS8_FORMAT, 0.30),
]


def rel_err(ref, approx):
    if ref == 0:
        return abs(approx - ref)
    return abs(ref - approx) / abs(ref)


@pytest.mark.parametrize("name,to_lns,from_lns,add,mul,mac,fmt,tol", FORMATS)
def test_zero_roundtrip(name, to_lns, from_lns, add, mul, mac, fmt, tol):
    assert from_lns(to_lns(0.0)) == 0.0
    assert from_lns(to_lns(-0.0)) == 0.0


@pytest.mark.parametrize("name,to_lns,from_lns,add,mul,mac,fmt,tol", FORMATS)
def test_roundtrip_positive_negative(name, to_lns, from_lns, add, mul, mac, fmt, tol):
    for x in [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.75, -3.75]:
        y = from_lns(to_lns(x))
        assert rel_err(x, y) < tol, f"{name} roundtrip failed for {x}: got {y}"
        assert (y < 0) == (x < 0)


@pytest.mark.parametrize("name,to_lns,from_lns,add,mul,mac,fmt,tol", FORMATS)
def test_sign_preserved(name, to_lns, from_lns, add, mul, mac, fmt, tol):
    a = to_lns(2.0)
    b = to_lns(-2.0)
    assert from_lns(mul(a, b)) < 0
    assert from_lns(mul(b, b)) > 0


@pytest.mark.parametrize("name,to_lns,from_lns,add,mul,mac,fmt,tol", FORMATS)
def test_multiplication_accuracy(name, to_lns, from_lns, add, mul, mac, fmt, tol):
    pairs = [(2.0, 3.0), (-2.0, 3.0), (2.0, -3.0), (-2.0, -3.0), (0.5, 4.0), (7.0, 7.0)]
    for x, y in pairs:
        ref = x * y
        got = from_lns(mul(to_lns(x), to_lns(y)))
        assert rel_err(ref, got) < tol, f"{name} mul({x},{y}) ref={ref} got={got}"


@pytest.mark.parametrize("name,to_lns,from_lns,add,mul,mac,fmt,tol", FORMATS)
def test_multiplication_by_zero(name, to_lns, from_lns, add, mul, mac, fmt, tol):
    assert from_lns(mul(to_lns(5.0), to_lns(0.0))) == 0.0
    assert from_lns(mul(to_lns(0.0), to_lns(-5.0))) == 0.0


@pytest.mark.parametrize("name,to_lns,from_lns,add,mul,mac,fmt,tol", FORMATS)
def test_addition_accuracy(name, to_lns, from_lns, add, mul, mac, fmt, tol):
    pairs = [(2.0, 3.0), (-2.0, -3.0), (5.0, -2.0), (-5.0, 2.0), (1.0, 1.0), (10.0, 0.01)]
    for x, y in pairs:
        ref = x + y
        got = from_lns(add(to_lns(x), to_lns(y)))
        assert rel_err(ref, got) < tol, f"{name} add({x},{y}) ref={ref} got={got}"


@pytest.mark.parametrize("name,to_lns,from_lns,add,mul,mac,fmt,tol", FORMATS)
def test_addition_with_zero(name, to_lns, from_lns, add, mul, mac, fmt, tol):
    a = to_lns(3.25)
    z = to_lns(0.0)
    assert rel_err(3.25, from_lns(add(a, z))) < tol
    assert rel_err(3.25, from_lns(add(z, a))) < tol


@pytest.mark.parametrize("name,to_lns,from_lns,add,mul,mac,fmt,tol", FORMATS)
def test_exact_cancellation(name, to_lns, from_lns, add, mul, mac, fmt, tol):
    a = to_lns(4.0)
    b = to_lns(-4.0)
    assert from_lns(add(a, b)) == 0.0


@pytest.mark.parametrize("name,to_lns,from_lns,add,mul,mac,fmt,tol", FORMATS)
def test_mac_accuracy(name, to_lns, from_lns, add, mul, mac, fmt, tol):
    # Note: (-2, 3, 5) is a near-cancellation case (-6 + 5 = -1): a small
    # *relative* error in the large intermediate product a*b becomes a much
    # larger relative error in the small final result. This is an inherent
    # property of limited-precision arithmetic under cancellation (not
    # specific to LNS), so it gets a looser tolerance here.
    triples = [(2.0, 3.0, 1.0), (0.5, 0.5, -0.1), (4.0, -4.0, 16.0)]
    for a, b, c in triples:
        ref = c + a * b
        got = from_lns(mac(to_lns(a), to_lns(b), to_lns(c)))
        assert rel_err(ref, got) < tol, f"{name} mac({a},{b},{c}) ref={ref} got={got}"

    cancellation_tol = 0.05 if name == "LNS16" else 0.6
    a, b, c = -2.0, 3.0, 5.0
    ref = c + a * b
    got = from_lns(mac(to_lns(a), to_lns(b), to_lns(c)))
    assert rel_err(ref, got) < cancellation_tol, f"{name} mac({a},{b},{c}) ref={ref} got={got}"


@pytest.mark.parametrize("name,to_lns,from_lns,add,mul,mac,fmt,tol", FORMATS)
def test_overflow_saturates(name, to_lns, from_lns, add, mul, mac, fmt, tol):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bits = to_lns(1e300)
        assert any("overflow" in str(x.message) for x in w)
    val = from_lns(bits)
    assert math.isfinite(val)
    assert rel_err(fmt.max_repr, abs(val)) < 1e-6


@pytest.mark.parametrize("name,to_lns,from_lns,add,mul,mac,fmt,tol", FORMATS)
def test_underflow_flushes_to_zero(name, to_lns, from_lns, add, mul, mac, fmt, tol):
    # Use a value that is still representable in FP32 (min normal ~1.18e-38)
    # but well below both LNS16's (~2.3e-10) and LNS8's (~0.0039) minimum
    # representable magnitude, so the underflow path in `encode` is
    # exercised rather than FP32 itself already flushing to zero.
    tiny = 1e-20
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bits = to_lns(tiny)
        assert any("underflow" in str(x.message) for x in w)
    assert from_lns(bits) == 0.0


def test_bit_width_is_respected():
    assert LNS16_FORMAT.total_bits == 16
    assert LNS8_FORMAT.total_bits == 8
    # No packed word should ever require more bits than the format allows.
    for x in [1234.5, -0.0001, 1e9, -1e9]:
        b16 = fp32_to_lns16(x)
        b8 = fp32_to_lns8(x)
        assert 0 <= b16 < (1 << 16)
        assert 0 <= b8 < (1 << 8)


def test_oo_wrapper_matches_functional_api():
    x = LNS16(3.0)
    y = LNS16(4.0)
    assert rel_err(7.0, float(x + y)) < 0.01
    assert rel_err(12.0, float(x * y)) < 0.01

    xs = LNS8(3.0)
    ys = LNS8(4.0)
    assert rel_err(7.0, float(xs + ys)) < 0.3
    assert rel_err(12.0, float(xs * ys)) < 0.3


def rel_err(ref, approx):
    if ref == 0:
        return abs(approx - ref)
    return abs(ref - approx) / abs(ref)
