"""
core.py
=======
Format-agnostic core of the Logarithmic Number System (LNS) library.

Representation
--------------
A real number x is represented as

    x = (-1)^sign * 2^L        if x != 0
    x = 0                      if the zero flag is set

L ("the log value") is stored as a fixed-point two's-complement integer
("L-code") with `int_bits` bits before the binary point and `frac_bits`
bits after it. A word is packed, MSB first, as:

    [ sign (1 bit) | zero_flag (1 bit) | L-code (int_bits + frac_bits bits) ]

so total_bits = 2 + int_bits + frac_bits.

The zero flag is a dedicated bit rather than a reserved L-code so that
the full L-code range is available for genuine (nonzero) magnitudes.

All arithmetic (add, mul, mac) is performed on the packed integer
representation / L-codes directly -- operands are never converted back
to a linear floating point number as part of computing a result. This
mirrors how LNS arithmetic units are built in hardware:

  * Multiplication of two LNS numbers is *exact* integer addition of
    their L-codes (log(a*b) = log(a) + log(b)).

  * Addition of two LNS numbers requires the "Gaussian logarithm"
    correction functions

        sb(d) = log2(1 + 2^d)      (same-sign addition)
        db(d) = log2(1 - 2^d)      (opposite-sign addition / subtraction)

    where d = Lb - La <= 0 is the (log-domain) exponent *difference*
    between the two operands. In real LNS hardware these functions are
    realised with a lookup table (LUT) or a piecewise-polynomial
    approximation, indexed purely by d; this module evaluates them
    directly for numerical accuracy, but note that the only
    transcendental evaluation involved operates on the log-domain
    difference `d`, never on the original linear operands -- exactly
    the quantity a LUT-based hardware unit would be indexed by.
"""

import math
import warnings


class LNSFormat:
    """Describes one LNS bit layout (e.g. LNS16, LNS8) and implements
    encode/decode and log-domain add/mul/mac for that layout."""

    def __init__(self, name, total_bits, int_bits, frac_bits):
        if total_bits != 2 + int_bits + frac_bits:
            raise ValueError("total_bits must equal 2 + int_bits + frac_bits")
        self.name = name
        self.total_bits = total_bits
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.L_bits = int_bits + frac_bits          # bits used for the L-code
        self.scale = 1 << frac_bits                  # fixed-point scale factor

        # L-code is a two's-complement signed integer with L_bits bits.
        self.L_code_max = (1 << (self.L_bits - 1)) - 1
        self.L_code_min = -(1 << (self.L_bits - 1))
        self.L_val_max = self.L_code_max / self.scale     # largest representable log2|x|
        self.L_val_min = self.L_code_min / self.scale      # smallest representable log2|x|
        self.max_repr = 2.0 ** self.L_val_max
        self.min_repr = 2.0 ** self.L_val_min

        self._L_mask = (1 << self.L_bits) - 1

    # ------------------------------------------------------------------ #
    # bit packing
    # ------------------------------------------------------------------ #
    def pack(self, sign, zero_flag, L_code):
        L_code = max(self.L_code_min, min(self.L_code_max, L_code))
        bits = (sign & 1) << (self.total_bits - 1)
        bits |= (zero_flag & 1) << (self.total_bits - 2)
        bits |= L_code & self._L_mask
        return bits

    def unpack(self, bits):
        """Returns (sign, zero_flag, L_code) where L_code is a signed int."""
        sign = (bits >> (self.total_bits - 1)) & 1
        zero_flag = (bits >> (self.total_bits - 2)) & 1
        raw = bits & self._L_mask
        if raw >= (1 << (self.L_bits - 1)):
            raw -= (1 << self.L_bits)
        return sign, zero_flag, raw

    # ------------------------------------------------------------------ #
    # encode (float -> LNS) / decode (LNS -> float)
    # ------------------------------------------------------------------ #
    def encode(self, x):
        x = float(x)
        if math.isnan(x):
            warnings.warn(f"{self.name}: NaN input encoded as zero")
            return self.pack(0, 1, 0)
        if x == 0.0:
            return self.pack(0, 1, 0)

        sign = 1 if x < 0 else 0
        mag = abs(x)

        if math.isinf(mag):
            warnings.warn(f"{self.name}: +/-inf input saturated to max representable magnitude")
            L_val = self.L_val_max
        else:
            L_val = math.log2(mag)
            if L_val > self.L_val_max:
                warnings.warn(
                    f"{self.name}: overflow encoding {x!r} "
                    f"(|x| > {self.max_repr:.4g}); saturated to max representable value"
                )
                L_val = self.L_val_max
            elif L_val < self.L_val_min:
                warnings.warn(
                    f"{self.name}: underflow encoding {x!r} "
                    f"(|x| < {self.min_repr:.4g}); flushed to zero"
                )
                return self.pack(0, 1, 0)

        L_code = round(L_val * self.scale)
        return self.pack(sign, 0, L_code)

    def decode(self, bits):
        sign, zero_flag, L_code = self.unpack(bits)
        if zero_flag:
            return 0.0
        mag = 2.0 ** (L_code / self.scale)
        return -mag if sign else mag

    # ------------------------------------------------------------------ #
    # log-domain arithmetic
    # ------------------------------------------------------------------ #
    def mul(self, a_bits, b_bits):
        sa, za, La = self.unpack(a_bits)
        sb, zb, Lb = self.unpack(b_bits)
        if za or zb:
            return self.pack(0, 1, 0)

        sign = sa ^ sb
        L_code = La + Lb  # integer addition of L-codes == multiplication in linear domain

        if L_code > self.L_code_max:
            warnings.warn(f"{self.name}: multiply overflow, result saturated")
            L_code = self.L_code_max
        elif L_code < self.L_code_min:
            warnings.warn(f"{self.name}: multiply underflow, result flushed to zero")
            return self.pack(0, 1, 0)

        return self.pack(sign, 0, L_code)

    def add(self, a_bits, b_bits):
        sa, za, La = self.unpack(a_bits)
        sb, zb, Lb = self.unpack(b_bits)

        if za and zb:
            return self.pack(0, 1, 0)
        if za:
            return b_bits
        if zb:
            return a_bits

        # Work with the larger-magnitude operand as the "base" (La >= Lb).
        if La < Lb or (La == Lb and sa > sb):
            La, Lb = Lb, La
            sa, sb = sb, sa

        d = (Lb - La) / self.scale  # <= 0 : log-domain exponent difference

        if sa == sb:
            # same sign: |sum| = |big| * (1 + 2^d)  ->  L = La + log2(1 + 2^d)
            correction = math.log2(1.0 + 2.0 ** d)
            result_sign = sa
        else:
            if La == Lb:
                # equal magnitude, opposite sign: exact cancellation
                return self.pack(0, 1, 0)
            # opposite sign: |sum| = |big| * (1 - 2^d)  ->  L = La + log2(1 - 2^d)
            correction = math.log2(1.0 - 2.0 ** d)
            result_sign = sa

        L_val = La / self.scale + correction

        if L_val > self.L_val_max:
            warnings.warn(f"{self.name}: add overflow, result saturated")
            L_val = self.L_val_max
        elif L_val < self.L_val_min:
            warnings.warn(f"{self.name}: add underflow, result flushed to zero")
            return self.pack(0, 1, 0)

        L_code = round(L_val * self.scale)
        return self.pack(result_sign, 0, L_code)

    def mac(self, a_bits, b_bits, c_bits):
        """Computes c + a*b entirely in the log domain."""
        prod_bits = self.mul(a_bits, b_bits)
        return self.add(prod_bits, c_bits)

    def __repr__(self):
        return (f"LNSFormat({self.name}, total_bits={self.total_bits}, "
                f"int_bits={self.int_bits}, frac_bits={self.frac_bits}, "
                f"range=[{-self.max_repr:.4g}, {self.max_repr:.4g}])")
