# lns-arithmetic

A small importable Python library implementing the **Logarithmic Number
System (LNS)** in two fixed-width formats, **LNS16** (16-bit) and
**LNS8** (8-bit), intended as a lightweight numeric backend for
experimenting with low-precision arithmetic in DNN training/inference.

The library supports:

- Conversion **FP32 → LNS16/LNS8** and **FP16 → LNS16/LNS8**, and back.
- **Addition**, **multiplication**, and **multiply–accumulate (MAC)**
  performed **entirely in the log domain** (operands are never converted
  back to linear floating point mid-computation).
- Correct handling of **sign, zero, overflow and underflow**, with
  rounding of the quantized log value.
- A test suite (`pytest`) and an **error-analysis script** that reports
  mean/max relative error against FP32/FP16 references.

---

## 1. Representation

A nonzero real number `x` is represented as

```
x = (-1)^sign * 2^L
```

`L = log2(|x|)` is stored as a **fixed-point two's-complement integer**
("L-code") with `int_bits` integer bits and `frac_bits` fractional bits.
Zero is represented with a dedicated flag bit rather than a reserved
L-code, so the full L-code range stays available for real magnitudes.
Each word is packed MSB-first as:

```
[ sign (1 bit) | zero_flag (1 bit) | L-code (int_bits + frac_bits bits) ]
```

| Format | Total bits | Sign | Zero flag | L-code bits | Int.frac | Dynamic range `|x|` | L resolution |
|---|---|---|---|---|---|---|---|
| **LNS16** | 16 | 1 | 1 | 14 | Q6.8 | `[2.3e-10, 4.28e9]` | 1/256 ≈ 0.0039 |
| **LNS8**  | 8  | 1 | 1 | 6  | Q4.2 | `[3.9e-3, 215.3]`   | 1/4 ≈ 0.25 |

The integer bits of the L-code set the **dynamic range**; the
fractional bits set the **precision** (quantization step of `log2|x|`,
which translates to a roughly constant *relative* error, unlike FP
formats where the *relative* error is set by the mantissa width and is
independent of range, but the range itself is set by the exponent
width — see §4 for a direct comparison).

## 2. Arithmetic in the log domain

- **Multiplication** is exact integer addition of L-codes plus XOR of
  sign bits, since `log(a·b) = log(a) + log(b)`. Only final saturation
  to the format's range introduces error.

- **Addition/subtraction** cannot be done by simply adding L-codes;
  LNS addition uses the standard **Gaussian logarithm** (a.k.a.
  co-transformation) identities. With `a`, `b` such that `|a| ≥ |b|`
  and `d = log2|b| - log2|a| ≤ 0`:

  ```
  same sign:      log2|a + b| = log2|a| + log2(1 + 2^d)
  opposite sign:   log2|a - b| = log2|a| + log2(1 - 2^d)   (a ≠ b)
  ```

  The correction functions `sb(d) = log2(1+2^d)` and `db(d) = log2(1-2^d)`
  are evaluated as a function of the **log-domain exponent difference
  `d`** only — never on the original linear-domain operands. This is
  exactly the quantity a hardware LNS ALU would use to index a
  lookup table (LUT) or piecewise-polynomial approximator for `sb`/`db`;
  this simulation evaluates them directly (`math.log2`) for numerical
  fidelity, and the LUT/approximation angle is called out in
  `core.py` as the natural hardware-realizable substitute.

- **MAC** (`c + a*b`) chains the two: multiply `a*b` in the log domain,
  then add the product to `c` using the Gaussian-logarithm addition
  above.

- **Overflow** (`log2|result|` exceeds the format's max L) **saturates**
  to the format's maximum representable magnitude (with a `warnings.warn`).
- **Underflow** (`log2|result|` is below the format's min L) **flushes
  to zero** (with a `warnings.warn`).
- **Exact cancellation** (`a + (-a)`) returns exact zero.
- **Zero operands** are handled explicitly (`0 * x = 0`, `0 + x = x`)
  without going through the log-domain formulas above (log(0) is
  undefined).

## 3. Repository layout

```
lns-arithmetic/
├── lns_arithmetic/            # the importable package
│   ├── __init__.py            # public API re-exports
│   ├── core.py                # LNSFormat: bit packing, encode/decode, add/mul/mac
│   ├── lns16.py                # LNS16 format instance + convenience functions/class
│   └── lns8.py                 # LNS8 format instance + convenience functions/class
├── tests/
│   ├── test_correctness.py     # pytest unit tests: round-trip, sign, zero,
│   │                            # overflow/underflow, add/mul/mac accuracy
│   ├── error_analysis.py       # generates error_report.md (see §4)
│   └── error_report.md          # generated report (mean/max error tables)
├── examples/
│   └── demo.py                  # minimal runnable usage example
├── pyproject.toml               # packaging metadata (pip install -e .)
├── requirements.txt
└── README.md                    # this file
```

### File-by-file details

- **`lns_arithmetic/core.py`** — The engine. Defines `LNSFormat`, a
  class parameterized by `(total_bits, int_bits, frac_bits)` so that
  LNS16 and LNS8 share one implementation. Implements bit-level
  `pack`/`unpack`, `encode` (float → packed LNS word, with
  overflow/underflow/NaN handling), `decode` (packed LNS word → float),
  and the log-domain `add`, `mul`, `mac` methods described in §2.

- **`lns_arithmetic/lns16.py`** — Instantiates `LNSFormat("LNS16", 16, 6, 8)`
  and exposes: `fp32_to_lns16`, `fp16_to_lns16`, `lns16_to_fp32`,
  `lns16_to_fp16`, `lns16_add`, `lns16_mul`, `lns16_mac`, plus an
  object-oriented `LNS16` wrapper with `+`, `*`, `.mac()`, `float()`.

- **`lns_arithmetic/lns8.py`** — Same as above, instantiating
  `LNSFormat("LNS8", 8, 4, 2)`, exposing the `lns8_*` functions and the
  `LNS8` class.

- **`lns_arithmetic/__init__.py`** — Re-exports the public API so users
  can `from lns_arithmetic import fp32_to_lns16, lns16_add, LNS16, ...`.

- **`tests/test_correctness.py`** — `pytest` suite covering: zero
  round-trip, positive/negative round-trip, sign propagation through
  multiplication, multiplication/addition/MAC numerical accuracy
  against float references (within format-appropriate tolerance),
  addition with zero, exact cancellation, overflow saturation
  (with warning check), underflow flush-to-zero (with warning check),
  bit-width containment (packed words never exceed 16/8 bits), and
  equivalence of the functional vs. object-oriented APIs.

- **`tests/error_analysis.py`** — Generates positive, negative, zero,
  small, large, and randomly sampled FP32/FP16 test vectors; runs them
  through conversion, add, mul, and mac for both LNS16 and LNS8;
  computes relative error `|x_ref - x_LNS| / |x_ref|` (absolute error
  when `x_ref == 0`); and writes `tests/error_report.md` with mean/max
  error tables. It reports **two** populations per test: the full
  random population (which includes values outside a format's dynamic
  range, showing saturation effects) and an **in-range-only**
  population (isolating pure quantization precision from range
  effects) — this split matters a lot for LNS8, whose range is tiny.

- **`examples/demo.py`** — A short, directly runnable script showing
  both the functional and object-oriented APIs, and overflow/underflow
  behavior.

- **`pyproject.toml`** — Makes the package `pip install`-able
  (`pip install -e .`), with `numpy` as the only runtime dependency
  (used only for casting reference/test values to FP32/FP16 precision).

## 4. Error analysis results

Generated by `python tests/error_analysis.py` (see `tests/error_report.md`
for the raw output; summarized below). 2,000–3,000 test vectors per row,
covering positive, negative, zero, small (`~2^-20`), large (`~2^20`,
and uniform up to `1e6`), and random values.

**Full population** (includes out-of-range values → saturation/flush effects included):

| Test | Mean rel. error | Max rel. error |
|---|---|---|
| LNS16 conversion (vs FP32) | 6.6e-04 | 1.4e-03 |
| LNS16 add (vs FP32) | 3.0e-03 | 1.0 * |
| LNS16 mul (vs FP32) | 9.1e-04 | 2.6e-03 |
| LNS16 mac (vs FP32) | 1.1e-03 | 5.5e-03 |
| LNS8 conversion (vs FP32) | 3.3e-01 | 1.0 |
| LNS8 add (vs FP32) | 8.1e-01 | 15.9 |
| LNS8 mul (vs FP32) | 9.9e-01 | 1.0 |
| LNS8 mac (vs FP32) | 9.97e-01 | 3.0 |

**In-range population only** (isolates quantization precision from range/saturation):

| Test | Mean rel. error | Max rel. error |
|---|---|---|
| LNS16 conversion (vs FP32) | 6.6e-04 | 1.4e-03 |
| LNS16 add (vs FP32) | 3.6e-03 | 1.0 * |
| LNS16 mul (vs FP32) | 9.1e-04 | 2.7e-03 |
| LNS16 mac (vs FP32) | 3.3e-03 | 1.0 * |
| LNS8 conversion (vs FP32) | 4.1e-02 | 9.0e-02 |
| LNS8 add (vs FP32) | 1.5e-01 | 10.7 * |
| LNS8 mul (vs FP32) | 6.0e-02 | 1.8e-01 |
| LNS8 mac (vs FP32) | 3.4e-01 | 286 * |

\* The occasional very large max errors on `add`/`mac` come from
**catastrophic cancellation**: when two nearly-equal-magnitude,
opposite-sign operands are added, the true result is close to zero,
so even a tiny absolute quantization error produces a huge *relative*
error (this is a property of relative error under cancellation, not
specific to LNS — it affects any finite-precision system, including
FP16/FP32, though LNS's coarser mantissa-equivalent precision makes it
show up more often and more severely). It is exactly why the report
also gives the *mean*, not just the max.

### Discussion: LNS vs. FP32/FP16

- **Dynamic range.** LNS16's 6 integer L-bits give a huge dynamic range
  (`~2.3e-10` to `~4.3e9`), comparable to or exceeding FP32's normal
  range, in only 16 bits — because in LNS, range is bought with
  *integer* L-bits which cost far less than FP32's 8 exponent bits +
  23 mantissa bits per unit of range. LNS8, by contrast, has only 4
  integer L-bits, so its range (`~0.004` to `~215`) is much narrower
  than FP16's (`~6e-5` to `~65504`) — most "everyday" DNN activation/
  weight magnitudes will fit, but large accumulator sums or very small
  gradients will saturate or flush to zero, as the "full population"
  numbers above show.

- **Precision.** In LNS, the *relative* quantization error is
  approximately constant across the entire dynamic range (≈ `ln(2) ·
  2^-frac_bits / 2`), unlike FP formats where relative error is
  constant *within* one exponent range but is set independently by the
  mantissa width. LNS16's 8 fractional L-bits give ≈0.07–0.3% relative
  error per operation (roughly comparable to an 8-9 bit mantissa, i.e.
  coarser than FP16's ~11-bit mantissa but with far greater range).
  LNS8's 2 fractional L-bits give a much coarser ≈4–15% relative error
  per operation — usable for very aggressive quantization experiments
  or gating/attention-style low-precision paths, but not for
  accumulating long reduction chains without expecting visible drift.

- **Multiplication is (comparatively) "free" and exact up to rounding**
  in both formats, since it's a single fixed-point addition — this is
  LNS's classic advantage for MAC-heavy DNN workloads. **Addition is
  the expensive/lossy operation**, since it requires the Gaussian-log
  correction and is where most of the error budget (and all of the
  cancellation blow-ups) comes from — the opposite trade-off from FP,
  where addition is "free" (align-and-add) and multiplication requires
  a full mantissa multiply.

## 5. Installation & usage

```bash
git clone <this-repo-url>
cd lns-arithmetic
pip install -e .            # installs the package (numpy is the only dependency)
pip install pytest          # only needed to run the test suite
```

```python
from lns_arithmetic import fp32_to_lns16, lns16_to_fp32, lns16_add, lns16_mul

a = fp32_to_lns16(3.5)
b = fp32_to_lns16(-1.25)
print(lns16_to_fp32(lns16_add(a, b)))   # -> 2.25 (approx)
print(lns16_to_fp32(lns16_mul(a, b)))   # -> -4.375 (approx)
```

or with the object-oriented wrapper:

```python
from lns_arithmetic import LNS16
x, y = LNS16(3.5), LNS16(-1.25)
print(float(x + y), float(x * y))
```

Run the tests:

```bash
pytest -q
```

Regenerate the error-analysis report:

```bash
python tests/error_analysis.py
```

## 6. Known limitations / next steps

- The Gaussian-log correction functions (`sb`, `db`) are evaluated
  directly via `math.log2`/exponentiation rather than via a
  hardware-realistic LUT or piecewise-polynomial approximator; a LUT-
  based `add` (indexed and interpolated on `d`) would be a natural
  extension for anyone wanting to simulate actual hardware error, and
  the interface (`LNSFormat.add`) is written so that swap is
  localized to one method.
- Subnormal-style "graceful underflow" (keeping reduced precision
  below the normal range, as FP32/FP16 subnormals do) is not
  implemented; this library flushes to zero on underflow.
- No NumPy-vectorized batch API yet (all functions operate on scalars);
  wrapping `encode`/`decode`/`add`/`mul` in `np.vectorize` or a real
  vectorized implementation would help throughput for the eventual
  training/inference integration mentioned in the assignment.
