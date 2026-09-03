"""
error_analysis.py
==================
Generates positive, negative, zero, small, large and random test
vectors, runs them through the LNS16 and LNS8 pipelines, and reports
mean/max relative error (absolute error when the reference is zero)
against FP32 and FP16 references for:

  1. value conversion       (FP -> LNS -> FP)
  2. addition                a + b
  3. multiplication           a * b
  4. multiply-accumulate      c + a*b

Run with:   python tests/error_analysis.py
Writes a Markdown table to  tests/error_report.md
"""
import warnings
import numpy as np

from lns_arithmetic import (
    LNS16_FORMAT, LNS8_FORMAT,
    fp32_to_lns16, fp16_to_lns16, lns16_to_fp32, lns16_to_fp16,
    lns16_add, lns16_mul, lns16_mac,
    fp32_to_lns8, fp16_to_lns8, lns8_to_fp32, lns8_to_fp16,
    lns8_add, lns8_mul, lns8_mac,
)

warnings.filterwarnings("ignore")  # overflow/underflow warnings expected for extreme test values
RNG = np.random.default_rng(0)


def rel_err(ref, approx):
    ref = float(ref)
    approx = float(approx)
    if ref == 0.0:
        return abs(approx - ref)
    return abs(ref - approx) / abs(ref)


def gen_conversion_values():
    """Positive, negative, zero, small, large, and random values."""
    vals = [0.0, -0.0, 1.0, -1.0]
    vals += [2.0 ** k for k in range(-20, 21)]        # powers of two, small & large
    vals += [-2.0 ** k for k in range(-20, 21)]
    vals += list(RNG.uniform(-1.0, 1.0, 1000))          # small random
    vals += list(RNG.uniform(-1000.0, 1000.0, 1000))    # medium random
    vals += list(RNG.uniform(-1e6, 1e6, 200))           # large random
    return vals


def gen_op_pairs(n=3000):
    """Wide-range pairs, used for the 'full population' pass (exercises overflow/underflow)."""
    xs = RNG.uniform(-1000.0, 1000.0, n)
    ys = RNG.uniform(-1000.0, 1000.0, n)
    return xs, ys


def gen_op_pairs_scaled(fmt, op, n=3000):
    """Pairs scaled to `fmt`'s dynamic range, used for the 'in-range' pass so
    enough samples actually land inside the format's representable range
    (this matters most for LNS8, whose range is only ~[0.004, 215])."""
    if op == "mul":
        lim = max(1.0, (fmt.max_repr ** 0.5) * 0.9)
    else:
        lim = max(1.0, fmt.max_repr * 0.45)
    xs = RNG.uniform(-lim, lim, n)
    ys = RNG.uniform(-lim, lim, n)
    return xs, ys


def summarize(label, errs, n_total=None):
    errs = np.asarray(errs, dtype=np.float64)
    finite = np.isfinite(errs)
    n_total = n_total if n_total is not None else len(errs)
    return {
        "label": label,
        "mean": float(np.mean(errs[finite])) if finite.any() else float("nan"),
        "max": float(np.max(errs[finite])) if finite.any() else float("nan"),
        "n": int(finite.sum()),
        "n_total": n_total,
    }


def eval_conversion(to_lns, from_lns, ref_cast, fmt, in_range_only):
    errs = []
    for x in gen_conversion_values():
        ref = float(ref_cast(x))
        if in_range_only and not (fmt.min_repr <= abs(ref) <= fmt.max_repr):
            continue
        got = from_lns(to_lns(x))
        errs.append(rel_err(ref, got))
    return errs


def _in_range(fmt, *vals):
    return all((v == 0.0) or (fmt.min_repr <= abs(v) <= fmt.max_repr) for v in vals)


def eval_binary_op(op_name, to_lns, from_lns, lns_op, ref_cast, fmt, in_range_only):
    xs, ys = gen_op_pairs_scaled(fmt, op_name) if in_range_only else gen_op_pairs()
    errs = []
    for x, y in zip(xs, ys):
        rx, ry = float(ref_cast(x)), float(ref_cast(y))
        ref = (rx + ry) if op_name == "add" else (rx * ry)
        if in_range_only and not _in_range(fmt, rx, ry, ref):
            continue
        got = from_lns(lns_op(to_lns(x), to_lns(y)))
        errs.append(rel_err(ref, got))
    return errs


def eval_mac(to_lns, from_lns, lns_mac, ref_cast, fmt, in_range_only):
    xs, ys = gen_op_pairs_scaled(fmt, "mul", n=3000) if in_range_only else gen_op_pairs()
    if in_range_only:
        zs = RNG.uniform(-fmt.max_repr * 0.45, fmt.max_repr * 0.45, len(xs))
    else:
        zs = RNG.uniform(-1000.0, 1000.0, len(xs))
    errs = []
    for x, y, z in zip(xs, ys, zs):
        rx, ry, rz = float(ref_cast(x)), float(ref_cast(y)), float(ref_cast(z))
        ref = rz + rx * ry
        if in_range_only and not _in_range(fmt, rx, ry, rz, ref, rx * ry):
            continue
        got = from_lns(lns_mac(to_lns(x), to_lns(y), to_lns(z)))
        errs.append(rel_err(ref, got))
    return errs


def build_results(in_range_only):
    results = []

    # ---- LNS16 vs FP32 -------------------------------------------------
    results.append(summarize("LNS16 conversion (vs FP32)",
                              eval_conversion(fp32_to_lns16, lns16_to_fp32, np.float32, LNS16_FORMAT, in_range_only)))
    results.append(summarize("LNS16 add (vs FP32)",
                              eval_binary_op("add", fp32_to_lns16, lns16_to_fp32, lns16_add, np.float32, LNS16_FORMAT, in_range_only)))
    results.append(summarize("LNS16 mul (vs FP32)",
                              eval_binary_op("mul", fp32_to_lns16, lns16_to_fp32, lns16_mul, np.float32, LNS16_FORMAT, in_range_only)))
    results.append(summarize("LNS16 mac (vs FP32)",
                              eval_mac(fp32_to_lns16, lns16_to_fp32, lns16_mac, np.float32, LNS16_FORMAT, in_range_only)))

    # ---- LNS16 vs FP16 -------------------------------------------------
    results.append(summarize("LNS16 conversion (vs FP16)",
                              eval_conversion(fp16_to_lns16, lns16_to_fp16, np.float16, LNS16_FORMAT, in_range_only)))

    # ---- LNS8 vs FP32 ---------------------------------------------------
    results.append(summarize("LNS8 conversion (vs FP32)",
                              eval_conversion(fp32_to_lns8, lns8_to_fp32, np.float32, LNS8_FORMAT, in_range_only)))
    results.append(summarize("LNS8 add (vs FP32)",
                              eval_binary_op("add", fp32_to_lns8, lns8_to_fp32, lns8_add, np.float32, LNS8_FORMAT, in_range_only)))
    results.append(summarize("LNS8 mul (vs FP32)",
                              eval_binary_op("mul", fp32_to_lns8, lns8_to_fp32, lns8_mul, np.float32, LNS8_FORMAT, in_range_only)))
    results.append(summarize("LNS8 mac (vs FP32)",
                              eval_mac(fp32_to_lns8, lns8_to_fp32, lns8_mac, np.float32, LNS8_FORMAT, in_range_only)))

    # ---- LNS8 vs FP16 ----------------------------------------------------
    results.append(summarize("LNS8 conversion (vs FP16)",
                              eval_conversion(fp16_to_lns8, lns8_to_fp16, np.float16, LNS8_FORMAT, in_range_only)))
    return results


def print_and_collect_md(title, results, lines_md):
    header = f"{'Test':32s} {'n':>6s} {'mean err':>14s} {'max err':>14s}"
    print(f"\n{title}")
    print(header)
    print("-" * len(header))
    lines_md.append(f"\n## {title}\n")
    lines_md.append("| Test | n | Mean error | Max error |")
    lines_md.append("|---|---|---|---|")
    for r in results:
        print(f"{r['label']:32s} {r['n']:6d} {r['mean']:14.6e} {r['max']:14.6e}")
        lines_md.append(f"| {r['label']} | {r['n']} | {r['mean']:.6e} | {r['max']:.6e} |")


def main():
    lines_md = []

    full = build_results(in_range_only=False)
    print_and_collect_md("Full test population (includes values outside each format's "
                          "dynamic range, i.e. overflow/underflow/saturation effects)", full, lines_md)

    in_range = build_results(in_range_only=True)
    print_and_collect_md("In-range test population only (isolates pure quantization "
                          "precision, excluding saturation)", in_range, lines_md)

    with open("tests/error_report.md", "w") as f:
        f.write("# LNS Error Analysis Report\n\n")
        f.write("Relative error = |x_ref - x_LNS| / |x_ref|; absolute error used when x_ref == 0.\n")
        f.write(f"\nLNS16 dynamic range: [{LNS16_FORMAT.min_repr:.4g}, {LNS16_FORMAT.max_repr:.4g}]  "
                f"(L resolution 1/{LNS16_FORMAT.scale})\n")
        f.write(f"\nLNS8 dynamic range: [{LNS8_FORMAT.min_repr:.4g}, {LNS8_FORMAT.max_repr:.4g}]  "
                f"(L resolution 1/{LNS8_FORMAT.scale})\n")
        f.write("\n".join(lines_md) + "\n")

    print("\nWrote tests/error_report.md")


if __name__ == "__main__":
    main()
