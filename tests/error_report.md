# LNS Error Analysis Report

Relative error = |x_ref - x_LNS| / |x_ref|; absolute error used when x_ref == 0.

LNS16 dynamic range: [2.328e-10, 4.283e+09]  (L resolution 1/256)

LNS8 dynamic range: [0.003906, 215.3]  (L resolution 1/4)

## Full test population (includes values outside each format's dynamic range, i.e. overflow/underflow/saturation effects)

| Test | n | Mean error | Max error |
|---|---|---|---|
| LNS16 conversion (vs FP32) | 2286 | 6.648153e-04 | 1.354123e-03 |
| LNS16 add (vs FP32) | 3000 | 3.018761e-03 | 1.000000e+00 |
| LNS16 mul (vs FP32) | 3000 | 9.123881e-04 | 2.632461e-03 |
| LNS16 mac (vs FP32) | 3000 | 1.097694e-03 | 5.537882e-03 |
| LNS16 conversion (vs FP16) | 2092 | 6.289832e-04 | 1.736111e-03 |
| LNS8 conversion (vs FP32) | 2286 | 3.285307e-01 | 1.000000e+00 |
| LNS8 add (vs FP32) | 3000 | 8.078910e-01 | 1.591866e+01 |
| LNS8 mul (vs FP32) | 3000 | 9.912730e-01 | 9.997834e-01 |
| LNS8 mac (vs FP32) | 3000 | 9.968303e-01 | 3.047053e+00 |
| LNS8 conversion (vs FP16) | 2093 | 2.740274e-01 | 1.000000e+00 |

## In-range test population only (isolates pure quantization precision, excluding saturation)

| Test | n | Mean error | Max error |
|---|---|---|---|
| LNS16 conversion (vs FP32) | 2284 | 6.645050e-04 | 1.354072e-03 |
| LNS16 add (vs FP32) | 3000 | 3.623490e-03 | 1.000000e+00 |
| LNS16 mul (vs FP32) | 3000 | 9.067199e-04 | 2.679883e-03 |
| LNS16 mac (vs FP32) | 2989 | 3.252931e-03 | 1.000000e+00 |
| LNS16 conversion (vs FP16) | 2082 | 6.418034e-04 | 1.736111e-03 |
| LNS8 conversion (vs FP32) | 1267 | 4.108477e-02 | 9.048249e-02 |
| LNS8 add (vs FP32) | 3000 | 1.539248e-01 | 1.072289e+01 |
| LNS8 mul (vs FP32) | 2997 | 5.964093e-02 | 1.778366e-01 |
| LNS8 mac (vs FP32) | 2984 | 3.356081e-01 | 2.860986e+02 |
| LNS8 conversion (vs FP16) | 1246 | 4.357227e-02 | 9.042077e-02 |
