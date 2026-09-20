# Fixed-point feature extraction

`docs/quantization.md` establishes that the **network** survives int16
quantisation at 0.010% score error. It quantises the 21 features as *inputs*,
but does not address **computing** them from the 208x3 window — which is the
larger half of the work (~5,000 float ops vs ~3,000 for inference) and, on a
CPU without an FPU, the more expensive half by far.

This document covers that missing half: computing `get_slice_params()` in
integer arithmetic only.

Implementation: `lib/fixed_features.py`. Validation: `scripts/validate_fixed.py`.

---

## 1. Why it matters

The target CPU (Simply-V / CV32E40P) is built with `FPU = 0`. Every `float`
operation therefore becomes a libgcc soft-float call (`__mulsf3`, `__divsf3`,
…) costing ~25–60 cycles instead of ~1. Feature extraction in fp32 is
roughly 10 ms per window; in fixed point it is roughly 0.5 ms — a ~20x
improvement that costs **no custom hardware and no FPU**.

---

## 2. Format

Samples are **int16 in Q2.14** (value = raw / 16384, range ±2.0).

The FHI data sets peak at `|x| = 1.0159`, so Q1.15 (±1.0) would clip and Q2.14
fits with headroom. A power-of-two scale keeps conversion to shifts rather
than multiplies. All 21 output features are returned in the same Q2.14 format.

---

## 3. The one non-obvious result: use the exact moment forms

The natural formulation computes means first, then subtracts:

```
var = E[X^2] - E[X]^2
cov = E[XY]  - E[X]E[Y]
```

This rounds the means to Q2.14 **before** the subtraction. For the z axis
(mean 0.846, std 0.291) and for all three correlations, that subtraction
cancels away most of the significant bits, so the rounding error in the mean
dominates what is left.

Measured on 2,166 FHI / FHI-BREAK windows:

| Feature | Naive form | Exact form | Improvement |
|---|---|---|---|
| `corr_xz` | **10.400%** | **0.052%** | 200x |
| `corr_yz` | 10.814% | 0.075% | 144x |
| `corr_xy` | 1.315% | 0.102% | 13x |
| `std_z` | 0.424% | 0.014% | 30x |

The fix is to use forms that never round before subtracting:

```
N^2 * var = N*sum(x^2) - sum(x)^2
N^2 * cov = N*sum(xy)  - sum(x)*sum(y)
```

Both right-hand sides are **exact integers** — every term is an exact sum of
products of int16 values. And because `corr = cov / (sx * sy)`, the `N^2`
cancels entirely, so correlation needs no division by N at all:

```
corr = (N*sum(xy) - sum(x)sum(y)) / sqrt(Vx * Vy),   Vx = N*sum(x^2) - sum(x)^2
```

A smaller version of the same idea applies to `rms`: `isqrt(sum(x^2) * N) / N`
keeps log2(N) more bits than `isqrt(sum(x^2) / N)`.

---

## 4. Operations required

| Features | Method | Notes |
|---|---|---|
| 0–8 peak/min/max | comparisons | no arithmetic |
| 12–14 mean | `sum(x)`, reciprocal multiply | `round(2^22/208) = 20165` |
| 9–11 rms | `isqrt(sum(x^2)*N)/N` | int64 accumulator |
| 15–17 std | `isqrt(V)/N`, `V = N*sum(x^2)-sum(x)^2` | reuses `sum(x^2)` |
| 18–20 corr | `(num << 14) / (isqrt(Vi)*isqrt(Vj))` | one divide each |

No floating point. No division except three correlation denominators (RV32M
provides `div` in hardware). Integer square root is the classic bit-by-bit
digit recurrence: ~16 iterations of shift/compare/subtract, no multiply, no
divide.

### Range analysis (all fit in int64)

```
sum(x^2)     <= 208 * 2^30   ~ 2^37.7
N*sum(x^2)   <= 208 * 2^37.7 ~ 2^45.4
(sum(x))^2   <= (208*2^15)^2 ~ 2^45.4
cov << 14                    ~ 2^59.4
```

`sum(x^2)` overflows int32, so the accumulators **must** be int64. On RV32 that
is `mul` + `mulh` + a two-word add — about 5 instructions per sample, with no
library call.

---

## 5. Results

2,166 windows from FHI (10 rides) and FHI-BREAK (10 rides), hop 104.

### 5.1 Per-feature accuracy

Worst relative error **0.264%** (`mean_y`, whose absolute error is 4.1e-5 on a
feature with absmax 0.0156). Every feature's absolute error is at or near the
Q2.14 quantisation floor of 3.05e-5 (half an LSB), i.e. the features are
limited by **input quantisation, not by the algorithm**.

| Group | Max abs error | Max rel error |
|---|---|---|
| peak / min / max | 3.05e-05 | 0.012% |
| rms | 4.31e-05 | 0.034% |
| mean | 4.16e-05 | 0.264% |
| std | 4.29e-05 | 0.034% |
| corr | 6.90e-04 | 0.102% |

### 5.2 End-to-end separation

Per-ride scores, 10 healthy vs 10 faulty:

| Config | healthy | faulty | margin | AUC | score err |
|---|---|---|---|---|---|
| float feats + fp32 net (reference) | 0.5896 | 1.1360 | +0.2181 | 1.0000 | — |
| fixed feats + fp32 net | 0.5895 | 1.1359 | +0.2180 | 1.0000 | 0.008% |
| **fixed feats + int16 net (full integer)** | **0.5895** | **1.1359** | **+0.2180** | **1.0000** | **0.008%** |

Full integer path: healthy max 0.6765, faulty min 0.8945, **margin +0.2180 —
separation holds**, with perfect AUC and per-ride deltas of at most 0.0001.

**Conclusion: the entire pipeline — feature extraction and inference — runs in
integer arithmetic with no measurable loss of detection quality.**

---

## 6. Accumulator width: int16 x int16 does not fit int32

Measured on the same 2,166 windows, using per-channel int16 weights and int16
activations, the raw MAC accumulator reaches:

| Layer | shape | `|acc|` max | int32 headroom |
|---|---|---|---|
| 1 | 21 -> 32 | 1.54e9 | 1.4x |
| 2 | 32 -> 2 | **3.37e9** | **0.6x — overflows** |
| 3 | 2 -> 32 | 2.13e9 | 1.0x |
| 4 | 32 -> 21 | 1.68e9 | 1.3x |

Algebraic worst case is `32 * 32767^2 = 3.4e10 = 2^35`. This matters because
`MUL_DOT16` accumulates into a **32-bit** `rd`, so a naive int16 x int16
datapath cannot use it — and would silently saturate rather than fail loudly.

Sweeping widths (all with the fixed-point feature front end, per-ride margin
and AUC over 10 healthy + 10 faulty rides):

| Weights | Activations | `|acc|` max | int32 headroom | margin | AUC |
|---|---|---|---|---|---|
| int16 | int16 | 4.40e9 | 0.5x | +0.2180 | 1.0000 | **overflows** |
| int16 | int12 | 2.75e8 | 7.8x | +0.2181 | 1.0000 |
| **int12** | **int12** | **1.72e7** | **125x** | **+0.2181** | **1.0000** |
| int8 | int12 | 1.07e6 | 2016x | +0.2229 | 1.0000 |

Reference (fp32 network, fixed-point features) margin is +0.2180, so every
configuration except int16 x int16 preserves separation exactly.

**Recommendation: int12 weights and int12 activations**, stored in int16
containers. Its algebraic worst case is `32 * 2047^2 = 1.34e8`, a 16x margin
under int32 even for pathological input — so it cannot overflow, not merely
does not overflow on this data. `sv.dot16` still applies: the instruction does
int16 x int16, the values simply happen to be small.

This independently matches `docs/quantization.md`'s recommendation of
`ap_fixed<12,2>`, arrived at there from accuracy and DSP considerations rather
than accumulator width.

The *activation* width is what governs accumulator growth (it is multiplied by
the summation length). int16 weights with int12 activations is also safe in
practice at 7.8x measured headroom, but its algebraic worst case
(`32 * 32767 * 2047 = 2.15e9`) sits exactly at the int32 limit, so it has no
guaranteed margin.

---

## 7. Caveats

- Validated at hop 104 (the training hop). Other hops change the window set but
  not the arithmetic.
- `lib/fixed_features.py` uses Python arbitrary-precision ints for the int64
  intermediates. The range analysis in §4 shows every intermediate fits in
  int64, but the C port should be checked against this reference **bit-exactly**
  rather than assumed correct.
- Correlation clamps to ±1.0 (Q2.14 `±16384`). This never triggered on the FHI
  data but is present for safety.
- Biases and the network's own quantisation are covered by
  `docs/quantization.md`, not here.
