"""Fixed-point reference implementation of get_slice_params().

This is the bit-exact model of what the RISC-V/FPGA implementation computes.
It uses only integer arithmetic: adds, multiplies, shifts, compares, one
reciprocal multiply and an integer square root. No floating point, no division
except the correlation denominator.

Format
------
Input samples are int16 in Q2.14 (value = raw / 16384, range +-2.0). The FHI
data set peaks at |x| = 1.016, so Q2.14 fits with headroom while Q1.15 would
clip. All 21 output features are returned as Q2.14 raw ints.

Why the exact moment forms matter
---------------------------------
The obvious formulation

    var = E[X^2] - E[X]^2
    cov = E[XY]  - E[X]E[Y]

rounds the means to Q2.14 *before* subtracting. For the z axis (mean 0.846,
std 0.29) and for all three correlations the subtraction cancels away most of
the significant bits, so the rounding error in the mean dominates the result.
Measured on the FHI windows that gives **10.4% error on corr_xz**.

Using the exact integer forms instead, which never round before subtracting:

    N^2 * var = N*sum(x^2) - sum(x)^2
    N^2 * cov = N*sum(xy)  - sum(x)*sum(y)

and noting that corr = cov/(sx*sy) lets the N^2 cancel entirely, the same
measurement gives **0.052%** -- a 200x improvement, and now at the Q2.14
quantisation floor rather than algorithm-limited.

Range analysis (all fit in int64):
    sum(x^2)        <= 208 * 2^30      ~ 2^37.7
    N*sum(x^2)      <= 208 * 2^37.7    ~ 2^45.4
    (sum(x))^2      <= (208*2^15)^2    ~ 2^45.4
    cov << 14                          ~ 2^59.4
"""
import numpy as np

SHIFT = 14
SCALE = 1 << SHIFT          # Q2.14
N = 208                     # WINDOW_SIZE
RECIP_SHIFT = 22
RECIP_N = ((1 << RECIP_SHIFT) + N // 2) // N    # round(2^22 / 208) = 20165

FEATURE_NAMES = [
    "peak_x", "peak_y", "peak_z", "min_x", "min_y", "min_z",
    "max_x", "max_y", "max_z", "rms_x", "rms_y", "rms_z",
    "mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z",
    "corr_xy", "corr_xz", "corr_yz",
]


def to_fixed(x):
    """float -> int16 raw in Q2.14, saturating."""
    return np.clip(np.round(np.asarray(x) * SCALE), -32768, 32767).astype(np.int64)


def isqrt(n):
    """floor(sqrt(n)). Bit-by-bit digit recurrence: shifts, compares, subtracts
    only -- no division, no multiplication. ~16 iterations for a 32-bit input."""
    n = int(n)
    if n <= 0:
        return 0
    res = 0
    bit = 1 << ((n.bit_length() - 1) & ~1)      # highest even power of 4 <= n
    while bit:
        if n >= res + bit:
            n -= res + bit
            res = (res >> 1) + bit
        else:
            res >>= 1
        bit >>= 2
    return res


def div_n(v):
    """v / 208, round-to-nearest, via reciprocal multiply (no divide instruction)."""
    v = int(v)
    half = 1 << (RECIP_SHIFT - 1)
    if v >= 0:
        return (v * RECIP_N + half) >> RECIP_SHIFT
    return -((-v * RECIP_N + half) >> RECIP_SHIFT)


def fixed_features(slice_raw):
    """slice_raw: (208, 3) integer array of int16 Q2.14 samples.
    Returns (21,) int64 of Q2.14 raw features, ordered as FEATURE_NAMES."""
    f = [0] * 21
    sums, V = [], []

    for a in range(3):
        x = np.asarray(slice_raw)[:, a].astype(np.int64)
        s1 = int(np.sum(x))
        s2 = int(np.sum(x * x))
        sums.append(s1)

        f[0 + a] = int(np.max(np.abs(x)))
        f[3 + a] = int(np.min(x))
        f[6 + a] = int(np.max(x))
        f[12 + a] = div_n(s1)

        # rms = sqrt(sum(x^2)/N). isqrt(s2*N)/N keeps log2(N) extra bits of
        # precision versus isqrt(s2/N).
        f[9 + a] = div_n(isqrt(s2 * N))

        v = N * s2 - s1 * s1                    # == N^2 * variance, exact
        V.append(max(v, 0))
        f[15 + a] = div_n(isqrt(V[a]))          # std = sqrt(V)/N

    for k, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
        xi = np.asarray(slice_raw)[:, i].astype(np.int64)
        xj = np.asarray(slice_raw)[:, j].astype(np.int64)
        num = N * int(np.sum(xi * xj)) - sums[i] * sums[j]   # == N^2 * cov
        den = isqrt(V[i]) * isqrt(V[j])                      # == N^2 * sx * sy
        if den == 0:
            corr = 0
        else:
            corr = (num << SHIFT) // den if num >= 0 else -((-num << SHIFT) // den)
            corr = max(-SCALE, min(SCALE, corr))             # corr in [-1, 1]
        f[18 + k] = corr

    return np.array(f, dtype=np.int64)


def fixed_features_float(slice_float):
    """Convenience wrapper: float (208,3) window -> dequantised float features,
    for comparing against lib.utils.get_slice_params()."""
    return fixed_features(to_fixed(slice_float)).astype(np.float64) / SCALE
