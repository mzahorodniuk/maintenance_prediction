# Numeric-format study for FPGA deployment

Goal: pick a numeric format to run `DAEParam` (the fault-detection autoencoder
from `deep_ae_param_fault_test()`) on a **Xilinx Artix-7 XC7A100T** (Arty
A7-100T / Nexys A7-100T — same die, different board peripherals). This
documents what was tested, how it was verified, and the result.

---

## 1. What's being quantized

`DAEParam(21)` (`lib/model.py`): `21→32→2` encoder, `2→32→21` decoder,
**1,559 parameters** (1,472 weights + 87 biases), **1,472 MACs** per inference.
Input is the 21 hand-crafted features from `get_slice_params()` — peak/min/
max/RMS/mean/std per axis plus 3 correlations, all `|x| ≤ 1.02`, weights
`O(0.1–1)`.

That bounded range is the central fact driving every result below: **dynamic
range is never the constraint here, mantissa/step precision is** — which is
why fixed point does well and wide-exponent floats don't.

---

## 2. Formats tested

| Format | Bits | Structure |
|---|---|---|
| `bf16` | 16 | 1 sign, 8 exponent, 7 mantissa (fp32's exponent, truncated mantissa) |
| `fp16` | 16 | 1 sign, 5 exponent, 10 mantissa (IEEE 754 binary16) |
| `int8` / `int10` / `int12` / `int16` | 8–16 | Symmetric fixed point, range `[-(2^(n-1)-1), 2^(n-1)-1]`, scale `= max\|x\| / qmax` |
| `mxfp8_e4m3` | 8 (+ shared scale) | OCP Microscaling: 32-element blocks share one E8M0 power-of-two scale; elements are FP8 E4M3 |
| `mxfp8_e5m2` | 8 (+ shared scale) | Same MX blocking, elements are FP8 E5M2 |

int8/int12/etc. support both **per-tensor** (one scale for the whole tensor)
and **per-channel** (one scale per output neuron, `axis=0`) granularity.
Per-channel matters here because the 21 input features span a **51.6× dynamic
range** across dimensions (0.0197 to 1.016 absmax).

Implementation: `lib/quantize.py`. Every format function round-trips a
float32 tensor through the target format and back, so a "quantized" model
still runs in ordinary PyTorch but produces bit-accurate values for that
format.

---

## 3. How the emulators were verified

Getting bit-level format semantics right (rounding mode, subnormals, overflow
behaviour, MX block-scale rules) is easy to get subtly wrong, so each format
went through an independent-implementation-plus-adversarial-review process
before any model-level number was trusted:

1. **Two agents independently implemented each format** from a written spec,
   without seeing each other's work.
2. **A third agent adversarially verified** both: derived the spec
   independently, then diffed the two implementations against each other and
   against ground truth on random tensors, adversarial edge cases
   (subnormals, exact rounding ties, overflow boundaries, all-zero tensors),
   and the model's own real weights and features.
3. Reconciled into one canonical implementation, re-tested.

Concrete results from that process:

| Format | How it was checked | Result |
|---|---|---|
| `bf16` | Exhaustive sweep over **all 2³² float32 bit patterns** vs. `torch.bfloat16` | 0 mismatches |
| `fp16` | Exhaustive sweep over **all 2³² float32 bit patterns** vs. `torch.float16` | 0 mismatches |
| `int8` (per-tensor) | Bit-exact vs. an independent exact-rational (`Fraction`) reference, on the real 1,559-parameter state dict and all 224k real feature vectors | 0 mismatches |
| `int8` (per-channel) | Two independent implementations agreed bit-exactly; one bug found (rounding via a materialized float32 scale flips ~40% of exact midpoints vs. computing in float64) and fixed | Confirmed |
| `mxfp8_e4m3` / `mxfp8_e5m2` | Spec re-derived from the source paper (arXiv:2310.10537, "Microscaling Data Formats for Deep Learning") and cross-read against Microsoft's `microxcaling` reference implementation, rather than trusting a web summary (one fetched summary had a wrong exponent table and was rejected) | Confirmed; caught the MX-specific rule that elements **clamp** to the format max rather than overflowing to infinity |

One of the six verification agents was interrupted by an account spend limit
mid-run. Rather than re-spend, its output was recovered from the run's
journal and **cross-checked locally**: `lib/quantize.py`'s implementations
are bit-exact against every recovered independent implementation, except
`int8` per-tensor, which differs by 1 ulp (6×10⁻⁸) from a scale-computation
convention difference — confirmed to produce **identical integer codes** on
real weights and features, i.e. the same hardware result, just a different
last-bit of the Python-side dequantized float.

---

## 4. Model-level test harness

Weight error alone doesn't say whether the model still works — what matters
is whether **healthy vs. faulty separation survives**. `harness2.py`
(scratch, not in the repo — see §7) trains `DAEParam` from scratch on **5
different random seeds**, quantizes each trained model to every format, and
reports:

- **AUC** — healthy vs. faulty ROC AUC (Mann-Whitney U)
- **Ride margin** — `min(faulty ride scores) - max(healthy ride scores)`; positive means every ride is still classified correctly by a threshold
- **Spearman correlation** of quantized scores vs. that seed's own fp32 scores
- **Score error %** — mean absolute score error vs. fp32, as a percentage

Two scenarios per format:

- **W** — only weights quantized (optimistic — activations stay fp32)
- **WAI** — weights **and** activations **and** the input feature vector all quantized (realistic for actual hardware, since a real datapath doesn't have an fp32 activation path sitting alongside an int8 weight path)

All numbers below are **WAI**, since that's what a real deployment looks like.

---

## 5. Results

### 5.1 The four requested formats

| Format | Score error | Spearman | Ride margin kept |
|---|---|---|---|
| **fp16** | **0.044% ± 0.003** | 1.00000 | 100.0% |
| bf16 | 0.367% ± 0.023 | 0.99993 | 99.6% |
| int8 (per-channel) | 2.517% ± 0.166 | 0.99702 | 98.8% |
| int8 (per-tensor) | 2.689% ± 0.234 | 0.99668 | 98.2% |
| mxfp8_e4m3 | 7.328% ± 0.759 | 0.97770 | 92.7% |
| mxfp8_e5m2 | 15.183% ± 1.104 | 0.93358 | 77.5% |

**fp16 beats bf16 by ~8× at the same 16 bits**, because bf16's extra 3
exponent bits buy range this bounded-input model never uses, at the direct
cost of mantissa precision.

**MXFP8 underperforms even int8** here because its 32-element block size is
**larger than every layer dimension** (21, 32, 2) — every "block" degenerates
to one whole row, so the shared scale provides no local adaptivity, and
zero-padding partial blocks pushes effective density to ~15.6 bits/weight,
worse than plain int8.

### 5.2 Fixed-point bit-width sweep

| Width | Score error | vs. float formats |
|---|---|---|
| `int16` | 0.010% | 4× better than fp16 |
| `int14` | 0.039% | ≈ fp16, 2 fewer bits |
| `int12` | 0.155% | beats bf16, 4 fewer bits |
| `int10` | 0.622% | — |
| `int8` | 2.517% | = int8 above |
| `int6` | 12.0% | separation degraded but survives |
| `int4` | 91.4% | **separation breaks** — the only format that fails |

Fixed point dominates every float format at equal or lower width. Detection
survives quantization in **every format down to int6** — this task is
genuinely robust; the numbers above are margin erosion, not failure, except
at int4.

---

## 6. FPGA resource assessment (XC7A100T)

Six independent agents assessed each format for DSP48E1 usage, LUT/FF
overhead, and toolchain support, several backing their claims with their own
simulation of this exact topology.

| Resource | Available on XC7A100T | int16 fixed point needs |
|---|---|---|
| Block RAM | 4,860 Kb | 25.7 Kb (**0.53%**) |
| DSP48E1 | 240 | 1 (sequential engine) to 46 (hls4ml RF=32 parallel) |
| LUTs | 63,400 | ~4–12k depending on parallelism |

Key findings:

- **The DSP48E1 is a native 25×18 fixed-point multiplier.** int8 through
  int16 all fit in one DSP per multiply — bit width in that range doesn't
  change DSP count. Floating formats (bf16/fp16) have **no native support**:
  a soft-float datapath (mantissa multiply + exponent add + normalize +
  round) in fabric, or a soft FPU core, is required.
- **bf16 is not supported by the 7-series toolchain at all** —
  `ap_float<16,8>` in Vitis HLS targets Versal/UltraScale+ only.
- **Timing is a non-issue.** A new 208-sample window arrives roughly every
  250 ms (416 Hz sample rate, 104-sample hop). Even the smallest, fully
  sequential single-DSP MAC engine finishes one inference in ~14.7 µs at
  100 MHz — about 17,000× faster than required. Design choice should
  therefore optimize for **simplicity and verifiability**, not throughput.
- **MXFP8 has no viable NN-to-FPGA flow on this part**: Vitis AI's DPU
  doesn't target Artix-7, FINN targets quantized/binarized nets, and
  hls4ml's native citizen is `ap_fixed`/`ap_int`.

**Recommendation: `ap_fixed<12,2>` (or `<16,2>` if you'd rather not think
about the tradeoff — the model is small enough that the extra 4 bits cost
nothing meaningful on this chip; see §5.2 and the BRAM/DSP table above).**
Both are simpler to implement and verify than any float format on this
board, and more accurate at equal or lower bit width.

---

## 7. Modeling caveats (stated so the thesis can cite them precisely)

- Accumulation in the harness is **fp32** throughout. Real int8/intN hardware
  accumulates in a wider fixed-point accumulator (exact for the ≤32-term
  sums here), so the int-format numbers above are realistic; a genuine
  float datapath on FPGA would typically also accumulate in-format, so the
  float-format numbers here are *slightly* optimistic relative to real
  float hardware.
- Biases are kept fp32 for the integer formats in both the harness and
  `lib/quantize.py`, modeling a fixed-point accumulator with a fp32/wide
  bias add — standard practice, but worth stating as a choice.
- The 5-seed multi-seed study and the FPGA resource study were run as
  scratch scripts during this investigation and are **not committed to the
  repo** — only the final, reusable pieces are (§8). Re-running the full
  study (multi-seed sweep + independent-agent verification) would reproduce
  the same conclusions but isn't checked in as a script.

---

## 8. What's in the repo

| File | Purpose |
|---|---|
| `lib/quantize.py` | The verified format emulators (`bf16`, `fp16`, `fixed(nbits)` → `int8`/`int10`/`int12`/`int16`, `mxfp8_e4m3`, `mxfp8_e5m2`) and `quantize_model()` / `quantize_activations()` |
| `lib/utils.py` — `save_model()` / `load_model()` | Persist/restore a trained model's `state_dict()` |
| `scripts/quantize_export.py` | `train_and_save()` trains and saves `DAEParam` on FHI; `quantize_export()` quantizes a saved model to every format in `FORMATS`, saves each to `models/quantized/`, and prints the healthy/faulty score comparison |

Usage:

```python
from scripts.quantize_export import quantize_export
quantize_export()   # trains models/daeparam_fhi.pt if it doesn't exist yet
```

```
format        healthy   faulty     ratio  score err%    params
--------------------------------------------------------------
fp32           0.5879   1.1245    1.913x      0.000%      1559
bf16           0.5884   1.1256    1.913x      0.387%      1559
fp16           0.5878   1.1244    1.913x      0.050%      1559
int8           0.6000   1.1336    1.889x      3.322%      1559
int10          0.5887   1.1255    1.912x      0.826%      1559
int12          0.5880   1.1245    1.912x      0.212%      1559
int16          0.5879   1.1245    1.913x      0.013%      1559
mxfp8_e4m3     0.6430   1.1382    1.770x      8.222%      1559
mxfp8_e5m2     0.7441   1.2387    1.665x     17.451%      1559
```

(Single-run numbers; the multi-seed table in §5 is the one to cite, this is
here to match what `quantize_export()` actually prints when you run it.)
