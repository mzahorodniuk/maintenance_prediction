"""Export DAEParam to C headers for the Simply-V / RISC-V port.

Emits three headers into an output directory:

  weights_f32.h  fp32 weights + biases           (for the Phase-0 baseline)
  weights_i12.h  int12 codes, int32 biases,      (for the integer port)
                 per-channel requantise multipliers
  golden.h       golden windows: raw int16 samples, expected Q2.14 features,
                 expected Q2.14 reconstruction, expected score

Integer datapath modelled here (and what the C must implement)
--------------------------------------------------------------
Per layer, with activations in scale s_x and per-row weight scale s_w[r]:

    acc[r]  = sum_i  x_code[i] * w_code[r][i]        int32
    acc[r] += bias_acc[r]                            bias pre-scaled into
                                                     accumulator units
    if relu: acc[r] = max(acc[r], 0)
    y_code[r] = (acc[r] * requant_mul[r] + round) >> REQUANT_SHIFT

int12 codes in int16 containers keep the accumulator far inside int32:
algebraic worst case is 32 * 2047^2 = 1.34e8, a 16x margin (see
docs/fixed_point_features.md section 6).

The last layer requantises directly to Q2.14, the same scale the features use,
so the reconstruction error is a plain integer subtraction.

Usage:
    from scripts.export_weights_c import export_weights_c
    export_weights_c("/home/misha/Simply-V/sw/SoC/examples/common_ae")
"""
import os
import numpy as np
import torch

from lib.utils import (get_indexed_sample_from_data, get_window_slices_from_sample,
                       get_slice_params, load_model)
from lib.model import DAEParam
from lib.fixed_features import fixed_features, to_fixed, SCALE, SHIFT

WBITS = 12
ABITS = 12
WQMAX = 2 ** (WBITS - 1) - 1        # 2047
AQMAX = 2 ** (ABITS - 1) - 1        # 2047
REQUANT_SHIFT = 30
WINDOW, STEP = 208, 104
MODEL_PATH = "models/daeparam_fhi.pt"
RELU_AFTER = (True, False, True, False)   # DAEParam: 21->32 R, 32->2, 2->32 R, 32->21


def _all_windows():
    w, tags = [], []
    for lab, npz in [(0, "data/FHI.npz"), (1, "data/FHI-BREAK.npz")]:
        for n in range(1, 11):
            ws = get_window_slices_from_sample(
                get_indexed_sample_from_data(npz, n), WINDOW, STEP)
            w += ws
            tags += [(lab, n)] * len(ws)
    return w, np.array(tags)


def _qcode_per_row(W, qmax):
    a = np.abs(W).max(axis=1, keepdims=True)
    a = np.where(a == 0, 1.0, a)
    code = np.clip(np.round(W * qmax / a), -qmax, qmax).astype(np.int64)
    return code, (a / qmax).reshape(-1)


def _calibrate(layers, biases, feats_q214):
    """Forward the real data in float to pick one fixed activation scale per
    layer boundary. Hardware cannot use per-inference dynamic scales."""
    scales = []
    x = feats_q214.astype(np.float64) / SCALE
    s_in = np.abs(x).max() / AQMAX
    scales.append(s_in)
    for li, l in enumerate(layers):
        W = l.weight.detach().numpy().astype(np.float64)
        x = x @ W.T + biases[li][None, :]
        if RELU_AFTER[li]:
            x = np.maximum(x, 0.0)
        if li == len(layers) - 1:
            scales.append(1.0 / SCALE)          # last layer -> Q2.14
        else:
            m = np.abs(x).max()
            scales.append((m if m > 0 else 1.0) / AQMAX)
    return scales


def build(model_path=MODEL_PATH):
    ae = load_model(DAEParam(21), model_path)
    layers = [m for m in list(ae.encoder) + list(ae.decoder)
              if isinstance(m, torch.nn.Linear)]
    biases = [l.bias.detach().numpy().astype(np.float64) for l in layers]

    windows, tags = _all_windows()
    feats_q214 = np.stack([fixed_features(to_fixed(w)) for w in windows]).astype(np.int64)

    scales = _calibrate(layers, biases, feats_q214)

    L = []
    for li, l in enumerate(layers):
        W = l.weight.detach().numpy().astype(np.float64)
        wc, ws = _qcode_per_row(W, WQMAX)
        s_x, s_y = scales[li], scales[li + 1]
        bias_acc = np.round(biases[li] / (s_x * ws)).astype(np.int64)
        M = (s_x * ws) / s_y
        mul = np.round(M * (1 << REQUANT_SHIFT)).astype(np.int64)
        L.append(dict(wc=wc, ws=ws, bias_acc=bias_acc, mul=mul,
                      s_x=s_x, s_y=s_y, relu=RELU_AFTER[li],
                      n_in=W.shape[1], n_out=W.shape[0]))
    return ae, layers, biases, L, scales, feats_q214, windows, tags


def _rshift_round(v, shift):
    """Round-half-away-from-zero arithmetic shift, matching the C:
        v >= 0 : (v + half) >> shift
        v <  0 : -((-v + half) >> shift)
    Deliberately NOT np.round, which rounds half to even and would disagree
    with the C on exact ties."""
    half = 1 << (shift - 1)
    return np.where(v >= 0, (v + half) >> shift, -((-v + half) >> shift))


def in_scale_recip(scales):
    """Q30 reciprocal that converts a Q2.14 feature into a layer-0 input code."""
    return int(round((1.0 / scales[0]) / SCALE * (1 << 30)))


def int_forward(feats_q214, L, scales):
    """Bit-exact simulation of the C datapath. Returns Q2.14 reconstruction
    and the largest |accumulator| seen. Pure integer throughout, so the C port
    can be checked against it bit-for-bit."""
    h = np.clip(_rshift_round(feats_q214.astype(np.int64) * in_scale_recip(scales), 30),
                -AQMAX, AQMAX)
    accmax = 0
    for lay in L:
        acc = h @ lay["wc"].T + lay["bias_acc"][None, :]
        accmax = max(accmax, int(np.abs(acc).max()))
        if lay["relu"]:
            acc = np.maximum(acc, 0)
        h = _rshift_round(acc * lay["mul"][None, :], REQUANT_SHIFT)
        if lay is not L[-1]:
            h = np.clip(h, -AQMAX, AQMAX)
    return h, accmax


def _carr(name, arr, ctype, per_line=12):
    flat = np.asarray(arr).reshape(-1)
    out = ["static const %s %s = {" % (ctype, name)]
    for i in range(0, len(flat), per_line):
        out.append("    " + ", ".join(str(int(v)) for v in flat[i:i + per_line]) + ",")
    out.append("};")
    return "\n".join(out)


def _farr(name, arr, per_line=6):
    flat = np.asarray(arr).reshape(-1)
    out = ["static const float %s = {" % name]
    for i in range(0, len(flat), per_line):
        out.append("    " + ", ".join("%.9ef" % v for v in flat[i:i + per_line]) + ",")
    out.append("};")
    return "\n".join(out)


def export_weights_c(out_dir, model_path=MODEL_PATH, n_golden=8):
    ae, layers, biases, L, scales, feats_q214, windows, tags = build(model_path)
    os.makedirs(out_dir, exist_ok=True)

    recon_q, accmax = int_forward(feats_q214, L, scales)
    with torch.no_grad():
        ref = ae(torch.tensor(feats_q214.astype(np.float64) / SCALE,
                              dtype=torch.float32)).numpy().astype(np.float64)

    int_score = np.abs(recon_q - feats_q214).sum(1)                 # Q2.14 units
    ref_score = np.abs(ref - feats_q214 / SCALE).sum(1) * SCALE

    def ride_means(v):
        h = [v[(tags[:, 0] == 0) & (tags[:, 1] == n)].mean() for n in range(1, 11)]
        f = [v[(tags[:, 0] == 1) & (tags[:, 1] == n)].mean() for n in range(1, 11)]
        return np.array(h), np.array(f)

    ih, iff = ride_means(int_score)
    rh, rf = ride_means(ref_score)

    print("accumulator |max| = %d  (int32 headroom %.1fx)" % (accmax, (2**31 - 1) / accmax))
    print("reference  margin %+.2f Q2.14 (= %+.4f)" % (rf.min() - rh.max(), (rf.min() - rh.max()) / SCALE))
    print("integer    margin %+.2f Q2.14 (= %+.4f)" % (iff.min() - ih.max(), (iff.min() - ih.max()) / SCALE))
    print("integer    healthy max %.1f, faulty min %.1f -> %s"
          % (ih.max(), iff.min(), "SEPARATION HOLDS" if iff.min() > ih.max() else "BROKEN"))

    # ---------------- weights_f32.h ----------------
    p = ["// Generated by scripts/export_weights_c.py -- do not edit.",
         "// fp32 DAEParam weights, for the Phase-0 soft-float baseline.",
         "#ifndef AE_WEIGHTS_F32_H", "#define AE_WEIGHTS_F32_H", "",
         "#define AE_N_LAYERS 4", ""]
    for i, l in enumerate(layers):
        W = l.weight.detach().numpy()
        p.append("#define AE_L%d_IN  %d" % (i, W.shape[1]))
        p.append("#define AE_L%d_OUT %d" % (i, W.shape[0]))
        p.append(_farr("ae_w%d[AE_L%d_OUT * AE_L%d_IN]" % (i, i, i), W))
        p.append(_farr("ae_b%d[AE_L%d_OUT]" % (i, i), biases[i]))
        p.append("")
    p += ["#endif", ""]
    open(os.path.join(out_dir, "weights_f32.h"), "w").write("\n".join(p))

    # ---------------- weights_i12.h ----------------
    p = ["// Generated by scripts/export_weights_c.py -- do not edit.",
         "// int%d weights / int%d activations. Accumulator stays in int32:" % (WBITS, ABITS),
         "// algebraic worst case 32*%d^2 = %d, measured max %d."
         % (WQMAX, 32 * WQMAX * WQMAX, accmax),
         "#ifndef AE_WEIGHTS_I12_H", "#define AE_WEIGHTS_I12_H", "",
         "#include <stdint.h>", "",
         "#define AE_N_LAYERS      4",
         "#define AE_REQUANT_SHIFT %d" % REQUANT_SHIFT,
         "#define AE_ACT_QMAX      %d" % AQMAX,
         "#define AE_FEAT_SHIFT    %d   // features are Q2.14" % SHIFT,
         "#define AE_IN_SCALE_RECIP_Q30 %d  // round((1/s_in)/2^14 * 2^30)"
         % in_scale_recip(scales), ""]
    for i, lay in enumerate(L):
        p.append("#define AE_L%d_IN   %d" % (i, lay["n_in"]))
        p.append("#define AE_L%d_OUT  %d" % (i, lay["n_out"]))
        p.append("#define AE_L%d_RELU %d" % (i, 1 if lay["relu"] else 0))
        p.append(_carr("ae_qw%d[AE_L%d_OUT * AE_L%d_IN]" % (i, i, i), lay["wc"], "int16_t", 16))
        p.append(_carr("ae_qb%d[AE_L%d_OUT]" % (i, i), lay["bias_acc"], "int32_t", 8))
        p.append(_carr("ae_qm%d[AE_L%d_OUT]" % (i, i), lay["mul"], "int32_t", 8))
        p.append("")
    p += ["#endif", ""]
    open(os.path.join(out_dir, "weights_i12.h"), "w").write("\n".join(p))

    # ---------------- golden.h ----------------
    step = max(1, len(windows) // n_golden)
    idx = list(range(0, len(windows), step))[:n_golden]
    raw = np.stack([to_fixed(windows[i]) for i in idx])          # (g,208,3) int16
    fq = feats_q214[idx]
    rq = recon_q[idx]
    sc = int_score[idx]
    fref = np.stack([get_slice_params(windows[i]) for i in idx])

    p = ["// Generated by scripts/export_weights_c.py -- do not edit.",
         "// %d golden windows with expected intermediate values." % len(idx),
         "#ifndef AE_GOLDEN_H", "#define AE_GOLDEN_H", "",
         "#include <stdint.h>", "",
         "#define AE_GOLDEN_N   %d" % len(idx),
         "#define AE_WINDOW     %d" % WINDOW,
         "#define AE_N_FEAT     21", "",
         "// raw int16 Q2.14 samples, [window][208][3]",
         _carr("ae_golden_raw[AE_GOLDEN_N][AE_WINDOW][3]", raw, "int16_t", 15), "",
         "// expected Q2.14 features from fixed_features()",
         _carr("ae_golden_feat[AE_GOLDEN_N][AE_N_FEAT]", fq, "int16_t", 11), "",
         "// expected Q2.14 reconstruction from the integer network",
         _carr("ae_golden_recon[AE_GOLDEN_N][AE_N_FEAT]", rq, "int16_t", 11), "",
         "// expected score = sum |recon - feat|, Q2.14 units",
         _carr("ae_golden_score[AE_GOLDEN_N]", sc, "int32_t", 8), "",
         "// fp32 reference features, for the Phase-0 baseline",
         _farr("ae_golden_feat_f32[AE_GOLDEN_N][AE_N_FEAT]", fref), "",
         "#endif", ""]
    open(os.path.join(out_dir, "golden.h"), "w").write("\n".join(p))

    for f in ("weights_f32.h", "weights_i12.h", "golden.h"):
        print("wrote %s (%d bytes)" % (os.path.join(out_dir, f),
                                       os.path.getsize(os.path.join(out_dir, f))))
    return iff.min() > ih.max()
