"""Quantize DAEParam with Brevitas (lib/qmodel.py: power-of-two scales, FLOOR)
and export it to QONNX, the input of the FPGA generators (FINN, hls4ml) of the
RISC-V co-processor (~/riscv_co-processor_nexys, step 18).

Pipeline:
    trained float weights -> QDAEParam -> calibration (PTQ)
    -> quantization-aware training (QAT) -> export_qonnx -> qonnx cleanup

QAT follows scripts/export_tvm.py (healthy FHI rides 1-7, window 208, step 104,
MSE, Adam lr 1e-4, batch 64, 30 epochs, seed 0), single-threaded and with
deterministic algorithms so a re-run gives the same weights.

Checks (any failure stops the export):
  - every Quant node has a power-of-two scale and zero point 0, and every
    activation rounds with FLOOR;
  - lib/qonnx_int.py (integer codes) equals QONNX's own execution of the model
    in float64 on the dataset and on random vectors (float32 accumulation is not
    exact: 16 bit x 8 bit products summed 21 times exceed 2^24);
  - the per-ride healthy/faulty separation holds.

Writes into out_dir:
  daeparam_fhi_w<W>a<A>.qonnx   the cleaned QONNX model
  qat_state.pt                  the QAT checkpoint (QDAEParam state dict)
  golden_w<W>a<A>.npz           Q2.14 features, recon and score of the dataset
                                (with ride tags) and of random vectors

Needs Brevitas and QONNX (the python3.10 venv ~/tvm-env), run from the repo root:
    ~/tvm-env/bin/python -m scripts.export_qonnx --model models/daeparam_fhi.pt --out <dir>
"""
import argparse
import os
import tempfile

import numpy as np
import onnx
import torch
import torch.nn as nn

from lib.utils import load_model
from lib.model import DAEParam
from lib.fixed_features import fixed_features, to_fixed, SCALE
from lib.qmodel import QDAEParam
from lib.qonnx_int import QonnxInt, score as int_score
from scripts.export_weights_c import MODEL_PATH, _all_windows

QAT_EPOCHS, QAT_LR, QAT_BATCH = 30, 1e-4, 64
N_RANDOM = 4096


def _features():
    windows, tags = _all_windows()
    feats = np.stack([fixed_features(to_fixed(w)) for w in windows]).astype(np.int64)
    return feats, np.array(tags)


def ride_separation(tags, score):
    """(healthy max, faulty min) of the per-ride mean score"""
    h = [score[(tags[:, 0] == 0) & (tags[:, 1] == n)].mean() for n in range(1, 11)]
    f = [score[(tags[:, 0] == 1) & (tags[:, 1] == n)].mean() for n in range(1, 11)]
    return max(h), min(f)


def _torch_score(model, x):
    with torch.no_grad():
        y = model(torch.from_numpy(x))
    y = (y.value if hasattr(y, "value") else y).numpy().astype(np.float64)
    return np.abs(y - x).sum(1) * SCALE


def qat(ae, x, tags, epochs=QAT_EPOCHS, lr=QAT_LR, a_bits=8):
    """Brevitas model from the float weights: calibrated, then fine-tuned.
    Returns the model and its PTQ score (before fine-tuning)."""
    from brevitas.graph.calibrate import calibration_mode

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    m = QDAEParam(21, a_bits=a_bits).load_float(ae)
    m.eval()
    with torch.no_grad(), calibration_mode(m):
        for i in range(0, len(x), 256):
            m(torch.from_numpy(x[i:i + 256]))
    m.eval()
    ptq = _torch_score(m, x)

    train = torch.from_numpy(x[(tags[:, 0] == 0) & (tags[:, 1] <= 7)])
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    m.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(train), generator=gen).split(QAT_BATCH):
            opt.zero_grad()
            loss = loss_fn(m(train[idx]), train[idx])
            loss.backward()
            opt.step()
    m.eval()
    return m, ptq


def check_qonnx(path):
    """Power-of-two scales, zero points 0, FLOOR on every activation"""
    model = onnx.load(path)
    init = {i.name for i in model.graph.initializer}
    n_act = 0
    for n in model.graph.node:
        if n.op_type != "Quant":
            continue
        a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
        if n.input[0] not in init:
            n_act += 1
            if a.get("rounding_mode", b"ROUND") not in (b"FLOOR", "FLOOR"):
                raise SystemExit("%s: activation rounding %s, expected FLOOR" % (n.name, a.get("rounding_mode")))
    QonnxInt(path)          # raises on a scale that is not a power of two or a zero point != 0
    return n_act


def qonnx_exec_f64(path, feats):
    """QONNX's own execution (Quant in numpy, the rest in ONNX Runtime), every
    tensor in float64 and the batch dimension set to len(feats)"""
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx
    from qonnx.custom_op.general import quant as qmod

    def quant_f64(self, context, graph):         # Quant.execute_node casts its result to float32
        n = self.onnx_node
        context[n.output[0]] = np.asarray(qmod.quant(
            *(context[i] for i in n.input), self.get_nodeattr("signed"), self.get_nodeattr("narrow"),
            self.get_nodeattr("rounding_mode")), dtype=np.float64)

    model = onnx.load(path)
    g = model.graph
    for t in list(g.input) + list(g.output) + list(g.value_info):
        tt = t.type.tensor_type
        if tt.elem_type == onnx.TensorProto.FLOAT:
            tt.elem_type = onnx.TensorProto.DOUBLE
        if len(tt.shape.dim) == 2 and tt.shape.dim[0].dim_value == 1:
            tt.shape.dim[0].dim_value = len(feats)
    for i in g.initializer:
        if i.data_type == onnx.TensorProto.FLOAT:
            i.CopyFrom(onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(i).astype(np.float64), i.name))
    mw = ModelWrapper(model)
    saved = qmod.Quant.execute_node
    qmod.Quant.execute_node = quant_f64
    try:
        out = execute_onnx(mw, {g.input[0].name: feats.astype(np.float64) / SCALE})[g.output[0].name]
    finally:
        qmod.Quant.execute_node = saved
    code = out * SCALE
    if not np.array_equal(code, np.round(code)):
        raise SystemExit("QONNX float64 output is not on the Q2.14 grid")
    return np.round(code).astype(np.int64)


def export_qonnx(out_dir, model_path=MODEL_PATH, epochs=QAT_EPOCHS, lr=QAT_LR, a_bits=8):
    from brevitas.export import export_qonnx as brevitas_export_qonnx
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.util.cleanup import cleanup_model

    os.makedirs(out_dir, exist_ok=True)
    tag = "w8a%d" % a_bits
    ae = load_model(DAEParam(21), model_path)
    feats, tags = _features()
    x = (feats / SCALE).astype(np.float32)
    score_f = _torch_score(ae, x)

    m, score_ptq = qat(ae, x, tags, epochs, lr, a_bits)
    score_qat = _torch_score(m, x)
    torch.save(m.state_dict(), os.path.join(out_dir, "qat_state.pt"))

    path = os.path.join(out_dir, "daeparam_fhi_%s.qonnx" % tag)
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw.onnx")
        brevitas_export_qonnx(m, torch.zeros(1, 21), export_path=raw)
        cleanup_model(ModelWrapper(raw)).save(path)
    n_act = check_qonnx(path)

    ref = QonnxInt(path)
    rng = np.random.default_rng(1)
    rfeat = rng.integers(-32768, 32768, size=(N_RANDOM, 21), dtype=np.int64)
    rfeat[: N_RANDOM // 4] //= 16           # a quarter in the dataset's range
    recon_d, recon_r = ref.forward(feats), ref.forward(rfeat)
    for name, f, r in (("dataset", feats, recon_d), ("random", rfeat, recon_r)):
        q = qonnx_exec_f64(path, f)
        bad = int((q != r).any(1).sum())
        if bad:
            raise SystemExit("lib/qonnx_int.py differs from QONNX float64 on %d %s vectors" % (bad, name))
    score_d = int_score(recon_d, feats)
    # the Brevitas model in float32 only for information: ties may differ
    torch_diff = int((np.abs(score_qat - score_d) > 0.5).sum())

    np.savez_compressed(os.path.join(out_dir, "golden_%s.npz" % tag),
                        feats=feats, tags=tags, recon=recon_d, score=score_d,
                        rfeats=rfeat, rrecon=recon_r, rscore=int_score(recon_r, rfeat))

    print("QONNX %s: %d activation quantizers (FLOOR), every scale a power of two" % (path, n_act))
    for i, (w, ew, b) in enumerate(ref.layers()):
        print("  layer %d: %2d x %2d, weight scale 2^%d, |w| <= %d, |b| <= %d"
              % (i, w.shape[0], w.shape[1], ew, np.abs(w).max(), np.abs(b).max()))
    print("  activation exponents: %s" % [ref.quant[k][0] for k in ref.quant])
    print("  lib/qonnx_int.py == QONNX float64 on %d dataset and %d random vectors" % (len(feats), N_RANDOM))
    print("  Brevitas float32 score != integer score on %d of %d windows (float32 ties)" % (torch_diff, len(feats)))
    print("%-8s %9s %9s %9s %10s" % ("model", "healthy", "faulty", "margin", "score err"))
    ok = True
    for name, s in (("float", score_f), ("ptq", score_ptq), ("qat", score_qat), ("integer", score_d)):
        h, f = ride_separation(tags, s)
        err = np.mean(np.abs(s - score_f) / score_f) * 100
        print("%-8s %9.1f %9.1f %+9.1f %9.2f%%" % (name, h, f, f - h, err))
        if name == "integer":
            ok = f > h
    print("(per-ride mean score, Q2.14 units; healthy = highest healthy ride, faulty = lowest faulty"
          " ride; score err = mean |score - float score| / float score)")
    if not ok:
        raise SystemExit("accuracy gate: the integer model does not separate healthy from faulty rides")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Brevitas QAT -> QONNX for the FPGA generators")
    ap.add_argument("--model", default=MODEL_PATH, help="float DAEParam state dict")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--epochs", type=int, default=QAT_EPOCHS)
    ap.add_argument("--lr", type=float, default=QAT_LR)
    ap.add_argument("--a-bits", type=int, default=8, help="hidden activation width")
    a = ap.parse_args()
    export_qonnx(a.out, a.model, a.epochs, a.lr, a.a_bits)
