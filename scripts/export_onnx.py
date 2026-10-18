"""Export DAEParam to ONNX as it is (fp32) and generate C with emx-onnx-cgen,
for the Simply-V / RISC-V port: the fp32 baseline the quantized models
(scripts/export_tvm.py) are compared against.

The model takes the 21 features as float (the Q2.14 fixed_features() of a
window divided by 2^14, as the device computes them) and returns the 21
reconstructed features as float.

Writes into out_dir: daeparam_fp32.onnx and ae_model_fp32.c with
  void ae_model_fp32(const float feat[1][21], float recon[1][21]).
Prints the per-ride mean score (ONNX Runtime) to compare with the device.

Usage:
    from scripts.export_onnx import export_onnx
    export_onnx("/home/misha/simplyv_ddr_nexys/sw/SoC/examples/ae_onnx/gen")
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import torch

from lib.utils import load_model
from lib.model import DAEParam
from lib.fixed_features import fixed_features, to_fixed, SCALE
from scripts.export_weights_c import MODEL_PATH, _all_windows

FORMATS = ("fp32",)


def _torch_export(model, path):
    torch.onnx.export(model, torch.zeros(1, 21), path, input_names=["feat"],
                      output_names=["recon"], dynamo=False)


def _cgen(onnx_path, c_path, name):
    exe = shutil.which("emx-onnx-cgen") or os.path.join(os.path.dirname(sys.executable), "emx-onnx-cgen")
    # --large-temp-threshold 0: every temporary on the stack, no malloc
    subprocess.run([exe, "compile", "--model-name", name, "--large-temp-threshold", "0",
                    onnx_path, c_path], check=True, stdout=subprocess.DEVNULL)


def _run(path, x):
    """Reconstruction of every window with ONNX Runtime"""
    s = ort.InferenceSession(path)
    return np.concatenate([s.run(None, {"feat": v[None]})[0] for v in x])


def export_onnx(out_dir, model_path=MODEL_PATH, formats=FORMATS):
    os.makedirs(out_dir, exist_ok=True)
    ae = load_model(DAEParam(21), model_path)
    windows, tags = _all_windows()
    tags = np.array(tags)
    x = (np.stack([fixed_features(to_fixed(w)) for w in windows]) / SCALE).astype(np.float32)

    path = os.path.join(out_dir, "daeparam_fp32.onnx")
    _torch_export(ae, path)
    _cgen(path, os.path.join(out_dir, "ae_model_fp32.c"), "ae_model_fp32")

    with torch.no_grad():
        ref_score = np.abs(ae(torch.from_numpy(x)).numpy() - x).sum(1) * SCALE
    score = np.abs(_run(path, x) - x).sum(1) * SCALE
    h = [score[(tags[:, 0] == 0) & (tags[:, 1] == n)].mean() for n in range(1, 11)]
    f = [score[(tags[:, 0] == 1) & (tags[:, 1] == n)].mean() for n in range(1, 11)]
    err = np.mean(np.abs(score - ref_score) / ref_score) * 100

    print("%-11s %9s %9s %9s %10s  %s" % ("format", "healthy", "faulty", "margin", "score err", "by"))
    print("%-11s %9.1f %9.1f %+9.1f %9.2f%%  %s" % ("fp32", max(h), min(f), min(f) - max(h), err, "ort"))
    print("(healthy = highest healthy ride mean, faulty = lowest faulty ride mean, Q2.14 units;"
          " score err = mean |score - PyTorch fp32 score| / score)")
    print("wrote %s/ae_model_fp32.c" % out_dir)
