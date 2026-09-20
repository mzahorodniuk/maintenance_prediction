"""Validate lib/fixed_features.py against the float reference.

Two checks:
  1. per-feature error of the fixed-point features vs get_slice_params()
  2. end-to-end: does healthy/faulty separation survive fixed-point features,
     both with the fp32 network and with the int16-quantised network

Usage:
    from scripts.validate_fixed import validate_fixed
    validate_fixed()
"""
import numpy as np
import torch

from lib.utils import (get_slice_params, get_indexed_sample_from_data,
                       get_window_slices_from_sample, load_model)
from lib.model import DAEParam
from lib.quantize import quantize_model, quantize_activations
from lib.fixed_features import fixed_features_float, FEATURE_NAMES

WINDOW_SIZE = 208
STEP_SIZE = 104
MODEL_PATH = "models/daeparam_fhi.pt"


def _ride_windows(npz, n):
    return get_window_slices_from_sample(
        get_indexed_sample_from_data(npz, n), WINDOW_SIZE, STEP_SIZE)


def _scores(model, feats):
    x = torch.tensor(np.asarray(feats), dtype=torch.float32)
    with torch.no_grad():
        return (model(x) - x).abs().sum(1).numpy()


def _auc(h, f):
    """ROC AUC via Mann-Whitney U."""
    h, f = np.asarray(h), np.asarray(f)
    r = np.concatenate([h, f]).argsort().argsort().astype(float) + 1
    u = r[:len(h)].sum() - len(h) * (len(h) + 1) / 2.0
    return 1.0 - u / (len(h) * len(f))


def validate_fixed(model_path=MODEL_PATH):
    print("loading windows...")
    rides = {}
    for tag, npz in [("healthy", "data/FHI.npz"), ("faulty", "data/FHI-BREAK.npz")]:
        for n in range(1, 11):
            w = _ride_windows(npz, n)
            rides[(tag, n)] = {
                "float": np.stack([get_slice_params(s) for s in w]),
                "fixed": np.stack([fixed_features_float(s) for s in w]),
            }

    ref = np.concatenate([rides[k]["float"] for k in rides])
    fix = np.concatenate([rides[k]["fixed"] for k in rides])
    err = np.abs(fix - ref)

    print("\n--- per-feature error (%u windows) ---" % len(ref))
    print("%-9s %11s %11s %11s %9s" % ("feature", "absmax|ref|", "max abs err", "mean abs err", "rel%"))
    print("-" * 58)
    worst = 0.0
    for i, nm in enumerate(FEATURE_NAMES):
        sc = np.abs(ref[:, i]).max()
        rel = 100.0 * err[:, i].max() / sc if sc > 0 else 0.0
        worst = max(worst, rel)
        print("%-9s %11.6f %11.2e %11.2e %8.3f%%"
              % (nm, sc, err[:, i].max(), err[:, i].mean(), rel))
    print("worst relative error: %.3f%%" % worst)

    ae = load_model(DAEParam(21), model_path)
    ae_i16 = quantize_activations(quantize_model(ae, "int16"), "int16")

    configs = [
        ("float feats + fp32 net (reference)", "float", ae),
        ("FIXED feats + fp32 net", "fixed", ae),
        ("FIXED feats + int16 net (full integer)", "fixed", ae_i16),
    ]
    results = {}
    for name, key, model in configs:
        h = np.array([_scores(model, rides[("healthy", n)][key]).mean() for n in range(1, 11)])
        f = np.array([_scores(model, rides[("faulty", n)][key]).mean() for n in range(1, 11)])
        results[name] = (h, f)

    ref_all = np.concatenate(results[configs[0][0]])

    print("\n--- end-to-end separation (per-ride scores) ---")
    print("%-40s %8s %8s %9s %7s %8s" % ("config", "healthy", "faulty", "margin", "AUC", "err%"))
    print("-" * 86)
    for name, _, _ in configs:
        h, f = results[name]
        e = 100.0 * np.abs(np.concatenate([h, f]) - ref_all).mean() / ref_all.mean()
        print("%-40s %8.4f %8.4f %+9.4f %7.4f %7.3f%%"
              % (name, h.mean(), f.mean(), f.min() - h.max(), _auc(h, f), e))

    h, f = results[configs[-1][0]]
    ok = f.min() > h.max()
    print("\nfull integer path: healthy max %.4f, faulty min %.4f, margin %+.4f -> %s"
          % (h.max(), f.min(), f.min() - h.max(),
             "SEPARATION HOLDS" if ok else "SEPARATION BROKEN"))
    return ok
