import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.model import *
from lib.utils import *
from lib.dataset import *
from lib.quantize import FORMATS, quantize_model, quantize_activations

WINDOW_SIZE = 208
STEP_SIZE = 104
MODEL_PATH = "models/daeparam_fhi.pt"


def _slices(npz, ride_range, step):
    slices = []
    for n in ride_range:
        sample = get_indexed_sample_from_data(npz, n)
        slices += get_window_slices_from_sample(sample, WINDOW_SIZE, step)
    return slices


def _features(npz, ride_range, step):
    dataset = ElevatorSliceParams(_slices(npz, ride_range, step))
    return torch.stack([dataset[i][0] for i in range(len(dataset))])


def train_and_save(path=MODEL_PATH, epochs=100):
    """Train DAEParam on the healthy FHI rides and save it."""
    dataset = ElevatorSliceParams(_slices("data/FHI.npz", range(1, 8), STEP_SIZE))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    print_traindata_parameter(loader)

    ae = DAEParam(21)
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
    train(ae, epochs, nn.MSELoss(), optimizer, loader)
    save_model(ae, path)
    return ae


def quantize_export(path=MODEL_PATH, out_dir="models/quantized", quantize_acts=True):
    """Quantize a saved model to every supported format, save each, and report
    how much the healthy/faulty separation degrades.

    quantize_acts also quantizes layer-boundary activations and the input, which
    is what real hardware does. Weight-only numbers are optimistic.
    """
    if not os.path.exists(path):
        print("%s not found, training first" % path)
        train_and_save(path)

    ae = load_model(DAEParam(21), path)

    healthy = _features("data/FHI.npz", range(1, 11), STEP_SIZE)
    faulty = _features("data/FHI-BREAK.npz", range(1, 11), STEP_SIZE)
    print("evaluating on %u healthy and %u faulty windows\n" % (len(healthy), len(faulty)))

    def scores(model, x):
        with torch.no_grad():
            return (model(x) - x).abs().sum(1)

    ref = torch.cat([scores(ae, healthy), scores(ae, faulty)])

    print("%-12s %8s %8s %9s %11s %9s" % ("format", "healthy", "faulty", "ratio", "score err%", "params"))
    print("-" * 62)

    for fmt in FORMATS:
        q = quantize_model(ae, fmt)
        save_model_quiet = os.path.join(out_dir, "daeparam_%s.pt" % fmt)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(q.state_dict(), save_model_quiet)

        model = quantize_activations(q, fmt) if quantize_acts else q
        s_h, s_f = scores(model, healthy), scores(model, faulty)
        err = 100.0 * (torch.cat([s_h, s_f]) - ref).abs().mean() / ref.mean()
        print("%-12s %8.4f %8.4f %8.3fx %10.3f%% %9u"
              % (fmt, s_h.mean(), s_f.mean(), s_f.mean() / s_h.mean(), err, get_n_params(q)))

    print("\nsaved %u quantized models to %s/" % (len(FORMATS), out_dir))
