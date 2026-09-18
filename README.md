# AutoencoderCPU

Unsupervised **predictive-maintenance** experiments for elevators, based on
autoencoder reconstruction error over 3-axis accelerometer recordings.

The premise is the standard anomaly-detection one: train an autoencoder to
reconstruct vibration data recorded from a **healthy** elevator. A healthy ride
is reconstructed well (low error); a ride with a developing fault contains
structure the model never learned, so the reconstruction error rises. The error
is used directly as a health score — no fault labels are needed for training.

Everything runs on CPU (PyTorch CPU build), hence the name.

---

## 1. Quick start

```bash
cd /home/misha/AutoencoderCPU

# One-time setup (a .venv already exists in the repo)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Run whichever experiment is enabled in main.py
python main.py
```

The existing `.venv/` is Python 3.10.12 with torch 2.14.0+cpu, numpy 2.2.6,
scipy, matplotlib and tqdm already installed, so `source .venv/bin/activate`
is usually enough.

> **Requires a display.** Most experiments call `plt.show()`, which blocks until
> the window is closed. Over SSH use `ssh -X`, or set `MPLBACKEND=Agg` and
> replace `plt.show()` with `plt.savefig(...)` in the script you are running.

---

## 2. How you "configure" it

There is **no CLI and no config file**. The two knobs are:

### 2.1 Which experiment runs — `main.py`

`main.py` imports every experiment and calls exactly one; the rest are commented
out. To switch experiments, move the comment:

```python
if __name__ == "__main__":
    #deep_ae()
    #deep_ae_fault_test()
    #conv_ae()
    #conv_ae_full_sample()
    #conv_ae_full_sample_fault_test()
    #conv_ae_amp_signals()
    #conv_ae_amp_signals_fault_test()
    #deep_ae_param()
    deep_ae_param_fault_test()      # <- currently active
```

### 2.2 Hyperparameters — edited in place

Window size, step, epochs, learning rate, batch size, which recordings are used
for training and which for test are **hardcoded inside each function** in
`scripts/`. There is no argument passing. The recurring constants are:

| Constant | Value | Meaning |
|---|---|---|
| `WINDOW_SIZE` | `208` | Samples per analysis window (0.5 s at 416 Hz) |
| `STEP_SIZE` | `104` | Hop between **training** windows (50 % overlap) |
| eval hop | `1` or `10` | Hop used when scoring a ride (dense sliding window) |
| batch size | `64` (train) / `1` (eval) | `DataLoader` batch |
| epochs | `100` (fault tests) / `5000` (early exploratory runs) | |
| optimizer | `Adam`, `lr=0.001` | `SGD` variants are present but commented out |
| loss | `nn.MSELoss()` | `L1Loss` present but commented out |

---

## 3. The data

`data/*.npz` — each file is one measurement campaign / site. Each archive holds
`arr_0 … arr_{n-1}`, one array per **ride** (Fahrt), shaped `(T, 3)` `float64`,
columns = `x, y, z` acceleration in g.

| File | Rides | Typical length | Role |
|---|---|---|---|
| `GTZH.npz` | 46 | ~10 000–11 000 samples | Largest healthy set; used for the "normal behaviour" experiments |
| `HS.npz` | 10 | ~8 700–13 500 | Second site, healthy |
| `OVGU.npz` | 5 | ~6 000–7 000 | Third site, used as an out-of-distribution check |
| `FHI.npz` | 10 | ~10 200–14 000 | **Healthy** reference elevator for the fault study |
| `FHI-BREAK.npz` | 10 | ~10 700–12 500 | **Same elevator with an induced brake fault** |

Sampling rate is **416 Hz** (it appears explicitly only in
`get_slice_envelope`, as `fs = 416`). Amplitudes are roughly ±1.4 g on GTZH/OVGU
and ±0.9 g on the FHI sets.

Indexing helper: `get_indexed_sample_from_data(file, number)` is **1-based** —
`number=1` returns `arr_0`.

`convert_data_R()` in `scripts/convert_data_R.py` dumps one ride to a long-format
`data.csv` (`sample, accel, axis`) for plotting in R.

---

## 4. Pipeline

```
 .npz ride (T,3)
      │
      ├─ get_window_slices_from_sample(data, size, step)   → list of (208,3) windows
      │
      ├─ optional: z_score_slices()   per-window, per-axis z-score  (present, mostly disabled)
      ├─ optional: get_slice_envelope()  |x|,|y| + 104 Hz Butterworth LP, z LP only
      │
      ├─ Dataset wrapper  (chooses the representation)
      │     ElevatorSlices        → flatten (208,3) column-major → 624-vector      [DAE]
      │     ElevatorSlicesChannels→ transpose to (3,208) channels-first            [CAE*]
      │     ElevatorSliceParams   → get_slice_params() → 21 hand-crafted features  [DAEParam]
      │
      ├─ train(model, epochs, MSELoss, Adam, loader)       (auto-uses CUDA if present)
      │
      └─ get_sample_error(model, loader)  → per-ride health score
```

### 4.1 The 21 features (`get_slice_params`)

Computed per window, in this exact order — 7 statistics × 3 axes, then 3
correlations:

```
0-2   peak   max(|x|), max(|y|), max(|z|)
3-5   min    min(x),   min(y),   min(z)
6-8   max    max(x),   max(y),   max(z)
9-11  rms    sqrt(sum(x^2)/N), …
12-14 mean
15-17 std
18    corr(x, y)
19    corr(x, z)
20    corr(y, z)
```

> Note: these features are fed to the network **unnormalised**. `corr` lives in
> [-1, 1] while `rms`/`std` are ~0.0x — relevant if you ever quantise this model.

### 4.2 Scoring (`get_sample_error`)

For each window: `error = recon - input`, and the **sum of absolute errors over
all elements** is accumulated. The reported value is that sum averaged over the
number of windows in the ride:

```
Average window error = Σ_windows Σ_elements |recon - input| / n_windows
Average window score = 100 / (1 + error)          # cosmetic, printed in bold
```

Because it is a *sum* over elements, the scale depends on the representation:
21 elements for `DAEParam`, 624 for `DAE`/`CAE`. Errors are **not comparable
across models**, only across rides for a fixed model.

---

## 5. Models (`lib/model.py`)

| Class | Input | Architecture (encoder → latent → decoder) | Params | MACs / window |
|---|---|---|---|---|
| `DAEParam(21)` | 21 features | `21→32→2` / `2→32→21` | **1 559** | **1 472** |
| `DAE` | 624 raw | `624→256→32→2` / mirrored | 337 202 | 336 000 |
| `CAETest` | `(3,208)` | Conv1d `3→16→3` (s=2) / ConvT mirrored | 614 | ~67 400 |
| `CAE` | `(3,208)` | Conv1d `3→16→32→64→8` (s=2) / ConvT mirrored | 18 955 | ~823 700 |
| `CAEDenseDecoder` | `(3,208)` | Conv encoder → dense decoder | ~950 k | — |

All use `ReLU` between layers and a **linear output layer** (no activation on the
reconstruction), latent dimension 2 for the dense models.

`get_n_params(model)` prints the parameter count; several experiments do this.

---

## 6. The experiments

### `deep_ae_param_fault_test()` — the working fault detector *(currently enabled)*

Trains `DAEParam(21)` on FHI rides 1–7 (healthy), 100 epochs, then scores all 10
FHI (healthy) and all 10 FHI-BREAK (faulty) rides with a dense hop-1 sliding
window. Writes `deep_ae_param_fault_test.csv` (`model, elevator_status, error`).

This is the result that actually separates. From the committed CSV:

| Class | n | min | max | mean |
|---|---|---|---|---|
| FHI (healthy) | 10 | 0.627 | **0.777** | 0.669 |
| FHI-BREAK (faulty) | 10 | **0.902** | 1.373 | 1.130 |

No overlap — any threshold in (0.78, 0.90) classifies all 20 rides correctly.

### `deep_ae_param()`
Same model, trained on GTZH rides 1–31, scored on all 46 GTZH rides and the 5
OVGU rides. Exploratory: does the model flag a different site as anomalous?

### `deep_ae()` / `deep_ae_fault_test()`
`DAE` on the raw 624-sample flattened window. `deep_ae()` trains 5000 epochs on a
single ride (GTZH #24) and then animates the reconstruction window by window.

### `conv_ae()` / `conv_ae_full_sample()` / `conv_ae_full_sample_fault_test()`
`CAE`/`CAETest` on channels-first raw windows. `conv_ae_full_sample()` also
scores an all-zeros window as a sanity baseline, and evaluates against OVGU and
HS. Writes `conv_ae_fault_test.csv`.

### `conv_ae_amp_signals()` / `conv_ae_amp_signals_fault_test()`
Same as above but the input is the **envelope** (`get_slice_envelope`): rectify
x and y, then a 2nd-order Butterworth low-pass at 104 Hz; z is low-passed
without rectification. Writes `conv_ae_amp_signals_fault_test.csv`.

`conv_ae_amp_signals_test()` is a visual check of the envelope only — no
training.

---

## 7. Outputs

| Path | Produced by |
|---|---|
| `deep_ae_param_fault_test.csv` | `deep_ae_param_fault_test()` (present in the repo) |
| `conv_ae_fault_test.csv` | `conv_ae_full_sample_fault_test()` |
| `conv_ae_amp_signals_fault_test.csv` | `conv_ae_amp_signals_fault_test()` |
| `data.csv` | `convert_data_R()` |
| `plots/<SITE>-<n>.png` | Rendered by hand from the reconstruction plots; **not** written by the scripts |

Interactive figures: `plot_slice_reconstruction()` (single window, input/output/
error per axis) and `plot_slice_reconstruction_continuous(fps, ae, loader)`
(animated sweep along a ride).

---

## 8. Known limitations

Worth knowing before building anything on top of this:

1. **No model persistence.** Nothing calls `torch.save` / `load_state_dict`.
   Every run retrains from scratch, and the trained weights are lost on exit.
   Exporting weights is the first thing needed for any deployment.
2. **Train/test overlap in the fault tests.** `deep_ae_param_fault_test` trains
   on FHI rides 1–7 and then scores FHI rides 1–10 — 7 of the 10 "healthy" test
   rides were seen during training. The healthy/faulty gap is large enough that
   the conclusion probably survives, but the healthy scores are optimistic.
   Rides 8, 9, 10 (0.777, 0.669, 0.631) are the only clean healthy numbers.
3. **No fixed seed.** `torch.manual_seed` is never called, so numbers move
   between runs.
4. **No threshold is ever computed.** The separation is read off the CSV by eye;
   there is no decision rule in code.
5. **`plt.show()` blocks** in the middle of several experiments — the evaluation
   loop only continues after you close the window.
6. **Features are unnormalised** (see §4.1), and `z_score_slices` is commented
   out of the raw-window paths.
7. Single-window `Dataset`s rebuild the whole slice list in memory; a dense
   hop-1 evaluation of a 10 000-sample ride creates ~9 800 windows.
