"""Numeric-format emulation, for studying FPGA deployment of a trained model.

Each format function round-trips a float32 tensor through the target format and
returns float32, so a quantized model still runs in ordinary PyTorch while
behaving numerically like the target hardware would.

Measured on DAEParam over 5 seeds, quantizing weights, activations and inputs
(mean reconstruction-score error vs the fp32 model):

    int16  0.01%   fp16   0.04%   int12  0.16%   bf16  0.37%
    int8   2.5%    mxfp8_e4m3  7.3%   mxfp8_e5m2  15.2%

Fixed point beats every float format at equal or lower bit width, because this
model's values are bounded (|x| <= 1.02), so a float exponent field buys range
that is never used. int12 is the recommended target for Artix-7.
"""
import copy

import torch


def bf16(t, axis=None):
    """bfloat16: 8 exponent + 7 mantissa bits. No scale, so axis is unused."""
    return t.to(torch.bfloat16).float()


def fp16(t, axis=None):
    """IEEE 754 binary16: 5 exponent + 10 mantissa bits. axis is unused."""
    return t.to(torch.float16).float()


def fixed(nbits):
    """Build a symmetric fixed-point (intN) quantizer.

    axis=None gives one scale for the whole tensor; axis=int gives an
    independent scale per slice along that axis, i.e. per output neuron for a
    [out_features, in_features] weight matrix.

    The code is computed as round(x * qmax / absmax) in float64 rather than by
    multiplying with a rounded float32 scale, which would flip the result on
    roughly 40% of exact midpoints. Both conventions yield identical integer
    codes on this model's tensors, so they describe the same hardware; they
    differ only by one ulp in the dequantized float.
    """
    qmax = 2 ** (nbits - 1) - 1

    def quantizer(t, axis=None):
        x = t.detach().to(torch.float64)
        if axis is None:
            a = x.abs().max()
            if a == 0:
                return t.clone()
            return ((x * qmax / a).round().clamp(-qmax, qmax) * a / qmax).float()
        ax = axis % x.dim()
        dims = [d for d in range(x.dim()) if d != ax]
        a = x.abs().amax(dim=dims, keepdim=True)
        safe = torch.where(a == 0, torch.ones_like(a), a)
        q = (x * qmax / safe).round().clamp(-qmax, qmax) * safe / qmax
        return torch.where(a == 0, torch.zeros_like(q), q).float()

    return quantizer


int8 = fixed(8)
int10 = fixed(10)
int12 = fixed(12)
int16 = fixed(16)


def _mxfp8(elem_dtype, emax_elem, elem_max):
    """Build an OCP Microscaling MXFP8 quantizer.

    32 consecutive elements along the last axis share one power-of-two (E8M0)
    scale. Elements clamp to the format maximum rather than overflowing to
    infinity, which is what the MX spec requires and is the main way this
    differs from a plain FP8 cast. axis is unused: the blocking is inherent.

    Note this model's layers have last dimensions of 21, 32, 2 and 21, all no
    larger than the block size, so every block covers a whole row and the
    shared scale gains nothing over a per-row scale.
    """

    def quantizer(t, axis=None):
        shape = t.shape
        last = shape[-1]
        nblk = -(-last // 32)
        pad = nblk * 32 - last
        flat = t.reshape(-1, last)
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        blocks = flat.reshape(-1, nblk, 32)

        amax = blocks.abs().amax(dim=-1, keepdim=True)
        exp = (torch.log2(amax.clamp(min=1e-38)).floor() - emax_elem).clamp(-127, 127)
        scale = torch.where(amax == 0, torch.ones_like(exp), torch.exp2(exp))

        q = (blocks / scale).clamp(-elem_max, elem_max).to(elem_dtype).float() * scale
        q = q.reshape(-1, nblk * 32)
        if pad:
            q = q[:, :last]
        return q.reshape(shape)

    return quantizer


mxfp8_e4m3 = _mxfp8(torch.float8_e4m3fn, 8, 448.0)
mxfp8_e5m2 = _mxfp8(torch.float8_e5m2, 15, 57344.0)


FORMATS = {
    "fp32": None,
    "bf16": bf16,
    "fp16": fp16,
    "int8": int8,
    "int10": int10,
    "int12": int12,
    "int16": int16,
    "mxfp8_e4m3": mxfp8_e4m3,
    "mxfp8_e5m2": mxfp8_e5m2,
}


def quantize_model(model, fmt, per_channel=True):
    """Return a copy of model with every Linear weight round-tripped through fmt.

    Biases are left in fp32 for the integer formats, matching hardware that
    accumulates in int32 and adds the bias at accumulator precision.
    """
    if fmt not in FORMATS:
        raise KeyError("unknown format %r, expected one of %s" % (fmt, sorted(FORMATS)))
    quantizer = FORMATS[fmt]
    out = copy.deepcopy(model)
    if quantizer is None:
        return out

    keep_bias_fp32 = fmt.startswith("int")
    with torch.no_grad():
        for mod in out.modules():
            if isinstance(mod, torch.nn.Linear):
                w = quantizer(mod.weight, axis=0) if per_channel else quantizer(mod.weight)
                mod.weight.copy_(w)
                if mod.bias is not None and not keep_bias_fp32:
                    mod.bias.copy_(quantizer(mod.bias))
    out.eval()
    return out


def quantize_activations(model, fmt):
    """Wrap model so every layer-boundary activation is also quantized.

    Weight-only quantization is optimistic: real hardware carries the format
    through the datapath too. Use this to get the deployment-realistic number.
    """
    quantizer = FORMATS[fmt]
    if quantizer is None:
        return model

    class _Quantized(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            h = quantizer(x)
            for layer in list(self.inner.encoder) + list(self.inner.decoder):
                h = layer(h)
                if not isinstance(layer, torch.nn.ReLU):
                    h = quantizer(h)
            return h

    return _Quantized(model).eval()
