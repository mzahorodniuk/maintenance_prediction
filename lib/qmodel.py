"""DAEParam quantized with Brevitas for the FPGA generators (FINN, hls4ml).

Every scale is a power of two and every zero point is 0, so each tensor is a
plain fixed-point number: hls4ml maps it to an ap_fixed and FINN to integer
thresholds, with no per-channel rescaling left over. Activations round with
FLOOR (an arithmetic shift in hardware), which QONNX, hls4ml (AP_TRN) and FINN
(thresholds moved by half a step) all compute the same way; ROUND would leave
the tools to disagree on exact ties.

    tensor      quantizer                                  codes
    input       16 bit, scale 2^-14 (constant)             Q2.14, the features as computed
    weights     int8 narrow, per tensor, power-of-two      int8
    biases      int32, scale = input scale * weight scale  int32
    after ReLU  uint8, power-of-two scale, FLOOR           uint8
    latent      int8, power-of-two scale, FLOOR            int8
    output      16 bit, scale 2^-14 (constant), FLOOR      Q2.14 recon

Used by scripts/fpga_qonnx.py.
"""
import torch.nn as nn

from brevitas.inject.enum import FloatToIntImplType, ScalingImplType
from brevitas.nn import QuantIdentity, QuantLinear, QuantReLU
from brevitas.quant import (Int8ActPerTensorFixedPoint, Int8WeightPerTensorFixedPoint, Int32Bias,
                            Uint8ActPerTensorFixedPoint)


class Q214Act(Int8ActPerTensorFixedPoint):
    """Q2.14: 16 bit, scale fixed at 2^-14 (max_val 2.0), floor"""
    bit_width = 16
    scaling_impl_type = ScalingImplType.CONST
    scaling_const = 2.0
    max_val = 2.0
    min_val = -2.0
    float_to_int_impl_type = FloatToIntImplType.FLOOR


class Uint8FloorAct(Uint8ActPerTensorFixedPoint):
    float_to_int_impl_type = FloatToIntImplType.FLOOR


class Int8FloorAct(Int8ActPerTensorFixedPoint):
    float_to_int_impl_type = FloatToIntImplType.FLOOR


def _linear(n_in, n_out, w_bits):
    return QuantLinear(n_in, n_out, bias=True, weight_quant=Int8WeightPerTensorFixedPoint,
                       weight_bit_width=w_bits, bias_quant=Int32Bias, return_quant_tensor=True)


class QDAEParam(nn.Module):
    """DAEParam(21) (21 -> 32 ReLU -> 2 -> 32 ReLU -> 21) with the quantizers above"""

    LAYERS = (("l0", "encoder.0"), ("l1", "encoder.2"), ("l2", "decoder.0"), ("l3", "decoder.2"))

    def __init__(self, inout=21, w_bits=8, a_bits=8):
        super().__init__()
        self.inp = QuantIdentity(act_quant=Q214Act, return_quant_tensor=True)
        self.l0 = _linear(inout, 32, w_bits)
        self.r0 = QuantReLU(act_quant=Uint8FloorAct, bit_width=a_bits, return_quant_tensor=True)
        self.l1 = _linear(32, 2, w_bits)
        self.q1 = QuantIdentity(act_quant=Int8FloorAct, bit_width=a_bits, return_quant_tensor=True)
        self.l2 = _linear(2, 32, w_bits)
        self.r2 = QuantReLU(act_quant=Uint8FloorAct, bit_width=a_bits, return_quant_tensor=True)
        self.l3 = _linear(32, inout, w_bits)
        self.out = QuantIdentity(act_quant=Q214Act, return_quant_tensor=False)

    def forward(self, x):
        x = self.r0(self.l0(self.inp(x)))
        x = self.q1(self.l1(x))
        x = self.r2(self.l2(x))
        return self.out(self.l3(x))

    def load_float(self, ae):
        """Copy the weights and biases of a trained float DAEParam."""
        for dst, src in self.LAYERS:
            getattr(self, dst).weight.data.copy_(ae.get_submodule(src).weight.data)
            getattr(self, dst).bias.data.copy_(ae.get_submodule(src).bias.data)
        return self

    def linears(self):
        return [getattr(self, n) for n, _ in self.LAYERS]

    def acts(self):
        """The quantizer after each linear layer"""
        return [self.r0, self.q1, self.r2, self.out]
