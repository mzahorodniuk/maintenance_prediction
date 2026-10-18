"""Integer forward pass of a power-of-two QONNX model (lib/qmodel.py exported by
scripts/export_qonnx.py), with onnx and numpy only.

It is the reference the hardware generators check against: FINN, hls4ml and the
RTL wrappers must give exactly these reconstructions. Every tensor is an int64
code with a power-of-two exponent (value = code * 2^exp), so nothing is rounded
except where a Quant node says so:

    Quant on a constant   code = clip(round_half_even(w / scale))     (weights, biases)
    Quant on activations  code = clip(floor or round(x * 2^(e_x - e))) (shift, no float)
    Gemm / MatMul + Add   int64 accumulation, exponent e_x + e_w
    Relu                  max(code, 0)

Input and output are the Q2.14 feature codes (int16 range, exponent -14).
scripts/export_qonnx.py checks it against QONNX's own execution in float64.
"""
import numpy as np
import onnx
from onnx import numpy_helper

IN_EXP = -14


def _exp2(scale, what):
    s = np.asarray(scale, dtype=np.float64).reshape(-1)
    if s.size != 1:
        raise ValueError("%s: per-channel scale, expected one power-of-two scale" % what)
    e = np.log2(float(s[0]))
    if e != round(e):
        raise ValueError("%s: scale %r is not a power of two" % (what, float(s[0])))
    return int(round(e))


def _limits(bits, signed, narrow):
    if signed:
        return -(2 ** (bits - 1)) + (1 if narrow else 0), 2 ** (bits - 1) - 1
    return 0, 2 ** bits - 1 - (1 if narrow else 0)


def _shift(code, by, mode):
    """code * 2^by, rounded as the Quant node says (by < 0 drops bits)"""
    if by >= 0:
        return code << by
    d = -by
    if mode == "FLOOR":
        return code >> d
    if mode == "ROUND":                       # half to even, as np.round
        q, r = code >> d, code & ((1 << d) - 1)
        half = 1 << (d - 1)
        return q + ((r > half) | ((r == half) & ((q & 1) == 1)))
    raise ValueError("rounding mode %s" % mode)


class QonnxInt:
    """Parsed once; forward() on any number of feature vectors."""

    def __init__(self, path):
        model = onnx.load(path)
        g = model.graph
        self.init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
        self.nodes = list(g.node)
        self.input = g.input[0].name
        self.output = g.output[0].name
        self.quant = {}         # dynamic Quant nodes: output -> (exp, lo, hi, mode)
        self.const = {}         # constants after their Quant: name -> (code, exp)
        for n in self.nodes:
            if n.op_type != "Quant":
                continue
            a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
            mode = a.get("rounding_mode", b"ROUND")
            mode = (mode.decode() if isinstance(mode, bytes) else mode).upper()
            if np.any(self.init[n.input[2]] != 0):
                raise ValueError("%s: zero point is not 0" % n.name)
            e = _exp2(self.init[n.input[1]], n.name)
            lo, hi = _limits(int(self.init[n.input[3]]), a.get("signed", 1), a.get("narrow", 0))
            if n.input[0] in self.init:
                w = self.init[n.input[0]].astype(np.float64) / 2.0 ** e
                code = np.clip(np.round(w) if mode == "ROUND" else np.floor(w), lo, hi).astype(np.int64)
                self.const[n.output[0]] = (code, e)
            else:
                self.quant[n.output[0]] = (e, lo, hi, mode)

    def layers(self):
        """[(weight code [out][in], weight exp, bias code [out] in accumulator units)],
        in graph order (for the exporters' summaries)"""
        out = []
        for n in self.nodes:
            if n.op_type == "Gemm":
                w, ew = self.const[n.input[1]]
                b = np.zeros(w.shape[0], np.int64)
                if len(n.input) > 2:
                    code, eb = self.const[n.input[2]]
                    b = code << (eb - self._exp_in(n) - ew)
                out.append((w, ew, b))
        return out

    def _exp_in(self, gemm):
        """Exponent of a Gemm's activation input (its Quant node, or the graph input)"""
        src = gemm.input[0]
        return self.quant[src][0] if src in self.quant else IN_EXP

    def forward(self, feats):
        """feats: int Q2.14 codes [n][21] -> recon codes [n][21] (int64)"""
        t = {self.input: (np.asarray(feats, dtype=np.int64), IN_EXP)}
        for n in self.nodes:
            if n.op_type == "Quant":
                if n.output[0] in self.const:
                    continue
                code, ex = t[n.input[0]]
                e, lo, hi, mode = self.quant[n.output[0]]
                t[n.output[0]] = (np.clip(_shift(code, ex - e, mode), lo, hi), e)
            elif n.op_type == "Gemm":
                a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
                if a.get("transA", 0) or a.get("alpha", 1.0) != 1.0 or a.get("beta", 1.0) != 1.0:
                    raise ValueError("%s: only Gemm with transB" % n.name)
                x, ex = t[n.input[0]]
                w, ew = self.const[n.input[1]]
                acc = x @ (w.T if a.get("transB", 0) else w)
                if len(n.input) > 2:
                    b, eb = self.const[n.input[2]]
                    if eb < ex + ew:
                        raise ValueError("%s: bias finer than the accumulator" % n.name)
                    acc = acc + _shift(b, eb - (ex + ew), "FLOOR")
                t[n.output[0]] = (acc, ex + ew)
            elif n.op_type == "Relu":
                code, ex = t[n.input[0]]
                t[n.output[0]] = (np.maximum(code, 0), ex)
            else:
                raise ValueError("unsupported node %s (%s)" % (n.name, n.op_type))
        code, e = t[self.output]
        if e != IN_EXP:
            raise ValueError("output exponent %d, expected Q2.14" % e)
        return code


def score(recon, feats):
    """sum |recon - feature| over the 21 features, Q2.14 units"""
    return np.abs(np.asarray(recon, np.int64) - np.asarray(feats, np.int64)).sum(1)
