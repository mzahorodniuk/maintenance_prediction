"""Quantize DAEParam with Brevitas, export it through QONNX to standard
ONNX QDQ and compile it with Apache TVM 0.16 to integer-only bare-metal C for
the Simply-V RISC-V cores (AOT executor, C runtime, no heap).

    format    weights                           activations
    bw8a8     int8 per output channel           int8 (uint8 after ReLU)
    bw4a8     int4 per output channel (int8)    int8 (uint8 after ReLU)

Pipeline per format:
    trained float weights -> Brevitas model (QuantLinear, Int32Bias)
    -> calibration -> quantization-aware training (QAT)
    -> export_qonnx -> qonnx cleanup + QuantToQCDQ (+ 32-bit bias Quant
       rewritten as DequantizeLinear(int32)) -> ONNX QDQ
    -> relay.frontend.from_onnx -> FakeQuantizationToInteger
    -> relay.build target c, AOT -> Model Library Format -> C

QAT follows scripts/deep_ae_param.py (healthy FHI rides 1-7, window 208,
step 104, MSE, Adam, batch 64): it starts from the trained float weights,
calibrated, and fine-tunes at lr 1e-4 for 30 epochs. The accuracy before
fine-tuning (PTQ) is printed as well.

Each format is also built with integer input and output ("<format>-int"):
the graph is cut after the input quantize and before the output dequantize,
and the Q2.14 <-> int8 conversion is written to ae_tvm_io.h as fixed-point
multipliers, so the device needs no float at all.

Writes into out_dir, per format: <format>/*.c and tvmgen_default.h (from the
Model Library Format), model.pt (the QAT checkpoint), model_qdq.onnx, and
runtime_include/ (the TVM runtime headers the generated C includes); prints
the accuracy table: the Brevitas model, the same Relay module compiled for
the host with LLVM, and the PTQ model.

Needs the TVM 0.16 environment (python3.10 venv ~/tvm-env with
PYTHONPATH=~/tvm-0.16/python):
    from scripts.export_tvm import export_tvm
    export_tvm("/home/misha/simplyv_ddr_nexys/sw/SoC/examples/ae_tvm/gen")
"""
import math
import os
import shutil
import tarfile
import tempfile

import numpy as np
import onnx
import torch
import torch.nn as nn

import tvm
from tvm import relay
from tvm.relay.backend import Executor, Runtime

from lib.utils import load_model
from lib.model import DAEParam
from lib.fixed_features import fixed_features, to_fixed, SCALE
from scripts.export_weights_c import MODEL_PATH, _all_windows

FORMATS = ("bw8a8", "bw4a8")
QAT_EPOCHS, QAT_LR, QAT_BATCH = 30, 1e-4, 64
RUNTIME_HEADERS = ("tvm/runtime/c_runtime_api.h", "tvm/runtime/c_backend_api.h", "dlpack/dlpack.h")


# ---------------------------------------------------------------- data, scoring

def _data():
    windows, tags = _all_windows()
    x = (np.stack([fixed_features(to_fixed(w)) for w in windows]) / SCALE).astype(np.float32)
    return x, np.array(tags)


def _row(score, tags, ref):
    h = [score[(tags[:, 0] == 0) & (tags[:, 1] == n)].mean() for n in range(1, 11)]
    f = [score[(tags[:, 0] == 1) & (tags[:, 1] == n)].mean() for n in range(1, 11)]
    err = np.mean(np.abs(score - ref) / ref) * 100
    return max(h), min(f), err


def _score(recon, x):
    return np.abs(recon - x).sum(1) * SCALE


def _torch_recon(model, x):
    with torch.no_grad():
        y = model(torch.from_numpy(x))
    return (y.value if hasattr(y, "value") else y).numpy()


# ---------------------------------------------------------------- Brevitas

def _brevitas_model(ae, wbits):
    from brevitas.nn import QuantIdentity, QuantLinear, QuantReLU
    from brevitas.quant import (Int8ActPerTensorFloat, Int8WeightPerChannelFloat, Int32Bias,
                                Uint8ActPerTensorFloat)

    def lin(i, o):
        return QuantLinear(i, o, bias=True, weight_quant=Int8WeightPerChannelFloat,
                           weight_bit_width=wbits, bias_quant=Int32Bias, return_quant_tensor=True)

    class BrevitasDAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
            self.l0, self.r0 = lin(21, 32), QuantReLU(act_quant=Uint8ActPerTensorFloat, return_quant_tensor=True)
            self.l1, self.q1 = lin(32, 2), QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
            self.l2, self.r2 = lin(2, 32), QuantReLU(act_quant=Uint8ActPerTensorFloat, return_quant_tensor=True)
            self.l3, self.q3 = lin(32, 21), QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)

        def forward(self, x):
            x = self.r0(self.l0(self.inp(x)))
            x = self.q1(self.l1(x))
            x = self.r2(self.l2(x))
            return self.q3(self.l3(x))

    m = BrevitasDAE()
    for dst, src in (("l0", "encoder.0"), ("l1", "encoder.2"), ("l2", "decoder.0"), ("l3", "decoder.2")):
        getattr(m, dst).weight.data.copy_(ae.get_submodule(src).weight.data)
        getattr(m, dst).bias.data.copy_(ae.get_submodule(src).bias.data)
    return m


def _bias_quant_to_dq(model):
    """QuantToQCDQ converts only Quant nodes of <= 8 bits (QuantizeLinear has
    no int32); the 32-bit bias quantizers stay. Their inputs are constants, so
    replace each with its standard QDQ form, DequantizeLinear(int32, scale),
    as ONNX Runtime's quantizer writes biases."""
    from onnx import helper, numpy_helper
    init = {i.name: i for i in model.graph.initializer}
    nodes = []
    for n in model.graph.node:
        if n.op_type != "Quant":
            nodes.append(n)
            continue
        bias, scale, zero, bits = (numpy_helper.to_array(init[i]) for i in n.input)
        if int(bits) != 32 or np.any(zero != 0):
            raise RuntimeError("unexpected Quant node %s (%d bits)" % (n.name, int(bits)))
        q = np.clip(np.round(bias / scale), -2 ** 31, 2 ** 31 - 1).astype(np.int32)
        names = [n.name + "_int32", n.name + "_scale", n.name + "_zero"]
        scale = np.asarray(scale, dtype=np.float32)
        model.graph.initializer.extend([
            numpy_helper.from_array(q, names[0]),
            numpy_helper.from_array(scale.reshape(-1) if scale.size > 1 else scale.reshape(()), names[1]),
            numpy_helper.from_array(np.zeros(scale.shape if scale.size > 1 else (), np.int32), names[2])])
        attrs = {"axis": 0} if scale.size > 1 else {}
        nodes.append(helper.make_node("DequantizeLinear", names, list(n.output), name=n.name + "_dq", **attrs))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {i for n in model.graph.node for i in n.input}
    keep = [i for i in model.graph.initializer if i.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    keep = [o for o in model.opset_import if o.domain != "qonnx.custom_op.general"]
    del model.opset_import[:]
    model.opset_import.extend(keep)
    return model


def _brevitas(ae, x, tags, ref_score, wbits, workdir):
    from brevitas.export import export_qonnx
    from brevitas.graph.calibrate import calibration_mode
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.transformation.qonnx_to_qcdq import QuantToQCDQ
    from qonnx.util.cleanup import cleanup_model

    torch.manual_seed(0)
    m = _brevitas_model(ae, wbits)
    m.eval()
    with torch.no_grad(), calibration_mode(m):
        for i in range(0, len(x), 256):
            m(torch.from_numpy(x[i:i + 256]))
    ptq = _row(_score(_torch_recon(m.eval(), x), x), tags, ref_score)

    # QAT on the healthy rides the float model was trained on
    train = torch.from_numpy(x[(tags[:, 0] == 0) & (tags[:, 1] <= 7)])
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.Adam(m.parameters(), lr=QAT_LR)
    loss_fn = nn.MSELoss()
    m.train()
    for _ in range(QAT_EPOCHS):
        for idx in torch.randperm(len(train), generator=gen).split(QAT_BATCH):
            opt.zero_grad()
            loss = loss_fn(m(train[idx]), train[idx])
            loss.backward()
            opt.step()
    m.eval()
    recon = _torch_recon(m, x)

    qonnx_path = os.path.join(workdir, "model_qonnx.onnx")
    export_qonnx(m, torch.zeros(1, 21), export_path=qonnx_path)
    mw = cleanup_model(ModelWrapper(qonnx_path))
    mw = cleanup_model(mw.transform(QuantToQCDQ()))
    model = _bias_quant_to_dq(mw.model)
    old = model.graph.input[0].name             # name the input "feat", as the other formats
    for node in model.graph.node:
        node.input[:] = ["feat" if n == old else n for n in node.input]
    model.graph.input[0].name = "feat"
    onnx.save(model, os.path.join(workdir, "model_qdq.onnx"))

    mod, params = relay.frontend.from_onnx(model, shape={"feat": (1, 21)})
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.FakeQuantizationToInteger()(mod)
    return mod, params, recon, ptq, m


# ---------------------------------------------------------------- integer I/O

def _quantize_multiplier(real):
    """real = m * 2^shift / 2^31, m in [2^30, 2^31): TFLite's QuantizeMultiplier"""
    if real == 0:
        return 0, 0
    frac, shift = math.frexp(real)
    m = int(round(frac * (1 << 31)))
    if m == 1 << 31:
        m //= 2
        shift += 1
    return m, shift


def _const(e):
    return e.data.numpy().reshape(-1)


def _int_io(mod):
    """Cut the graph after the input qnn.quantize and before the output
    qnn.dequantize; return the new module and the I/O quantization."""
    func = mod["main"]
    feat = func.params[0]
    io = {}

    class Cut(relay.ExprMutator):
        def visit_call(self, call):
            if call.op.name == "qnn.quantize" and call.args[0] == feat:
                io["in"] = (float(_const(call.args[1])[0]), int(_const(call.args[2])[0]),
                            call.attrs.out_dtype)
                io["var"] = relay.var("feat_q", shape=(1, 21), dtype=call.attrs.out_dtype)
                return io["var"]
            return super().visit_call(call)

    body = func.body
    if not (isinstance(body, relay.Call) and body.op.name == "qnn.dequantize"):
        raise RuntimeError("output is not a qnn.dequantize: %s" % body.op)
    io["out"] = (float(_const(body.args[1])[0]), int(_const(body.args[2])[0]),
                 str(relay.transform.InferType()(tvm.IRModule.from_expr(body.args[0]))["main"].ret_type.dtype))
    new_body = Cut().visit(body.args[0])
    if "var" not in io:
        raise RuntimeError("no qnn.quantize on the input")
    # any other parameters (weights the frontend left as inputs) stay; params binds them
    return tvm.IRModule.from_expr(relay.Function([io["var"]] + list(func.params[1:]), new_body)), io


def _io_header(io, path):
    s_in, z_in, t_in = io["in"]
    s_out, z_out, t_out = io["out"]
    m_in, sh_in = _quantize_multiplier(1.0 / (SCALE * s_in))     # Q2.14 -> input
    m_out, sh_out = _quantize_multiplier(SCALE * s_out)          # output -> Q2.14
    ctype = {"int8": "int8_t", "uint8": "uint8_t", "int16": "int16_t"}
    lim = {"int8": (-128, 127), "uint8": (0, 255), "int16": (-32768, 32767)}
    with open(path, "w") as f:
        f.write("// Generated by scripts/export_tvm.py -- do not edit.\n"
                "// Integer I/O of the model: input = clamp(Q2.14 * in_mult * 2^in_shift / 2^31 + zero point),\n"
                "// Q2.14 = (output - zero point) * out_mult * 2^out_shift / 2^31.\n"
                "#ifndef AE_TVM_IO_H\n#define AE_TVM_IO_H\n\n#include <stdint.h>\n\n")
        f.write("typedef %s ae_tvm_in_t;\ntypedef %s ae_tvm_out_t;\n\n" % (ctype[t_in], ctype[t_out]))
        f.write("#define AE_TVM_IN_MULT   %dL\n#define AE_TVM_IN_SHIFT  %d\n#define AE_TVM_IN_ZERO   %d\n"
                "#define AE_TVM_IN_MIN    %d\n#define AE_TVM_IN_MAX    %d\n" % (m_in, sh_in, z_in, *lim[t_in]))
        f.write("#define AE_TVM_OUT_MULT  %dL\n#define AE_TVM_OUT_SHIFT %d\n#define AE_TVM_OUT_ZERO  %d\n\n#endif\n"
                % (m_out, sh_out, z_out))


# ---------------------------------------------------------------- compile

def _build_c(mod, params, out, runtime_include):
    """AOT, C interface, unpacked API, parameters linked in, static workspace"""
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True,
                                                        "tir.usmp.enable": True}):
        lib = relay.build(mod, target=tvm.target.Target("c"), params=params, runtime=Runtime("crt"),
                          executor=Executor("aot", {"interface-api": "c", "unpacked-api": True,
                                                    "link-params": True,
                                                    "workspace-byte-alignment": 8}))
    with tempfile.TemporaryDirectory() as tmp:
        tar = os.path.join(tmp, "mlf.tar")
        tvm.micro.export_model_library_format(lib, tar)
        with tarfile.open(tar) as t:
            t.extractall(tmp)
        if os.path.isdir(out):
            shutil.rmtree(out)
        os.makedirs(out)
        src = os.path.join(tmp, "codegen", "host", "src")
        for name in sorted(os.listdir(src)):
            shutil.copy(os.path.join(src, name), out)
        shutil.copy(os.path.join(tmp, "codegen", "host", "include", "tvmgen_default.h"), out)
        rt = os.path.join(tmp, "runtime", "include")
        for h in RUNTIME_HEADERS:         # only what the generated C includes
            os.makedirs(os.path.dirname(os.path.join(runtime_include, h)), exist_ok=True)
            shutil.copy(os.path.join(rt, h), os.path.join(runtime_include, h))
    return lib


def _host_recon(mod, params, x):
    """The same Relay module on the host (LLVM), every window"""
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="llvm", params=params)
    gm = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.cpu()))
    out = np.empty_like(x)
    for i, v in enumerate(x):
        gm.set_input(0, v[None])
        gm.run()
        out[i] = gm.get_output(0).numpy()[0]
    return out


def _float_ops(mod):
    """Names of float-typed calls left in a quantized graph (should be only
    the input quantize and output dequantize)"""
    mod = relay.transform.InferType()(mod)
    found = []

    def visit(e):
        if isinstance(e, relay.Call) and hasattr(e.op, "name"):
            if str(e.checked_type).endswith("float32]") and e.op.name not in ("qnn.dequantize",):
                found.append(e.op.name)
    relay.analysis.post_order_visit(mod["main"].body, visit)
    return found


def export_tvm(out_dir, model_path=MODEL_PATH, formats=FORMATS):
    os.makedirs(out_dir, exist_ok=True)
    rt_inc = os.path.join(out_dir, "runtime_include")
    ae = load_model(DAEParam(21), model_path)
    x, tags = _data()
    with torch.no_grad():
        ref_score = _score(ae(torch.from_numpy(x)).numpy(), x)

    rows = []
    for fmt in formats:
        work = tempfile.mkdtemp()
        wbits = {"bw8a8": 8, "bw4a8": 4}[fmt]
        mod, params, recon, ptq, m = _brevitas(ae, x, tags, ref_score, wbits, work)
        left = [o for o in _float_ops(mod) if o not in ("qnn.quantize",)]
        if left:
            print("%s: float ops left in the graph: %s" % (fmt, sorted(set(left))))

        _build_c(mod, params, os.path.join(out_dir, fmt), rt_inc)
        torch.save(m.state_dict(), os.path.join(out_dir, fmt, "model.pt"))
        shutil.copy(os.path.join(work, "model_qdq.onnx"), os.path.join(out_dir, fmt))
        host = _host_recon(mod, params, x)
        mod_i, io = _int_io(mod)
        _build_c(mod_i, params, os.path.join(out_dir, fmt + "-int"), rt_inc)
        _io_header(io, os.path.join(out_dir, fmt + "-int", "ae_tvm_io.h"))
        shutil.rmtree(work)

        rows.append((fmt, "torch", *_row(_score(recon, x), tags, ref_score)))
        rows.append((fmt, "tvm", *_row(_score(host, x), tags, ref_score)))
        rows.append((fmt, "ptq", *ptq))

    print("%-9s %-6s %9s %9s %9s %10s" % ("format", "by", "healthy", "faulty", "margin", "score err"))
    for fmt, by, h, f, err in rows:
        print("%-9s %-6s %9.1f %9.1f %+9.1f %9.2f%%" % (fmt, by, h, f, f - h, err))
    print("(healthy = highest healthy ride mean, faulty = lowest faulty ride mean, Q2.14 units;"
          " score err = mean |score - PyTorch fp32 score| / score; by: torch = the Brevitas"
          " model, tvm = the TVM module on the host, ptq = Brevitas before QAT)")
    print("wrote %s/{%s}" % (out_dir, ",".join(formats)))
