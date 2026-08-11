"""The RKNPU-only rewrite rows and the `rknn.config` extras derived beside them; no toolkit import needed."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx_ir as ir
from onnxscript.rewriter.pattern import MatchResult, RewriteRuleClassBase

from ..onnx._ir import make_init, make_node, sole_consumer
from ..onnx.lowering import HostCtcDecodePass

# the mean the DMA now owes the graph, carried on the graph so the compiler reads both off one file
_DMA_MEAN = "rknn_input_mean"


def rknn_config(prepared: Path) -> dict[str, Any]:
    """mean/std for `rknn.config`, read off the graph the rows PRODUCED: retiring the shift into the DMA is
    what deletes it, so a source graph that still shows the shift is no proof the rows retired it."""
    mean = ir.load(prepared).metadata_props.get(_DMA_MEAN)
    if mean is None:
        return {}
    values = json.loads(mean)
    return {"mean_values": [values], "std_values": [[1.0] * len(values)]}


def _dma_shift(graph: ir.Graph) -> tuple[ir.Node, list[float]] | None:
    """The `Sub` closing a uint8-NHWC `Cast -> Transpose -> Sub` preprocess, and the mean it subtracts."""
    if not graph.inputs:
        return None
    image = graph.inputs[0]
    dims = image.shape
    if image.dtype != ir.DataType.UINT8 or dims is None or len(dims) != 4:
        return None
    channels = dims[3]
    if not isinstance(channels, int) or not 0 < channels <= 4:
        return None
    cast = sole_consumer(image, "Cast")
    if cast is None or cast.attributes.get_int("to") != ir.DataType.FLOAT:
        return None
    transpose = sole_consumer(cast.outputs[0], "Transpose")
    if transpose is None or list(transpose.attributes.get_ints("perm", [])) != [0, 3, 1, 2]:
        return None
    sub = sole_consumer(transpose.outputs[0], "Sub")
    if sub is None:
        return None
    shift = next((i.const_value for i in sub.inputs if i is not None and i.const_value is not None), None)
    if shift is None or shift.size != 1:
        return None
    return sub, [float(shift.numpy().reshape(()))] * channels


class Uint8ImageInputPass(ir.passes.InPlacePass):
    """Retire a scalar-shift image preprocess into the NPU's native uint8 input path, which applies the
    shift in the input DMA instead of transferring floats. The ONNX is retyped float NCHW here only because
    rknn.load_onnx rejects a uint8 graph input -- the compiled binary's own input is uint8 NHWC again."""

    def requires(self, model: ir.Model) -> None:
        """A symbolic dim here is not a decline: `_input_spec` resolves it to a working-looking 1x1 binary."""
        free = [
            f"{inp.name}[{axis}]"
            for inp in model.graph.inputs
            for axis, dim in enumerate(inp.shape or [])
            if axis and not isinstance(dim, int)
        ]
        if free:
            raise ir.passes.PreconditionError(f"RKNPU is static-shape; unpinned input dim(s) {free}")

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        shift = _dma_shift(model.graph)
        if shift is None:
            return ir.passes.PassResult(model, False)
        sub, mean = shift
        image = model.graph.inputs[0]
        batch, height, width = image.shape[0], image.shape[1], image.shape[2]  # type: ignore[index]
        image.dtype = ir.DataType.FLOAT
        image.shape = ir.Shape([batch, len(mean), height, width])  # NHWC -> NCHW, matching the backbone conv
        sub.outputs[0].replace_all_uses_with(image)
        model.metadata_props[_DMA_MEAN] = json.dumps(mean)
        return ir.passes.PassResult(model, True)


class FloatImageInputPass(ir.passes.InPlacePass):
    """uint8 NHWC input -> float32, dropping the Cast: rknn rejects uint8 ("Not Support Dtype: 2"), and raw
    float pixels are identical."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        if not graph.inputs:
            return ir.passes.PassResult(model, False)
        image = graph.inputs[0]
        dims = image.shape
        channels = dims[-1] if dims is not None and len(dims) == 4 else None
        if image.dtype != ir.DataType.UINT8 or not isinstance(channels, int) or not 0 < channels <= 4:
            return ir.passes.PassResult(model, False)

        image.dtype = ir.DataType.FLOAT
        for usage in list(image.uses()):
            cast = usage.node
            if cast.op_type != "Cast" or cast.attributes.get_int("to") != ir.DataType.FLOAT:
                continue
            cast.outputs[0].replace_all_uses_with(image)
            graph.remove(cast, safe=True)
        return ir.passes.PassResult(model, True)


class SplitLargeReduction(RewriteRuleClassBase):
    """Split a MatMul whose fp16 weight column outgrows the RKNPU's high-utilization band into
    channel-parallel sub-MatMuls. ``threshold_bytes`` is the top of that band, NOT the userguide's knee."""

    def __init__(
        self, *, threshold_bytes: int = 1536, subtile_bytes: int = 1024, elem_bytes: int = 2, name: str | None = None
    ) -> None:
        super().__init__(name=name)
        assert subtile_bytes <= threshold_bytes, (
            "sub-filters must fall under the split threshold (else non-terminating)"
        )
        self._threshold = threshold_bytes
        self._subtile = subtile_bytes
        self._elem = elem_bytes

    def pattern(self, op: Any, x: Any, w: Any) -> Any:
        return op.MatMul(x, w, _outputs=["reduction"])

    def check(self, context: Any, x: Any, w: Any, reduction: Any) -> MatchResult:
        result = MatchResult()
        weight = w.const_value
        if weight is None or len(weight.shape) != 2:
            return result.fail("weight is not a 2-D constant")
        if int(weight.shape[0]) * self._elem <= self._threshold:
            return result.fail("filter is inside the high-utilization band")
        if x.shape is None:
            return result.fail("reduction input has no static rank")
        return result

    @staticmethod
    def _rows(tensor: ir.TensorProtocol, lo: int, hi: int, name: str) -> ir.TensorProtocol:
        """Rows [lo, hi) of a 2-D weight: a VIEW of the parent's own bytes where the parent provably
        stores exactly its own elements contiguously on disk, and a copy otherwise. Every condition is
        read off this tensor rather than assumed of the graph -- a sub-byte dtype has no whole-byte row,
        and a parent whose extent is not its element count is storing something this cannot address."""
        columns = int(tensor.shape[1])
        itemsize = tensor.dtype.itemsize
        stride = columns * int(itemsize)
        if (
            isinstance(tensor, ir.ExternalTensor)
            and tensor.offset is not None
            and float(itemsize).is_integer()
            and tensor.length == int(tensor.shape[0]) * stride
        ):
            return ir.ExternalTensor(
                location=tensor.location,
                offset=tensor.offset + lo * stride,
                length=(hi - lo) * stride,
                dtype=tensor.dtype,
                shape=ir.Shape((hi - lo, columns)),
                name=name,
                base_dir=tensor.base_dir,
            )
        return ir.tensor(np.ascontiguousarray(tensor.numpy()[lo:hi]), name=name)

    def rewrite(self, op: Any, x: Any, w: Any, reduction: Any) -> Any:
        weight = w.const_value
        c_in = int(weight.shape[0])
        splits = -(-c_in * self._elem // self._subtile)
        axis = len(x.shape) - 1
        bounds = [round(i * c_in / splits) for i in range(splits + 1)]
        base = w.name or reduction.name
        total: Any = None
        for i in range(splits):
            lo, hi = bounds[i], bounds[i + 1]
            piece = op.MatMul(
                op.Slice(
                    x,
                    op.Constant(value=ir.tensor(np.array([lo], np.int64))),
                    op.Constant(value=ir.tensor(np.array([hi], np.int64))),
                    op.Constant(value=ir.tensor(np.array([axis], np.int64))),
                ),
                op.initializer(self._rows(weight, lo, hi, f"{base}_split{i}"), name=f"{base}_split{i}"),
            )
            total = piece if total is None else op.Add(total, piece)
        return total


class RawCtcLogitsPass(ir.passes.InPlacePass):
    """Retire the greedy-CTC head and emit raw logits in the NPU's own ``[batch, classes, seq]`` layout,
    leaving the argmax to the host: `Exp` has no NPU kernel, and `convert_exmatmul_to_conv` emits that
    layout, so ``[batch, seq, classes]`` would add a de-tiling transpose the runtime runs on the CPU too.
    No `Softmax` goes back: `exSoftmax13` is exact to 8192 classes on C and wrong beyond, and the toolkit
    puts it on the NPU at any size."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        if not HostCtcDecodePass()(model).modified:
            return ir.passes.PassResult(model, False)
        softmax = graph.outputs[0].producer()
        logits = softmax.inputs[0] if softmax is not None else None
        if logits is None:
            return ir.passes.PassResult(model, False)
        transpose = make_node("Transpose", [logits], out="logits", perm=[0, 2, 1])
        graph.extend([transpose])
        graph.outputs.clear()
        graph.outputs.append(transpose.outputs[0])
        return ir.passes.PassResult(model, True)


class FloatifyNotEqual(RewriteRuleClassBase):
    """Cast(Not(Equal(int, scalar))) -> float(x != c), exact for integer ids: int32 Equal has no librknnrt
    kernel, and the XLM towers reuse the pad comparison as the pooling weight, so it outlives the mask."""

    def pattern(self, op: Any, x: Any, pad: Any) -> Any:
        return op.Cast(op.Not(op.Equal(x, pad)), _outputs=["indicator"])

    def check(self, context: Any, x: Any, pad: Any, indicator: Any) -> MatchResult:
        result = MatchResult()
        if x.producer() is not None or x.dtype not in (ir.DataType.INT32, ir.DataType.INT64):
            return result.fail("Equal operand is not an integer graph input")
        if pad.const_value is None or pad.const_value.size != 1:
            return result.fail("pad is not a scalar constant")
        to = indicator.producer().attributes.get_int("to")
        if to not in (int(ir.DataType.INT64), int(ir.DataType.INT32), int(ir.DataType.FLOAT)):
            return result.fail("indicator cast target is not a numeric count/weight type")
        return result

    def rewrite(self, op: Any, x: Any, pad: Any, indicator: Any) -> Any:
        def const(value: float) -> Any:
            return op.Constant(value=ir.tensor(np.array(value, np.float32)))

        delta = op.Sub(op.Cast(x, to=int(ir.DataType.FLOAT)), const(float(pad.const_value.numpy())))
        return op.Clip(op.Abs(delta), const(0.0), const(1.0))


class FloatifyPadKeep(RewriteRuleClassBase):
    """Compute the pad indicator instead of reading it out of the table `FoldPadMaskPass` builds:
    rknn-toolkit2's `fold_constant` merges Gathers sharing an index without comparing their tables, so the
    keep lookup aliases the token-embedding lookup and the build dies inside the toolkit."""

    def pattern(self, op: Any, table: Any, ids: Any) -> Any:
        return op.Gather(table, ids, _outputs=["keep"])

    def check(self, context: Any, table: Any, ids: Any, keep: Any) -> MatchResult:
        result = MatchResult()
        if keep.producer().attributes.get_int("axis", 0) != 0:
            return result.fail("Gather axis is not 0")
        if ids.producer() is not None or ids.dtype not in (ir.DataType.INT32, ir.DataType.INT64):
            return result.fail("index is not an integer graph input")
        const = table.const_value  # metadata first: the token-embedding table must not materialize here
        if const is None or len(const.shape) != 1 or const.dtype != ir.DataType.FLOAT:
            return result.fail("table is not a 1-D float32 constant")
        array = const.numpy()
        if np.count_nonzero(array == 0.0) != 1 or not (array[array != 0.0] == 1.0).all():
            return result.fail("table is not a keep indicator with a single zero row")
        return result

    def rewrite(self, op: Any, table: Any, ids: Any, keep: Any) -> Any:
        def const(value: float) -> Any:
            return op.Constant(value=ir.tensor(np.array(value, np.float32)))

        pad = float(np.flatnonzero(table.const_value.numpy() == 0.0)[0])
        delta = op.Sub(op.Cast(ids, to=int(ir.DataType.FLOAT)), const(pad))
        return op.Clip(op.Abs(delta), const(0.0), const(1.0))


class OpaqueZeroMul(RewriteRuleClassBase):
    """Mul(x, 0.0) -> Sub(x, x): identical zeros, but opaque to rknn-toolkit2's `fold_constant`, which
    otherwise folds the batch-zeros helper into a constant Q and crashes the toolkit's SDPA matcher."""

    def pattern(self, op: Any, x: Any, zero: Any) -> Any:
        return op.Mul(x, zero, _outputs=["zeroed"])

    def check(self, context: Any, x: Any, zero: Any, zeroed: Any) -> MatchResult:
        result = MatchResult()
        const = zero.const_value
        if const is None or const.size != 1 or float(const.numpy().reshape(())) != 0.0:
            return result.fail("multiplier is not the scalar 0.0")
        return result

    def rewrite(self, op: Any, x: Any, zero: Any, zeroed: Any) -> Any:
        return op.Sub(x, x)


_MASK_ISLAND_OPS = {
    "Equal", "Not", "Cast", "And", "GatherND", "Concat", "Range", "Unsqueeze", "Squeeze",
    "Reshape", "Shape", "Slice", "Add", "Mul", "Expand", "Gather", "Where",
}  # fmt: skip


class FloatifyPadMaskPass(ir.passes.InPlacePass):
    """Replace an in-graph bool pad mask feeding Attention with a float additive bias, killing the integer
    mask island: int32 Equal has no librknnrt kernel. The bias is bitwise Where(mask, 0, -1e4) either way."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        masks = {
            node.inputs[3]
            for node in graph
            if node.op_type == "Attention" and len(node.inputs) > 3 and node.inputs[3] is not None
            if node.inputs[3].dtype == ir.DataType.BOOL and node.inputs[3].producer() is not None
        }
        converted = False
        for mask in masks:
            equal = not_node = None
            roots: list[ir.Value] = []
            ok = True
            seen: set[int] = set()
            stack: list[ir.Value] = [mask]
            while stack and ok:
                value = stack.pop()
                node = value.producer()
                if node is None:
                    const = value.const_value
                    if const is None:
                        if value not in roots:  # the input is reached via several paths (Equal, ez helper)
                            roots.append(value)
                    elif const.dtype == ir.DataType.BOOL and not const.numpy().all():
                        ok = False  # a non-all-True bool const would add its own masking
                    continue
                if id(node) in seen:
                    continue
                seen.add(id(node))
                if node.op_type == "Equal":
                    equal = None if equal is not None else node
                    ok = equal is not None
                elif node.op_type == "Not":
                    not_node = None if not_node is not None else node
                    ok = not_node is not None
                elif node.op_type not in _MASK_ISLAND_OPS:
                    ok = False
                stack.extend(i for i in node.inputs if i is not None)
            if not ok or equal is None or not_node is None or len(roots) != 1:
                continue
            tokens = roots[0]
            pad = next((i.const_value for i in equal.inputs if i.const_value is not None), None)
            if pad is None or pad.size != 1 or tokens not in equal.inputs:
                continue
            if tokens.dtype not in (ir.DataType.INT32, ir.DataType.INT64) or tokens.shape is None:
                continue

            def const(value: float, name: str) -> ir.Value:
                return make_init(graph, name, np.array(value, np.float32))

            base = f"{mask.name}_padbias"
            axes = make_init(graph, f"{base}_axes", np.array([1, 2], np.int64))
            cast = make_node("Cast", [tokens], to=int(ir.DataType.FLOAT))
            sub = make_node("Sub", [cast.outputs[0], const(float(pad.numpy()), f"{base}_pad")])
            abs_ = make_node("Abs", [sub.outputs[0]])
            clip = make_node("Clip", [abs_.outputs[0], const(0.0, f"{base}_lo"), const(1.0, f"{base}_hi")])
            keep = make_node("Sub", [clip.outputs[0], const(1.0, f"{base}_one")])
            bias = make_node("Mul", [keep.outputs[0], const(1.0e4, f"{base}_scale")])
            unsq = make_node("Unsqueeze", [bias.outputs[0], axes], out=base)
            graph.extend([cast, sub, abs_, clip, keep, bias, unsq])
            mask.replace_all_uses_with(unsq.outputs[0])
            converted = True
        return ir.passes.PassResult(model, converted)


class PinOpsetPass(ir.passes.InPlacePass):
    def __init__(self, version: int) -> None:
        self.version = version

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        model.opset_imports.pop("ai.onnx", None)
        model.opset_imports[""] = self.version
        return ir.passes.PassResult(model, True)
