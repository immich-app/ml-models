"""The RKNPU-only rewrite rows and the `rknn.config` extras derived beside them; no toolkit import needed."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx_ir as ir
from onnxscript.rewriter.pattern import MatchResult, RewriteRuleClassBase

from ..onnx._ir import make_init, make_node, sole_consumer

# the mean the DMA now owes the graph, carried on the graph so the compiler reads both off one file
_DMA_MEAN = "rknn_input_mean"


# what the compiler is told for every model, rendered into the plans so a flag cannot change a binary
# without moving a fixture. model_pruning increases model size here and has no measurable benefit.
RKNN_CONFIG: dict[str, Any] = {"disable_rules": [], "enable_flash_attention": False, "model_pruning": False}
DO_QUANTIZATION = False


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
    """Retire a scalar-shift image preprocess into the NPU's uint8 input DMA. The ONNX is retyped float NCHW
    only because rknn.load_onnx rejects a uint8 input; the compiled binary's own input is uint8 NHWC again."""

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
        """Rows [lo, hi) of a 2-D weight: a view of the parent's bytes where it provably stores exactly its own
        elements contiguously on disk, and a copy otherwise."""
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


# the band the RKNPU keeps its MAC array busy in; either floor alone lets a partial conv fall out the bottom
_HIGH_UTILIZATION_BYTES = 6144
_SUBTILE_BYTES = 4096
_SPLIT_CHANNEL_FLOOR = 32
_BRANCH_CHANNEL_FLOOR = 16
# a conv with nothing to reuse its weight over is weight-streaming bound, where summing partials only adds nodes
_REUSE_POSITIONS = 64


def _tile_bytes(shape: Any) -> int:
    return int(shape[1]) * int(shape[2]) * int(shape[3]) * 2  # fp16 on the NPU whatever the graph declares


def _i64(op: Any, value: int) -> Any:
    return op.Constant(value=ir.tensor(np.array([value], np.int64)))


def _cannot_sum(w: Any, x: Any, conv: Any) -> str | None:
    """Why this Conv cannot be re-expressed as partial convs summed, or None if it can."""
    weight = w.const_value
    if weight is None or len(weight.shape) != 4:
        return "weight is not a 4-D constant"
    node = conv.producer()
    if node.attributes.get_int("group", 1) != 1:
        return "a grouped conv does not reduce over the whole channel axis"
    dims = x.shape
    if dims is None or len(dims) != 4 or any(not isinstance(d, int) for d in dims[2:]):
        return "conv input has no static spatial extent"
    strides = node.attributes.get_ints("strides", [1, 1])
    if math.prod(int(d) // s for d, s in zip(dims[2:], strides)) < _REUSE_POSITIONS:
        return "too few output positions to reuse the weight over"
    return None


def _sum_partials(op: Any, conv: Any, weight: np.ndarray, parts: Any, base: str) -> Any:
    """One conv per (source, channel range) of the weight, summed; the bias is per-output so it rides one."""
    node = conv.producer()
    bias = node.inputs[2] if len(node.inputs) > 2 else None
    total: Any = None
    for index, (source, lo, hi) in enumerate(parts):
        piece = op.Conv(
            source,
            op.initializer(
                ir.tensor(np.ascontiguousarray(weight[:, lo:hi]), name=f"{base}_part{index}"),
                name=f"{base}_part{index}",
            ),
            *([bias] if total is None and bias is not None else []),
            **node.attributes,
        )
        total = piece if total is None else op.Add(total, piece)
    return total


class SplitLargeConvReduction(RewriteRuleClassBase):
    """Split a dense Conv whose weight tile outgrows the high-utilization band into partial convs summed,
    the RKNPU stalling its MAC array on the reduction otherwise."""

    def pattern(self, op: Any, x: Any, w: Any) -> Any:
        return op.Conv(x, w, _allow_other_inputs=True, _allow_other_attributes=True, _outputs=["conv"])

    def check(self, context: Any, x: Any, w: Any, conv: Any) -> MatchResult:
        result = MatchResult()
        if reason := _cannot_sum(w, x, conv):
            return result.fail(reason)
        if _tile_bytes(w.const_value.shape) <= _HIGH_UTILIZATION_BYTES:
            return result.fail("filter is inside the high-utilization band")
        return result

    def rewrite(self, op: Any, x: Any, w: Any, conv: Any) -> Any:
        weight = w.const_value.numpy()
        c_in = int(weight.shape[1])
        # a wide kernel reaches the subtile at too few channels to fill the array, so the two bounds compete
        splits = min(-(-_tile_bytes(weight.shape) // _SUBTILE_BYTES), max(c_in // _SPLIT_CHANNEL_FLOOR, 1))
        edges = [round(i * c_in / splits) for i in range(splits + 1)]
        parts = [(op.Slice(x, _i64(op, lo), _i64(op, hi), _i64(op, 1)), lo, hi) for lo, hi in zip(edges, edges[1:])]
        return _sum_partials(op, conv, weight, parts, w.name or conv.name)


class FoldConcatIntoConv(RewriteRuleClassBase):
    """Fold a channel-axis Concat into the Conv consuming it, as partial convs summed: the copy is not free."""

    def pattern(self, op: Any, x: Any, w: Any) -> Any:
        cat = op.Concat(x, _allow_other_inputs=True, _outputs=["cat"])
        return op.Conv(cat, w, _allow_other_inputs=True, _allow_other_attributes=True, _outputs=["conv"])

    def check(self, context: Any, x: Any, w: Any, cat: Any, conv: Any) -> MatchResult:
        result = MatchResult()
        if reason := _cannot_sum(w, cat, conv):
            return result.fail(reason)
        node = cat.producer()
        if node.attributes.get_int("axis", 0) != 1 or len(node.inputs) < 2:
            return result.fail("not a multi-branch concatenation on the channel axis")
        widths = []
        for branch in node.inputs:
            width = branch.shape[1] if branch.shape is not None and len(branch.shape) == 4 else None
            if not isinstance(width, int) or width < _BRANCH_CHANNEL_FLOOR:
                return result.fail("a branch is not statically wide enough to be its own conv")
            widths.append(width)
        if sum(widths) != int(w.const_value.shape[1]):
            return result.fail("the branches do not account for the conv's input channels")
        return result

    def rewrite(self, op: Any, x: Any, w: Any, cat: Any, conv: Any) -> Any:
        parts, offset = [], 0
        for branch in cat.producer().inputs:
            parts.append((branch, offset, offset + int(branch.shape[1])))
            offset = parts[-1][2]
        return _sum_partials(op, conv, w.const_value.numpy(), parts, w.name or conv.name)


CLASS_TILE = 2048


class SeqMajorLogitsPass(ir.passes.InPlacePass):
    """A lone `Softmax` output -> its raw logits, seq-major, a class tile at a time. No Softmax goes back, being
    wrong past 8192 classes; the tiles the toolkit merges back are what keep the transpose off the CPU."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        if len(graph.outputs) != 1:
            return ir.passes.PassResult(model, False)
        softmax = graph.outputs[0].producer()
        if softmax is None or softmax.op_type != "Softmax":
            return ir.passes.PassResult(model, False)
        logits = softmax.inputs[0]
        classes = logits.shape[2] if logits is not None and logits.shape is not None else None
        if not isinstance(classes, int):
            return ir.passes.PassResult(model, False)

        native = make_node("Transpose", [logits], out="logits_class_major", perm=[0, 2, 1])
        axes = make_init(graph, "ctc_seq_axes", np.array([2], np.int64))
        widened = make_node("Unsqueeze", [native.outputs[0], axes], out="logits_4d")
        sizes = [CLASS_TILE] * (classes // CLASS_TILE) + ([classes % CLASS_TILE] if classes % CLASS_TILE else [])
        split = ir.node(
            "Split",
            inputs=[widened.outputs[0], make_init(graph, "ctc_tile_sizes", np.array(sizes, np.int64))],
            attributes={"axis": 1},
            num_outputs=len(sizes),
        )
        moved = [
            make_node("Transpose", [tile], out=f"logits_tile{i}", perm=[0, 3, 2, 1])
            for i, tile in enumerate(split.outputs)
        ]
        joined = ir.node("Concat", inputs=[node.outputs[0] for node in moved], attributes={"axis": 3}, num_outputs=1)
        joined.outputs[0].name = "ctc_logits"

        graph.extend([native, widened, split, *moved, joined])
        graph.outputs.clear()
        graph.outputs.append(joined.outputs[0])
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
    """Compute the pad indicator rather than look it up: rknn-toolkit2 merges Gathers sharing an index without
    comparing their tables, aliasing the keep lookup onto the token embedding."""

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
