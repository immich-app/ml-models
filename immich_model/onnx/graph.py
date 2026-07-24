"""Shared ONNX graph-surgery helpers for the fused-model exporters (insightface, ocr)."""

import logging
from collections.abc import Sequence
from fractions import Fraction
from typing import Any

import numpy as np
import onnx_ir as ir
import onnx_ir.passes.common as common_passes
import onnxscript.optimizer
from onnxscript.rewriter import RewritePass, RewriteRule, RewriteRuleSet
from onnxscript.rewriter.rules.common import (
    fuse_batchnorm_into_conv_rule,
    fuse_batchnorm_into_conv_transpose_rule,
    fuse_batchnorm_into_gemm_rule,
)
from onnxscript.values import OnnxFunction
from onnxscript.version_converter import ConvertVersionPass

from ._ir import ReinferShapesPass, const_array, make_init, pointwise, single_use

log = logging.getLogger(__name__)

BATCHNORM_FOLD_RULES = (
    fuse_batchnorm_into_conv_rule,
    fuse_batchnorm_into_conv_transpose_rule,
    fuse_batchnorm_into_gemm_rule,
)


class OptimizePass(ir.passes.InPlacePass):
    """onnxscript's optimizer, which is a pass sequence exposed only as a function."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        onnxscript.optimizer.optimize(model)
        return ir.passes.PassResult(model, True)


class PinnedRewritePass(RewritePass):
    """A rule set whose match count the source catalog pins."""

    def __init__(self, rules: Sequence[RewriteRule] | RewriteRuleSet, expected: int, subject: str) -> None:
        super().__init__(rules)
        self.expected, self.subject, self.applied = expected, subject, 0

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        self.applied = self.rules.apply_to_model(model)
        log.info("Rewrote %d %s", self.applied, self.subject)
        return ir.passes.PassResult(model, bool(self.applied))

    def ensures(self, model: ir.Model) -> None:
        if self.applied != self.expected:
            raise ir.passes.PostconditionError(f"Rewrote {self.applied} {self.subject}, expected {self.expected}")


class ConvertOpsetPass(ConvertVersionPass):
    """Migrate the stock graph to the exporter's opset; the onnx C-API fallback swallows a failure into a no-op."""

    def __init__(self, target: int) -> None:
        super().__init__(target_version=target, fallback=True)

    def ensures(self, model: ir.Model) -> None:
        opset = model.graph.opset_imports.get("")
        if opset != self.target_version:
            raise ir.passes.PostconditionError(f"Graph is at opset {opset}, expected {self.target_version}")


class DeclareInputDimsPass(ir.passes.InPlacePass):
    """Assert the exporter's contract on the first input's dims: a symbol names an extent, an int pins one."""

    def __init__(self, dims: dict[int, str | int]) -> None:
        self.dims = dims

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        value = model.graph.inputs[0]
        shape = list(value.shape)
        for axis, dim in self.dims.items():
            if isinstance(dim, int) and isinstance(shape[axis], int) and shape[axis] != dim:
                raise ValueError(f"Input dim {axis} is statically {shape[axis]}, cannot pin to {dim}")
            shape[axis] = dim
        value.shape = ir.Shape(shape)
        return ir.passes.PassResult(model, True)


class FoldInputScalePass(ir.passes.InPlacePass):
    """Fold the `* scale` of the backbone's normalization into the input Convs; the additive shift stays
    explicit because it does not commute with a zero pad."""

    def __init__(self, scale: float, flip_channels: bool = False) -> None:
        self.scale, self.flip_channels = scale, flip_channels

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        consumers = dict.fromkeys(use.node for use in model.graph.inputs[0].uses())
        not_conv = [node.op_type for node in consumers if node.op_type != "Conv"]
        if not consumers or not_conv:
            raise ValueError(f"Cannot fold input scale/perm: input consumed by {not_conv or 'nothing'}")

        for conv in consumers:
            value = conv.inputs[1]
            weight = const_array(value)  # [O, C, kH, kW]
            folded = (weight * self.scale).astype(weight.dtype)
            if self.flip_channels:
                folded = folded[:, ::-1]
            value.const_value = ir.tensor(np.ascontiguousarray(folded), name=value.name)
        return ir.passes.PassResult(model, True)


class ReinferPass(ir.passes.Sequential):
    """Re-infer shapes, then unify: inference mints a fresh `unk__N` per extent it cannot copy verbatim, so
    unpaired it grows the declared shape domain a symbol per pass."""

    def __init__(self) -> None:
        super().__init__(ReinferShapesPass(), UnifyDimSymbolsPass())


# ops whose output 0 aligns axis-for-axis with every equal-rank input
_ALIGNED = frozenset(
    {
        "Add",
        "BatchNormalization",
        "Cast",
        "Clip",
        "Div",
        "Erf",
        "Exp",
        "Gelu",
        "HardSigmoid",
        "HardSwish",
        "Identity",
        "InstanceNormalization",
        "LayerNormalization",
        "LogSoftmax",
        "Max",
        "Min",
        "Mul",
        "PRelu",
        "Pow",
        "QuickGelu",
        "Reciprocal",
        "Relu",
        "Sigmoid",
        "SimplifiedLayerNormalization",
        "SkipLayerNormalization",
        "Softmax",
        "Sqrt",
        "Sub",
        "Tanh",
        "Where",
    }
)

# ops that slide a window over the trailing axes
_WINDOWED = frozenset(
    {
        "AveragePool",
        "Conv",
        "ConvTranspose",
        "GlobalAveragePool",
        "GlobalMaxPool",
        "LpPool",
        "MaxPool",
    }
)

_ONE = Fraction(1)

_Shape = list[str | int]


def _symbol(shape: _Shape | None, axis: int) -> str:
    """The symbol at `axis`, or "" for a constant extent, a missing shape or an out-of-range axis."""
    if shape is None or not 0 <= axis < len(shape):
        return ""
    dim = shape[axis]
    return dim if isinstance(dim, str) else ""


def _keeps_window(node: ir.Node) -> bool:
    attributes = node.attributes
    kernel = attributes.get_ints("kernel_shape", [])  # the Global* pools carry none
    if not kernel or attributes.get_ints("output_shape", []) or any(attributes.get_ints("output_padding", [0])):
        return False
    spatial = [1] * len(kernel)
    # list() is load-bearing, for the reason `_ir.pointwise` gives: without it this fails open, silently
    if list(attributes.get_ints("strides", spatial)) != spatial:
        return False
    if list(attributes.get_ints("dilations", spatial)) != spatial:
        return False
    if attributes.get_string("auto_pad", "NOTSET").startswith("SAME"):
        return True
    pads = attributes.get_ints("pads", [0] * 2 * len(kernel))
    return all(pads[i] + pads[i + len(kernel)] == kernel[i] - 1 for i in range(len(kernel)))


def _dim_symbol_renames(graph: ir.Graph) -> dict[str, str]:
    """Rename each set of provably-equal symbolic dims to one symbol, preferring the exporter's own name.
    A forward walk cannot: the equalities are neither directional nor all rooted at an input."""
    values = [*graph.inputs, *graph.initializers.values(), *(out for node in graph for out in node.outputs)]
    shapes: dict[ir.Value, _Shape] = {
        value: [dim if isinstance(dim, int) else (dim.value or 0) for dim in value.shape]
        for value in values
        if value.shape is not None
    }
    constants = {
        value: value.const_value.numpy()
        for value in values
        if value.const_value is not None and value.const_value.shape.rank() == 1 and value.const_value.size <= 8
    }
    ranks = {key: len(shape) for key, shape in shapes.items()}
    parent: dict[str, str] = {}
    scale: dict[str, Fraction] = {}

    def find(symbol: str) -> tuple[str, Fraction]:
        """The class representative, and what `symbol` is as a multiple of it."""
        factor = _ONE
        while parent.setdefault(symbol, symbol) != symbol:
            factor *= scale[symbol]
            symbol = parent[symbol]
        return symbol, factor

    def link(
        source: ir.Value | None, source_axis: int, target: ir.Value | None, target_axis: int, ratio: Fraction = _ONE
    ) -> None:
        a, b = _symbol(shapes.get(source), source_axis), _symbol(shapes.get(target), target_axis)
        if not a or not b:
            return
        (root_a, factor_a), (root_b, factor_b) = find(a), find(b)
        if root_a == root_b:
            assert factor_b == ratio * factor_a, f"{target}[{target_axis}] contradicts {source}[{source_axis}]"
            return
        parent[root_b], scale[root_b] = root_a, ratio * factor_a / factor_b

    for node in graph:
        if not node.inputs or not node.outputs:
            continue
        first, out = node.inputs[0], node.outputs[0]
        rank = ranks.get(out, 0)

        if node.op_type in _ALIGNED:
            for name in node.inputs:
                if ranks.get(name) == rank:
                    for axis in range(rank):
                        link(name, axis, out, axis)
        elif node.op_type in _WINDOWED:
            link(first, 0, out, 0)  # batch never enters the window
            if _keeps_window(node):
                for axis in range(2, rank):
                    link(first, axis, out, axis)
        elif node.op_type == "Transpose":
            for axis, source_axis in enumerate(node.attributes.get_ints("perm", list(reversed(range(rank))))):
                link(first, source_axis, out, axis)
        elif node.op_type == "Resize":
            scales = constants.get(node.inputs[2]) if len(node.inputs) > 2 else None
            if scales is not None and len(scales) == rank:
                for axis in range(rank):
                    if float(scales[axis]) == 1.0:
                        link(first, axis, out, axis)
        elif node.op_type == "Concat":
            axis = _norm(node.attributes.get_int("axis", 0), rank)
            for name in node.inputs:
                for other in range(rank):
                    if other != axis:
                        link(name, other, out, other)
        elif node.op_type == "Split":
            axis = _norm(node.attributes.get_int("axis", 0), rank)
            for name in node.outputs:
                for other in range(rank):
                    if other != axis:
                        link(first, other, name, other)
        elif node.op_type == "Slice":
            # an omitted `axes` slices the leading len(starts)
            named = node.inputs[3] if len(node.inputs) > 3 else None
            sliced = constants.get(named) if named else constants.get(node.inputs[1])
            if sliced is not None:
                touched = {_norm(int(a), rank) for a in sliced} if named else set(range(len(sliced)))
                for axis in set(range(rank)) - touched:
                    link(first, axis, out, axis)
        elif node.op_type == "Pad":
            # pads is begin-then-end over the padded axes
            pads = constants.get(node.inputs[1]) if len(node.inputs) > 1 else None
            named = node.inputs[3] if len(node.inputs) > 3 else None
            axes = constants.get(named) if named else range(rank)
            if pads is not None and axes is not None and len(pads) == 2 * len(axes):
                padded = {_norm(int(a), rank) for a in axes}
                unpadded = {_norm(int(a), rank) for i, a in enumerate(axes) if pads[i] == 0 == pads[i + len(axes)]}
                for axis in (set(range(rank)) - padded) | unpadded:
                    link(first, axis, out, axis)
        elif node.op_type == "Squeeze":
            axes = constants.get(node.inputs[1]) if len(node.inputs) > 1 else None
            if axes is not None:
                dropped = {_norm(int(a), ranks.get(first, 0)) for a in axes}
                kept = [axis for axis in range(ranks.get(first, 0)) if axis not in dropped]
                for target_axis, source_axis in enumerate(kept):
                    link(first, source_axis, out, target_axis)
        elif node.op_type == "Unsqueeze":
            axes = constants.get(node.inputs[1]) if len(node.inputs) > 1 else None
            if axes is not None:
                inserted = {_norm(int(a), rank) for a in axes}
                for source_axis, target_axis in enumerate(a for a in range(rank) if a not in inserted):
                    link(first, source_axis, out, target_axis)
        elif node.op_type == "Reshape":
            for source_axis, target_axis, ratio in _reshape_pairs(node, constants, shapes.get(first), rank):
                link(first, source_axis, out, target_axis, ratio)
        elif node.op_type == "MatMul" and ranks.get(node.inputs[1]) == 2 and ranks.get(first) == rank:
            # equal-rank operands would be a batched matmul whose leading axes broadcast, unproven here
            for axis in range(rank - 1):
                link(first, axis, out, axis)
        elif node.op_type == "Attention" and ranks.get(first) == rank:
            # holds for the 3-D and 4-D spellings alike
            for name in node.inputs[1:3]:
                link(first, 0, name, 0)
            for axis in range(rank - 1):
                link(first, axis, out, axis)

    symbols = dict.fromkeys(dim for shape in shapes.values() for dim in shape if isinstance(dim, str))
    classes: dict[tuple[str, Fraction], list[str]] = {}
    for symbol in symbols:  # graph inputs first: the exporter's declared names lead their class
        classes.setdefault(find(symbol), []).append(symbol)

    names = {}
    for key, group in classes.items():
        declared = [symbol for symbol in group if not symbol.startswith("unk__")]
        assert len(declared) < 2, f"unified distinct declared dims {declared}"  # a rule proved a falsehood
        names[key] = declared[0] if declared else group[0]
    return {symbol: names[find(symbol)] for symbol in symbols}


def _norm(axis: int, rank: int) -> int:
    return axis + rank if axis < 0 else axis


def _reshape_pairs(
    node: ir.Node, constants: dict[ir.Value, np.ndarray], source: _Shape | None, rank: int
) -> list[tuple[int, int, Fraction]]:
    """Axis pairs `(source_axis, target_axis, ratio)` a Reshape provably carries, `out == ratio * in`."""
    target = constants.get(node.inputs[1]) if len(node.inputs) > 1 else None
    if source is None or target is None or len(target) != rank:
        return []

    copied = [] if node.attributes.get_int("allowzero", 0) == 1 else [a for a, e in enumerate(target) if e == 0]
    pairs = [(axis, axis, _ONE) for axis in copied if axis < len(source)]

    free_source = [axis for axis in range(len(source)) if axis not in copied]
    free_target = [axis for axis in range(rank) if axis not in copied]
    symbolic = [axis for axis in free_source if _symbol(source, axis)]
    unknown = [axis for axis in free_target if target[axis] < 0]
    known_source = [dim for dim in (source[axis] for axis in free_source) if isinstance(dim, int)]
    known_target = [target[axis] for axis in free_target if axis not in unknown]
    if len(symbolic) == 1 and len(unknown) == 1 and all(known_source) and all(e > 0 for e in known_target):
        pairs.append((symbolic[0], unknown[0], Fraction(int(np.prod(known_source)), int(np.prod(known_target)))))
    return pairs


class UnifyDimSymbolsPass(ir.passes.InPlacePass):
    """`_dim_symbol_renames` over an `ir.Model`."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        renames = _dim_symbol_renames(graph)

        def renamed(dim: int | ir.SymbolicDim) -> int | str | None:
            return dim if isinstance(dim, int) else renames.get(dim.value or "", dim.value)

        for value in (*graph.inputs, *graph.initializers.values(), *(out for node in graph for out in node.outputs)):
            shape = value.shape
            if shape is not None and any(not isinstance(dim, int) for dim in shape):
                value.shape = ir.Shape([renamed(dim) for dim in shape])
        retired = len(renames) - len(set(renames.values()))
        log.info("Retired %d dim symbols the graph proved equal", retired)
        return ir.passes.PassResult(model, bool(retired))


class FoldPointwiseConvsPass(ir.passes.InPlacePass):
    """Compose back-to-back 1x1 Convs into one, a 1x1 stride-1 unpadded conv being a pure per-pixel matmul.
    Composed in float64 and cast back for one rounding, which is why it runs at export on the fp32 graph."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        folded, composed = 0, set()
        for second in tuple(graph):
            source = second.inputs[0]
            first = source.producer()
            if first is None or first in composed or not (pointwise(first) and pointwise(second)):
                continue
            if not single_use(source) or source.is_graph_output():
                continue
            weights = [const_array(node.inputs[1]) for node in (first, second)]
            if any(weight is None for weight in weights):
                continue
            biases = [
                const_array(node.inputs[2]) if len(node.inputs) > 2 else np.zeros(weight.shape[0], weight.dtype)
                for node, weight in zip((first, second), weights)
            ]
            m1, m2 = (weight[:, :, 0, 0].astype(np.float64) for weight in weights)
            name = f"{second.name}_folded"
            weight = make_init(graph, f"{name}_w", (m2 @ m1)[:, :, None, None].astype(weights[0].dtype))
            bias = make_init(
                graph, f"{name}_b", (m2 @ biases[0].astype(np.float64) + biases[1]).astype(biases[1].dtype)
            )
            second.resize_inputs(3)  # the second conv may have carried no bias; the composition always has one
            for index, value in enumerate((first.inputs[0], weight, bias)):
                second.replace_input_with(index, value)
            graph.remove(first, safe=True)
            composed.add(second)
            folded += 1
        log.info("Composed %d back-to-back pointwise convs", folded)
        return ir.passes.PassResult(model, bool(folded))


_FUSED = "immich.fused"


def _attach(model: ir.Model, fn: OnnxFunction[Any, Any]) -> ir.Function:
    """Register `fn` as a model-local function; the eDSL compiles to a proto, so this is the one crossing left."""
    graph = ir.from_proto(fn.to_model_proto()).graph
    function = ir.Function(domain=_FUSED, name=fn.name, graph=graph, attributes=())
    model.functions[function.identifier()] = function
    model.opset_imports.setdefault(_FUSED, 1)
    return function


def _call(function: ir.Function, inputs: list[ir.Value]) -> ir.Node:
    call = ir.node(function.name, domain=_FUSED, inputs=inputs, num_outputs=len(function.outputs))
    for produced, declared in zip(call.outputs, function.outputs):
        produced.name = declared.name
    return call


class WrapPass(ir.passes.Sequential):
    """Compose pre -> backbone -> post. The sort and optimizer are the composition, not a tail: the calls are
    appended after the nodes reading them, and the optimizer's opening InlinePass substitutes the bodies."""

    def __init__(self, pre: OnnxFunction[Any, Any], post: OnnxFunction[Any, Any]) -> None:
        super().__init__(_PrependPass(pre), _AppendPass(post), common_passes.TopologicalSortPass(), OptimizePass())


class WrapPrePass(ir.passes.Sequential):
    """Compose pre -> backbone with no post. For heads whose decode runs host-side."""

    def __init__(self, pre: OnnxFunction[Any, Any]) -> None:
        super().__init__(_PrependPass(pre), common_passes.TopologicalSortPass(), OptimizePass())


class _PrependPass(ir.passes.InPlacePass):
    def __init__(self, pre: OnnxFunction[Any, Any]) -> None:
        self.pre = pre

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        function = _attach(model, self.pre)
        inputs = [ir.Value(name=value.name, shape=value.shape, type=value.type) for value in function.inputs]
        call = _call(function, inputs)
        graph = model.graph
        for backbone_input, produced in zip(list(graph.inputs), call.outputs):
            produced.name = backbone_input.name
            ir.convenience.replace_all_uses_with(backbone_input, produced)
        graph.inputs.clear()
        graph.inputs.extend(inputs)
        graph.append(call)
        return ir.passes.PassResult(model, True)


class _AppendPass(ir.passes.InPlacePass):
    def __init__(self, post: OnnxFunction[Any, Any]) -> None:
        self.post = post

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        call = _call(_attach(model, self.post), list(graph.outputs))
        graph.outputs.clear()
        graph.outputs.extend(call.outputs)
        graph.append(call)
        return ir.passes.PassResult(model, True)


class NameOutputDimsPass(ir.passes.Sequential):
    """Assert the contract shape on each named graph output, then unify again. Must run after the post-wrap
    re-inference, which clears every output shape."""

    def __init__(self, shapes: dict[str, list[str | int]]) -> None:
        super().__init__(_DeclareOutputDimsPass(shapes), UnifyDimSymbolsPass())


class _DeclareOutputDimsPass(ir.passes.InPlacePass):
    def __init__(self, shapes: dict[str, list[str | int]]) -> None:
        self.shapes = shapes

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        outputs = {output.name: output for output in model.graph.outputs}
        if outputs.keys() != self.shapes.keys():
            raise ValueError(f"Graph outputs are {sorted(outputs)}, expected {sorted(self.shapes)}")
        for name, declared in self.shapes.items():
            shape = outputs[name].shape
            assert shape is not None and len(shape) == len(declared)
            outputs[name].shape = ir.Shape([dim if isinstance(dim, int) else d for dim, d in zip(shape, declared)])
        return ir.passes.PassResult(model, True)
