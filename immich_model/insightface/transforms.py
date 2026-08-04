"""Stock InsightFace ONNX -> Immich's fused format: uint8 RGB NHWC input, folded preprocessing and L2
normalization. Face alignment and the SCRFD anchor decode stay host-side, so detection emits nine raw heads."""

import logging
from typing import Any

import numpy as np
import onnx_ir.passes.common as common_passes
from onnxscript import ir
from onnxscript.rewriter import RewritePass
from onnxscript.rewriter.pattern import Constant, MatchResult, OrValue, RewriteRuleClassBase, RewriteRuleSet

from ..onnx._ir import FlushDenormalsPass, const_array, const_ints, single_use, sole_consumer
from ..onnx.graph import (
    BATCHNORM_FOLD_RULES,
    ConvertOpsetPass,
    DeclareInputDimsPass,
    FoldInputScalePass,
    FoldPointwiseConvsPass,
    NameOutputDimsPass,
    OptimizePass,
    PinnedRewritePass,
    ReinferPass,
    WrapPass,
    WrapPrePass,
)
from . import _dsl

log = logging.getLogger(__name__)


class _FoldBatchNormIntoGemm(RewriteRuleClassBase):
    """Fold the ArcFace tail's BatchNormalization -> Flatten -> Gemm into the Gemm's weights; the existing BN
    folds all want the BN adjacent to it. Flatten(axis=1) makes each channel a contiguous block of k."""

    def pattern(self, op: Any, x: Any, gamma: Any, beta: Any, mean: Any, var: Any, w: Any, b: Any) -> Any:
        normalized = op.BatchNormalization(
            x, gamma, beta, mean, var, _allow_other_attributes=True, _outputs=["normalized"]
        )
        flat = op.Flatten(normalized, axis=1, _outputs=["flat"])
        return op.Gemm(flat, w, b, alpha=1.0, beta=1.0, transB=1, _outputs=["gemm"])

    def check(
        self,
        context: Any,
        gamma: Any,
        beta: Any,
        mean: Any,
        var: Any,
        w: Any,
        b: Any,
        normalized: Any,
        flat: Any,
        **_: Any,
    ) -> MatchResult:
        result = MatchResult()
        if any(const_array(value) is None for value in (gamma, beta, mean, var, w, b)):
            return result.fail("BatchNormalization statistics or Gemm weights are not constant")
        if len(normalized.producer().outputs) != 1:
            return result.fail("BatchNormalization is in training mode")
        if not (single_use(normalized) and single_use(flat)):
            return result.fail("the normalized feature map is read outside the Gemm")
        return result

    def rewrite(
        self,
        op: Any,
        x: Any,
        gamma: Any,
        beta: Any,
        mean: Any,
        var: Any,
        w: Any,
        b: Any,
        normalized: Any,
        flat: Any,
        gemm: Any,
        **_: Any,
    ) -> Any:
        eps = normalized.producer().attributes.get_float("epsilon", 1e-5)
        g, shift, mu, sigma = (const_array(value).astype(np.float64) for value in (gamma, beta, mean, var))
        scale = g / np.sqrt(sigma + eps)
        offset = shift - mu * scale
        weight, bias = const_array(w), const_array(b)
        weight64 = weight.astype(np.float64)
        block = weight.shape[1] // scale.size
        folded_w = (weight64 * np.repeat(scale, block)).astype(weight.dtype)
        folded_b = (bias.astype(np.float64) + weight64 @ np.repeat(offset, block)).astype(bias.dtype)
        return op.Gemm(
            op.Flatten(x, **flat.producer().attributes),
            op.initializer(ir.tensor(folded_w, name=w.name + "_bn")),
            op.initializer(ir.tensor(folded_b, name=b.name + "_bn")),
            **gemm.producer().attributes,
        )


_BN_RULES = RewriteRuleSet([_FoldBatchNormIntoGemm.rule(), *BATCHNORM_FOLD_RULES])


_ANCHORS_PER_CELL = 2  # SCRFD keypoint models place 2 anchors per feature-map cell
_HEAD_KINDS = ("cls", "reg", "kps")
_HEAD_NAMES = ("scores", "boxes", "kps")  # what the contract calls the same three, in channel order
_HEAD_CHANNELS = (1, 4, 10)  # per anchor; also what tells the three head branches apart

# no dim expression can say 2 * (height/stride) * (width/stride), so the anchor count takes a name
_HEADS = {
    f"{name}{stride}": [_dsl.Batch, f"anchors{stride}", channels]
    for name, channels in zip(_HEAD_NAMES, _HEAD_CHANNELS)
    for stride in _dsl.DET_STRIDES
}


def transform_detection(model: ir.Model) -> ir.Model:
    if len(model.graph.outputs) != 9:
        raise ValueError(f"Expected a 9-output SCRFD keypoint model, got {len(model.graph.outputs)} outputs")

    ir.passes.Sequential(
        ConvertOpsetPass(_dsl.OPSET),
        # height and width stay symbolic: pinning a resolution is a load-time choice, not an artifact one
        DeclareInputDimsPass({0: _dsl.Batch, 2: _dsl.Height, 3: _dsl.Width}),
        FoldInputScalePass(scale=1.0 / 128.0),
        _FixFpnResizesPass(),
        _MergeScrfdHeadsPass(),
        _NameHeadsPass(),
        ReinferPass(),
        OptimizePass(),
        FoldPointwiseConvsPass(),
        WrapPrePass(_dsl.det_preprocess),
        ReinferPass(),
        NameOutputDimsPass(_HEADS),
        FlushDenormalsPass(),  # last: the folds above are what set the weights that ship
        common_passes.CheckerPass(),
    )(model)
    _check_detection(model)
    return model


def _check_detection(model: ir.Model) -> None:
    """Reject an unusable artifact: some stock packs declare no head shapes, so nothing before this can."""

    def shape(value: ir.Value) -> list[Any]:
        return [dim if isinstance(dim, int) else dim.value for dim in value.shape]

    if shape(model.graph.inputs[0]) != [_dsl.Batch, _dsl.Height, _dsl.Width, 3]:
        raise ValueError(f"Fused input is {shape(model.graph.inputs[0])}, expected [batch, height, width, 3]")
    heads = {output.name: shape(output) for output in model.graph.outputs}
    if heads != _HEADS:
        raise ValueError(f"Fused detection heads are {heads}, expected {_HEADS}")


def transform_recognition(model: ir.Model) -> ir.Model:
    return ir.passes.Sequential(
        ConvertOpsetPass(_dsl.OPSET),
        DeclareInputDimsPass({0: _dsl.Batch}),
        _FoldTailBatchNormPass(),
        FoldInputScalePass(scale=1.0 / 127.5),
        ReinferPass(),
        OptimizePass(),
        FoldPointwiseConvsPass(),
        WrapPass(_dsl.rec_preprocess, _dsl.l2_normalize),
        ReinferPass(),
        FlushDenormalsPass(),
        common_passes.CheckerPass(),
    )(model).model


class _FoldTailBatchNormPass(RewritePass):
    """Fold the ArcFace tail's BN into the fc Gemm; the IResNet residual BNs follow Adds, so they stay."""

    def __init__(self) -> None:
        super().__init__(_BN_RULES)

    def ensures(self, model: ir.Model) -> None:
        """Reject an artifact whose ArcFace tail did not collapse -- the rule matching nothing is silent."""
        gemm = next(node for node in model.graph if node.op_type == "Gemm")
        source = gemm.inputs[0].producer().inputs[0].producer()
        if source.op_type == "BatchNormalization":
            raise ir.passes.PostconditionError(f"{source.name} still sits between the backbone and the fc Gemm")


class _MergeScrfdHeads(RewriteRuleClassBase):
    """Merge one SCRFD stride's three head convs into one Conv + Transpose/Reshape/Split, anchor-major so
    each anchor's channels stay contiguous and the flatten IS the anchor unpack. Only the exporter can do
    it: conv output-channel order is observable data layout. The flatten targets are pinned in the pattern,
    not read in `check`: kps is a strict subgraph of the cls chain and the matcher does not backtrack, so
    checking there binds all three branches to cls and the rule declines every stride."""

    def _head(self, op: Any, x: Any, weight: Any, scale: Any, channels: int, tag: str) -> Any:
        conv = op.Conv(x, weight, _allow_other_inputs=True, _allow_other_attributes=True, _outputs=[f"conv_{tag}"])
        # OrValue, not a second rule: the scale is a box-branch thing, and only on some packs
        scaled = OrValue([op.Mul(conv, scale), conv], name=f"scaled_{tag}")
        rows = op.Transpose(scaled, _allow_other_attributes=True)
        target = Constant([-1, channels])
        return op.Reshape(rows, target, _allow_other_attributes=True, _outputs=[f"head_{tag}"])

    def pattern(self, op: Any, x: Any, wc: Any, wr: Any, wk: Any, sc: Any, sr: Any, sk: Any) -> Any:
        branches = zip((wc, wr, wk), (sc, sr, sk), _HEAD_CHANNELS, _HEAD_KINDS)
        cls, reg, kps = (self._head(op, x, weight, scale, channels, tag) for weight, scale, channels, tag in branches)
        return op.Sigmoid(cls), reg, kps

    def check(self, context: Any, **bound: Any) -> MatchResult:
        result = MatchResult()
        convs = [bound[f"conv_{tag}"].producer() for tag in _HEAD_KINDS]
        if any(const_array(conv.inputs[1]) is None for conv in convs):
            return result.fail("a head conv weight is not constant")
        if any(len(conv.inputs) > 2 and const_array(conv.inputs[2]) is None for conv in convs):
            return result.fail("a head conv bias is not constant")
        scaled = [bound.get(f"scaled_{tag}") for tag in _HEAD_KINDS]
        if any(_is_rescale(value) and _head_scale(value) is None for value in scaled):
            return result.fail("a head branch's Mul is not a scalar rescale")
        return result

    def rewrite(self, op: Any, x: Any, **bound: Any) -> Any:
        weights, biases = [], []
        for tag in _HEAD_KINDS:
            conv = bound[f"conv_{tag}"].producer()
            weight = const_array(conv.inputs[1])
            bias = const_array(conv.inputs[2]) if len(conv.inputs) > 2 else np.zeros(weight.shape[0], weight.dtype)
            scale = _head_scale(bound.get(f"scaled_{tag}"))
            if scale is not None:
                weight = (weight.astype(np.float64) * scale).astype(weight.dtype)
                bias = (bias.astype(np.float64) * scale).astype(bias.dtype)
            weights.append(np.split(weight, _ANCHORS_PER_CELL))
            biases.append(np.split(bias, _ANCHORS_PER_CELL))

        head = bound["conv_cls"].producer()  # the conv attributes are shared across the triple
        anchors = range(_ANCHORS_PER_CELL)
        stacked = [np.concatenate([part[a] for a in anchors for part in group]) for group in (weights, biases)]
        params = (op.initializer(ir.tensor(array, name=f"{head.name}_{tag}")) for array, tag in zip(stacked, "wb"))
        flat = op.initializer(ir.tensor(np.array([0, -1, sum(_HEAD_CHANNELS)], np.int64), name=f"{head.name}_flat"))
        sizes = op.initializer(ir.tensor(np.array(_HEAD_CHANNELS, np.int64), name=f"{head.name}_split"))
        cols = op.Reshape(op.Transpose(op.Conv(x, *params, **head.attributes), perm=[0, 2, 3, 1]), flat)
        cls, reg, kps = op.Split(cols, sizes, axis=2, _outputs=3)
        return op.Sigmoid(cls), reg, kps


def _is_rescale(value: ir.Value | None) -> bool:
    """True where the OrValue took the scaled arm, i.e. this branch has a `Mul` to fold away."""
    return value is not None and value.producer() is not None and value.producer().op_type == "Mul"


def _head_scale(value: ir.Value | None) -> float | None:
    """The scalar that `Mul` applies; None for no Mul at all AND for a non-scalar one, hence `_is_rescale`."""
    if not _is_rescale(value):
        return None
    factors = [array for v in value.producer().inputs if (array := const_array(v)) is not None]
    return float(factors[0]) if factors and factors[0].size == 1 else None


class _MergeScrfdHeadsPass(PinnedRewritePass):
    def __init__(self) -> None:
        super().__init__([_MergeScrfdHeads.rule()], len(_dsl.DET_STRIDES), "SCRFD head triples")


def _feature_strides(graph: ir.Graph) -> dict[ir.Value, int]:
    """How far the input is downsampled at each value."""
    strides = {graph.inputs[0]: 1}
    for node in graph:
        source = next((value for value in node.inputs if value in strides), None)
        if source is None:
            continue
        stride = strides[source]
        if node.op_type in ("Conv", "MaxPool", "AveragePool"):
            stride *= node.attributes.get_ints("strides", [1])[0]
        elif node.op_type == "Resize":
            stride //= int(const_array(node.inputs[2])[2])
        for output in node.outputs:
            strides[output] = stride
    return strides


class _NameHeadsPass(ir.passes.InPlacePass):
    """Name the nine heads `{kind}{stride}` for the host decode; the stride comes off the graph, not the shape."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        strides = _feature_strides(model.graph)
        for split in model.graph:
            if split.op_type != "Split" or const_ints(split.inputs[1]) != list(_HEAD_CHANNELS):
                continue
            scores = sole_consumer(split.outputs[0], "Sigmoid").outputs[0]
            for value, kind in zip((scores, *split.outputs[1:]), _HEAD_NAMES):
                value.name = f"{kind}{strides[value]}"
        return ir.passes.PassResult(model, True)

    def ensures(self, model: ir.Model) -> None:
        named = [output.name for output in model.graph.outputs]
        if sorted(named) != sorted(_HEADS):
            raise ir.passes.PostconditionError(f"Merged heads are named {named}, expected {sorted(_HEADS)}")


class _ConstantFpnScale(RewriteRuleClassBase):
    """Replace an FPN upsample's runtime-computed scales with a constant, which the Shape/Slice/Concat chain
    otherwise folds to with a stale batch=1 baked in. roi is dropped, not kept as the packs' empty
    initializer: a materialized tensor makes importers take their roi path, and dropping it also terminates
    the rule, a plain `Var` refusing to bind an absent optional input."""

    def pattern(self, op: Any, x: Any, roi: Any, scales: Any) -> Any:
        return op.Resize(x, roi, scales, _allow_other_inputs=True, _allow_other_attributes=True, _outputs=["resized"])

    def check(self, context: Any, resized: Any, **_: Any) -> MatchResult:
        mode = resized.producer().attributes.get_string("mode")
        return MatchResult() if mode == "nearest" else MatchResult().fail(f"Resize mode {mode} is not nearest")

    def rewrite(self, op: Any, x: Any, roi: Any, scales: Any, resized: Any) -> Any:
        node = resized.producer()
        # per-node name: `op.initializer` OVERWRITES on a collision, orphaning the earlier upsample
        doubled = op.initializer(ir.tensor(np.array([1, 1, 2, 2], np.float32), name=f"{node.name}_scales_2x"))
        return op.Resize(x, None, doubled, _name=node.name, **node.attributes)


class _FixFpnResizesPass(PinnedRewritePass):
    """`_ConstantFpnScale`, pinned on the incoming Resize census too so a declined mode fails as loudly."""

    def __init__(self) -> None:
        super().__init__([_ConstantFpnScale.rule()], len(_dsl.DET_STRIDES) - 1, "FPN Resize scales")

    def requires(self, model: ir.Model) -> None:
        resizes = sum(1 for node in model.graph if node.op_type == "Resize")
        if resizes != self.expected:
            raise ir.passes.PreconditionError(f"Expected {self.expected} FPN Resize nodes, found {resizes}")


__all__ = ["transform_detection", "transform_recognition"]
