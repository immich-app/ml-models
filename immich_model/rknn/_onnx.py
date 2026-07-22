from pathlib import Path
from typing import Any

import numpy as np
import onnx_ir as ir
from onnx_ir.passes.common import (
    DeduplicateInitializersPass,
    LiftConstantsToInitializersPass,
    RemoveUnusedNodesPass,
    ShapeInferencePass,
)
from onnxscript.rewriter import RewriteRuleSet
from onnxscript.rewriter.pattern import MatchResult, RewriteRuleClassBase

from ..onnx._ir import clear_cached_annotations, make_init, make_node
from ..onnx.lowering import DecomposeAttention, DecomposeGelu, FoldConstantGatherElements, PatchEmbedToMatMul


def prepare_for_rknn(onnx_path: Path, work_dir: Path) -> Path:
    """Normalize an ONNX graph so rknn.build can ingest and run it on the NPU: decompose Attention/Gelu
    so the toolkit re-fuses them into NPU kernels (no kernel otherwise, and opset>19 is rejected),
    im2col the ViT patch-embed conv (else a ~40x-slower quadrant split), float the uint8 image, pin opset 19.
    A float opset<=19 model with none of these is returned untouched.
    """
    model = ir.load(onnx_path)
    opset = max(model.opset_imports.get("", 0), model.opset_imports.get("ai.onnx", 0))

    changed = _PATCH_EMBED_REWRITE.apply_to_model(model) > 0
    changed |= _float_image_input(model)
    changed |= _floatify_pad_mask(model)
    changed |= _FLOAT_INDICATOR_RULES.apply_to_model(model) > 0
    changed |= _CONST_GATHER_RULES.apply_to_model(model) > 0  # token-type lookup: toolkit crash + dead island
    changed |= _RKNN_DECOMPOSITIONS.apply_to_model(model) > 0

    if opset <= 19 and not changed:
        return onnx_path

    # sweeps unused initializers (the im2col-retired conv weight), then dedups the re-pointed duplicates
    RemoveUnusedNodesPass()(model)
    LiftConstantsToInitializersPass()(model)
    DeduplicateInitializersPass()(model)
    _pin_opset(model, 19)
    model.graph.sort()
    model = _reinfer(model)

    out_path = work_dir / "model.onnx"
    ir.save(model, out_path, external_data="model.onnx.data")
    return out_path


# keep pre-`lowering` rule names (stamped into node metadata) so AOT output stays byte-stable
_RKNN_DECOMPOSITIONS = RewriteRuleSet(
    [DecomposeGelu.rule(name="_DecomposeGelu"), DecomposeAttention.rule(name="_DecomposeAttention")]
)
_PATCH_EMBED_REWRITE = RewriteRuleSet([PatchEmbedToMatMul.rule(name="_PatchEmbedToMatMul")])


def _float_image_input(model: ir.Model) -> bool:
    """uint8 NHWC image input -> float32, dropping the redundant leading Cast.

    rknn rejects the uint8 input ("Not Support Dtype: 2", toolkit 2.3.2) since the fused contracts fold
    normalization into the graph; feeding float raw pixels is bit-identical (the Cast becomes a no-op).
    """
    graph = model.graph
    if not graph.inputs:
        return False
    image = graph.inputs[0]
    dims = image.shape
    channels = dims[-1] if dims is not None and len(dims) == 4 else None
    if image.dtype != ir.DataType.UINT8 or not isinstance(channels, int) or not 0 < channels <= 4:
        return False

    image.dtype = ir.DataType.FLOAT
    for usage in list(image.uses()):
        cast = usage.node
        if cast.op_type != "Cast" or cast.attributes.get_int("to") != ir.DataType.FLOAT:
            continue
        cast.outputs[0].replace_all_uses_with(image)
        graph.remove(cast, safe=True)
    return True


class _FloatifyNotEqual(RewriteRuleClassBase):
    """Cast(Not(Equal(int, scalar))) -> Clip(Abs(Cast(x)-c),0,1) = float(x != c), exact for integer ids.

    openclip XLM towers reuse the pad comparison as the mean-pool weight, so retiring the attention
    mask alone leaves int32 Equal alive — which has no librknnrt kernel."""

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


class _OpaqueZeroMul(RewriteRuleClassBase):
    """Mul(x, 0.0) -> Sub(x, x): identical zeros but opaque to rknn-toolkit2's fold_constant.

    Folding collapses batch_zeros_1d (batch pinned to 1) into a constant Q, which crashes the SDPA
    matcher (ValueError: inputs or 'outputs' must be set); the opaque form fuses all sites (13/13 on
    SigLIP B-16). Un-folded cost is bounded: ez_slice 154us + Sub 61us + query Add 19us = 0.18% of
    SigLIP2-B16 visual, and even a perfect restructure caps at ~0.5% (674us MAP-site exSDPAttention)."""

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


_FLOAT_INDICATOR_RULES = RewriteRuleSet([_FloatifyNotEqual.rule(), _OpaqueZeroMul.rule()])
_CONST_GATHER_RULES = RewriteRuleSet([FoldConstantGatherElements.rule()])


_MASK_ISLAND_OPS = {
    "Equal", "Not", "Cast", "And", "GatherND", "Concat", "Range", "Unsqueeze", "Squeeze",
    "Reshape", "Shape", "Slice", "Add", "Mul", "Expand", "Gather", "Where",
}  # fmt: skip


def _floatify_pad_mask(model: ir.Model) -> bool:
    """Replace an in-graph bool pad mask feeding Attention with an equivalent all-float additive bias,
    killing the whole integer mask island.

    NLLB-style towers derive the mask in-graph (Equal(text, pad) -> Not -> broadcast); int32 Equal has
    no librknnrt kernel (RKNN_ERR_FAIL, XLM-R precedent). (clip(|cast(text)-pad|,0,1)-1)*1e4 equals
    Where(mask, 0, -1e4) bitwise for every int id, so the score Add is unchanged and exSDPAttention re-fuses."""
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
    return converted


def _pin_opset(model: ir.Model, version: int) -> None:
    model.opset_imports.pop("ai.onnx", None)
    model.opset_imports[""] = version


def _reinfer(model: ir.Model) -> ir.Model:
    """Drop cached shape/type annotations and re-infer from the retyped inputs.

    Floating the image input leaves value_info stale (e.g. a Slice off the uint8 image still annotated
    UINT8), which strict type-checkers (ORT session load) reject.
    """
    clear_cached_annotations(model.graph)
    return ShapeInferencePass()(model).model
