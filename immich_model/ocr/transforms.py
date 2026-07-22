"""Turn stock PP-OCR ONNX (detection + recognition) into Immich's fused format: uint8 RGB NHWC
input, (x - 127.5) / 127.5 normalization and the greedy-CTC head folded into the graph.

Both transforms fold Conv+BatchNormalization at export: a hard fp16 prerequisite — stock det BN
variance constants reach ~1e8, which clamp to inf/65504 in fp16 and corrupt the backbone.
"""

import logging
from typing import Any

import numpy as np
import onnx_ir.passes.common as common_passes
import onnxscript.optimizer
from onnx import ModelProto, version_converter
from onnxscript import ir
from onnxscript.rewriter.pattern import MatchResult, OrValue, RewriteRuleClassBase, RewriteRuleSet, Var
from onnxscript.rewriter.rules.common import (
    fuse_batchnorm_into_conv_rule,
    fuse_batchnorm_into_conv_transpose_rule,
)
from onnxscript.rewriter.rules.fusion._layer_norm import layer_normalization_ruleset

from ..onnx.graph import fold_input_scale, reinfer, wrap
from ..onnx.transforms import _const_ints
from . import _dsl

log = logging.getLogger(__name__)

_BN_RULES = RewriteRuleSet([fuse_batchnorm_into_conv_rule, fuse_batchnorm_into_conv_transpose_rule])


def transform_detection(model: ModelProto, affine_folds: int) -> ModelProto:
    _name_input_dims(model, {0: _dsl.Batch, 2: _dsl.Height, 3: _dsl.Width})

    # optimize first: stock Identity nodes + standalone deconv bias Adds block the BN-fusion patterns
    model = onnxscript.optimizer.optimize(reinfer(model))
    model = _fold_bias_adds(model)
    model = _fuse_batchnorm(model, require_all=True)
    fold_input_scale(model, scale=1.0 / 127.5, flip_channels=True)
    model = _decompose_hardswish(model)
    _relax_pool_ceil_mode(model)

    model = reinfer(model)
    model = reinfer(version_converter.convert_version(model, _dsl.OPSET))
    model = _fold_learnable_affine(model, expected=affine_folds)

    io_map = [(model.graph.output[0].name, "probs_raw")]
    model = wrap(model, _dsl.det_preprocess, _dsl.det_postprocess, io_map)
    _name_output_dims(model, 0, [_dsl.Batch, _dsl.Height, _dsl.Width])
    return model


def transform_recognition(model: ModelProto, affine_folds: int, layernorms: int, shape_domains: int) -> ModelProto:
    _name_input_dims(model, {0: _dsl.Batch, 3: _dsl.Width})
    _pin_input_dim(model, 2, _dsl.REC_HEIGHT)

    # require_all=False: SVTR-neck BNs don't follow convs, and rec has no fp16-hazardous BN
    # constants (verified). Folding the standalone [1,C,1,1] bias Adds lets the affine blocks fold.
    model = onnxscript.optimizer.optimize(reinfer(model))
    model = _fuse_batchnorm(model, require_all=False)
    model = _fold_bias_adds(model)
    fold_input_scale(model, scale=1.0 / 127.5, flip_channels=True)
    model = _decompose_hardswish(model)
    _relax_pool_ceil_mode(model)

    model = reinfer(model)
    model = reinfer(version_converter.convert_version(model, _dsl.OPSET))
    model = _fold_learnable_affine(model, expected=affine_folds)
    ir_model = _fuse_layernorm(model, expected=layernorms)
    _simplify_rec_shape_domain(ir_model, expected=shape_domains)
    _elide_ctc_softmax(ir_model)
    model = ir.to_proto(ir_model)

    io_map = [(model.graph.output[0].name, "logits")]
    model = wrap(model, _dsl.rec_preprocess, _dsl.rec_postprocess, io_map)
    _name_output_dims(model, 0, [_dsl.Batch, _dsl.Seq])
    _name_output_dims(model, 1, [_dsl.Batch, _dsl.Seq])
    return model


def _name_input_dims(model: ModelProto, names: dict[int, str]) -> None:
    dims = model.graph.input[0].type.tensor_type.shape.dim
    for i, name in names.items():
        dims[i].ClearField("dim_value")
        dims[i].dim_param = name


def _pin_input_dim(model: ModelProto, axis: int, value: int) -> None:
    """Pin a symbolic input dim to its contract value. Variant rec exports declare the 48-px height
    dynamically; a symbolic height keeps the backbone height symbolic so shape inference can't prove
    the SVTR flatten collapses it to 1 and the shape-domain guard trips. All PP-OCRv5 rec is trained
    at 48, so pinning is exact; fail-closed on a conflicting static value."""
    dim = model.graph.input[0].type.tensor_type.shape.dim[axis]
    if dim.HasField("dim_value") and dim.dim_value != value:
        raise ValueError(f"Input dim {axis} is statically {dim.dim_value}, cannot pin to {value}")
    dim.ClearField("dim_param")
    dim.dim_value = value


def _name_output_dims(model: ModelProto, output: int, names: list[str]) -> None:
    dims = model.graph.output[output].type.tensor_type.shape.dim
    assert len(dims) == len(names)
    for dim, name in zip(dims, names):
        if not dim.dim_value:
            dim.ClearField("dim_value")
            dim.dim_param = name


class _FoldBiasAdd(RewriteRuleClassBase):
    """Fold Add(Conv[Transpose](x, W[, b]), per-channel const c) into the conv's bias. Paddle emits
    the head deconvs' bias as a standalone Add that blocks BN fusion. Exact: same math, one fewer op."""

    def pattern(self, op: Any, x: Any, w: Any, c: Any) -> Any:
        b = Var("b", can_match_none=True)
        conv_out = OrValue(
            [
                op.Conv(x, w, b, _allow_other_attributes=True),
                op.ConvTranspose(x, w, b, _allow_other_attributes=True),
            ],
            name="conv_out",
        )
        return op.Add(conv_out, c)

    def check(self, context: Any, c: Any, b: Any, conv_out: Any, **_: Any) -> MatchResult:
        result = MatchResult()
        if c.const_value is None:
            return result.fail("add operand is not constant")
        arr = c.const_value.numpy()
        if arr.shape not in ((arr.size,), (arr.size, 1, 1), (1, arr.size, 1, 1)):
            return result.fail("add constant is not a per-channel bias")
        if b is not None and b.const_value is None:
            return result.fail("existing conv bias is not constant")
        return result

    def rewrite(self, op: Any, x: Any, w: Any, c: Any, b: Any, conv_out: Any, **_: Any) -> Any:
        arr = c.const_value.numpy()
        bias = arr.reshape(-1)
        if b is not None:
            bias = bias + b.const_value.numpy()
        conv = conv_out.producer()
        bias_init = op.initializer(ir.tensor(bias.astype(arr.dtype), name=f"{conv_out.name}_bias"))
        return getattr(op, conv.op_type)(x, w, bias_init, **conv.attributes)


_BIAS_ADD_RULES = RewriteRuleSet([_FoldBiasAdd.rule()], commute=True)


def _fold_bias_adds(model: ModelProto) -> ModelProto:
    ir_model = ir.from_proto(model)
    folded = _BIAS_ADD_RULES.apply_to_model(ir_model)
    log.info("Folded %d standalone bias Adds into convs", folded)
    return ir.to_proto(ir_model)


class _DecomposeHardSwish(RewriteRuleClassBase):
    """HardSwish(x) -> x * HardSigmoid(x, 1/6, 0.5), exact per spec. CoreML EP (and some NPUs) lack
    HardSwish; keeping it fragments the MobileNetV3 backbone into a partition per activation."""

    def pattern(self, op: Any, x: Any) -> Any:
        return op.HardSwish(x)

    def rewrite(self, op: Any, x: Any) -> Any:
        return op.Mul(x, op.HardSigmoid(x, alpha=1.0 / 6.0, beta=0.5))


_HARDSWISH_RULES = RewriteRuleSet([_DecomposeHardSwish.rule()])


def _decompose_hardswish(model: ModelProto) -> ModelProto:
    ir_model = ir.from_proto(model)
    decomposed = _HARDSWISH_RULES.apply_to_model(ir_model)
    log.info("Decomposed %d HardSwish ops", decomposed)
    return ir.to_proto(ir_model)


def _relax_pool_ceil_mode(model: ModelProto) -> None:
    """Zero ceil_mode on SAME-padded pooling: CoreML MLProgram rejects ceil_mode=True with SAME
    padding, and with auto_pad=SAME the output is ceil(input/stride) anyway so it's redundant (exact).
    The PP-OCRv5_server MaxPool otherwise drops the whole model to CPU."""
    for node in model.graph.node:
        if node.op_type not in ("MaxPool", "AveragePool"):
            continue
        auto_pad = next((a.s.decode() for a in node.attribute if a.name == "auto_pad"), "NOTSET")
        if not auto_pad.startswith("SAME"):
            continue
        for attr in node.attribute:
            if attr.name == "ceil_mode" and attr.i == 1:
                attr.i = 0


def _fuse_batchnorm(model: ModelProto, require_all: bool) -> ModelProto:
    ir_model = ir.from_proto(model)
    applied = _BN_RULES.apply_to_model(ir_model)
    remaining = sum(1 for node in ir_model.graph if node.op_type == "BatchNormalization")
    log.info("Fused %d Conv+BatchNormalization pairs (%d BN ops remain)", applied, remaining)
    if require_all and remaining:
        raise ValueError(f"{remaining} BatchNormalization ops did not fold into convs")
    return ir.to_proto(ir_model)


def _scalar(value: ir.Value | None) -> float | None:
    const = value.const_value if value is not None else None
    if const is None or const.size != 1:
        return None
    return float(const.numpy().reshape(()))


def _single_use_const(value: ir.Value | None) -> bool:
    return value is not None and value.const_value is not None and len(list(value.uses())) == 1


class _FoldAffineAfterConv(RewriteRuleClassBase):
    """Fold scalar affine block Conv(x)->Mul(a)->Add(b) into the conv: W'=a*W, bias'=a*bias+b. Exact
    (per-out-channel scale/shift after the conv, padding irrelevant) when the conv output feeds only
    this affine and already carries a bias. Fold in fp64, cast back to the weights' dtype."""

    def pattern(self, op: Any, x: Any, w: Any, b: Any, a: Any, b_add: Any) -> Any:
        conv = op.Conv(x, w, b, _allow_other_attributes=True, _outputs=["conv"])
        return op.Add(op.Mul(conv, a, _outputs=["mul"]), b_add, _outputs=["shifted"])

    def check(self, context: Any, w: Any, b: Any, a: Any, b_add: Any, conv: Any, mul: Any, **_: Any) -> MatchResult:
        result = MatchResult()
        if _scalar(a) is None or _scalar(b_add) is None:
            return result.fail("affine scale/shift is not a scalar constant")
        if not (_single_use_const(w) and _single_use_const(b)):
            return result.fail("conv weight/bias is not a single-use constant")
        if len(list(conv.uses())) != 1 or len(list(mul.uses())) != 1:
            return result.fail("conv output feeds consumers outside the affine block")
        return result

    def rewrite(self, op: Any, x: Any, w: Any, b: Any, a: Any, b_add: Any, conv: Any, **_: Any) -> Any:
        scale, shift = _scalar(a), _scalar(b_add)
        w_arr, b_arr = w.const_value.numpy(), b.const_value.numpy()
        w_new = (w_arr.astype(np.float64) * scale).astype(w_arr.dtype)
        b_new = (b_arr.astype(np.float64) * scale + shift).astype(b_arr.dtype)
        w_init = op.initializer(ir.tensor(w_new, name=w.name + "_lab"))
        b_init = op.initializer(ir.tensor(b_new, name=b.name + "_lab"))
        return op.Conv(x, w_init, b_init, **conv.producer().attributes)


class _FoldAffineBeforeConv(RewriteRuleClassBase):
    """Fold post-activation affine Mul(a)->Add(b)->Conv into the following conv, only when pads=0:
    W'=a*W, bias'[o]=bias[o]+b*sum(W[o]). A padded conv reads a zero border, not a*0+b, so the shift
    doesn't commute with the pad (same as fold_input_scale). Affine feeds only this conv; conv has a bias."""

    def pattern(self, op: Any, x: Any, w: Any, b: Any, a: Any, b_add: Any) -> Any:
        affine = op.Add(op.Mul(x, a, _outputs=["mul"]), b_add, _outputs=["affine"])
        return op.Conv(affine, w, b, pads=[0, 0, 0, 0], _allow_other_attributes=True, _outputs=["conv"])

    def check(
        self, context: Any, w: Any, b: Any, a: Any, b_add: Any, mul: Any, affine: Any, conv: Any, **_: Any
    ) -> MatchResult:
        result = MatchResult()
        if _scalar(a) is None or _scalar(b_add) is None:
            return result.fail("affine scale/shift is not a scalar constant")
        if not (_single_use_const(w) and _single_use_const(b)):
            return result.fail("conv weight/bias is not a single-use constant")
        if len(list(affine.uses())) != 1 or len(list(mul.uses())) != 1:
            return result.fail("affine feeds consumers outside the single following conv")
        if conv.producer().attributes.get_string("auto_pad", "").startswith("SAME"):
            return result.fail("conv pads at runtime via auto_pad despite pads=0")
        return result

    def rewrite(self, op: Any, x: Any, w: Any, b: Any, a: Any, b_add: Any, conv: Any, **_: Any) -> Any:
        scale, shift = _scalar(a), _scalar(b_add)
        w_arr, b_arr = w.const_value.numpy(), b.const_value.numpy()
        w64 = w_arr.astype(np.float64)
        w_new = (w64 * scale).astype(w_arr.dtype)
        b_new = (b_arr.astype(np.float64) + (w64 * shift).sum(axis=(1, 2, 3))).astype(b_arr.dtype)
        w_init = op.initializer(ir.tensor(w_new, name=w.name + "_lab"))
        b_init = op.initializer(ir.tensor(b_new, name=b.name + "_lab"))
        return op.Conv(x, w_init, b_init, **conv.producer().attributes)


# commute: PP-OCR emits the affine scale at Mul input 0, shift at Add input 1
_LAB_RULES = RewriteRuleSet([_FoldAffineAfterConv.rule(), _FoldAffineBeforeConv.rule()], commute=True)


def _fold_learnable_affine(model: ModelProto, expected: int) -> ModelProto:
    """Fold the foldable scalar affine-block pairs into adjacent convs (pre-act into the preceding
    conv, post-act into a following pads=0 conv); SE-boundary and padded pairs stay. Double-guarded:
    a structural scan must agree pair-for-pair and the total must equal the source's pinned count."""
    scanned = _count_foldable_lab(model.graph)
    if scanned != expected:
        raise ValueError(f"Found {scanned} foldable learnable-affine pairs, expected {expected}")
    ir_model = ir.from_proto(model)
    applied = _LAB_RULES.apply_to_model(ir_model)
    if applied != scanned:
        raise ValueError(f"Folded {applied} learnable-affine pairs, but the structural scan found {scanned}")
    common_passes.RemoveUnusedNodesPass()(ir_model)
    log.info("Folded %d learnable-affine-block pairs into convs", applied)
    return ir.to_proto(ir_model)


def _count_foldable_lab(graph: Any) -> int:
    """Count the LAB pairs _LAB_RULES must fold by plain graph walking, so a rewrite-rule regression
    can't silently skip folds. Mirrors the rules' scope exactly."""
    producers = {o: n for n in graph.node for o in n.output}
    consumers: dict[str, list[Any]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    sizes = {i.name: int(np.prod(i.dims)) for i in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            sizes[node.output[0]] = int(np.prod(node.attribute[0].t.dims))

    def scalar_const(name: str) -> bool:
        return sizes.get(name) == 1

    def single_use_const(name: str) -> bool:
        return name in sizes and len(consumers.get(name, [])) == 1

    def conv_foldable(conv: Any) -> bool:
        return len(conv.input) > 2 and single_use_const(conv.input[1]) and single_use_const(conv.input[2])

    def sole_consumer(name: str, op_type: str) -> Any:
        nodes = consumers.get(name, [])
        return nodes[0] if len(nodes) == 1 and nodes[0].op_type == op_type else None

    count = 0
    for mul in graph.node:
        if mul.op_type != "Mul":
            continue
        a = next((i for i in mul.input if scalar_const(i)), None)
        if a is None:
            continue
        x = mul.input[1] if mul.input[0] == a else mul.input[0]
        add = sole_consumer(mul.output[0], "Add")
        if add is None or not any(scalar_const(i) for i in add.input if i != mul.output[0]):
            continue
        # pre-activation Conv -> Mul -> Add
        conv = producers.get(x)
        if conv is not None and conv.op_type == "Conv" and len(consumers.get(x, [])) == 1 and conv_foldable(conv):
            count += 1
            continue
        # post-activation Mul -> Add -> Conv(pads=0)
        conv = sole_consumer(add.output[0], "Conv")
        if conv is not None and conv_foldable(conv) and _explicit_zero_pads(conv):
            count += 1
    return count


def _explicit_zero_pads(conv: Any) -> bool:
    attrs = {a.name: a for a in conv.attribute}
    pads = attrs.get("pads")
    auto_pad = attrs.get("auto_pad")
    return (
        pads is not None
        and list(pads.ints) == [0, 0, 0, 0]
        and (auto_pad is None or not auto_pad.s.decode().startswith("SAME"))
    )


def _elide_ctc_softmax(model: ir.Model) -> None:
    """Delete the stock CTC head's trailing last-axis Softmax so the [batch, seq, classes] softmax
    never materializes; the fused head recomputes the winning prob from raw logits (argmax is
    softmax-invariant). Also drops the charset-wide (18k-class) Softmax for backends where it's slow
    or unsupported (RKNN). Two tail forms: opset-14 plain Softmax(axis=2), and the opset-10 server
    export's version-converter Flatten(2)->Softmax(-1)->Reshape sandwich (same math). Fail-closed on
    anything else over rank-3 data."""
    graph = model.graph
    if len(graph.outputs) != 1:
        raise ValueError(f"Expected a single recognition output, found {len(graph.outputs)}")
    output = graph.outputs[0]
    rank = len(output.shape) if output.shape is not None else None
    if rank != 3:
        raise ValueError(f"Recognition output is rank {rank}, expected 3")

    def sole_use(value: ir.Value, op_type: str) -> ir.Node:
        uses = value.uses()
        if len(uses) != 1 or uses[0].node.op_type != op_type or value in graph.outputs:
            raise ValueError(f"{value.name} must feed exactly one {op_type}, found {[u.node.op_type for u in uses]}")
        return uses[0].node

    if output.uses():
        raise ValueError(f"Recognition output feeds {[u.node.name for u in output.uses()]}, expected no consumers")
    tail = output.producer()
    if tail is not None and tail.op_type == "Softmax":
        if tail.attributes.get_int("axis", -1) not in (2, -1):
            raise ValueError(f"CTC-head Softmax is over axis {tail.attributes.get_int('axis', -1)}, expected the last")
        logits, removed = tail.inputs[0], [tail]
    elif tail is not None and tail.op_type == "Reshape":
        softmax = tail.inputs[0].producer()
        if softmax is None or softmax.op_type != "Softmax" or softmax.attributes.get_int("axis", -1) not in (1, -1):
            raise ValueError("Reshape tail is not fed by a last-axis Softmax")
        flatten = softmax.inputs[0].producer()
        if flatten is None or flatten.op_type != "Flatten" or flatten.attributes.get_int("axis", 1) != 2:
            raise ValueError("Sandwiched Softmax is not fed by Flatten(axis=2)")
        if not _is_shape_of(tail.inputs[1].producer(), flatten.inputs[0]):
            raise ValueError("Reshape tail does not restore the pre-Flatten shape")
        sole_use(flatten.outputs[0], "Softmax")
        sole_use(softmax.outputs[0], "Reshape")
        logits, removed = flatten.inputs[0], [tail, softmax, flatten]
    else:
        raise ValueError(f"Recognition output is produced by {tail and tail.op_type}, expected a Softmax tail")

    graph.outputs[0] = logits
    for node in removed:
        graph.remove(node, safe=True)
    # dangling helpers (the sandwich's Shape) are swept by wrap()'s optimizer DCE
    log.info("Elided the CTC head's trailing Softmax (graph output is now raw logits)")


def _fuse_layernorm(model: ModelProto, expected: int) -> ir.Model:
    """Fuse the SVTR neck's 9-op LayerNorm decomposition into LayerNormalization (axis=-1, opset 19).
    ORT re-fuses at runtime, so this only helps EPs matching the raw graph (CoreML, RKNN); equivalent
    (fp32 reassociation)."""
    ir_model = ir.from_proto(model)
    # LN pattern keys on ReduceMean(axes=[-1], keepdims=1); version_converter un-inlines the axes and
    # drops default keepdims, so surface both first
    common_passes.AddDefaultAttributesPass()(ir_model)
    onnxscript.optimizer.fold_constants(ir_model)
    layer_normalization_ruleset.apply_to_model(ir_model)
    fused = sum(1 for node in ir_model.graph if node.op_type == "LayerNormalization")
    if fused != expected:
        raise ValueError(f"Fused {fused} LayerNormalization ops, expected {expected}")
    common_passes.RemoveUnusedNodesPass()(ir_model)
    log.info("Fused %d LayerNormalization ops", fused)
    return ir_model


class _FlattenToSqueeze(RewriteRuleClassBase):
    """Reshape(x, Concat(Slice(Shape(x),[0],[2]), [-1])) on [B,C,1,W] is Squeeze(axis=2). The shared
    x enforces same value; the shape-helper chain stays (also feeds the unflatten target) and dies in
    cleanup (remove_nodes=False)."""

    def __init__(self) -> None:
        super().__init__(remove_nodes=False)

    def pattern(self, op: Any, data: Any) -> Any:
        lead = op.Slice(op.Shape(data, _outputs=["shp"]), [0], [2], _allow_other_inputs=True, _outputs=["lead"])
        return op.Reshape(data, op.Concat(lead, [-1], _allow_other_attributes=True))

    def check(self, context: Any, data: Any, shp: Any, lead: Any, **_: Any) -> MatchResult:
        result = MatchResult()
        if not _is_shape_of(shp.producer(), data):
            return result.fail("Shape is windowed (start/end)")
        extras = list(lead.producer().inputs[3:])  # optional Slice axes/steps
        if extras and not (_const_ints(extras[0]) == [0] and (len(extras) < 2 or _const_ints(extras[1]) == [1])):
            return result.fail("Slice is not a plain [0:2] window")
        if data.shape is None or len(data.shape) != 4 or data.shape[2] != 1:
            return result.fail("data is not [B,C,1,W]")
        return result

    def rewrite(self, op: Any, data: Any, **_: Any) -> Any:
        return op.Squeeze(data, op.initializer(ir.tensor(np.array([2], np.int64), name="rec_squeeze_axis")))


class _UnflattenToUnsqueeze(RewriteRuleClassBase):
    """Reshape(x, Concat([0],[1],W,[C])) on [B,W,C] with only width dynamic is Unsqueeze(axis=1);
    the target's [C] must equal the static channel dim."""

    def __init__(self) -> None:
        super().__init__(remove_nodes=False)

    def pattern(self, op: Any, data: Any, w: Any, c: Any) -> Any:
        return op.Reshape(data, op.Concat([0], [1], w, c, _allow_other_attributes=True))

    def check(self, context: Any, data: Any, w: Any, c: Any, **_: Any) -> MatchResult:
        result = MatchResult()
        if w.const_value is not None:
            return result.fail("width entry is constant — not the dynamic-width unflatten")
        if data.shape is None or len(data.shape) != 3 or not isinstance(data.shape[2], int):
            return result.fail("data is not [B,W,C] with static C")
        if _const_ints(c) != [data.shape[2]]:
            return result.fail("target channel entry does not match the data's channel dim")
        return result

    def rewrite(self, op: Any, data: Any, **_: Any) -> Any:
        return op.Unsqueeze(data, op.initializer(ir.tensor(np.array([1], np.int64), name="rec_unsqueeze_axis")))


class _ElideNoopUnsqueezeAddSqueeze(RewriteRuleClassBase):
    """Squeeze(Add(Unsqueeze(x,[0]), 0.0), [0]) -> x: every deleted op is a copy or a zero-add."""

    def __init__(self) -> None:
        super().__init__(remove_nodes=False)

    def pattern(self, op: Any, x: Any) -> Any:
        return op.Squeeze(op.Add(op.Unsqueeze(x, [0]), 0.0), [0])

    def rewrite(self, op: Any, x: Any) -> Any:
        return op.Identity(x)


_FLATTEN_RULES = RewriteRuleSet([_FlattenToSqueeze.rule()])
_UNFLATTEN_RULES = RewriteRuleSet([_UnflattenToUnsqueeze.rule()])
_NOOP_RULES = RewriteRuleSet([_ElideNoopUnsqueezeAddSqueeze.rule()], commute=True)


def _simplify_rec_shape_domain(model: ir.Model, expected: int) -> None:
    """Devitalize rec's backbone->SVTR flatten/unflatten shape domain. Only width is dynamic, so both
    reshapes are rank ops; dropping the runtime targets kills the graph's only Shape op (the CoreML-EP
    rec partition blocker). Width stays symbolic."""
    flattened = _FLATTEN_RULES.apply_to_model(model)
    unflattened = _UNFLATTEN_RULES.apply_to_model(model)
    # no-op count unpinned: each removal is provably exact, and the quirk is v5-only (absent in v6)
    noops = _NOOP_RULES.apply_to_model(model)
    common_passes.IdentityEliminationPass()(model)
    if (flattened, unflattened) != (expected, expected):
        raise ValueError(
            f"rec shape-domain simplification matched flatten={flattened} "
            f"unflatten={unflattened}, expected {expected} of each"
        )
    log.info(
        "Simplified rec shape domain (%d flatten/%d unflatten -> rank ops, %d no-op passes)",
        flattened,
        unflattened,
        noops,
    )


def _is_shape_of(node: "ir.Node | None", data: ir.Value) -> bool:
    """True if node is a full Shape(data): no end, start absent or the explicit default 0
    (AddDefaultAttributesPass materializes start=0)."""
    if node is None or node.op_type != "Shape" or node.inputs[0] is not data:
        return False
    return node.attributes.get_int("start", 0) == 0 and "end" not in node.attributes


__all__ = ["transform_detection", "transform_recognition"]
