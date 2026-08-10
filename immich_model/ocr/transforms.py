"""Stock PP-OCR ONNX (detection + recognition) -> Immich's fused format: uint8 RGB NHWC input, folded
normalization, greedy-CTC head in the graph. Folding Conv+BN is an fp16 prerequisite: stock variances
overflow it."""

import logging
import math
from typing import Any, NamedTuple

import numpy as np
import onnx_ir.passes.common as common_passes
from onnxscript import ir
from onnxscript.rewriter import RewritePass
from onnxscript.rewriter.pattern import MatchResult, OrValue, RewriteRuleClassBase, RewriteRuleSet, Var
from onnxscript.rewriter.rules.fusion._layer_norm import layer_normalization_ruleset

from ..onnx._ir import (
    FlushDenormalsPass,
    const_array,
    const_ints,
    make_init,
    make_node,
    pointwise,
    producer_of,
    single_use,
    sole_consumer,
)
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
)
from . import _dsl

log = logging.getLogger(__name__)


def transform_detection(
    model: ir.Model,
    affine_folds: int,
    se_residuals: int,
    se_merges: int,
    gelus: int,
    head_scale: int = 1,
    asym_folds: int = 0,
    affine_scales: int = 0,
) -> ir.Model:
    return ir.passes.Sequential(
        ConvertOpsetPass(_dsl.OPSET),
        DeclareInputDimsPass({0: _dsl.Batch, 2: _dsl.Height, 3: _dsl.Width}),
        ReinferPass(),
        # first: stock Identity nodes block the BN-fusion patterns
        OptimizePass(),
        _FoldBiasAddsPass(),
        _FuseBatchNormPass(require_all=True),
        FoldInputScalePass(scale=1.0 / 127.5, flip_channels=True),
        RewritePass([_DecomposeHardSwish.rule()]),
        _RelaxPoolCeilModePass(),
        FoldPointwiseConvsPass(),
        ReinferPass(),
        _FuseGeluPass(gelus),
        _FoldLearnableAffinePass(affine_folds),
        _FoldAffineScalePass(affine_scales),
        _RescaleDetHeadPass(head_scale),
        _FoldSeResidualPass(se_residuals),
        _MergeSeBranchesPass(se_merges),
        _FoldAsymmetricConvsPass(asym_folds),
        WrapPass(_dsl.det_preprocess, _dsl.det_postprocess),
        ReinferPass(),
        NameOutputDimsPass({"probs": [_dsl.Batch, _dsl.Height, _dsl.Width]}),
        FlushDenormalsPass(),  # last: the folds above are what set the weights that ship
        common_passes.CheckerPass(),
    )(model).model


def transform_recognition(
    model: ir.Model,
    affine_folds: int,
    layernorms: int,
    shape_domains: int,
    qkv_unpacks: int,
    gelus: int,
    affine_scales: int = 0,
    pool_affines: int = 0,
) -> ir.Model:
    return ir.passes.Sequential(
        ConvertOpsetPass(_dsl.OPSET),
        # the height is pinned: left symbolic, inference cannot prove the SVTR flatten collapses it
        DeclareInputDimsPass({0: _dsl.Batch, 2: _dsl.REC_HEIGHT, 3: _dsl.Width}),
        ReinferPass(),
        OptimizePass(),
        # require_all=False: the SVTR neck's BNs don't follow convs
        _FuseBatchNormPass(require_all=False),
        _FoldBiasAddsPass(),
        FoldInputScalePass(scale=1.0 / 127.5, flip_channels=True),
        RewritePass([_DecomposeHardSwish.rule()]),
        _RelaxPoolCeilModePass(),
        FoldPointwiseConvsPass(),
        ReinferPass(),
        _FuseGeluPass(gelus),
        # eliminate, then relocate, then partially fold: moving an affine changes what it neighbours, so
        # it goes after the folds that would retire it outright and before the one that only halves it
        _FoldLearnableAffinePass(affine_folds),
        _MoveAffinePastPoolPass(pool_affines),
        _FoldAffineScalePass(affine_scales),
        # before the two passes below: they key on attributes the stock graph leaves implicit
        common_passes.AddDefaultAttributesPass(),
        _FuseLayerNormPass(layernorms),
        _SimplifyRecShapeDomainPass(shape_domains),
        _RestructureSvtrAttentionPass(qkv_unpacks),
        _ElideCtcSoftmaxPass(),
        WrapPass(_dsl.rec_preprocess, _dsl.rec_postprocess),
        ReinferPass(),
        NameOutputDimsPass({"indices": [_dsl.Batch, _dsl.Seq], "probs": [_dsl.Batch, _dsl.Seq]}),
        FlushDenormalsPass(),
        common_passes.CheckerPass(),
    )(model).model


class _FoldBiasAdd(RewriteRuleClassBase):
    """Fold a per-channel Add into the conv's bias; Paddle spells the head deconv bias so, blocking BN fusion."""

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


class _FoldBiasAddsPass(RewritePass):
    def __init__(self) -> None:
        super().__init__(RewriteRuleSet([_FoldBiasAdd.rule()], commute=True))


class _DecomposeHardSwish(RewriteRuleClassBase):
    """HardSwish -> Mul(x, HardSigmoid), exact: CoreML and some NPUs have none and fragment the backbone."""

    def pattern(self, op: Any, x: Any) -> Any:
        return op.HardSwish(x)

    def rewrite(self, op: Any, x: Any) -> Any:
        return op.Mul(x, op.HardSigmoid(x, alpha=1.0 / 6.0, beta=0.5))


class _FuseErfGelu(RewriteRuleClassBase):
    """The erf-GELU chain -> Gelu, which the backends take natively but do not fuse for us. Constants
    compare rounded to their own dtype: a tolerance tight enough for fp32 misses fp16's sqrt(2)."""

    def pattern(self, op: Any, x: Any, sqrt2: Any, one: Any, half: Any) -> Any:
        return op.Mul(op.Mul(x, op.Add(op.Erf(op.Div(x, sqrt2)), one)), half)

    def check(self, context: Any, sqrt2: Any, one: Any, half: Any, **_: Any) -> MatchResult:
        result = MatchResult()
        for value, want in ((sqrt2, math.sqrt(2.0)), (one, 1.0), (half, 0.5)):
            const = value.const_value
            if const is None or const.size != 1:
                return result.fail("gelu constant is not a scalar constant")
            array = const.numpy()
            if array.reshape(()) != array.dtype.type(want):
                return result.fail("gelu constant is not the wanted value in the tensor's own dtype")
        return result

    def rewrite(self, op: Any, x: Any, **_: Any) -> Any:
        return op.Gelu(x, approximate="none")


class _FuseGeluPass(PinnedRewritePass):
    def __init__(self, expected: int) -> None:
        super().__init__(RewriteRuleSet([_FuseErfGelu.rule()], commute=True), expected, "erf-GELU chains into Gelu")


class _FoldSeResidual(RewriteRuleClassBase):
    """Add(x, Mul(x, gate)) -> Mul(x, Add(gate, 1)) on the RSE-FPN residual: the +1 moves onto the pooled
    gate, retiring a full-resolution pass. Not bit-exact, which is why no runtime does it for us."""

    def pattern(self, op: Any, x: Any, gate: Any) -> Any:
        return op.Add(x, op.Mul(x, gate, _outputs=["gated"]))

    def check(self, context: Any, gate: Any, gated: Any, **_: Any) -> MatchResult:
        result = MatchResult()
        if gate.shape is None or len(gate.shape) != 4 or list(gate.shape)[2:] != [1, 1]:
            return result.fail("gate is not a pooled [N,C,1,1] tensor")
        if len(gated.uses()) != 1:
            return result.fail("the gated product feeds consumers outside the residual")
        return result

    def rewrite(self, op: Any, x: Any, gate: Any, **_: Any) -> Any:
        one = ir.tensor(np.array(1.0, gate.dtype.numpy()), name=f"{gate.name}_one")
        return op.Mul(x, op.Add(gate, op.initializer(one)))


class _FoldSeResidualPass(PinnedRewritePass):
    def __init__(self, expected: int) -> None:
        rules = RewriteRuleSet([_FoldSeResidual.rule()], commute=True)
        super().__init__(rules, expected, "SE residuals into their gate")


class _SeBranch(NamedTuple):
    pool: ir.Node
    down: ir.Node
    up: ir.Node
    gate: ir.Value  # the HardSigmoid output the block multiplies by


def _se_branches(graph: ir.Graph) -> list[_SeBranch]:
    branches = []
    for pool in graph:
        if pool.op_type != "GlobalAveragePool":
            continue
        chain, value = [], pool.outputs[0]
        for op_type in ("Conv", "Relu", "Conv", "HardSigmoid"):
            node = sole_consumer(value, op_type)
            if node is None:
                break
            chain.append(node)
            value = node.outputs[0]
        if len(chain) == 4 and all(pointwise(conv) and len(conv.inputs) > 2 for conv in (chain[0], chain[2])):
            branches.append(_SeBranch(pool, chain[0], chain[2], value))
    return branches


def _stack_weights(graph: ir.Graph, name: str, values: list[ir.Value]) -> ir.Value:
    return make_init(graph, name, np.concatenate([const_array(value) for value in values], axis=0))


class _MergeSeBranchesPass(ir.passes.InPlacePass):
    """Merge sibling SE branches into one grouped-conv pair (bit-exact: the grouped conv IS the
    block-diagonal of their GEMMs). No backend does it -- horizontal conv fusion wants a shared input."""

    def __init__(self, expected: int) -> None:
        self.expected, self.merged = expected, 0

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        order = {node: i for i, node in enumerate(graph)}
        groups: dict[Any, list[_SeBranch]] = {}
        for branch in _se_branches(graph):
            gate = branch.gate.producer()
            key = (
                const_array(branch.down.inputs[1]).shape,
                const_array(branch.up.inputs[1]).shape,
                gate.attributes.get_float("alpha"),
                gate.attributes.get_float("beta"),
            )
            groups.setdefault(key, []).append(branch)

        merged = 0
        for members in groups.values():
            if len(members) < 2:
                continue
            members.sort(key=lambda branch: order[branch.pool])
            count = len(members)
            tag = f"se_group{merged}"
            stacked = [
                _stack_weights(graph, f"{tag}_{conv}{index}", [getattr(b, conv).inputs[index] for b in members])
                for conv, index in (("down", 1), ("down", 2), ("up", 1), ("up", 2))
            ]
            attributes = dict(group=count, kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1])
            pooled = make_node("Concat", [branch.pool.outputs[0] for branch in members], out=f"{tag}_pooled", axis=1)
            down = make_node("Conv", [pooled.outputs[0], *stacked[:2]], out=f"{tag}_down", **attributes)
            relu = make_node("Relu", [down.outputs[0]], out=f"{tag}_relu")
            up = make_node("Conv", [relu.outputs[0], *stacked[2:]], out=f"{tag}_up", **attributes)
            gate = members[0].gate.producer()
            gates = make_node(
                "HardSigmoid",
                [up.outputs[0]],
                out=f"{tag}_gates",
                alpha=gate.attributes.get_float("alpha"),
                beta=gate.attributes.get_float("beta"),
            )
            width = const_array(members[0].up.inputs[1]).shape[0]
            sizes = make_init(graph, f"{tag}_sizes", np.full(count, width, np.int64))
            split = ir.node("Split", inputs=[gates.outputs[0], sizes], attributes={"axis": 1}, num_outputs=count)
            # independent siblings, so the last pool is an insertion point for all of them; consumers
            # left sitting before their producer stay that way until WrapPass sorts the graph
            graph.insert_after(members[-1].pool, [pooled, down, relu, up, gates, split])
            for branch, gate_out in zip(members, split.outputs):
                gate_out.name = f"{tag}_{branch.gate.name}"
                gate_out.shape, gate_out.type = branch.gate.shape, branch.gate.type
                branch.gate.replace_all_uses_with(gate_out)
            merged += 1

        log.info("Merged %d groups of sibling SE branches into grouped convs", merged)
        self.merged = merged
        return ir.passes.PassResult(model, bool(merged))

    def ensures(self, model: ir.Model) -> None:
        if self.merged != self.expected:
            raise ir.passes.PostconditionError(
                f"Merged {self.merged} groups of sibling SE branches, expected {self.expected}"
            )


class _AsymBlock(NamedTuple):
    square: ir.Node
    strips: list[ir.Node]
    adds: list[ir.Node]  # the binary Add tree summing the branches, root last


def _same_conv_geometry(a: ir.Node, b: ir.Node) -> bool:
    return all(
        list(a.attributes.get_ints(name, default)) == list(b.attributes.get_ints(name, default))
        for name, default in (("strides", [1, 1]), ("dilations", [1, 1]))
    ) and a.attributes.get_int("group", 1) == b.attributes.get_int("group", 1)


def _centered_pads(conv: ir.Node, kernel: list[int]) -> bool:
    """SAME padding for this kernel, so every branch of the group agrees on output extent."""
    pads = list(conv.attributes.get_ints("pads", [0, 0, 0, 0]))
    return pads == [(kernel[0] - 1) // 2, (kernel[1] - 1) // 2] * 2


def _sum_tree(branches: list[ir.Node]) -> list[ir.Node] | None:
    """The binary Add tree summing exactly these outputs, or None if any of them goes anywhere else."""
    pending = {branch.outputs[0] for branch in branches}
    if any(not single_use(value) for value in pending):
        return None
    adds: list[ir.Node] = []
    while len(pending) > 1:
        found = next(
            (
                use.node
                for value in pending
                for use in value.uses()
                if use.node.op_type == "Add"
                and len(use.node.inputs) == 2
                and all(operand in pending for operand in use.node.inputs)
            ),
            None,
        )
        if found is None:
            return None
        pending.difference_update(found.inputs)
        pending.add(found.outputs[0])
        adds.append(found)
        if len(pending) > 1 and not single_use(found.outputs[0]):
            return None
    return adds


def _asym_blocks(graph: ir.Graph) -> list[_AsymBlock]:
    by_source: dict[ir.Value, list[ir.Node]] = {}
    for node in graph:
        if node.op_type == "Conv" and node.inputs and node.inputs[0] is not None:
            by_source.setdefault(node.inputs[0], []).append(node)

    blocks = []
    for siblings in by_source.values():
        kernels = {node: list(node.attributes.get_ints("kernel_shape", [])) for node in siblings}
        square = [node for node in siblings if len(kernels[node]) == 2 and kernels[node][0] == kernels[node][1] > 1]
        if len(square) != 1:
            continue
        size = kernels[square[0]][0]
        strips = [node for node in siblings if kernels[node] in ([size, 1], [1, size])]
        members = [square[0], *strips]
        if len(strips) != 2 or kernels[strips[0]] == kernels[strips[1]]:
            continue
        if not all(_same_conv_geometry(square[0], node) and _centered_pads(node, kernels[node]) for node in members):
            continue
        if any(const_array(node.inputs[1]) is None for node in members):
            continue
        adds = _sum_tree(members)
        if adds is not None:
            blocks.append(_AsymBlock(square[0], strips, adds))
    return blocks


class _FoldAsymmetricConvsPass(ir.passes.InPlacePass):
    """Fold each KxK + Kx1 + 1xK branch group into one KxK conv -- the asymmetric-convolution
    reparameterization PP-OCR's LKPAN neck ships, summed with no activation between. Nothing collapses
    it for us: horizontal conv fusion wants identical kernels. Their SAME padding is exactly what
    aligns each strip to the square kernel's centre row or column. Accumulated in float64 so the three
    branches round once instead of three times."""

    def __init__(self, expected: int) -> None:
        self.expected, self.folded = expected, 0

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        for block in _asym_blocks(graph):
            size = list(block.square.attributes.get_ints("kernel_shape", []))[0]
            original = const_array(block.square.inputs[1])
            weight = original.astype(np.float64).copy()
            square_bias = const_array(block.square.inputs[2]) if len(block.square.inputs) > 2 else None
            bias = np.zeros(weight.shape[0], np.float64) if square_bias is None else square_bias.astype(np.float64)
            centre = (size - 1) // 2
            for strip in block.strips:
                strip_weight = const_array(strip.inputs[1]).astype(np.float64)
                if strip_weight.shape[-1] == 1:
                    weight[:, :, :, centre] += strip_weight[:, :, :, 0]
                else:
                    weight[:, :, centre, :] += strip_weight[:, :, 0, :]
                strip_bias = const_array(strip.inputs[2]) if len(strip.inputs) > 2 else None
                if strip_bias is not None:
                    bias = bias + strip_bias.astype(np.float64)

            name = block.square.name or block.square.outputs[0].name
            block.square.replace_input_with(1, make_init(graph, f"{name}_asym_w", weight.astype(original.dtype)))
            folded_bias = make_init(graph, f"{name}_asym_b", bias.astype(original.dtype))
            if len(block.square.inputs) > 2:
                block.square.replace_input_with(2, folded_bias)
            else:
                block.square.append_input(folded_bias)

            root = block.adds[-1].outputs[0]
            ir.convenience.replace_all_uses_with(root, block.square.outputs[0])
            for index, output in enumerate(graph.outputs):
                if output is root:
                    graph.outputs[index] = block.square.outputs[0]
            self.folded += 1

        if self.folded != self.expected:
            raise ir.passes.PostconditionError(f"Folded {self.folded} asymmetric conv groups, expected {self.expected}")
        if self.folded:
            common_passes.RemoveUnusedNodesPass()(model)  # the strips and their Add tree are now dead
        return ir.passes.PassResult(model, self.folded > 0)


class _RelaxPoolCeilModePass(ir.passes.InPlacePass):
    """Zero the redundant ceil_mode on SAME-padded pooling: CoreML rejects the pair and demotes the graph.
    A loop rather than a rule: a pattern cannot leave the op type open, and two pooling spellings qualify."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        relaxed = 0
        for node in model.graph:
            if node.op_type not in ("MaxPool", "AveragePool"):
                continue
            attributes = node.attributes
            if attributes.get_string("auto_pad", "NOTSET").startswith("SAME") and attributes.get_int("ceil_mode", 0):
                attributes["ceil_mode"] = ir.AttrInt64("ceil_mode", 0)
                relaxed += 1
        return ir.passes.PassResult(model, bool(relaxed))


def _squeeze_axes(node: ir.Node) -> list[int] | None:
    return const_ints(node.inputs[1]) if len(node.inputs) > 1 else None


class _HoistBatchNormOverSqueeze(RewriteRuleClassBase):
    """Hoist BatchNormalization back over a Squeeze to rank 4, where it folds into the preceding conv. ORT's
    CUDA fp16 BatchNorm reads the un-normalized shape's channel dim and kills the process at rank 3
    (core/providers/cuda/nn/batch_norm.cc)."""

    def __init__(self) -> None:
        super().__init__(remove_nodes=False)

    def pattern(self, op: Any, x: Any, scale: Any, bias: Any, mean: Any, var: Any) -> Any:
        squeezed = op.Squeeze(x, _allow_other_inputs=True, _allow_other_attributes=True, _outputs=["squeezed"])
        return op.BatchNormalization(
            squeezed, scale, bias, mean, var, _allow_other_attributes=True, _outputs=["normalized"]
        )

    def check(self, context: Any, x: Any, squeezed: Any, normalized: Any, **_: Any) -> MatchResult:
        result = MatchResult()
        if x.shape is None:
            return result.fail("Squeeze input has no inferred rank")
        axes = _squeeze_axes(squeezed.producer())
        if axes is None:
            return result.fail("Squeeze axes are not constant")
        if any(axis % len(x.shape) < 2 for axis in axes):
            return result.fail("Squeeze drops the batch or channel axis, so it does not commute with BN")
        if len(normalized.producer().outputs) != 1:
            return result.fail("BatchNormalization is in training mode")
        return result

    def rewrite(
        self, op: Any, x: Any, scale: Any, bias: Any, mean: Any, var: Any, squeezed: Any, normalized: Any, **_: Any
    ) -> Any:
        squeeze = squeezed.producer()
        hoisted = op.BatchNormalization(x, scale, bias, mean, var, **normalized.producer().attributes)
        return op.Squeeze(hoisted, *squeeze.inputs[1:], **squeeze.attributes)


_BN_RULES = RewriteRuleSet([_HoistBatchNormOverSqueeze.rule(), *BATCHNORM_FOLD_RULES])


class _FuseBatchNormPass(RewritePass):
    def __init__(self, require_all: bool) -> None:
        super().__init__(_BN_RULES)
        self.require_all = require_all

    def ensures(self, model: ir.Model) -> None:
        survivors = [node for node in model.graph if node.op_type == "BatchNormalization"]
        if self.require_all and survivors:
            raise ir.passes.PostconditionError(f"{len(survivors)} BatchNormalization ops did not fold into convs")
        ranks = {len(n.inputs[0].shape) for n in survivors if n.inputs[0].shape is not None} - {4}
        if ranks:
            raise ir.passes.PostconditionError(
                f"BatchNormalization survives at rank {sorted(ranks)}; ORT's CUDA fp16 kernel needs rank 4"
            )


def _scalar(value: ir.Value | None) -> float | None:
    const = value.const_value if value is not None else None
    if const is None or const.size != 1:
        return None
    return float(const.numpy().reshape(()))


def _single_use_const(value: ir.Value | None) -> bool:
    return value is not None and value.const_value is not None and single_use(value)


class _FoldAffineAfterConv(RewriteRuleClassBase):
    """Fold a scalar affine block Conv(x)->Mul(a)->Add(b) into the conv's weight and bias."""

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
    """Fold a post-activation affine Mul(a)->Add(b) into the following conv. Only at pads=0: a padded conv
    reads a zero border, not a*0+b, so the shift does not commute with the pad."""

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


class _MoveAffinePastPool(RewriteRuleClassBase):
    """`AveragePool(a*x + b) == a*AveragePool(x) + b`: the mean of an affine is the affine of the mean, so
    the pair can run on the pooled tensor instead of the full one. Holds while every window averages real
    elements only, which zero pads and `count_include_pad=0` give together -- with either absent, a window
    that reaches past the edge averages in a 0 that is not `a*0+b`."""

    def pattern(self, op: Any, x: Any, a: Any, b: Any) -> Any:
        affine = op.Add(op.Mul(x, a, _outputs=["mul"]), b, _outputs=["affine"])
        return op.AveragePool(affine, _allow_other_attributes=True, _outputs=["pool"])

    def check(self, context: Any, a: Any, b: Any, mul: Any, affine: Any, pool: Any, **_: Any) -> MatchResult:
        result = MatchResult()
        if _scalar(a) is None or _scalar(b) is None:
            return result.fail("affine scale/shift is not a scalar constant")
        if len(list(affine.uses())) != 1 or len(list(mul.uses())) != 1:
            return result.fail("affine feeds consumers besides the pool, so moving it would duplicate it")
        node = pool.producer()
        if not _explicit_zero_pads(node) or node.attributes.get_int("count_include_pad", 0):
            return result.fail("pool averages over padded positions")
        return result

    def rewrite(self, op: Any, x: Any, a: Any, b: Any, pool: Any, **_: Any) -> Any:
        return op.Add(op.Mul(op.AveragePool(x, **pool.producer().attributes), a), b)


class _MoveAffinePastPoolPass(PinnedRewritePass):
    """Relocation is worth less than elimination, so this runs on what the conv folds have already left."""

    def __init__(self, expected: int) -> None:
        rules = RewriteRuleSet([_MoveAffinePastPool.rule()], commute=True)
        super().__init__(rules, expected, "affine blocks past an average pool")


class _AffineSite(NamedTuple):
    add: ir.Node
    mul: ir.Node
    source: ir.Value  # the tensor the affine scales
    scale: float
    shift: float
    consumers: list[ir.Node]


def _affine_scale_sites(graph: ir.Graph) -> list[_AffineSite]:
    """Scalar `Mul(a)->Add(b)` blocks feeding convs only, that `_FoldAffineBeforeConv` will not take.

    It declines two shapes, and both are here: a PADDED consumer (its zero border does not carry the
    shift), and MULTIPLE consumers (a rewrite rule sees one). PP-OCRv5_mobile's remaining blocks are
    exactly these -- three of them fan out to a padded 3x3 and an unpadded 1x1 at once. The SE gate's
    block stays: its GlobalAveragePool consumer is not a conv, so there are no weights to carry."""
    sites = []
    for add in graph:
        if add.op_type != "Add" or len(add.inputs) < 2:
            continue
        mul = next((producer_of(operand, "Mul") for operand in add.inputs[:2] if producer_of(operand, "Mul")), None)
        shift = next((_scalar(operand) for operand in add.inputs[:2] if _scalar(operand) is not None), None)
        if mul is None or shift is None or not single_use(mul.outputs[0]):
            continue
        scale_value = next((operand for operand in mul.inputs[:2] if _scalar(operand) is not None), None)
        source = next((operand for operand in mul.inputs[:2] if operand is not scale_value), None)
        scale = _scalar(scale_value)
        if scale is None or source is None or scale == 0.0:
            continue
        consumers = [use.node for use in add.outputs[0].uses()]
        # every consumer has to be a conv with its own weight: a shared one would be scaled twice
        if not consumers or any(node.op_type != "Conv" for node in consumers):
            continue
        if any(len(node.inputs) < 2 or not _single_use_const(node.inputs[1]) for node in consumers):
            continue
        if len(consumers) == 1 and _explicit_zero_pads(consumers[0]):
            continue
        sites.append(_AffineSite(add, mul, source, scale, shift, consumers))
    return sites


class _FoldAffineScalePass(ir.passes.InPlacePass):
    """Move a leftover affine's SCALE into every consumer conv's weights, leaving the shift as an Add.

    `Conv(a*x + b, W) == Conv(x + b/a, a*W)` at ANY padding: both spellings zero-pad in the same
    places, and at a border tap `sum a*W*(x + b/a) == sum W*(a*x + b)`. The shift rides the weights it
    is actually multiplied by instead of a bias the pad never sees, which is why this holds where
    folding into the bias does not. Retires the Mul; the Add stays."""

    def __init__(self, expected: int) -> None:
        self.expected, self.folded = expected, 0

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        for site in _affine_scale_sites(graph):
            dtype = const_array(site.consumers[0].inputs[1]).dtype
            for conv in site.consumers:
                weight = conv.inputs[1]
                array = const_array(weight)
                scaled = (array.astype(np.float64) * site.scale).astype(array.dtype)
                conv.replace_input_with(1, make_init(graph, f"{weight.name}_labscale", scaled))

            name = site.add.outputs[0].name
            offset = make_init(graph, f"{name}_laboffset", np.array(site.shift / site.scale, dtype))
            shift = make_node("Add", [site.source, offset], out=f"{name}_shifted")
            graph.insert_before(site.mul, shift)
            for conv in site.consumers:
                conv.replace_input_with(0, shift.outputs[0])
            self.folded += 1

        if self.folded != self.expected:
            raise ir.passes.PostconditionError(f"Folded {self.folded} leftover affine scales, expected {self.expected}")
        if self.folded:
            common_passes.RemoveUnusedNodesPass()(model)
        return ir.passes.PassResult(model, self.folded > 0)


class _FoldLearnableAffinePass(PinnedRewritePass):
    """Fold the foldable scalar affine-block pairs into adjacent convs; SE-boundary and padded pairs stay."""

    def __init__(self, expected: int) -> None:
        # commute: PP-OCR emits the affine scale at Mul input 0, shift at Add input 1
        rules = RewriteRuleSet([_FoldAffineAfterConv.rule(), _FoldAffineBeforeConv.rule()], commute=True)
        super().__init__(rules, expected, "learnable-affine-block pairs into convs")

    def requires(self, model: ir.Model) -> None:
        scanned = _count_foldable_lab(model.graph)
        if scanned != self.expected:
            raise ir.passes.PreconditionError(
                f"Found {scanned} foldable learnable-affine pairs, expected {self.expected}"
            )


def _count_foldable_lab(graph: ir.Graph) -> int:
    """Count the affine pairs the rules must fold, by plain graph walking. The predicates below duplicate
    the rules' own `check()` deliberately: sharing them would blind rules and witness at once."""

    def scalar(value: ir.Value | None) -> bool:
        const = value.const_value if value is not None else None
        return const is not None and const.size == 1

    def sole_const(value: ir.Value | None) -> bool:
        return value is not None and value.const_value is not None and len(tuple(value.uses())) == 1

    def foldable(conv: ir.Node | None) -> bool:
        return conv is not None and len(conv.inputs) > 2 and sole_const(conv.inputs[1]) and sole_const(conv.inputs[2])

    count = 0
    for mul in graph:
        if mul.op_type != "Mul":
            continue
        a = next((value for value in mul.inputs if scalar(value)), None)
        if a is None:
            continue
        x = mul.inputs[1] if mul.inputs[0] is a else mul.inputs[0]
        add = sole_consumer(mul.outputs[0], "Add")
        if add is None or not any(scalar(v) for v in add.inputs if v is not mul.outputs[0]):
            continue
        if single_use(x) and foldable(producer_of(x, "Conv")):
            count += 1
            continue
        conv = sole_consumer(add.outputs[0], "Conv")
        if foldable(conv) and _explicit_zero_pads(conv):
            count += 1
    return count


def _explicit_zero_pads(node: ir.Node) -> bool:
    attributes = node.attributes
    return (
        "pads" in attributes
        and list(attributes.get_ints("pads")) == [0, 0, 0, 0]
        and not attributes.get_string("auto_pad", "").startswith("SAME")
    )


# f(c*x) == c*f(x) for c > 0, given every data input is scaled
_HOMOGENEOUS = {"Add", "Concat", "Relu", "Resize"}


def _data_inputs(node: ir.Node) -> list[ir.Value | None]:
    """The inputs a rescale propagates through; weights and Resize's roi/scales/sizes are not data."""
    return [node.inputs[0]] if node.op_type in ("Conv", "ConvTranspose", "Resize") else list(node.inputs)


class _RescaleDetHeadPass(ir.passes.InPlacePass):
    """Split the fp16 range between the DBNet neck and head: divide the head's input cone by `scale` and
    multiply the head conv's weights back. Folding the head's BN leaves some necks overflowing fp16 while
    the head's own weights go subnormal."""

    def __init__(self, scale: int) -> None:
        self.scale = scale

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        scale = self.scale
        if scale == 1:
            return ir.passes.PassResult(model, False)
        graph = model.graph
        heads = [
            node
            for node in graph
            if node.op_type == "Conv"
            and (concat := node.inputs[0].producer()) is not None
            and concat.op_type == "Concat"
            and len(concat.inputs) == 4
        ]
        if len(heads) != 1:
            raise ValueError(f"Expected one head Conv fed by the 4-level Concat, found {len(heads)}")
        head = heads[0]

        scaled = {head.inputs[0]}
        changed = True
        while changed:
            changed = False
            for node in graph:
                if node is head:
                    continue
                outputs, inputs = set(node.outputs), set(_data_inputs(node))
                if (inputs & scaled and not outputs <= scaled) or (
                    outputs & scaled and node.op_type in _HOMOGENEOUS and not inputs <= scaled
                ):
                    scaled |= outputs | (inputs if node.op_type in _HOMOGENEOUS else set())
                    changed = True
        if scaled & set(graph.outputs):
            raise ValueError("The head's input cone reaches a graph output; rescaling it would be observable")

        def rescale(value: ir.Value, factor: float) -> None:
            # const_array is None on anything but a weight, and the AttributeError that follows is the guard
            array = const_array(value)
            value.const_value = ir.tensor((array.astype(np.float64) * factor).astype(array.dtype), name=value.name)

        entries = 0
        for node in graph:
            if node is head or not set(node.outputs) & scaled:
                continue
            if node.op_type in _HOMOGENEOUS:
                continue
            if node.op_type not in ("Conv", "ConvTranspose"):
                raise ValueError(f"{node.op_type} in the head's input cone is not positively homogeneous")
            entry = node.inputs[0] not in scaled
            entries += entry
            for value in node.inputs[1:] if entry else node.inputs[2:]:
                rescale(value, 1.0 / scale)
        rescale(head.inputs[1], float(scale))
        log.info("Rescaled the head's input cone (%d values, %d entry convs) by 1/%d", len(scaled), entries, scale)
        return ir.passes.PassResult(model, True)


class _ElideCtcSoftmaxPass(ir.passes.InPlacePass):
    """Delete the CTC head's trailing Softmax; the fused head recomputes the winning probability itself."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
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
                raise ValueError(
                    f"{value.name} must feed exactly one {op_type}, found {[u.node.op_type for u in uses]}"
                )
            return uses[0].node

        if output.uses():
            raise ValueError(f"Recognition output feeds {[u.node.name for u in output.uses()]}, expected no consumers")
        tail = output.producer()
        if tail is not None and tail.op_type == "Softmax":
            if tail.attributes.get_int("axis", -1) not in (2, -1):
                raise ValueError(
                    f"CTC-head Softmax is over axis {tail.attributes.get_int('axis', -1)}, expected the last"
                )
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
        # dangling helpers are swept by WrapPass' optimizer DCE
        log.info("Elided the CTC head's trailing Softmax (graph output is now raw logits)")
        return ir.passes.PassResult(model, True)


class _FuseLayerNormPass(ir.passes.InPlacePass):
    """Fuse the SVTR neck's LayerNorm decomposition, for the EPs that match the raw graph (ORT re-fuses)."""

    def __init__(self, expected: int) -> None:
        self.expected = expected

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        fused = layer_normalization_ruleset.apply_to_model(model)
        log.info("Fused %d LayerNormalization ops", fused)
        return ir.passes.PassResult(model, bool(fused))

    def ensures(self, model: ir.Model) -> None:
        fused = sum(1 for node in model.graph if node.op_type == "LayerNormalization")
        if fused != self.expected:
            raise ir.passes.PostconditionError(f"Fused {fused} LayerNormalization ops, expected {self.expected}")


class _FlattenToSqueeze(RewriteRuleClassBase):
    """Squeeze(axis=2) for the [B,C,1,W] flatten; the shape chain stays (remove_nodes=False) for the unflatten."""

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
        if extras and not (const_ints(extras[0]) == [0] and (len(extras) < 2 or const_ints(extras[1]) == [1])):
            return result.fail("Slice is not a plain [0:2] window")
        if data.shape is None or len(data.shape) != 4 or data.shape[2] != 1:
            return result.fail("data is not [B,C,1,W]")
        return result

    def rewrite(self, op: Any, data: Any, **_: Any) -> Any:
        return op.Squeeze(data, op.initializer(ir.tensor(np.array([2], np.int64), name="rec_squeeze_axis")))


class _UnflattenToUnsqueeze(RewriteRuleClassBase):
    """Reshape(x, Concat([0],[1],W,[C])) on [B,W,C] with only width dynamic is Unsqueeze(axis=1)."""

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
        if const_ints(c) != [data.shape[2]]:
            return result.fail("target channel entry does not match the data's channel dim")
        return result

    def rewrite(self, op: Any, data: Any, **_: Any) -> Any:
        return op.Unsqueeze(data, op.initializer(ir.tensor(np.array([1], np.int64), name="rec_unsqueeze_axis")))


class _ElideNoopUnsqueezeAddSqueeze(RewriteRuleClassBase):
    """Squeeze(Add(Unsqueeze(x,[0]), 0.0), [0]) -> x. The zero is bound and checked, not written as a
    pattern literal: the graph spells it rank-1 `[0.0]`, which a literal (a scalar) silently misses."""

    def __init__(self) -> None:
        super().__init__(remove_nodes=False)

    def pattern(self, op: Any, x: Any, zero: Any) -> Any:
        return op.Squeeze(op.Add(op.Unsqueeze(x, [0]), zero), [0])

    def check(self, context: Any, zero: Any, **_: Any) -> MatchResult:
        result = MatchResult()
        if zero.const_value is None:
            return result.fail("add operand is not constant")
        value = zero.const_value.numpy()
        if value.size != 1 or value.reshape(-1)[0] != 0:
            return result.fail("add operand is not a single zero")
        return result

    def rewrite(self, op: Any, x: Any, **_: Any) -> Any:
        return op.Identity(x)


class _SimplifyRecShapeDomainPass(ir.passes.Sequential):
    """Devitalize rec's flatten/unflatten shape domain: only width is dynamic, so both reshapes are rank
    ops, and dropping their runtime targets kills the Shape op that blocks the CoreML EP on rec."""

    def __init__(self, expected: int) -> None:
        super().__init__(
            PinnedRewritePass([_FlattenToSqueeze.rule()], expected, "SVTR flattens into a Squeeze"),
            PinnedRewritePass([_UnflattenToUnsqueeze.rule()], expected, "SVTR unflattens into an Unsqueeze"),
            # unpinned: the quirk is v5-only, and each removal is provably exact
            RewritePass(RewriteRuleSet([_ElideNoopUnsqueezeAddSqueeze.rule()], commute=True)),
        )


class _QkvUnpack(NamedTuple):
    packed: ir.Value
    reshape: ir.Node
    branches: list[ir.Node]  # the three Squeezes, in packing order (q, k, v)
    heads: int
    head_dim: int


def _qkv_unpack_sites(graph: ir.Graph) -> list[_QkvUnpack]:
    """The SVTR block's hand-rolled packed-QKV unpack: rank-5 Reshape -> Transpose -> Slice+Squeeze x3."""
    sites = []
    for reshape in graph:
        target = const_ints(reshape.inputs[1]) if reshape.op_type == "Reshape" else None
        if target is None or len(target) != 5 or target[:3] != [0, -1, 3]:
            continue
        transpose = sole_consumer(reshape.outputs[0], "Transpose")
        if transpose is None or list(transpose.attributes.get_ints("perm")) != [2, 0, 3, 1, 4]:
            continue
        branches: dict[int, ir.Node] = {}
        for use in transpose.outputs[0].uses():
            if use.node.op_type != "Slice":
                continue
            starts, ends = const_ints(use.node.inputs[1]), const_ints(use.node.inputs[2])
            axes = const_ints(use.node.inputs[3]) if len(use.node.inputs) > 3 else [0]
            squeeze = sole_consumer(use.node.outputs[0], "Squeeze")
            if starts is None or ends is None or axes != [0] or ends[0] - starts[0] != 1:
                continue
            if squeeze is not None and _squeeze_axes(squeeze) == [0]:
                branches[starts[0]] = squeeze
        if sorted(branches) != [0, 1, 2]:
            continue
        sites.append(_QkvUnpack(reshape.inputs[0], reshape, [branches[i] for i in range(3)], target[3], target[4]))
    return sites


def _fold_qkv_scale(graph: ir.Graph, sites: list[_QkvUnpack]) -> int:
    """Fold the attention's Mul(1/sqrt(head_dim)) into the packed projection's columns for that branch."""
    folded = 0
    for site in sites:
        add = producer_of(site.packed, "Add")
        matmul = producer_of(add.inputs[0], "MatMul") if add is not None else None
        if add is None or matmul is None:
            continue
        weight, bias = matmul.inputs[1], add.inputs[1]
        w, b = const_array(weight), const_array(bias)
        if w is None or b is None or not (_single_use_const(weight) and _single_use_const(bias)):
            continue
        width = site.heads * site.head_dim
        for i, squeeze in enumerate(site.branches):
            mul = sole_consumer(squeeze.outputs[0], "Mul")
            scale = _scalar(next(v for v in mul.inputs if v is not squeeze.outputs[0])) if mul is not None else None
            if scale is None:
                continue
            scaled_w, scaled_b = w.astype(np.float64), b.astype(np.float64)
            scaled_w[:, i * width : (i + 1) * width] *= scale
            scaled_b[i * width : (i + 1) * width] *= scale
            matmul.replace_input_with(1, make_init(graph, f"{weight.name}_qkv_scaled", scaled_w.astype(w.dtype)))
            add.replace_input_with(1, make_init(graph, f"{bias.name}_qkv_scaled", scaled_b.astype(b.dtype)))
            mul.outputs[0].replace_all_uses_with(squeeze.outputs[0])
            graph.remove(mul, safe=True)
            folded += 1
            break
    return folded


def _unpack_qkv(graph: ir.Graph, sites: list[_QkvUnpack]) -> int:
    """Replace the rank-5 unpack with Reshape -> Transpose -> Split on the head axis: the projection is laid
    out (qkv, head, head_dim), so chunk i IS projection i and no per-branch reshape is needed."""
    for site in sites:
        name = site.packed.name
        shape = np.array([0, 0, 3 * site.heads, site.head_dim], np.int64)
        reshape = make_node(
            "Reshape", [site.packed, make_init(graph, f"{name}_qkv_shape", shape)], out=f"{name}_qkv", allowzero=0
        )
        transpose = make_node("Transpose", [reshape.outputs[0]], out=f"{name}_qkv_heads", perm=[0, 2, 1, 3])
        sizes = make_init(graph, f"{name}_qkv_sizes", np.full(3, site.heads, np.int64))
        split = ir.node("Split", inputs=[transpose.outputs[0], sizes], attributes={"axis": 1}, num_outputs=3)
        graph.insert_before(site.reshape, [reshape, transpose, split])
        for squeeze, tag, projection in zip(site.branches, "qkv", split.outputs):
            projection.name = f"{name}_{tag}"
            squeeze.outputs[0].replace_all_uses_with(projection)
    return len(sites)


class _RestructureSvtrAttentionPass(ir.passes.InPlacePass):
    """Rebuild the SVTR blocks' packed-QKV unpack, scale fold first: the fold reads the Squeeze->Mul the
    unpack deletes. Not an `Attention` op: at this head_dim CoreML's SDPA declines and EPs decompose back."""

    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.counts = (0, 0)

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        sites = _qkv_unpack_sites(model.graph)
        self.counts = (_fold_qkv_scale(model.graph, sites), _unpack_qkv(model.graph, sites))
        log.info("Restructured %d SVTR packed-QKV unpacks and folded their 1/sqrt(head_dim) scale", self.counts[1])
        return ir.passes.PassResult(model, bool(sites))

    def ensures(self, model: ir.Model) -> None:
        scaled, unpacked = self.counts
        if (scaled, unpacked) != (self.expected, self.expected):
            raise ir.passes.PostconditionError(
                f"SVTR attention restructure folded {scaled} scales and unpacked {unpacked} packed-QKV "
                f"projections, expected {self.expected} of each"
            )


def _is_shape_of(node: "ir.Node | None", data: ir.Value) -> bool:
    """True if node is a full Shape(data): no end, and start absent or the explicit default 0."""
    if node is None or node.op_type != "Shape" or node.inputs[0] is not data:
        return False
    return node.attributes.get_int("start", 0) == 0 and "end" not in node.attributes


__all__ = ["transform_detection", "transform_recognition"]
