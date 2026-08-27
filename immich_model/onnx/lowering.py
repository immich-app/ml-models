"""Shared graph lowerings for backends that cannot ingest the exported graphs as-is; see runtime.REGISTRY."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import onnx_ir as ir
from onnxscript.rewriter import RewritePass
from onnxscript.rewriter.pattern import MatchResult, OrValue, RewriteRuleClassBase, RewriteRuleSet

# patch-embed conv has kernel==stride >= this; a CNN stem's 3 or 7 never reaches it
_MIN_PATCH_STRIDE = 8

_CTC_CLASS_AXIS = 2
_POOL_SEQ_AXIS = 1

# the exporter's merged SCRFD head, restated: it packs the channel axis anchor-major, 2 anchors of (cls, box, kps)
_SCRFD_ANCHORS_PER_CELL = 2
_SCRFD_HEAD_CHANNELS = (1, 4, 10)


class FoldConstantGatherElements(RewriteRuleClassBase):
    """GatherElements(broadcast(const_row), const_indices) -> the gathered constant. Retires the XLM-R token-type
    chain that fragments the CoreML partition, and dodges rknn-toolkit2's `_p_gatherelements_to_einsum` crash."""

    def pattern(self, op: Any, data: Any, indices: Any) -> Any:
        return op.GatherElements(data, indices, _outputs=["gathered"])

    @staticmethod
    def _const_row(value: Any) -> np.ndarray | None:
        def resolves_to_zero(v: Any) -> bool:
            wrappers = ("Unsqueeze", "Cast", "Reshape")
            while v is not None and v.producer() is not None and v.producer().op_type in wrappers:
                v = v.producer().inputs[0]
            if v is None:
                return False
            node = v.producer()
            if node is None:
                return v.const_value is not None and not np.any(v.const_value.numpy())
            if node.op_type == "Sub" and node.inputs[0] is node.inputs[1]:
                return True
            if node.op_type == "Mul":
                consts = [i.const_value for i in node.inputs if i is not None and i.const_value is not None]
                return any(c.size == 1 and not np.any(c.numpy()) for c in consts)
            return False

        for _ in range(3):  # row | Add(row, zeros) | Add(Add(row, zeros), batch_col)
            if value.const_value is not None:
                arr = value.const_value.numpy()
                return arr if arr.ndim == 2 and arr.shape[0] == 1 else None
            node = value.producer()
            if node is None or node.op_type != "Add":
                return None
            carriers = [i for i in node.inputs if not resolves_to_zero(i)]
            if len(carriers) == 1:
                value = carriers[0]
            elif not carriers:  # dedup can alias an all-zero row and its zeros broadcast: Add(x, x)
                value = node.inputs[0]
            else:
                return None
        return None

    def check(self, context: Any, data: Any, indices: Any, gathered: Any) -> MatchResult:
        result = MatchResult()
        if gathered.producer().attributes.get_int("axis", 0) != 1:
            return result.fail("GatherElements axis is not 1")
        idx = indices.const_value
        if idx is None or idx.dtype.numpy().kind not in "iu":
            return result.fail("indices are not an integer constant")
        idx_arr = idx.numpy()
        if idx_arr.ndim != 2 or idx_arr.shape[0] != 1:
            return result.fail("indices are not a [1,S] row")
        row = self._const_row(data)
        if row is None:
            return result.fail("data is not a (batch-materialized) constant [1,N] row")
        if idx_arr.min() < -row.shape[1] or idx_arr.max() >= row.shape[1]:
            return result.fail("indices out of range for the data row")
        return result

    def rewrite(self, op: Any, data: Any, indices: Any, gathered: Any) -> Any:
        row = self._const_row(data)
        folded = np.take_along_axis(row, indices.const_value.numpy(), axis=1)
        return op.Constant(value=ir.tensor(folded))


_ZERO_PASSTHROUGH = ("Cast", "Reshape", "Unsqueeze", "Squeeze", "Expand", "Identity", "Flatten", "Transpose")


def _resolves_to_zero(value: Any, depth: int = 0) -> bool:
    """True if `value` is provably an all-zero tensor. Structural, no evaluation."""
    if value is None or depth > 40:
        return False
    if value.const_value is not None:
        return not np.any(value.const_value.numpy())
    node = value.producer()
    if node is None:
        return False
    if node.op_type in _ZERO_PASSTHROUGH:
        return _resolves_to_zero(node.inputs[0], depth + 1)
    if node.op_type in ("Gather", "GatherElements", "GatherND"):
        return _resolves_to_zero(node.inputs[0], depth + 1)  # selecting from an all-zero tensor stays zero
    if node.op_type == "Add":
        return all(_resolves_to_zero(i, depth + 1) for i in node.inputs)
    if node.op_type == "Sub":
        return node.inputs[0] is node.inputs[1] or all(_resolves_to_zero(i, depth + 1) for i in node.inputs)
    if node.op_type == "Mul":
        return any(_resolves_to_zero(i, depth + 1) for i in node.inputs)
    if node.op_type == "ConstantOfShape":
        fill = node.attributes.get("value")
        return fill is None or not np.any(fill.value.numpy())  # default fill is 0
    return False


class FoldZeroIndexGather(RewriteRuleClassBase):
    """Gather(const_2d_float_table, zero_index, axis=0) -> the constant row 0, broadcast. The token-type ids are
    always 0, but dynamo emits them as a batch-materialized zeros chain RKNPU cannot constant-fold."""

    def pattern(self, op: Any, table: Any, index: Any) -> Any:
        return op.Gather(table, index, _outputs=["gathered"])

    def check(self, context: Any, table: Any, index: Any, gathered: Any) -> MatchResult:
        result = MatchResult()
        if gathered.producer().attributes.get_int("axis", 0) != 0:
            return result.fail("Gather axis is not 0")
        value = table.const_value
        if value is None or len(value.shape) != 2 or value.dtype.numpy().kind != "f":
            return result.fail("table is not a 2D float constant")
        if index.shape is None or len(index.shape) != 2:
            return result.fail("index is not a [batch, seq] row")
        if not _resolves_to_zero(index):
            return result.fail("index does not provably resolve to zero")
        return result

    def rewrite(self, op: Any, table: Any, index: Any, gathered: Any) -> Any:
        row = table.const_value.numpy()[0].reshape(1, 1, -1)  # [1,1,H]: broadcasts against [batch,seq,H]
        return op.Constant(value=ir.tensor(row))


class DecomposeGelu(RewriteRuleClassBase):
    """Gelu -> the erf form rknn-toolkit2 fuses into exGelu; a survivor breaks its constant folding at the opset-19
    cap. approximate=tanh decomposes to erf too, deliberately: the tanh path takes RKNPU's coarse Tanh LUT."""

    def pattern(self, op: Any, x: Any) -> Any:
        return op.Gelu(x, _domain="", _outputs=["gelu"])

    def rewrite(self, op: Any, x: Any, gelu: Any) -> Any:
        dtype = x.dtype.numpy() if x.dtype is not None else np.float32

        def const(value: float) -> Any:
            return op.Constant(value=ir.tensor(np.array(value, dtype)))

        half_x = op.Mul(x, const(0.5))  # 0.5 x first, per the fused pattern
        return op.Mul(half_x, op.Add(op.Erf(op.Div(x, const(math.sqrt(2.0)))), const(1.0)))


class DecomposeGeluPass(RewritePass):
    def __init__(self) -> None:
        super().__init__([DecomposeGelu.rule()])

    def ensures(self, model: ir.Model) -> None:
        if stuck := sum(1 for node in model.graph if node.domain == "" and node.op_type == "Gelu"):
            raise ir.passes.PostconditionError(f"{stuck} Gelu op(s) survived the decomposition")


class DecomposeReduceL2(RewriteRuleClassBase):
    """ReduceL2(x, axes) -> Sqrt(ReduceSum(Mul(x, x), axes)). CoreML-only, a partitioning fix not a kernel one: no
    released EP has a ReduceL2 builder."""

    def pattern(self, op: Any, x: Any, axes: Any) -> Any:
        return op.ReduceL2(x, axes, _domain="", _outputs=["norm"])

    def rewrite(self, op: Any, x: Any, axes: Any, norm: Any) -> Any:
        keepdims = norm.producer().attributes.get_int("keepdims", 1)
        return op.Sqrt(op.ReduceSum(op.Mul(x, x), axes, keepdims=keepdims))


class DecomposeReduceL2Pass(RewritePass):
    def __init__(self) -> None:
        super().__init__([DecomposeReduceL2.rule()])

    def ensures(self, model: ir.Model) -> None:
        if stuck := sum(1 for node in model.graph if node.domain == "" and node.op_type == "ReduceL2"):
            raise ir.passes.PostconditionError(f"{stuck} ReduceL2 op(s) survived the decomposition")


class DecomposePRelu(RewriteRuleClassBase):
    """PRelu(x, slope) -> Relu(x) + slope*(x - Relu(x)), IEEE-exact. MIGraphX-only, for a fusion reason: its MLIR
    pointwise allowlist omits prelu and the fused module is all-or-nothing, so one PRelu evicts the bias Add too."""

    def pattern(self, op: Any, x: Any, slope: Any) -> Any:
        return op.PRelu(x, slope)

    def rewrite(self, op: Any, x: Any, slope: Any) -> Any:
        relu = op.Relu(x)  # one Relu, two uses: the negative branch is x minus the positive one
        return op.Add(relu, op.Mul(slope, op.Sub(x, relu)))


class DecomposePReluPass(RewritePass):
    def __init__(self) -> None:
        super().__init__([DecomposePRelu.rule()])


class DecomposeAttention(RewriteRuleClassBase):
    """opset-23 Attention -> pre-scaled Q, MatMul(Q',Kᵀ), [Add(mask)], Softmax(-1), MatMul(probs,V); this exact
    ordering is what rknn-toolkit2's fuse_..._to_sdpa matches. Boolean masks and is_causal become additive ones."""

    def pattern(self, op: Any, q: Any, k: Any, v: Any) -> Any:
        # _allow_other_inputs also matches the optional 4th (mask) input, so one rule covers both
        return op.Attention(q, k, v, _domain="", _allow_other_inputs=True, _outputs=["attn"])

    def check(self, context: Any, q: Any, k: Any, v: Any, attn: Any) -> MatchResult:
        result = MatchResult()
        node = attn.producer()
        has_mask = len(node.inputs) > 3 and node.inputs[3] is not None
        if node.attributes.get_int("is_causal", 0) != 0 and has_mask:
            return result.fail("Attention with both is_causal and an explicit mask is not handled")
        if node.attributes.get_float("softcap", 0.0) != 0.0:
            return result.fail("Attention with softcap is not handled by the RKNN decomposition")
        return result

    @staticmethod
    def _static_dim(value: Any, axis: int, rank: int = 3) -> int | None:
        if value.shape is None or len(value.shape) != rank:
            return None
        dim = value.shape[axis]
        return int(dim) if isinstance(dim, int) and dim > 0 else None

    def rewrite(self, op: Any, q: Any, k: Any, v: Any, attn: Any) -> Any:
        node = attn.producer()
        mask = node.inputs[3] if len(node.inputs) > 3 else None
        is_causal = node.attributes.get_int("is_causal", 0) != 0
        dtype = q.dtype.numpy() if q.dtype is not None else np.float32
        heads = node.attributes.get_int("q_num_heads")
        three_d = heads is not None and (q.shape is None or len(q.shape) == 3)

        if three_d:  # bracket the 3D contract into the per-head 4D layout the chain below expects
            kv_heads = node.attributes.get_int("kv_num_heads")
            if kv_heads is not None and kv_heads != heads:
                raise RuntimeError(f"grouped-query Attention {node.name} is not handled by the RKNN decomposition")
            q_seq, kv_seq = self._static_dim(q, 1), self._static_dim(k, 1)
            width, v_width = self._static_dim(q, -1), self._static_dim(v, -1)
            if None in (q_seq, kv_seq, width, v_width) or width % heads or v_width % heads:
                raise RuntimeError(f"could not resolve 3D head layout for Attention producing {attn.name}")
            head_dim, v_head_dim = width // heads, v_width // heads

            def to_heads(x: Any, seq: int, dim: int) -> Any:
                target = op.Constant(value=ir.tensor(np.array([-1, seq, heads, dim], np.int64)))
                return op.Transpose(op.Reshape(x, target), perm=[0, 2, 1, 3])

            q, k, v = to_heads(q, q_seq, head_dim), to_heads(k, kv_seq, head_dim), to_heads(v, kv_seq, v_head_dim)
        else:
            dims = [s[-1] for s in (q.shape, k.shape) if s is not None]
            head_dim = next((int(s) for s in dims if isinstance(s, int) and s > 0), None)
            q_seq, kv_seq = self._static_dim(q, -2, rank=4), self._static_dim(k, -2, rank=4)

        scale = node.attributes.get_float("scale")  # explicit scale needs no head_dim (survives failed inference)
        if scale is None:
            if head_dim is None:
                raise RuntimeError(f"could not determine head_dim for Attention producing {attn.name}")
            scale = 1.0 / math.sqrt(head_dim)
        scores = op.MatMul(
            op.Mul(q, op.Constant(value=ir.tensor(np.array(scale, dtype)))),
            op.Transpose(k, perm=[0, 1, 3, 2]),
        )
        if is_causal:  # materialize is_causal as the constant additive mask the score Add expects
            if q_seq is None or kv_seq is None or q_seq != kv_seq:
                raise RuntimeError(
                    f"cannot build a causal mask for Attention producing {attn.name}: "
                    f"q/kv sequence lengths {q_seq}/{kv_seq} are not static and equal"
                )
            causal = np.triu(np.full((q_seq, q_seq), -1.0e4, dtype), k=1).reshape(1, 1, q_seq, q_seq)
            mask = op.Constant(value=ir.tensor(causal))
        if mask is not None:
            if mask.dtype == ir.DataType.BOOL:
                keep = op.Constant(value=ir.tensor(np.array(0.0, dtype)))
                fill = op.Constant(value=ir.tensor(np.array(-1.0e4, dtype)))
                mask = op.Where(mask, keep, fill)
            scores = op.Add(scores, mask)
        context = op.MatMul(op.Softmax(scores, axis=-1), v)
        if not three_d:
            return context
        out_target = op.Constant(value=ir.tensor(np.array([-1, q_seq, v_width], np.int64)))
        return op.Reshape(op.Transpose(context, perm=[0, 2, 1, 3]), out_target)


def _fused_attention_count(model: ir.Model) -> int:
    return sum(1 for node in model.graph if node.op_type == "Attention")


class DecomposeAttentionPass(RewritePass):
    """`DecomposeAttention` with its sites enforced: every gated target either miscomputes the fused op or cannot
    build it, so a survivor is a wrong or unbuildable graph rather than a missed optimisation."""

    def __init__(self) -> None:
        super().__init__([DecomposeAttention.rule()])

    def ensures(self, model: ir.Model) -> None:
        if stuck := _fused_attention_count(model):
            raise ir.passes.PostconditionError(f"{stuck} fused Attention op(s) survived the decomposition")


class ElideMaskQueryAxis(RewriteRuleClassBase):
    """Drop the zero-Add materializing a padding mask's query axis and feed [b,1,1,S] to the decomposed score Add,
    which broadcasts it itself; exact because the mask is square. Its own row: DecomposeAttention ships wider."""

    def pattern(self, op: Any, q: Any, k: Any, mask: Any, zeros: Any) -> Any:
        return op.Add(op.MatMul(q, k, _outputs=["scores"]), op.Add(mask, zeros))

    def check(self, context: Any, q: Any, k: Any, mask: Any, zeros: Any, scores: Any) -> MatchResult:
        result = MatchResult()
        fill = zeros.const_value
        if fill is None or np.any(fill.numpy()):
            return result.fail("the mask's second Add operand is not an all-zero constant")
        dims = mask.shape
        if dims is None or len(dims) != 4 or dims[-2] != 1 or not isinstance(dims[-1], int):
            return result.fail("the mask is not a rank-4 key row with a static unit query axis")
        stretch = [1] * (4 - len(fill.shape)) + list(fill.shape)
        if len(fill.shape) > 4 or stretch != [1, 1, dims[-1], 1]:
            return result.fail("the zero-Add stretches an axis other than the query axis")
        return result

    def rewrite(self, op: Any, q: Any, k: Any, mask: Any, zeros: Any, scores: Any) -> Any:
        return op.Add(scores, mask)


class PatchEmbedToMatMul(RewriteRuleClassBase):
    """Non-overlapping ViT patch-embed Conv -> im2col Reshape/Transpose + MatMul + bias Add: a Conv with kernel==stride
    and no pad IS im2col plus a linear projection, and as a GEMM it skips the NPU's quadrant-split conv lowering."""

    def __init__(self, batch_dynamic: bool = False, crop_ragged: bool = False, name: str | None = None) -> None:
        super().__init__(name)
        self.batch_dynamic = batch_dynamic  # False hard-codes batch 1 for the toolkit
        self.crop_ragged = crop_ragged

    def pattern(self, op: Any, nhwc: Any, nchw: Any, weight: Any, target_shape: Any) -> Any:
        # exactly one arm binds: nhwc = the image before the contract's layout transpose, nchw = the conv input
        image = OrValue([op.Transpose(nhwc, perm=[0, 3, 1, 2]), nchw])
        conv = op.Conv(image, weight, _allow_other_inputs=True, _outputs=["conv"])
        return op.Transpose(op.Reshape(conv, target_shape), perm=[0, 2, 1])

    @staticmethod
    def _image_dims(nhwc: Any, nchw: Any) -> tuple[int, int, int] | None:
        dims = (nhwc if nhwc is not None else nchw).shape
        if dims is None or len(dims) != 4:
            return None
        spatial = (dims[1], dims[2], dims[3]) if nhwc is not None else (dims[2], dims[3], dims[1])
        return spatial if all(isinstance(d, int) for d in spatial) else None

    def check(self, context: Any, nhwc: Any, nchw: Any, weight: Any, target_shape: Any, conv: Any) -> MatchResult:
        result = MatchResult()
        w = weight.const_value
        if w is None:
            return result.fail("patch weight is not a constant")
        if len(w.shape) != 4 or w.shape[2] != w.shape[3] or w.shape[2] < _MIN_PATCH_STRIDE:
            return result.fail("not a patch-sized square kernel")
        kernel = w.shape[2]

        attributes = conv.producer().attributes
        strides = attributes.get_ints("strides")
        if strides is None or list(strides) != [kernel, kernel]:
            return result.fail("stride != kernel, not a disjoint patchify")
        if any(attributes.get_ints("pads", [])):
            return result.fail("padded conv is not a pure im2col")
        if attributes.get_string("auto_pad", "NOTSET") != "NOTSET":
            return result.fail("auto_pad conv has implicit padding, not a pure im2col")
        if any(d != 1 for d in attributes.get_ints("dilations", [])):
            return result.fail("dilated conv is not a pure im2col")
        if attributes.get_int("group", 1) != 1:
            return result.fail("grouped conv is not a pure im2col")
        bias = conv.producer().inputs[2] if len(conv.producer().inputs) > 2 else None
        if bias is not None and bias.const_value is None:
            return result.fail("conv bias is not a constant")

        dims = self._image_dims(nhwc, nchw)
        if dims is None:
            return result.fail("image is not a rank-4 tensor with static spatial dims")
        height, width, channels = dims
        if channels != w.shape[1]:
            return result.fail("image channels don't match the patch kernel")
        # a ragged grid drops the trailing partial patch, so the crop is what makes the im2col reshape exact
        if bool(height % kernel or width % kernel) is not self.crop_ragged:
            return result.fail("image dims don't tile into the patch kernel")
        return result

    def rewrite(self, op: Any, nhwc: Any, nchw: Any, weight: Any, target_shape: Any, conv: Any) -> Any:
        w = weight.const_value.numpy()  # [embed, C, K, K]
        embed, channels, kernel, _ = w.shape
        height, width, _ = self._image_dims(nhwc, nchw)  # type: ignore[misc]  # check() proved it static
        x = nhwc if nhwc is not None else op.Transpose(nchw, perm=[0, 2, 3, 1])
        bias_value = conv.producer().inputs[2] if len(conv.producer().inputs) > 2 else None
        bias = bias_value.const_value.numpy() if bias_value is not None else np.zeros(embed, w.dtype)
        height, width = height - height % kernel, width - width % kernel  # no-op unless ragged
        num_patches = (height // kernel) * (width // kernel)
        w_patch = w.transpose(2, 3, 1, 0).reshape(kernel * kernel * channels, embed)

        # named initializers: a model has one patchify, so names can't collide
        def init(array: np.ndarray, name: str) -> Any:
            return op.initializer(ir.tensor(array), name=name)

        if self.crop_ragged:  # see check(): the crop is what makes the im2col reshape exact
            x = op.Slice(
                x,
                init(np.array([0, 0], np.int64), "crop_start"),
                init(np.array([height, width], np.int64), "crop_end"),
                init(np.array([1, 2], np.int64), "crop_axes"),
            )

        if self.batch_dynamic:
            shape1 = np.array([-1, kernel, width // kernel, kernel * channels], np.int64)
            shape2 = np.array([-1, num_patches, kernel * kernel * channels], np.int64)
        else:  # the RKNN graphs are batch-1: keep every dim literal for the toolkit
            shape1 = np.array([height // kernel, kernel, width // kernel, kernel * channels], np.int64)
            shape2 = np.array([1, num_patches, kernel * kernel * channels], np.int64)
        grid = op.Reshape(x, init(shape1, "patch_reshape1_shape"))
        grouped = op.Transpose(grid, perm=[0, 2, 1, 3])
        cols = op.Reshape(grouped, init(shape2, "patch_reshape2_shape"))
        # split_large_reduction needs known shapes to apply
        batch = ir.SymbolicDim(None) if self.batch_dynamic else 1
        dtype = weight.const_value.dtype
        cols.shape, cols.type = ir.Shape([batch, num_patches, kernel * kernel * channels]), ir.TensorType(dtype)
        proj = op.MatMul(cols, init(w_patch.astype(w.dtype), "patch_weight"))
        proj.shape, proj.type = ir.Shape([batch, num_patches, embed]), ir.TensorType(dtype)
        return op.Add(proj, init(bias.astype(w.dtype).reshape(1, 1, embed), "patch_bias"))


class PatchEmbedToMatMulPass(RewritePass):
    def __init__(self, batch_dynamic: bool = True, crop_ragged: bool = False) -> None:
        super().__init__([PatchEmbedToMatMul.rule(batch_dynamic=batch_dynamic, crop_ragged=crop_ragged)])


class FuseGreedyCtcTopK(RewriteRuleClassBase):
    """The greedy-CTC head scans the vocab axis twice on the same logits (ArgMax for the class, ReduceMax for the
    softmax-stability max); one exact TopK(k=1) returns both. CoreML rejects TopK and RKNN runs it slower."""

    def pattern(self, op: Any, logits: Any, axes: Any) -> Any:
        return op.ArgMax(logits, axis=2, keepdims=0), op.ReduceMax(logits, axes, keepdims=1)

    def check(self, context: Any, logits: Any, axes: Any) -> MatchResult:
        result = MatchResult()
        value = axes.const_value
        if value is None or list(value.numpy().reshape(-1)) != [2]:
            return result.fail("ReduceMax does not reduce the vocab axis (2)")
        return result

    def rewrite(self, op: Any, logits: Any, axes: Any) -> Any:
        k = op.initializer(ir.tensor(np.array([1], np.int64), name="ctc_topk_k"))
        values, indices = op.TopK(logits, k, axis=2, largest=1, sorted=1, _outputs=2)
        squeeze_axes = op.initializer(ir.tensor(np.array([2], np.int64), name="ctc_topk_squeeze_axes"))
        return op.Squeeze(indices, squeeze_axes), values


class FuseGreedyCtcTopKPass(RewritePass):
    def __init__(self) -> None:
        super().__init__([FuseGreedyCtcTopK.rule()])


class FuseHardSwish(RewriteRuleClassBase):
    """`Mul(x, HardSigmoid(x, alpha=1/6, beta=1/2))` -> `HardSwish(x)`, exact by definition, so the attribute check
    is the whole correctness argument -- the identical HardSigmoid over a DIFFERENT tensor is an SE gate, not this."""

    _ALPHA, _BETA, _TOL = 1.0 / 6.0, 0.5, 1e-6

    def pattern(self, op: Any, x: Any) -> Any:
        return op.Mul(x, op.HardSigmoid(x, _outputs=["gate"]))

    def check(self, context: Any, x: Any, gate: Any) -> MatchResult:
        result = MatchResult()
        attributes = gate.producer().attributes
        alpha = attributes.get_float("alpha", 0.2)  # ONNX defaults, not the hard-swish ones
        beta = attributes.get_float("beta", 0.5)
        if abs(alpha - self._ALPHA) > self._TOL or abs(beta - self._BETA) > self._TOL:
            return result.fail("HardSigmoid is not the (1/6, 1/2) hard-swish gate")
        if gate.is_graph_output() or len(gate.uses()) != 1:  # retiring a shared gate would break its other reader
            return result.fail("the HardSigmoid feeds more than this Mul")
        return result

    def rewrite(self, op: Any, x: Any, gate: Any) -> Any:
        return op.HardSwish(x)


class FuseHardSwishPass(RewritePass):
    def __init__(self) -> None:
        # commute: upstream varies which operand carries the gate, within one graph
        super().__init__(RewriteRuleSet([FuseHardSwish.rule()], commute=True))

    def ensures(self, model: ir.Model) -> None:
        if stuck := _unfused_hard_swish_count(model):
            raise ir.passes.PostconditionError(f"{stuck} hard-swish site(s) survived the fusion")


def _unfused_hard_swish_count(model: ir.Model) -> int:
    """Hard-swish sites `FuseHardSwish` had to fuse and did not; blind to which Mul operand carries the gate."""
    stuck = 0
    for node in model.graph:
        if node.op_type != "Mul" or len(node.inputs) != 2:
            continue
        for idx, gate in enumerate(node.inputs):
            if gate is None:
                continue
            producer = gate.producer()
            if producer is None or producer.op_type != "HardSigmoid":
                continue
            attributes = producer.attributes
            alpha = attributes.get_float("alpha", 0.2)  # ONNX defaults, not the hard-swish ones
            beta = attributes.get_float("beta", 0.5)
            if abs(alpha - FuseHardSwish._ALPHA) > FuseHardSwish._TOL:
                continue
            if abs(beta - FuseHardSwish._BETA) > FuseHardSwish._TOL:
                continue
            if producer.inputs[0] is not node.inputs[1 - idx]:  # an SE gate over a different tensor
                continue
            stuck += not gate.is_graph_output() and len(gate.uses()) == 1
    return stuck


def _corner_slice(node: ir.Node, rank: int) -> bool:
    """The zero-column seed `x[:, 0:1, 0:1, ...]`, read off every non-batch axis so a permutation still matches."""
    if node.op_type != "Slice" or len(node.inputs) != 4:
        return False
    consts = [value.const_value if value is not None else None for value in node.inputs[1:4]]
    if any(const is None for const in consts):
        return False
    starts, ends, axes = ([int(v) for v in const.numpy().reshape(-1)] for const in consts)  # type: ignore[union-attr]
    return starts == [0] * (rank - 1) and ends == [1] * (rank - 1) and axes == list(range(1, rank))


def _channel_shift(value: ir.Value, channels: int) -> ir.Node | None:
    """A sole-consumer Add/Sub/Mul/Div over a private constant on the NHWC channel axis, between `value` and the
    layout Transpose. Only the padded CLIP stems keep one: a shift does not commute with a zero pad."""
    uses = list(value.uses())
    if len(uses) != 1:
        return None
    node = uses[0].node
    if node.op_type not in ("Add", "Sub", "Mul", "Div") or len(node.inputs) != 2:
        return None
    if node.inputs[0] is not value or len(node.outputs) != 1:
        return None
    const = node.inputs[1]
    tensor = const.const_value if const is not None else None
    # rank<=1 only ever broadcasts over the last (channel) axis
    if tensor is None or tensor.shape.rank() > 1 or len(list(const.uses())) != 1:  # type: ignore[union-attr]
        return None
    return node if tensor.size in (1, channels) else None


class NchwImageInputPass(ir.passes.InPlacePass):
    """Retype the uint8 NHWC image input to NCHW and delete the layout Transpose, handing the backend what its
    first conv wants. Mutates graph IO: the caller must feed NCHW."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        declined = ir.passes.PassResult(model, False)
        graph = model.graph
        image = graph.inputs[0] if graph.inputs else None
        if image is None or image.dtype != ir.DataType.UINT8 or image.shape is None or len(image.shape) != 4:
            return declined
        dims = list(image.shape)
        channels = dims[3]
        if not isinstance(channels, int) or not 0 < channels <= 4:
            return declined
        uses = list(image.uses())
        if len(uses) != 1 or uses[0].node.op_type != "Cast":
            return declined
        nhwc = [uses[0].node]
        shift = _channel_shift(nhwc[0].outputs[0], channels)
        if shift is not None:
            nhwc.append(shift)
        head = nhwc[-1].outputs[0]
        consumers = {use.node for use in head.uses()}
        transposes = [
            n for n in consumers if n.op_type == "Transpose" and list(n.attributes.get_ints("perm", [])) == [0, 3, 1, 2]
        ]
        if len(transposes) != 1 or any(n is not transposes[0] and not _corner_slice(n, 4) for n in consumers):
            return declined
        transpose = transposes[0]
        if transpose.outputs[0].is_graph_output() or head.is_graph_output():
            return declined

        nchw = [dims[0], channels, dims[1], dims[2]]
        image.shape = ir.Shape(nchw)
        for node in nhwc:  # the only stale annotations: everything downstream was already NCHW
            node.outputs[0].shape = ir.Shape(nchw)
        if shift is not None:
            const = shift.inputs[1]
            array = const.const_value.numpy().reshape(-1, 1, 1)  # type: ignore[union-attr]
            const.const_value = ir.tensor(array, name=const.name)
            const.shape = ir.Shape(array.shape)
        transpose.outputs[0].replace_all_uses_with(head)
        graph.remove(transpose, safe=True)
        return ir.passes.PassResult(model, True)


def _const_ints(value: ir.Value | None) -> list[int] | None:
    tensor = value.const_value if value is not None else None
    return [int(v) for v in tensor.numpy().reshape(-1)] if tensor is not None else None


def _reduce_axes(node: ir.Node) -> list[int] | None:
    return _const_ints(node.inputs[1]) if len(node.inputs) > 1 else None


def _axis(axes: list[int] | None, rank: int) -> int | None:
    """The single axis `axes` names, normalised against `rank`; exporters spell it -1 or 1, so a literal fails."""
    if axes is None or len(axes) != 1:
        return None
    return axes[0] + rank if axes[0] < 0 else axes[0]


def _walk_back(value: ir.Value, op_types: Sequence[str]) -> tuple[ir.Value, list[ir.Node]] | None:
    """Walk back from `value` through `op_types`, consumer first; None unless every step is privately consumed."""
    nodes: list[ir.Node] = []
    for depth, op_type in enumerate(op_types):
        if depth and (value.is_graph_output() or len(value.uses()) != 1):
            return None
        node = value.producer()
        if node is None or node.op_type != op_type or len(node.outputs) != 1:
            return None
        if not node.inputs or node.inputs[0] is None:
            return None
        nodes.append(node)
        value = node.inputs[0]
    return value, nodes


class HostCtcDecodePass(ir.passes.InPlacePass):
    """Retire the OCR-recognition greedy-CTC head for a last-axis Softmax; the host argmaxes. Mutates graph IO:
    2 outputs -> 1. The Softmax is load-bearing: the export elides the stock one, the fused head recomputing it."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        declined = ir.passes.PassResult(model, False)
        graph = model.graph
        if len(graph.outputs) != 2:
            return declined
        indices, probs = graph.outputs
        # fp16 too: the derived artifact's `probs` IS fp16, and gating on fp32 would decline it in silence
        if indices.dtype != ir.DataType.INT32 or probs.dtype not in (ir.DataType.FLOAT, ir.DataType.FLOAT16):
            return declined

        prob_producer = probs.producer()
        casts_probs = prob_producer is not None and prob_producer.op_type == "Cast"
        index_chain = _walk_back(indices, ("Cast", "ArgMax"))
        prob_chain = _walk_back(probs, (*(("Cast",) if casts_probs else ()), "Reciprocal", "ReduceSum", "Exp", "Sub"))
        if index_chain is None or prob_chain is None:
            return declined
        logits, index_nodes = index_chain
        shifted_logits, prob_nodes = prob_chain
        argmax, reduce_sum, sub = index_nodes[-1], prob_nodes[-3], prob_nodes[-1]
        max_logits = sub.inputs[1]
        reduce_max = max_logits.producer() if max_logits is not None else None
        if max_logits is None or reduce_max is None:
            return declined
        if (
            logits is not shifted_logits
            or logits.dtype not in (ir.DataType.FLOAT, ir.DataType.FLOAT16)
            or logits.shape is None
            or len(logits.shape) != 3
            or argmax.attributes.get_int("axis", 0) != _CTC_CLASS_AXIS
            or argmax.attributes.get_int("keepdims", 1) != 0
            or argmax.attributes.get_int("select_last_index", 0) != 0  # numpy argmax takes the first
            or _reduce_axes(reduce_sum) != [_CTC_CLASS_AXIS]
            or reduce_sum.attributes.get_int("keepdims", 1) != 0
            or reduce_max.op_type != "ReduceMax"
            or reduce_max.inputs[0] is not logits
            or _reduce_axes(reduce_max) != [_CTC_CLASS_AXIS]
            or reduce_max.attributes.get_int("keepdims", 1) != 1
            or max_logits.is_graph_output()
            or len(max_logits.uses()) != 1  # the Sub, so removing the ReduceMax is safe
        ):
            return declined

        # the output keeps the logits' own dtype: an fp16 graph hands back fp16, like every other output
        softmax = ir.node("Softmax", inputs=[logits], attributes={"axis": _CTC_CLASS_AXIS}, num_outputs=1)
        softmax.outputs[0].shape, softmax.outputs[0].type = logits.shape, logits.type
        output = softmax.outputs[0]
        output.name = "logits_softmax"
        graph.append(softmax)
        graph.outputs.clear()
        graph.outputs.append(output)
        for node in (*index_nodes, *prob_nodes, reduce_max):
            graph.remove(node, safe=True)
        return ir.passes.PassResult(model, True)


_MS_DOMAIN = "com.microsoft"
_FUSED_SKIP_DTYPES = (ir.DataType.FLOAT, ir.DataType.FLOAT16)


class FuseSkipLayerNorm(RewriteRuleClassBase):
    """Residual `Add` + `LayerNormalization` -> `com.microsoft::SkipLayerNormalization`, the sum coming back off the
    4th output. ORT's own fusion wants the Add single-use, true of post-norm towers and never of pre-norm ones."""

    def __init__(self, shared: bool, name: str | None = None) -> None:
        super().__init__(name)
        self.shared = shared

    def pattern(self, op: Any, x: Any, skip: Any, gamma: Any, beta: Any) -> Any:
        total = op.Add(x, skip, _outputs=["total"])
        normed = op.LayerNormalization(total, gamma, beta, _allow_other_attributes=True, _outputs=["normed"])
        return (total, normed) if self.shared else normed

    def check(self, context: Any, x: Any, skip: Any, gamma: Any, beta: Any, total: Any, normed: Any) -> MatchResult:
        result = MatchResult()
        norm = normed.producer()
        if normed.is_graph_output() or total.is_graph_output():
            return result.fail("the norm or the sum it reads is a graph output")
        if self.shared != any(use.node is not norm for use in total.uses()):
            return result.fail("the sum's reader count belongs to the other arm")
        if len(norm.outputs) != 1 or norm.attributes.get_int("stash_type", 1) != 1:
            return result.fail("LayerNormalization emits its training statistics or stashes a wider dtype")
        if norm.attributes.get_int("axis", -1) not in (-1, 2) or total.dtype not in _FUSED_SKIP_DTYPES:
            return result.fail("not an fp32/fp16 norm over the last of three axes")
        if any(v.shape is None or len(v.shape) != 3 or v.dtype != total.dtype for v in (x, skip, total)):
            return result.fail("the sum and its two addends are not rank-3 tensors of one dtype")
        if any(v.shape is None or len(v.shape) != 1 for v in (gamma, beta)):
            return result.fail("scale/bias are not rank-1")
        return result

    def rewrite(self, op: Any, x: Any, skip: Any, gamma: Any, beta: Any, total: Any, normed: Any) -> Any:
        fused = op.SkipLayerNormalization(
            x,
            skip,
            gamma,
            beta,
            epsilon=normed.producer().attributes.get_float("epsilon", 1e-5),
            _domain=_MS_DOMAIN,
            _outputs=4 if self.shared else 1,
        )
        return (fused[3], fused[0]) if self.shared else fused


class FuseSkipLayerNormPass(RewritePass):
    """Apply `FuseSkipLayerNorm` and blank the two training-only outputs its shared arm declares only to reach the
    third: ORT allocates a buffer for every NAMED output, and the ruleset's NameFixPass renames anything anonymous."""

    def __init__(self) -> None:
        super().__init__([FuseSkipLayerNorm.rule(shared=False), FuseSkipLayerNorm.rule(shared=True)])

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        result = super().call(model)
        for node in model.graph:
            if node.op_type == "SkipLayerNormalization" and len(node.outputs) == 4:
                node.outputs[1].name = node.outputs[2].name = ""
        return result


class KeepdimsMeanPool(RewriteRuleClassBase):
    """Keep the reduced axis on the masked mean-pool's ReduceSums and Squeeze after the Div, so no shape transform
    stands between a reduction and its consumer for MIGraphX's `simplify_reshapes` to rewrite across."""

    def pattern(self, op: Any, weighted: Any, mask: Any, sum_axes: Any, count_axes: Any, unsq_axes: Any) -> Any:
        summed = op.ReduceSum(weighted, sum_axes, _allow_other_attributes=True, _outputs=["summed"])
        count = OrValue(
            [
                op.Unsqueeze(
                    op.ReduceSum(mask, count_axes, _allow_other_attributes=True, _outputs=["dropped"]), unsq_axes
                ),
                op.ReduceSum(mask, count_axes, _allow_other_attributes=True, _outputs=["kept"]),
            ],
            name="count",
        )
        return op.Div(summed, count, _outputs=["pooled"])

    def check(
        self,
        context: Any,
        weighted: Any,
        mask: Any,
        unsq_axes: Any,
        summed: Any,
        count: Any,
        pooled: Any,
        dropped: Any = None,  # the arm that did not match binds no output at all, not None
        kept: Any = None,
        **_: Any,
    ) -> MatchResult:
        result = MatchResult()
        sum_reduce = summed.producer()
        count_reduce = (dropped if dropped is not None else kept).producer()
        if summed.is_graph_output() or count.is_graph_output():
            return result.fail("a Div operand is a graph output, so the reduction cannot be re-shaped")
        if unsq_axes is not None and _axis(_const_ints(unsq_axes), 2) != _POOL_SEQ_AXIS:
            return result.fail("the count's Unsqueeze does not rebuild the sequence axis")
        pool_axes = (_axis(_reduce_axes(sum_reduce), 3), _axis(_reduce_axes(count_reduce), 2))
        if any(axis != _POOL_SEQ_AXIS for axis in pool_axes):
            return result.fail("sum and count do not both reduce the sequence axis")
        if sum_reduce.attributes.get_int("keepdims", 1) != 0:
            return result.fail("the weighted sum already keeps the axis it reduces")
        # the count reaches [batch, 1] either way: kept by the Unsqueeze, or by keepdims itself
        if count_reduce.attributes.get_int("keepdims", 1) != (0 if dropped is not None else 1):
            return result.fail("the count's keepdims contradicts the arm that matched it")
        if weighted.shape is None or len(weighted.shape) != 3 or mask.shape is None or len(mask.shape) != 2:
            return result.fail("not a rank-3 weighted sum over a rank-2 mask")
        if pooled.shape is None or len(pooled.shape) != 2 or pooled.is_graph_output():
            return result.fail("the pooled value is not an internal rank-2 embedding")
        return result

    def rewrite(
        self,
        op: Any,
        weighted: Any,
        mask: Any,
        sum_axes: Any,
        count_axes: Any,
        unsq_axes: Any,
        summed: Any,
        count: Any,
        pooled: Any,
        dropped: Any = None,
        kept: Any = None,
        **_: Any,
    ) -> Any:
        def keeping(node: ir.Node) -> dict[str, Any]:
            return {**node.attributes, "keepdims": 1}

        def like(value: ir.Value, template: ir.Value, dims: list[Any]) -> ir.Value:
            """`value` typed as `template` and shaped `dims`; an untyped one is dropped from value_info."""
            value.type, value.shape = template.type, ir.Shape(dims)
            return value

        counted = dropped if dropped is not None else kept
        batch, width = pooled.shape
        total = like(op.ReduceSum(weighted, sum_axes, **keeping(summed.producer())), summed, [batch, 1, width])
        # open_clip's arm has no Unsqueeze axes to reuse, so it borrows the numerator's; `check` proved them equal
        column = like(
            op.Unsqueeze(
                like(op.ReduceSum(mask, count_axes, **keeping(counted.producer())), counted, [batch, 1]),
                unsq_axes if unsq_axes is not None else sum_axes,
            ),
            count,
            [batch, 1, 1],
        )
        return op.Squeeze(like(op.Div(total, column), pooled, [batch, 1, width]), sum_axes)


class BroadcastShapeWorkaroundPass(RewritePass):
    """Both edits, in either order; either one alone still fails to compile. See the registry gate."""

    def __init__(self) -> None:
        super().__init__([ElideMaskQueryAxis.rule(remove_nodes=False), KeepdimsMeanPool.rule()])

    def requires(self, model: ir.Model) -> None:
        # the mask half anchors on the score Add `DecomposeAttentionPass` produces, so it must run first
        if stuck := _fused_attention_count(model):
            raise ir.passes.PreconditionError(f"{stuck} fused Attention op(s) are still to be decomposed")

    def ensures(self, model: ir.Model) -> None:
        """Sites this had to move and did not, per half: in one sum a live half masks a dead one."""
        if stuck := _unkept_mean_pool_count(model) + _live_zero_mask_add_count(model):
            raise ir.passes.PostconditionError(f"{stuck} broadcast-shape site(s) survived the workaround")


def _unkept_mean_pool_count(model: ir.Model) -> int:
    """Masked mean-pools `KeepdimsMeanPool` had to reach and did not; reads the numerator, spelled alike by both."""
    stuck = 0
    for div in model.graph:
        if div.op_type != "Div" or len(div.inputs) != 2:
            continue
        summed = div.inputs[0]
        if summed is None:
            continue
        reduce = summed.producer()
        if reduce is None or reduce.op_type != "ReduceSum" or not reduce.inputs or reduce.inputs[0] is None:
            continue
        weighted = reduce.inputs[0]
        if weighted.shape is None or len(weighted.shape) != 3:
            continue
        if _axis(_reduce_axes(reduce), 3) == _POOL_SEQ_AXIS and reduce.attributes.get_int("keepdims", 1) == 0:
            stuck += 1
    return stuck


def _live_zero_mask_add_count(model: ir.Model) -> int:
    """Zero-Adds `ElideMaskQueryAxis` had to bypass and did not; blind to the operand order the rule pins."""
    stuck = 0
    for node in model.graph:
        if node.op_type != "Add" or len(node.inputs) != 2:
            continue
        producers = [value.producer() for value in node.inputs if value is not None]
        if not any(p is not None and p.op_type == "MatMul" for p in producers):
            continue
        for adder in producers:
            if adder is None or adder.op_type != "Add":
                continue
            fills = [value.const_value for value in adder.inputs if value is not None]
            if any(fill is not None and not np.any(fill.numpy()) for fill in fills):
                stuck += 1
    return stuck


class SymmetrizeConvPads(RewriteRuleClassBase):
    """Widen an even-kernel stride-1 SAME_UPPER conv to the next odd kernel with symmetric pads, zero-extending the
    weights: MIGraphX splits an asymmetrically-padded conv into a standalone `pad` plus a zero-padded conv."""

    def pattern(self, op: Any, x: Any, weight: Any) -> Any:
        return op.Conv(x, weight, _allow_other_inputs=True, _allow_other_attributes=True, _outputs=["conv"])

    def check(self, context: Any, weight: Any, conv: Any, **_: Any) -> MatchResult:
        result = MatchResult()
        attributes = conv.producer().attributes
        if attributes.get_string("auto_pad", "NOTSET") != "SAME_UPPER":
            return result.fail("the conv does not carry SAME_UPPER padding")
        kernel = list(attributes.get_ints("kernel_shape", []))
        steps = [step for name in ("strides", "dilations") for step in attributes.get_ints(name, [])]
        if not kernel or any(size % 2 for size in kernel) or any(step != 1 for step in steps):
            return result.fail("not an even kernel at unit stride and dilation")
        # a shared weight would be widened once and read at two kernel sizes
        if weight.const_value is None or len(weight.uses()) != 1:
            return result.fail("the weight is not a privately-owned constant")
        return result

    def rewrite(self, op: Any, x: Any, weight: Any, conv: Any, **_: Any) -> Any:
        node = conv.producer()
        kernel = list(node.attributes.get_ints("kernel_shape"))
        widened = np.pad(weight.const_value.numpy(), [(0, 0), (0, 0)] + [(1, 0)] * len(kernel))
        attributes = {
            **node.attributes,
            "kernel_shape": [size + 1 for size in kernel],
            "pads": [size // 2 for size in kernel] * 2,
        }
        del attributes["auto_pad"]
        odd = op.initializer(ir.tensor(widened, name=f"{weight.name}_symmetric"))
        return op.Conv(x, odd, *node.inputs[2:], _name=node.name, **attributes)


class SymmetrizeConvPadsPass(RewritePass):
    def __init__(self) -> None:
        super().__init__([SymmetrizeConvPads.rule()])

    def ensures(self, model: ir.Model) -> None:
        if stuck := _asymmetric_pad_count(model):
            raise ir.passes.PostconditionError(f"{stuck} asymmetrically padded conv(s) survived the widening")


def _asymmetric_pad_count(model: ir.Model) -> int:
    """Convs `SymmetrizeConvPads` had to widen and did not; blind to whether the padding is spelled `auto_pad`."""
    stuck = 0
    for node in model.graph:
        if node.op_type != "Conv" or len(node.inputs) < 2:
            continue
        weight = node.inputs[1]
        shape = weight.shape if weight is not None else None
        if weight is None or shape is None or len(shape) < 3 or weight.const_value is None:
            continue
        kernel = [dim for dim in shape[2:] if isinstance(dim, int)]
        if len(kernel) != len(shape) - 2 or not kernel or any(size % 2 for size in kernel):
            continue
        steps = [step for name in ("strides", "dilations") for step in node.attributes.get_ints(name, [])]
        if any(step != 1 for step in steps) or len(weight.uses()) != 1:
            continue
        auto_pad = node.attributes.get_string("auto_pad", "NOTSET")
        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):  # even kernel: SAME puts it all on one side
            stuck += 1
            continue
        pads = list(node.attributes.get_ints("pads", [0] * 2 * len(kernel)))
        stuck += pads[: len(pads) // 2] != pads[len(pads) // 2 :]
    return stuck


class UnpackScrfdHeads(RewriteRuleClassBase):
    """Cut the merged SCRFD head's branches apart while the channels are still an axis: [B,A*C,H,W] -> [B,A,C,HW]
    is a free view of the anchor-major pack. Gated, not exported: it trades one wide Transpose for three narrow."""

    def pattern(self, op: Any, x: Any, flat: Any, sizes: Any) -> Any:
        rows = op.Reshape(op.Transpose(x, perm=[0, 2, 3, 1]), flat)
        return op.Split(rows, sizes, axis=2, _outputs=3)

    def check(self, context: Any, x: Any, flat: Any, sizes: Any) -> MatchResult:
        result = MatchResult()
        cell = sum(_SCRFD_HEAD_CHANNELS)
        if _const_ints(sizes) != list(_SCRFD_HEAD_CHANNELS):
            return result.fail("the Split does not cut a SCRFD head's cls/box/kps channels")
        if _const_ints(flat) != [0, -1, cell]:
            return result.fail("the Reshape does not flatten the head to per-anchor rows")
        dims = x.shape
        if dims is None or len(dims) != 4 or dims[1] != _SCRFD_ANCHORS_PER_CELL * cell:
            return result.fail("the head conv does not emit a whole cell of anchor-major head channels")
        return result

    def rewrite(self, op: Any, x: Any, flat: Any, sizes: Any) -> Any:
        cell = np.array([0, _SCRFD_ANCHORS_PER_CELL, sum(_SCRFD_HEAD_CHANNELS), -1], np.int64)
        branches = op.Split(op.Reshape(x, op.Constant(value=ir.tensor(cell))), sizes, axis=2, _outputs=3)
        return tuple(
            op.Reshape(
                op.Transpose(branch, perm=[0, 3, 1, 2]),
                op.Constant(value=ir.tensor(np.array([0, -1, channels], np.int64))),
            )
            for branch, channels in zip(branches, _SCRFD_HEAD_CHANNELS)
        )
