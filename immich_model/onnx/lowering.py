"""Shared graph lowerings for backends that can't ingest the exported opset-23 graphs as-is.

Two consumers: rknn._onnx.prepare_for_rknn (AOT, opset-19 pin + RKNN extras) and runtime
(on-device rewrites for ORT EPs that mishandle fused ops; opset 23, batch stays dynamic).
"""

import math
from typing import Any

import numpy as np
import onnx_ir as ir
from onnxscript.rewriter.pattern import MatchResult, RewriteRuleClassBase

# patch-embed conv has kernel==stride >= this; a CNN stem (kernel 3/7) never reaches it -> selects only ViT patchify
_MIN_PATCH_STRIDE = 8


class FoldConstantGatherElements(RewriteRuleClassBase):
    """GatherElements(broadcast(const_row), const_indices) -> the gathered constant; exact for every
    batch (output shape is the constant indices shape, so the materialized batch dim never reaches it).
    Retires XLM-R's token-type materialization chain incl. the int32 batch-seed Slice island (the one
    island keeping XLM textual at two CoreML partitions), and dodges rknn-toolkit2 2.3.2's
    _p_gatherelements_to_einsum, which crashes (IndexError shape[2]) on the rank-2 form."""

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


class DecomposeGelu(RewriteRuleClassBase):
    """Gelu -> erf form [Mul(x,0.5), Div(x,sqrt2), Erf, Add(+1), Mul]; rknn-toolkit2 fuses this into
    exGelu/ConvExGelu. Erf is opset-9 so it clears the opset-19 cap (a surviving Gelu breaks the
    toolkit: fold_constant ORT refuses "Gelu with domain_version of 19"). approximate=tanh is
    DELIBERATELY decomposed to erf too: the tanh path routes through RKNPU's Tanh LUT (~0.08 rel-L2,
    collapsed fp16 SigLIP2 to cos 0.82), while erf<->tanh is <=3e-3 pointwise / <=3e-4 cos e2e
    (0.9997+). Bit-exact for approximate=none. RKNN-only: released ORT runs Gelu natively."""

    def pattern(self, op: Any, x: Any) -> Any:
        return op.Gelu(x, _domain="", _outputs=["gelu"])

    def rewrite(self, op: Any, x: Any, gelu: Any) -> Any:
        dtype = x.dtype.numpy() if x.dtype is not None else np.float32

        def const(value: float) -> Any:
            return op.Constant(value=ir.tensor(np.array(value, dtype)))

        half_x = op.Mul(x, const(0.5))  # 0.5 x first, per the fused pattern
        return op.Mul(half_x, op.Add(op.Erf(op.Div(x, const(math.sqrt(2.0)))), const(1.0)))


class DecomposeAttention(RewriteRuleClassBase):
    """opset-23 Attention(q,k,v,[mask]) -> pre-scaled Q, MatMul(Q',Kᵀ), [Add(mask)], Softmax(-1),
    MatMul(probs,V); this exact ordering is what rknn-toolkit2's fuse_..._to_sdpa matches -> exSDPAttention.
    Exact: Q'·Kᵀ = scale·(Q·Kᵀ). Boolean mask -> Where(mask,0,-1e4) (well-typed Add; -1e4 underflows
    softmax to 0). is_causal materialized as a constant [1,1,S,S] additive mask; the 3D [B,S,D] form is
    bracketed into the per-head 4D layout the chain expects, then inverted back to [B,S,D]."""

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


class PatchEmbedToMatMul(RewriteRuleClassBase):
    """Non-overlapping ViT patch-embed Transpose->Conv->Reshape->Transpose -> im2col Reshape/Transpose
    + MatMul + bias Add, reading NHWC directly. A Conv with kernel==stride and no pad IS im2col + a
    linear projection; as GEMM the NPU skips its quadrant-split conv lowering (4.2x faster e2e on RK3588)
    and slow-patchify GPU backends (OpenVINO on Arc, CUDA) run the plain matmul. Best-effort: doesn't
    fire unless the chain matches with static shapes. batch_dynamic=True folds batch into the leading -1
    of both Reshapes so one graph serves every batch; False hard-codes batch-1 shapes for the toolkit."""

    def __init__(self, batch_dynamic: bool = False, name: str | None = None) -> None:
        super().__init__(name)
        self.batch_dynamic = batch_dynamic

    def pattern(self, op: Any, x: Any, weight: Any, target_shape: Any) -> Any:
        nchw = op.Transpose(x, perm=[0, 3, 1, 2])
        conv = op.Conv(nchw, weight, _allow_other_inputs=True, _outputs=["conv"])
        return op.Transpose(op.Reshape(conv, target_shape), perm=[0, 2, 1])

    def check(self, context: Any, x: Any, weight: Any, target_shape: Any, conv: Any) -> MatchResult:
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
        if any(d != 1 for d in attributes.get_ints("dilations", [])):
            return result.fail("dilated conv is not a pure im2col")
        if attributes.get_int("group", 1) != 1:
            return result.fail("grouped conv is not a pure im2col")
        bias = conv.producer().inputs[2] if len(conv.producer().inputs) > 2 else None
        if bias is not None and bias.const_value is None:
            return result.fail("conv bias is not a constant")

        dims = x.shape
        if dims is None or len(dims) != 4:
            return result.fail("image is not a shaped NHWC tensor")
        _, height, width, channels = dims
        if not all(isinstance(d, int) for d in (height, width, channels)):
            return result.fail("image spatial dims are not static")
        if channels != w.shape[1] or height % kernel or width % kernel:
            return result.fail("image dims don't tile into the patch kernel")
        return result

    def rewrite(self, op: Any, x: Any, weight: Any, target_shape: Any, conv: Any) -> Any:
        w = weight.const_value.numpy()  # [embed, C, K, K]
        embed, channels, kernel, _ = w.shape
        _, height, width, _ = x.shape
        bias_value = conv.producer().inputs[2] if len(conv.producer().inputs) > 2 else None
        bias = bias_value.const_value.numpy() if bias_value is not None else np.zeros(embed, w.dtype)
        num_patches = (height // kernel) * (width // kernel)
        # patch-pixel order is (kh,kw,c); projection matches via transpose(kH,kW,C,embed)
        w_patch = w.transpose(2, 3, 1, 0).reshape(kernel * kernel * channels, embed)

        # named initializers: a model has one patchify, so names can't collide
        def init(array: np.ndarray, name: str) -> Any:
            return op.initializer(ir.tensor(array), name=name)

        # im2col: [B,H,W,C] -> [B(H/K),K,W/K,K*C] -> transpose -> [B(H/K),W/K,K,K*C] -> [B,np,K*K*C]
        if self.batch_dynamic:
            shape1 = np.array([-1, kernel, width // kernel, kernel * channels], np.int64)
            shape2 = np.array([-1, num_patches, kernel * kernel * channels], np.int64)
        else:  # the RKNN graphs are batch-1: keep every dim literal for the toolkit
            shape1 = np.array([height // kernel, kernel, width // kernel, kernel * channels], np.int64)
            shape2 = np.array([1, num_patches, kernel * kernel * channels], np.int64)
        grid = op.Reshape(x, init(shape1, "patch_reshape1_shape"))
        grouped = op.Transpose(grid, perm=[0, 2, 1, 3])
        cols = op.Reshape(grouped, init(shape2, "patch_reshape2_shape"))
        proj = op.MatMul(cols, init(w_patch.astype(w.dtype), "patch_weight"))
        return op.Add(proj, init(bias.astype(w.dtype).reshape(1, 1, embed), "patch_bias"))
