"""Per-target graph rewrites for released onnxruntime and rknn-toolkit2. The rewritten graph is a separate small .onnx
sharing the source's weight sidecar, not an in-memory patch: CoreML's compiled-model cache keys on file path."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import onnx_ir as ir
from onnx_ir.passes.common import DeduplicateInitializersPass, RemoveUnusedNodesPass, TopologicalSortPass
from onnxscript.rewriter import RewritePass

from .onnx._ir import CanonicalizeConstantsPass, ReinferShapesPass, save_with_external_data
from .onnx.lowering import (
    BroadcastShapeWorkaroundPass,
    DecomposeAttentionPass,
    DecomposeGeluPass,
    DecomposePReluPass,
    DecomposeReduceL2Pass,
    FoldConstantGatherElements,
    FuseGreedyCtcTopKPass,
    FuseHardSwishPass,
    FuseSkipLayerNormPass,
    HostCtcDecodePass,
    NchwImageInputPass,
    PatchEmbedToMatMulPass,
    SymmetrizeConvPadsPass,
)
from .rknn._onnx import (
    FloatifyNotEqual,
    FloatifyPadKeep,
    FloatifyPadMaskPass,
    FloatImageInputPass,
    OpaqueZeroMul,
    PinOpsetPass,
    RawCtcLogitsPass,
    SplitLargeReduction,
    Uint8ImageInputPass,
)

RKNPU = "RKNPU"  # the one target that is not an ORT execution provider: its rows run in `rknn compile`

# Hashed into the rewrite-plan digest alongside the package version, target and ordered rewrite names, all of
# which already move it. Bump ONLY when a row's behaviour changes with its name and gates fixed.
REWRITE_SET_VERSION = 22


@dataclass(frozen=True)
class RewriteContext:
    target: str
    ort_version: tuple[int, ...]


@dataclass(frozen=True)
class Rewrite:
    name: str
    # target -> what would retire this row there: documentation held as data so it cannot drift off its row.
    # Mostly toolkit versions an ORT-version bound cannot express, so do not reintroduce a version field.
    gates: Mapping[str, str]
    # a row whose miss is not survivable says so in its own pass's `ensures()`, hand-walked because a pattern
    # would reproduce the silent non-match it exists to catch; otherwise a miss costs latency
    transform: ir.passes.PassBase

    def applies(self, ctx: RewriteContext) -> bool:
        return ctx.target in self.gates


@dataclass(frozen=True)
class RewritePlan:
    rewrites: tuple[Rewrite, ...]
    digest: str

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(rewrite.name for rewrite in self.rewrites)


REGISTRY = (
    Rewrite(
        # first only because it must precede decompose_attention: it anchors on the fused op's 4th input
        name="floatify_pad_mask",
        gates={
            RKNPU: "a librknnrt with an int32 Equal kernel, which is what the whole mask island dies on",
        },
        transform=FloatifyPadMaskPass(),
    ),
    Rewrite(
        name="decompose_attention",
        gates={
            "CoreMLExecutionProvider": "an ORT release with the upstreamed CoreML Attention/is_causal builders",
            "MIGraphXExecutionProvider": "a MIGraphX release that parses opset-23 Attention and has a fused kernel",
            "OpenVINOExecutionProvider": "OpenVINO >= 2026.3, whose 3D Attention translation works",
            # both TensorRT EPs miscompute a single-row query against many K/V rows -- not head_dim 72, as an
            # earlier note here blamed -- and the decomposed form mirrors it above batch 1, where none runs
            "TensorrtExecutionProvider": "nothing version-shaped: 10.13 through 11.1 all fail identically",
            "NvTensorRTRTXExecutionProvider": "nothing version-shaped: 10.13 through 11.1 all fail identically",
            RKNPU: "an rknn-toolkit2 that parses the op at the opset it lives in; the chain IS its sdpa matcher's",
        },
        transform=DecomposeAttentionPass(),
    ),
    Rewrite(
        name="broadcast_shape_workaround",
        gates={
            # two upstream bugs, both hit, so both edits are required: simplify_reshapes throws an unchecked
            # multibroadcast at the mean-pool divide, simplify_algebra rank-promotes the mask ill-formed
            "MIGraphXExecutionProvider": "a MIGraphX release carrying fixes for both upstream bugs",
        },
        transform=BroadcastShapeWorkaroundPass(),
    ),
    Rewrite(
        name="decompose_reduce_l2",
        gates={
            # the release EP registers no ReduceL2 builder, so the norm and its Casts fall to a CPU partition
            "CoreMLExecutionProvider": "an ORT release carrying the upstreamed CoreML ReduceL2 builder",
        },
        transform=DecomposeReduceL2Pass(),
    ),
    Rewrite(
        name="decompose_prelu",
        gates={
            "MIGraphXExecutionProvider": "prelu joining MIGraphX's MLIR pointwise allowlist",
        },
        transform=DecomposePReluPass(),
    ),
    Rewrite(
        name="symmetrize_conv_pads",
        gates={
            # asymmetric padding makes MIGraphX insert launch-bound pad kernels around the conv
            "MIGraphXExecutionProvider": "a MIGraphX conv lowering that takes asymmetric padding, as its pooling does",
        },
        transform=SymmetrizeConvPadsPass(),
    ),
    Rewrite(
        name="greedy_ctc_topk",
        gates={
            "CUDAExecutionProvider": "a cuDNN bump that fixes reduce_tensor occupancy",
            "TensorrtExecutionProvider": "nothing: already neutral, Myelin fuses ArgMax->TopK itself",
            "NvTensorRTRTXExecutionProvider": "nothing: already neutral, same Myelin fusion",
        },
        transform=FuseGreedyCtcTopKPass(),
    ),
    Rewrite(
        name="hard_swish",
        gates={
            # fuses away a launch and an activation round-trip per site
            "CUDAExecutionProvider": "an ORT release whose own optimizers fuse the pair before the EP sees it",
            "OpenVINOExecutionProvider": "an OpenVINO release registering HardSigmoidDecomposition on x86",
        },
        transform=FuseHardSwishPass(),
    ),
    Rewrite(
        # never CoreML: mutually exclusive with the more valuable nchw_image_input, since im2col consumes the
        # layout Transpose that rewrite matches on, leaving the graph on CoreML's slow float-NHWC input path
        name="im2col_patchify",
        gates={
            # a 32x32/stride-32 filter falls off every NVIDIA tensor-core path, which the GEMM stays on
            "CUDAExecutionProvider": "a CUDA bump that flips the conv-vs-GEMM balance",
            "TensorrtExecutionProvider": "TensorRT no longer paying for the 32x32 filter",
            "NvTensorRTRTXExecutionProvider": "the same cliff clearing, here a batch-invariant `correlation` layer",
            "OpenVINOExecutionProvider": "intel_gpu fixing its patchify-conv kernels",
        },
        transform=PatchEmbedToMatMulPass(),
    ),
    Rewrite(
        name="im2col_patchify_ragged",
        gates={
            # ragged = the patch grid does not tile the image, so the conv silently drops an edge and a plain
            # im2col reshape would be wrong; slicing to the tiling sub-region first makes the same body exact
            "OpenVINOExecutionProvider": "an intel_gpu release whose selector stops preferring os_iyx_osv32 here",
        },
        transform=PatchEmbedToMatMulPass(crop_ragged=True),
    ),
    Rewrite(
        # ahead of the input rows, which delete the layout Transpose it reads its NHWC view from
        name="im2col_patchify_batch1",
        gates={
            RKNPU: "an RKNPU conv lowering that stops quadrant-splitting a kernel==stride patch conv",
        },
        transform=PatchEmbedToMatMulPass(batch_dynamic=False),
    ),
    Rewrite(
        name="skip_layer_norm",
        gates={
            # only the EPs that have the kernel AND run level-2 fusions over their own nodes; ORT's own
            # SkipLayerNormFusion reaches few of the sites (why, in fuse_skip_layer_norm)
            "CPUExecutionProvider": "an ORT SkipLayerNormFusion that emits the sum on output 3 rather than bailing",
            # the ORT floor in pyproject is this row's: below it the CUDA fp16 kernel comes back as beta (#28682)
            "CUDAExecutionProvider": "an ORT SkipLayerNormFusion that emits the sum on output 3 rather than bailing",
            # NEVER OpenVINO: it translates the op in its frontend and implements output 0 only, so the
            # 4-output form every pre-norm tower needs either fails session create or grows phantom
            # Parameters. A 1-output variant is safe but pointless, expanding to the unfused pair's nodes.
        },
        transform=FuseSkipLayerNormPass(),
    ),
    Rewrite(
        name="nchw_image_input",
        gates={
            # CoreML declines the uint8 Cast and its float NHWC input path is slow; the caller transposes
            "CoreMLExecutionProvider": "a CoreML EP whose float NHWC rank-4 input path stops being the slow one",
        },
        transform=NchwImageInputPass(),
    ),
    Rewrite(
        # carries the set's static-shape precondition: on a symbolic graph it declines, silently
        name="uint8_image_input",
        gates={
            RKNPU: "an rknn.load_onnx that takes a uint8 graph input, leaving the preprocess where it is",
        },
        transform=Uint8ImageInputPass(),
    ),
    Rewrite(
        # declines outright once the uint8 row has retyped the input, so the two are exclusive by contract
        name="float_image_input",
        gates={
            RKNPU: "the same rknn.load_onnx that takes a uint8 graph input",
        },
        transform=FloatImageInputPass(),
    ),
    Rewrite(
        name="host_ctc_decode",
        gates={
            # the only backend where the in-graph greedy-CTC head loses to a host decode at every batch
            "OpenVINOExecutionProvider": "ArgMax+ReduceMax costing less than the 18k-class readback",
        },
        transform=HostCtcDecodePass(),
    ),
    Rewrite(
        name="raw_ctc_logits",
        gates={
            RKNPU: "an RKNPU Exp kernel, which is the whole reason the softmax denominator cannot stay",
        },
        transform=RawCtcLogitsPass(),
    ),
    Rewrite(
        name="floatify_not_equal",
        gates={
            RKNPU: "the same librknnrt int32 Equal kernel floatify_pad_mask waits on",
        },
        transform=RewritePass([FloatifyNotEqual.rule()]),
    ),
    Rewrite(
        name="floatify_pad_keep",
        gates={
            RKNPU: "an rknn-toolkit2 fold_constant that compares two Gathers' tables before merging them",
        },
        transform=RewritePass([FloatifyPadKeep.rule()]),
    ),
    Rewrite(
        name="opaque_zero_mul",
        gates={
            RKNPU: "an rknn-toolkit2 whose SDPA matcher survives fold_constant collapsing the batch zeros",
        },
        transform=RewritePass([OpaqueZeroMul.rule()]),
    ),
    Rewrite(
        name="fold_gather_elements",
        gates={
            RKNPU: "an rknn-toolkit2 whose _p_gatherelements_to_einsum handles the rank-2 form",
        },
        transform=RewritePass([FoldConstantGatherElements.rule()]),
    ),
    Rewrite(
        name="decompose_gelu",
        gates={
            RKNPU: "an rknn-toolkit2 that ingests Gelu at the opset it pins, and a Tanh that is not a LUT",
        },
        transform=DecomposeGeluPass(),
    ),
    Rewrite(
        # last of the rewrites, so the MatMuls the two decompositions and im2col leave it all count
        name="split_large_reduction",
        gates={
            RKNPU: "an RKNPU whose MAC utilization stops falling away above a 1536-byte weight tile",
        },
        transform=RewritePass([SplitLargeReduction.rule()]),
    ),
    Rewrite(
        name="pin_opset",
        gates={
            RKNPU: "an rknn-toolkit2 that ingests the opset the exporter writes",
        },
        transform=PinOpsetPass(19),
    ),
    Rewrite(
        # the rows above leave new nodes unannotated and the toolkit reads the declared shapes. Sorted first:
        # floatify_pad_mask appends its island at the end of a graph it feeds in the middle.
        name="reinfer_shapes",
        gates={
            RKNPU: "nothing: a graph the rows have edited has to say what it now computes",
        },
        transform=ir.passes.Sequential(TopologicalSortPass(), ReinferShapesPass()),
    ),
)


def plan_rewrites(ctx: RewriteContext) -> RewritePlan:
    rewrites = tuple(rewrite for rewrite in REGISTRY if rewrite.applies(ctx))
    pkg = version("immich_model")
    # ort_version is dead for gating but load-bearing here: dropping it, or collapsing the fields into {ctx},
    # moves every digest and invalidates deployed rewrites and CoreML caches. Named fields, not the context
    # -- a new RewriteContext field left out of here lets two contexts that plan differently collide.
    facts = f"{pkg}\x1f{REWRITE_SET_VERSION}\x1f{ctx.target}\x1f{ctx.ort_version}\x1f{[r.name for r in rewrites]}"
    return RewritePlan(rewrites, hashlib.sha256(facts.encode()).hexdigest()[:12])


def apply_rewrites(src_path: Path, plan: RewritePlan, out_dir: Path | None = None, standalone: bool = False) -> Path:
    """Apply plan to src_path, writing <stem>.rw-<digest>.onnx beside it, or return it unwritten if nothing matched.
    Untouched initializers keep external refs into the source's sidecar; `standalone` writes the weights too, for a
    consumer that takes a file rather than a session."""
    src_path = Path(src_path)
    out_path = (src_path.parent if out_dir is None else Path(out_dir)) / f"{src_path.stem}.rw-{plan.digest}.onnx"

    if not plan.rewrites:
        return src_path
    model = ir.load(src_path)
    # Sequential, not any(): a short-circuit silently drops every rewrite after the first that matches
    if not ir.passes.Sequential(*(rewrite.transform for rewrite in plan.rewrites))(model).modified:
        return src_path
    ir.passes.Sequential(
        CanonicalizeConstantsPass(), RemoveUnusedNodesPass(), DeduplicateInitializersPass(), TopologicalSortPass()
    )(model)

    tmp = out_path.with_suffix(".tmp")  # same dir: keeps sidecar refs valid, makes replace atomic
    if standalone:
        save_with_external_data(model, tmp)
    else:
        ir.save(model, tmp)
    tmp.replace(out_path)
    return out_path
