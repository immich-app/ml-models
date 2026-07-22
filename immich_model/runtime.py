"""Per-EP runtime graph rewrites for released onnxruntime: plan_rewrites(ctx) picks +
digests the ordered rewrites, apply_rewrites writes model.rw-<digest>.onnx beside the source.
The rewritten graph is a separate small .onnx sharing the source's .onnx.data — CoreML's
compiled-model cache keys on file path, so it must be a stable on-disk artifact, not an
in-memory patch."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import onnx_ir as ir
from onnx_ir.passes.common import RemoveUnusedNodesPass
from onnxscript.rewriter import RewriteRuleSet

from .onnx.lowering import DecomposeAttention, PatchEmbedToMatMul
from .onnx.transforms import fuse_visual_input

# bumped when a registry row's semantics change without a package release; part of the digest.
# v3: apply_rewrites ran only the first matching ruleset (any() short-circuit), so
# earlier-digest artifacts may miss rewrites and must regenerate.
REWRITE_SET_VERSION = 3


@dataclass(frozen=True)
class RewriteContext:
    ep: str
    ort_version: tuple[int, ...]


@dataclass(frozen=True)
class VersionGate:
    """Needed only while ort_version < ort_below; None means "always for now". Bundled stacks
    (OpenVINO etc.) version in lockstep with their ORT wheel, so every drop condition reduces to
    an ORT bound."""

    ort_below: tuple[int, ...] | None = None

    def needed(self, ctx: "RewriteContext") -> bool:
        return self.ort_below is None or ctx.ort_version < self.ort_below


@dataclass(frozen=True)
class Rewrite:
    name: str
    gates: dict[str, VersionGate]  # keyed by execution provider name
    ruleset: RewriteRuleSet

    def applies(self, ctx: RewriteContext) -> bool:
        gate = self.gates.get(ctx.ep)
        return gate is not None and gate.needed(ctx)


@dataclass(frozen=True)
class RewritePlan:
    rewrites: tuple[Rewrite, ...]
    digest: str

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(rewrite.name for rewrite in self.rewrites)


REGISTRY = (
    Rewrite(
        name="decompose_attention",
        gates={
            # drop: first ORT release with the upstreamed CoreML Attention/is_causal builders
            "CoreMLExecutionProvider": VersionGate(),
            # drop: ORT release whose MIGraphX parser stops crashing on the fused op
            "MIGraphXExecutionProvider": VersionGate(),
            # drop: OVEP wheel paired with OpenVINO >= 2026.3 (working 3D Attention translation)
            "OpenVINOExecutionProvider": VersionGate(),
        },
        ruleset=RewriteRuleSet([DecomposeAttention.rule()]),
    ),
    Rewrite(
        # never CoreML: measured ANE regression on 16x16-patch models
        name="im2col_patchify",
        gates={
            "CUDAExecutionProvider": VersionGate(),  # drop: if a CUDA bump flips the conv-vs-GEMM balance
            "OpenVINOExecutionProvider": VersionGate(),  # drop: intel_gpu fixes its patchify-conv kernels
        },
        ruleset=RewriteRuleSet([PatchEmbedToMatMul.rule(batch_dynamic=True)]),
    ),
)


def plan_rewrites(ctx: RewriteContext) -> RewritePlan:
    rewrites = tuple(rewrite for rewrite in REGISTRY if rewrite.applies(ctx))
    pkg = version("immich_model")
    facts = f"{pkg}\x1f{REWRITE_SET_VERSION}\x1f{ctx.ep}\x1f{ctx.ort_version}\x1f{[r.name for r in rewrites]}"
    return RewritePlan(rewrites, hashlib.sha256(facts.encode()).hexdigest()[:12])


def apply_rewrites(src_path: Path, plan: RewritePlan, out_dir: Path | None = None) -> Path:
    """Apply plan to src_path, writing <stem>.rw-<digest>.onnx beside it. Untouched initializers
    keep their external refs into the source's .onnx.data, so all variants share one weight file;
    only rewrite-introduced tensors inline. If no rule matches, nothing is written and src_path is
    returned to serve as-is."""
    src_path = Path(src_path)
    out_path = (src_path.parent if out_dir is None else Path(out_dir)) / f"{src_path.stem}.rw-{plan.digest}.onnx"

    model = ir.load(src_path)
    # list, not any(generator): every ruleset must run — short-circuit dropped im2col from the OpenVINO plan
    applied = [rewrite.ruleset.apply_to_model(model) for rewrite in plan.rewrites]
    if not any(applied):
        return src_path
    RemoveUnusedNodesPass()(model)  # dead nodes AND initializers (the im2col-retired conv weight)
    model.graph.sort()

    tmp = out_path.with_suffix(".tmp")  # same dir: keeps sidecar refs valid, makes replace atomic
    ir.save(model, tmp)
    tmp.replace(out_path)
    return out_path


def adapt_legacy_visual(
    src_path: Path, mean: Sequence[float], std: Sequence[float], out_dir: Path | None = None
) -> Path:
    """Upgrade a legacy normalized-float-NCHW CLIP visual to the uint8-NHWC contract via
    fuse_visual_input: normalization folds into the stem convs (whole fold for unpadded ViT patch
    embeds, scale-only + in-graph shift for padded ResNet stems), input keeps legacy static batch-1
    shape. Numerically equivalent, not bit-exact (the fold reassociates input scaling); measured
    max|d| <= 7e-7, cos >= 0.9999999 on legacy ViT-B-32 and RN50."""
    src_path = Path(src_path)
    out_dir = src_path.parent if out_dir is None else Path(out_dir)
    model = fuse_visual_input(ir.load(src_path), list(mean), list(std))
    out_path = out_dir / f"{src_path.stem}.adapted.onnx"
    tmp = out_path.with_suffix(".tmp")  # atomic: a crash mid-save must not leave a torn file
    ir.save(model, tmp)
    tmp.replace(out_path)
    return out_path
