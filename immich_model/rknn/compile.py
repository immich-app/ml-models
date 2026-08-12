import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ..constants import RKNN_SOCS, SUBMODELS, canvas_label, canvas_sets_of, max_canvas, variant_dir
from ..onnx._ir import ReinferShapesPass
from ..runtime import RKNPU, RewriteContext, apply_rewrites, plan_rewrites
from ._onnx import rknn_config

# what `--cache` and the bench agree "compiled" means; `nodes` is absent, a foreign binary having no graph
SIDECAR_KEYS = ("canvases", "input")


def _export_platform(
    onnx_path: Path,
    output_dir: Path,
    target_platform: str,
    canvases: Sequence[Mapping[str, int]],
    inputs: list[str],
    shapes: list[list[list[int]]],
    config_extras: dict[str, Any] | None = None,
    fuse_matmul_softmax_matmul_to_sdpa: bool = True,
    variant: str = "",
) -> None:
    from rknn.api import RKNN

    output_path = output_dir / "rknpu" / target_platform / variant / "model.rknn"
    print(f"Exporting {onnx_path} to {output_path}")

    def check(ret: int, step: str) -> None:
        if ret != 0:
            raise RuntimeError(f"RKNN {step} failed for {target_platform} (code {ret})")

    rknn = RKNN(verbose=False)
    rknn.config(
        target_platform=target_platform,
        disable_rules=[] if fuse_matmul_softmax_matmul_to_sdpa else ["fuse_matmul_softmax_matmul_to_sdpa"],
        enable_flash_attention=False,
        model_pruning=True,
        # MaxShape is dynamic_input[0], and eval_perf/eval_memory only ever report that one
        dynamic_input=shapes,
        **(config_extras or {}),  # mean/std for a native uint8 image input (else empty)
    )
    check(rknn.load_onnx(model=onnx_path.as_posix(), inputs=inputs, input_size_list=shapes[0]), "load")
    check(rknn.build(do_quantization=False), "build")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    check(rknn.export_rknn(output_path.as_posix()), "export")
    # what was actually compiled: the .rknn carries none of it and the source .onnx disagrees
    import onnx_ir as ir

    prepared = ir.load(onnx_path)
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "nodes": len(list(ir.traversal.RecursiveGraphIterator(prepared.graph))),
                "canvases": [dict(canvas) for canvas in canvases],
                "input": {"name": inputs[0], "shapes": [shape[0] for shape in shapes]},
            }
        )
    )


def _export_platforms(
    input_dir: Path,
    output_dir: Path,
    cache: bool = True,
    target_socs: Sequence[str] | None = None,
) -> None:
    source = input_dir / "model.onnx"
    sets = canvas_sets_of(source)
    for group in sets:
        variant = variant_dir(sets, group.canvases[0])
        socs = []
        for soc in target_socs or RKNN_SOCS:
            model_path = output_dir / "rknpu" / soc / variant / "model.rknn"
            missing = stale(model_path)
            if cache and not missing:
                print(f"{model_path} already exists, skipping")
                continue
            print(f"compiling {model_path}" + (f": {', '.join(missing)} absent" if missing else ""))
            socs.append(soc)
        if not socs:
            continue
        _export_canvas(source, output_dir, socs, group.canvases, variant)


def _export_canvas(
    source: Path, output_dir: Path, socs: list[str], canvases: Sequence[Mapping[str, int]], variant: str
) -> None:
    with tempfile.TemporaryDirectory() as work_dir:
        # spec and DMA config come off the PREPARED graph, which is the one that retired the shift
        work = Path(work_dir)
        # every canvas is prepared, but only MaxShape's graph is compiled
        widest = max_canvas(canvases)
        onnx_path = _prepare(source, work / "max", widest)
        config_extras = rknn_config(onnx_path)
        inputs, max_shape = _input_spec(onnx_path)
        ordered = [max_shape] + [
            _input_spec(_prepare(source, work / canvas_label(c), c))[1] for c in canvases if c != widest
        ]

        def attempt(soc: str, fuse: bool) -> None:
            _export_platform(
                onnx_path,
                output_dir,
                soc,
                canvases,
                inputs,
                ordered,
                config_extras=config_extras,
                fuse_matmul_softmax_matmul_to_sdpa=fuse,
                variant=variant,
            )

        fuse = True
        failed: list[str] = []
        for soc in socs:
            try:
                attempt(soc, fuse)
            except Exception as e:
                # fusion isn't valid for every model; drop for this and later SoCs, then retry
                if fuse and "inputs or 'outputs' must be set" in str(e):
                    print(f"Retrying {soc} without fuse_matmul_softmax_matmul_to_sdpa")
                    fuse = False
                    try:
                        attempt(soc, fuse)
                        continue
                    except Exception as retry_error:
                        e = retry_error
                print(f"Failed to export {source.parent.name} for {soc}: {e}")
                failed.append(soc)

        if failed:
            raise RuntimeError(f"RKNN export failed for {source.parent.name} on: {', '.join(failed)}")


def stale(model_path: Path) -> list[str]:
    """What this compiled binary is missing: the binary itself, its sidecar, or the sidecar's keys."""
    sidecar = model_path.with_suffix(".json")
    if not model_path.is_file():
        return [model_path.name]
    if not sidecar.is_file():
        return [sidecar.name]
    recorded = json.loads(sidecar.read_text())
    return [key for key in SIDECAR_KEYS if key not in recorded]


def _prepare(source: Path, work_dir: Path, canvas: Mapping[str, int]) -> Path:
    """The graph the rows produce for one canvas: pinned, rewritten, and carrying its own weights."""
    work_dir.mkdir(parents=True, exist_ok=True)
    pinned = _pin(source, work_dir, canvas)
    return apply_rewrites(pinned, plan_rewrites(RewriteContext(RKNPU, ())), out_dir=work_dir, standalone=True)


def _pin(onnx_path: Path, work_dir: Path, canvas: Mapping[str, int]) -> Path:
    """Resolve the graph's named free input dims to `canvas`, ahead of the RKNPU rows: rknn.load_onnx takes
    the canvas as a shape list, so pinning only there leaves the rows reading a symbolic graph."""
    if not canvas:
        return onnx_path

    import onnx_ir as ir

    model = ir.load(onnx_path)
    for inp in model.graph.inputs:
        if inp.shape is not None:
            inp.shape = ir.Shape([canvas.get(str(dim), dim) for dim in inp.shape])
    pinned_path = work_dir / "pinned.onnx"
    ReinferShapesPass()(model)
    ir.save(model, pinned_path, external_data="pinned.onnx.data")
    return pinned_path


def _input_spec(onnx_path: Path) -> tuple[list[str], list[list[int]]]:
    """Names and batch-1-pinned shapes of a model's inputs, for rknn.load_onnx."""
    import onnx_ir as ir

    inputs = ir.load(onnx_path).graph.inputs
    names = cast(list[str], [inp.name for inp in inputs])  # graph inputs are always named
    sizes = [[dim if isinstance(dim, int) else 1 for dim in (inp.shape or [])] for inp in inputs]
    return names, sizes


def compile(input_dir: Path, output_dir: Path, cache: bool = True, socs: Sequence[str] | None = None) -> None:
    """Compile each ONNX submodel under input_dir into RKNN binaries under output_dir/<sub>/rknpu."""
    present = [sub for sub in SUBMODELS if (input_dir / sub).is_dir()]
    if not present:
        raise RuntimeError(f"No exportable model found under {input_dir}")

    errors: list[str] = []
    for sub in present:
        try:
            _export_platforms(input_dir / sub, output_dir / sub, cache=cache, target_socs=socs)
        except Exception as e:
            errors.append(str(e))

    if errors:
        raise RuntimeError("; ".join(errors))
