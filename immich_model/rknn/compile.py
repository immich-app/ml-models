import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ..constants import RKNN_SOCS, SUBMODELS, dim_sets_of, dims_label, max_dims, variant_dir
from ..onnx._ir import ReinferShapesPass
from ..runtime import RKNPU, RewriteContext, apply_rewrites, plan_rewrites
from ._onnx import rknn_config


def contract(group: Sequence[Mapping[str, int]]) -> str:
    """What a binary declares it serves, carried in `custom_string` and read back with `rknn_query`."""
    widest = max_dims(group)
    ordered = [widest, *(dims for dims in group if dims != widest)]
    return json.dumps({"dims": [dict(dims) for dims in ordered]}, separators=(",", ":"))


def _export_platform(
    onnx_path: Path,
    output_dir: Path,
    target_platform: str,
    group: Sequence[Mapping[str, int]],
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
        custom_string=contract(group),
        disable_rules=[] if fuse_matmul_softmax_matmul_to_sdpa else ["fuse_matmul_softmax_matmul_to_sdpa"],
        enable_flash_attention=False,
        model_pruning=True,
        # MaxShape is dynamic_input[0], and eval_perf/eval_memory only ever report that one. A single shape
        # has nothing to be dynamic about; asking anyway drops toolkit passes and only slows the compile.
        **({"dynamic_input": shapes} if len(shapes) > 1 else {}),
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
                "contract": contract(group),
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
    sets = dim_sets_of(source)
    for group in sets:
        variant = variant_dir(sets, group.dims[0])
        socs = []
        for soc in target_socs or RKNN_SOCS:
            model_path = output_dir / "rknpu" / soc / variant / "model.rknn"
            missing = stale(model_path, contract(group.dims))
            if cache and not missing:
                print(f"{model_path} already exists, skipping")
                continue
            print(f"compiling {model_path}" + (f": {', '.join(missing)} absent" if missing else ""))
            socs.append(soc)
        if not socs:
            continue
        _export_set(source, output_dir, socs, group.dims, variant)


def _export_set(
    source: Path, output_dir: Path, socs: list[str], group: Sequence[Mapping[str, int]], variant: str
) -> None:
    with tempfile.TemporaryDirectory() as work_dir:
        # spec and DMA config come off the PREPARED graph, which is the one that retired the shift
        work = Path(work_dir)
        # every shape is prepared, but only MaxShape's graph is compiled
        widest = max_dims(group)
        rest = [dims for dims in group if dims != widest]
        onnx_path = _prepare(source, work / "max", widest)
        config_extras = rknn_config(onnx_path)
        inputs, max_shape = _input_spec(onnx_path)
        ordered = [max_shape] + [_input_spec(_prepare(source, work / dims_label(c), c))[1] for c in rest]

        def attempt(soc: str, fuse: bool) -> None:
            _export_platform(
                onnx_path,
                output_dir,
                soc,
                group,
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


def stale(model_path: Path, expected: str) -> list[str]:
    """What this binary is missing or disagrees with; empty means fresh. The sidecar is written last."""
    sidecar = model_path.with_suffix(".json")
    if not model_path.is_file():
        return [model_path.name]
    if not sidecar.is_file():
        return [sidecar.name]
    if json.loads(sidecar.read_text()).get("contract") != expected:
        return ["contract"]
    return []


def _prepare(source: Path, work_dir: Path, dims: Mapping[str, int]) -> Path:
    """The graph the rows produce for one shape: pinned, rewritten, and carrying its own weights."""
    work_dir.mkdir(parents=True, exist_ok=True)
    pinned = _pin(source, work_dir, dims)
    return apply_rewrites(pinned, plan_rewrites(RewriteContext(RKNPU, ())), out_dir=work_dir, standalone=True)


def _pin(onnx_path: Path, work_dir: Path, dims: Mapping[str, int]) -> Path:
    """Resolve the graph's named free input dims to `dims`, ahead of the RKNPU rows: rknn.load_onnx takes
    them as a shape list, so pinning only there leaves the rows reading a symbolic graph."""
    if not dims:
        return onnx_path

    import onnx_ir as ir

    model = ir.load(onnx_path)
    for inp in model.graph.inputs:
        if inp.shape is not None:
            inp.shape = ir.Shape([dims.get(str(dim), dim) for dim in inp.shape])
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
