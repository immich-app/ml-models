import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from ..constants import RKNN_SOCS, SUBMODELS
from ._onnx import prepare_for_rknn


def _export_platform(
    onnx_path: Path,
    output_dir: Path,
    target_platform: str,
    inputs: list[str] | None = None,
    input_size_list: list[list[int]] | None = None,
    fuse_matmul_softmax_matmul_to_sdpa: bool = True,
) -> None:
    from rknn.api import RKNN

    output_path = output_dir / "rknpu" / target_platform / "model.rknn"
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
    )
    check(rknn.load_onnx(model=onnx_path.as_posix(), inputs=inputs, input_size_list=input_size_list), "load")
    check(rknn.build(do_quantization=False), "build")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    check(rknn.export_rknn(output_path.as_posix()), "export")


def _export_platforms(
    input_dir: Path,
    output_dir: Path,
    inputs: list[str] | None = None,
    input_size_list: list[list[int]] | None = None,
    cache: bool = True,
    target_socs: Sequence[str] | None = None,
) -> None:
    socs = []
    for soc in target_socs or RKNN_SOCS:
        if cache and (model_path := output_dir / "rknpu" / soc / "model.rknn").exists():
            print(f"{model_path} already exists, skipping")
        else:
            socs.append(soc)
    if not socs:
        return

    with tempfile.TemporaryDirectory() as work_dir:
        # normalise once for all SoCs (rknn.build rejects opset>19 / exact GELU)
        onnx_path = prepare_for_rknn(input_dir / "model.onnx", Path(work_dir))

        def attempt(soc: str, fuse: bool) -> None:
            _export_platform(
                onnx_path,
                output_dir,
                soc,
                inputs=inputs,
                input_size_list=input_size_list,
                fuse_matmul_softmax_matmul_to_sdpa=fuse,
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
                print(f"Failed to export {input_dir.name} for {soc}: {e}")
                failed.append(soc)

        if failed:
            raise RuntimeError(f"RKNN export failed for {input_dir.name} on: {', '.join(failed)}")


def _input_spec(onnx_path: Path) -> tuple[list[str], list[list[int]]]:
    """Names and batch-1-pinned shapes of a model's inputs, for rknn.load_onnx.

    RKNN needs a concrete shape and runs batch 1, so every symbolic dim is pinned to 1; dtype comes
    from the ONNX, so only shapes are pinned.
    """
    import onnx_ir as ir

    inputs = ir.load(onnx_path).graph.inputs
    names = cast(list[str], [inp.name for inp in inputs])  # graph inputs are always named
    sizes = [[d if isinstance(d, int) and d > 0 else 1 for d in (inp.shape or [])] for inp in inputs]
    return names, sizes


def compile(input_dir: Path, output_dir: Path, cache: bool = True, socs: Sequence[str] | None = None) -> None:
    """Compile each ONNX submodel under input_dir into RKNN binaries under output_dir/<sub>/rknpu.

    Input shape/dtype is read per-submodel from its ONNX (ViT visual is float, CNNs stay uint8);
    socs restricts targets (default all RKNN_SOCS).
    """
    present = [sub for sub in SUBMODELS if (input_dir / sub).is_dir()]
    if not present:
        raise RuntimeError(f"No exportable model found under {input_dir}")

    errors: list[str] = []
    for sub in present:
        try:
            inputs, input_size_list = _input_spec(input_dir / sub / "model.onnx")
            _export_platforms(
                input_dir / sub,
                output_dir / sub,
                inputs=inputs,
                input_size_list=input_size_list,
                cache=cache,
                target_socs=socs,
            )
        except Exception as e:
            errors.append(str(e))

    if errors:
        raise RuntimeError("; ".join(errors))
