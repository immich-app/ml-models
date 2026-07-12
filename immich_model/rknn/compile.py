from pathlib import Path

from ..constants import RKNN_SOCS


def _export_platform(
    model_dir: Path,
    target_platform: str,
    inputs: list[str] | None = None,
    input_size_list: list[list[int]] | None = None,
    fuse_matmul_softmax_matmul_to_sdpa: bool = True,
    cache: bool = True,
) -> None:
    from rknn.api import RKNN

    input_path = model_dir / "model.onnx"
    output_path = model_dir / "rknpu" / target_platform / "model.rknn"
    if cache and output_path.exists():
        print(f"{output_path} already exists, skipping")
        return

    print(f"Exporting {input_path} to {output_path}")

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
    check(rknn.load_onnx(model=input_path.as_posix(), inputs=inputs, input_size_list=input_size_list), "load")
    check(rknn.build(do_quantization=False), "build")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    check(rknn.export_rknn(output_path.as_posix()), "export")


def _export_platforms(
    model_dir: Path,
    inputs: list[str] | None = None,
    input_size_list: list[list[int]] | None = None,
    cache: bool = True,
) -> None:
    def attempt(soc: str, fuse: bool) -> None:
        _export_platform(
            model_dir,
            soc,
            inputs=inputs,
            input_size_list=input_size_list,
            fuse_matmul_softmax_matmul_to_sdpa=fuse,
            cache=cache,
        )

    fuse = True
    failed: list[str] = []
    for soc in RKNN_SOCS:
        try:
            attempt(soc, fuse)
        except Exception as e:
            # This fusion isn't valid for every model; drop it (for this and later SoCs) and retry.
            if fuse and "inputs or 'outputs' must be set" in str(e):
                print(f"Retrying {soc} without fuse_matmul_softmax_matmul_to_sdpa")
                fuse = False
                try:
                    attempt(soc, fuse)
                    continue
                except Exception as retry_error:
                    e = retry_error
            print(f"Failed to export {model_dir.name} for {soc}: {e}")
            failed.append(soc)

    if failed:
        raise RuntimeError(f"RKNN export failed for {model_dir.name} on: {', '.join(failed)}")


def compile(model_dir: Path, cache: bool = True) -> None:
    # (subdirectory, inputs, input_size_list) — inputs/sizes are only needed for the face models
    sub_models: list[tuple[Path, list[str] | None, list[list[int]] | None]] = [
        (model_dir / "textual", None, None),
        (model_dir / "visual", None, None),
        (model_dir / "detection", ["input.1"], [[1, 3, 640, 640]]),
        (model_dir / "recognition", ["input.1"], [[1, 3, 112, 112]]),
    ]
    present = [(d, inputs, sizes) for d, inputs, sizes in sub_models if d.is_dir()]
    if not present:
        raise RuntimeError(f"No exportable model found under {model_dir}")

    errors: list[str] = []
    for sub_dir, inputs, input_size_list in present:
        try:
            _export_platforms(sub_dir, inputs=inputs, input_size_list=input_size_list, cache=cache)
        except Exception as e:
            errors.append(str(e))

    if errors:
        raise RuntimeError("; ".join(errors))
