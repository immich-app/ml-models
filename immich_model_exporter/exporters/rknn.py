import subprocess
from pathlib import Path

from .constants import RKNN_SOCS

# rknn-toolkit2 needs protobuf <= 4.25, numpy <= 1.26 and onnx < 1.17, all incompatible with
# this project's environment, so conversion runs as a uv script with its own pinned deps.
_CONVERT_SCRIPT = Path(__file__).parent / "rknn_convert.py"


def _export_platform(
    model_dir: Path,
    target_platform: str,
    static_batch: bool = False,
    fuse_matmul_softmax_matmul_to_sdpa: bool = True,
    cache: bool = True,
) -> None:
    input_path = model_dir / "model.onnx"
    output_path = model_dir / "rknpu" / target_platform / "model.rknn"
    if cache and output_path.exists():
        print(f"Model {input_path} already exists at {output_path}, skipping")
        return

    print(f"Exporting model {input_path} to {output_path}")

    command = ["uv", "run", "--no-project", _CONVERT_SCRIPT.as_posix()]
    command += [input_path.as_posix(), output_path.as_posix(), target_platform]
    if static_batch:
        command.append("--static-batch")
    if not fuse_matmul_softmax_matmul_to_sdpa:
        command.append("--disable-sdpa-fuse")

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"RKNN conversion failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


def _export_platforms(model_dir: Path, static_batch: bool = False, cache: bool = True) -> None:
    fuse_matmul_softmax_matmul_to_sdpa = True
    for soc in RKNN_SOCS:
        try:
            _export_platform(
                model_dir,
                soc,
                static_batch=static_batch,
                fuse_matmul_softmax_matmul_to_sdpa=fuse_matmul_softmax_matmul_to_sdpa,
                cache=cache,
            )
        except Exception as e:
            print(f"Failed to export model for {soc}: {e}")
            if "inputs or 'outputs' must be set" in str(e):
                print("Retrying without fuse_matmul_softmax_matmul_to_sdpa")
                fuse_matmul_softmax_matmul_to_sdpa = False
                _export_platform(
                    model_dir,
                    soc,
                    static_batch=static_batch,
                    fuse_matmul_softmax_matmul_to_sdpa=fuse_matmul_softmax_matmul_to_sdpa,
                    cache=cache,
                )


def export(model_dir: Path, cache: bool = True) -> None:
    textual = model_dir / "textual"
    visual = model_dir / "visual"
    detection = model_dir / "detection"
    recognition = model_dir / "recognition"

    if textual.is_dir():
        _export_platforms(textual, cache=cache)

    if visual.is_dir():
        _export_platforms(visual, cache=cache)

    if detection.is_dir():
        _export_platforms(detection, static_batch=True, cache=cache)

    if recognition.is_dir():
        _export_platforms(recognition, static_batch=True, cache=cache)
