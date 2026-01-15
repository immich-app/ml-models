from pathlib import Path
import tempfile
import onnx
import shutil
from .constants import RKNN_SOCS
from .replace_cumsum import process_model as replace_cumsum_process

def _has_cumsum(model_path: Path) -> bool:
    if not model_path.exists():
        return False
    try:
        # Load only the model structure (no weights) for speed if possible, 
        # but onnx.load usually loads everything. 
        # We can use load_model from onnx with load_external_data=False if strictly needed,
        # but standard load is fine for checking nodes.
        model = onnx.load(str(model_path), load_external_data=False)
        for node in model.graph.node:
            if node.op_type == 'CumSum':
                return True
    except Exception as e:
        print(f"Warning: Failed to check for CumSum in {model_path}: {e}")
    return False

def _export_platform(
    model_path: Path,  # Changed: now accepts the actual model path to use
    model_dir: Path,
    target_platform: str,
    inputs: list[str] | None = None,
    input_size_list: list[list[int]] | None = None,
    fuse_matmul_softmax_matmul_to_sdpa: bool = True,
    cache: bool = True,
) -> None:
    from rknn.api import RKNN

    output_path = model_dir / "rknpu" / target_platform / "model.rknn"
    if cache and output_path.exists():
        print(f"Model {model_path} already exists at {output_path}, skipping")
        return

    print(f"Exporting model {model_path} to {output_path}")

    rknn = RKNN(verbose=False)

    rknn.config(
        target_platform=target_platform,
        disable_rules=["fuse_matmul_softmax_matmul_to_sdpa"] if not fuse_matmul_softmax_matmul_to_sdpa else [],
        enable_flash_attention=False,
        model_pruning=True,
    )
    
    ret = rknn.load_onnx(model=model_path.as_posix(), inputs=inputs, input_size_list=input_size_list)

    if ret != 0:
        raise RuntimeError("Load failed!")

    ret = rknn.build(do_quantization=False)

    if ret != 0:
        raise RuntimeError("Build failed!")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ret = rknn.export_rknn(output_path.as_posix())
    if ret != 0:
        raise RuntimeError("Export rknn model failed!")


def _export_platforms(
    model_dir: Path,
    inputs: list[str] | None = None,
    input_size_list: list[list[int]] | None = None,
    cache: bool = True,
    target_platform: str | None = None,
) -> None:
    input_path = model_dir / "model.onnx"

    # Filter platforms to only target platform if specified
    platforms = [target_platform] if target_platform else RKNN_SOCS

    # Check for CumSum once for all platforms
    temp_dir = None
    rknn_input_path = input_path

    if _has_cumsum(input_path):
        print(f"Detected CumSum ops in {input_path}, applying replacement...")
        temp_dir = tempfile.mkdtemp()
        temp_model_path = Path(temp_dir) / "model_replaced.onnx"
        try:
            replace_cumsum_process(str(input_path), str(temp_model_path))
            rknn_input_path = temp_model_path
            print(f"Using replaced model for RKNN export: {rknn_input_path}")
        except Exception as e:
            print(f"Error replacing CumSum: {e}. Proceeding with original model.")
            if temp_dir:
                shutil.rmtree(temp_dir)
            temp_dir = None
            rknn_input_path = input_path

    try:
        fuse_matmul_softmax_matmul_to_sdpa = True
        for soc in platforms:
            try:
                _export_platform(
                    rknn_input_path,
                    model_dir,
                    soc,
                    inputs=inputs,
                    input_size_list=input_size_list,
                    fuse_matmul_softmax_matmul_to_sdpa=fuse_matmul_softmax_matmul_to_sdpa,
                    cache=cache,
                )
            except Exception as e:
                print(f"Failed to export model for {soc}: {e}")
                if "inputs or 'outputs' must be set" in str(e):
                    print("Retrying without fuse_matmul_softmax_matmul_to_sdpa")
                    fuse_matmul_softmax_matmul_to_sdpa = False
                    _export_platform(
                        rknn_input_path,
                        model_dir,
                        soc,
                        inputs=inputs,
                        input_size_list=input_size_list,
                        fuse_matmul_softmax_matmul_to_sdpa=fuse_matmul_softmax_matmul_to_sdpa,
                        cache=cache,
                    )
    finally:
        # Clean up temporary directory after all platforms are done
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)


def export(model_dir: Path, cache: bool = True, target_platform: str | None = None) -> None:
    textual = model_dir / "textual"
    visual = model_dir / "visual"
    detection = model_dir / "detection"
    recognition = model_dir / "recognition"

    if textual.is_dir():
        _export_platforms(textual, cache=cache, target_platform=target_platform)

    if visual.is_dir():
        _export_platforms(visual, cache=cache, target_platform=target_platform)

    if detection.is_dir():
        _export_platforms(detection, inputs=["input.1"], input_size_list=[[1, 3, 640, 640]], cache=cache, target_platform=target_platform)

    if recognition.is_dir():
        _export_platforms(recognition, inputs=["input.1"], input_size_list=[[1, 3, 112, 112]], cache=cache, target_platform=target_platform)
