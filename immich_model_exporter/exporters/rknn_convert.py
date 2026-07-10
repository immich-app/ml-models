# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "rknn-toolkit2==2.3.2",
#   "numpy<=1.26.4",
#   "protobuf>=4.21.6,<=4.25.4",
#   "onnx>=1.16.1,<1.17",
#   "setuptools<81",
#   "torch<=2.4.0",
# ]
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cpu" }]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Convert an ONNX model to RKNN in an isolated legacy environment.

rknn-toolkit2 requires protobuf <= 4.25, numpy <= 1.26 and onnx < 1.17 (it still imports
onnx.mapping), all incompatible with the exporter's environment — so this runs as a
standalone uv script with its own pinned dependencies.
"""

import argparse
import sys
import tempfile
from pathlib import Path

# the sibling rknn.py would shadow rknn-toolkit2's package with this script's directory on sys.path
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).parent.resolve()]


def static_batch_copy(input_path: Path, work_dir: Path) -> Path:
    """Pin dynamic batch dims to 1 and retype uint8 inputs to float32.

    RKNN needs fully static shapes and rejects uint8 I/O in non-quantized builds. Retyping
    the input turns the graph's leading Cast(to=FLOAT) into a no-op; the RKNN runtime
    converts uint8 input feeds to the model dtype on its own.
    """
    import onnx

    model = onnx.load(input_path)
    for graph_input in model.graph.input:
        if graph_input.type.tensor_type.elem_type == onnx.TensorProto.UINT8:
            graph_input.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        dim = graph_input.type.tensor_type.shape.dim[0]
        if not dim.HasField("dim_value"):
            dim.ClearField("dim_param")
            dim.dim_value = 1
    del model.graph.value_info[:]
    for output in model.graph.output:
        output.type.tensor_type.ClearField("shape")
    model = onnx.shape_inference.infer_shapes(model)
    output_path = work_dir / input_path.name
    onnx.save(model, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("target_platform")
    parser.add_argument("--static-batch", action="store_true")
    parser.add_argument("--disable-sdpa-fuse", action="store_true")
    args = parser.parse_args()

    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    rknn.config(
        target_platform=args.target_platform,
        disable_rules=["fuse_matmul_softmax_matmul_to_sdpa"] if args.disable_sdpa_fuse else [],
        enable_flash_attention=False,
        model_pruning=True,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        load_path = static_batch_copy(args.input_path, Path(tmp_dir)) if args.static_batch else args.input_path
        if rknn.load_onnx(model=load_path.as_posix()) != 0:
            raise RuntimeError("Load failed!")
        if rknn.build(do_quantization=False) != 0:
            raise RuntimeError("Build failed!")
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        if rknn.export_rknn(args.output_path.as_posix()) != 0:
            raise RuntimeError("Export rknn model failed!")


if __name__ == "__main__":
    main()
