"""Shared ONNX graph-surgery helpers for the fused-model exporters (insightface, ocr).

Stay importable with only torch-free deps: immich_ml imports the transforms built on this to
upgrade cached face/OCR models in place.
"""

import tempfile
from pathlib import Path
from typing import Any

import google.protobuf.message
import onnx
import onnx.numpy_helper as nh
import onnxscript.optimizer
from onnx import ModelProto, compose
from onnxscript.values import OnnxFunction


def fold_input_scale(model: ModelProto, scale: float, flip_channels: bool = False) -> None:
    """Fold the `* scale` of the backbone's `(x - shift) * scale` normalization into the input Convs,
    leaving only `- shift` for the graph. Exact: scaling commutes with zero-padding (0*scale=0), so
    W'=scale*W reproduces outputs; the additive shift does NOT commute, so it stays explicit.
    flip_channels also reverses the weights' input-channel axis (BGR<->RGB).
    """
    graph = model.graph
    input_name = graph.input[0].name
    initializers = {i.name: i for i in graph.initializer}

    consumers = [n for n in graph.node if input_name in n.input]
    not_conv = [n.op_type for n in consumers if n.op_type != "Conv"]
    if not consumers or not_conv:
        raise ValueError(f"Cannot fold input scale/perm: input consumed by {not_conv or 'nothing'}")

    for conv in consumers:
        w_init = initializers[conv.input[1]]
        weight = nh.to_array(w_init)  # [O, C, kH, kW]
        folded = (weight * scale).astype(weight.dtype)
        if flip_channels:
            folded = folded[:, ::-1]
        w_init.CopyFrom(nh.from_array(folded, w_init.name))  # repeated-field items alias


def save_with_external_data(model: ModelProto, output_path: Path) -> None:
    """Save weights to a sidecar .data file: keeps model.onnx under the protobuf 2GB cap (larger CLIP
    variants exceed it) and lets graph-only revisions share it."""
    data_name = output_path.with_suffix(".onnx.data").name
    (output_path.parent / data_name).unlink(missing_ok=True)
    onnx.save(
        model,
        output_path.as_posix(),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
    )


def reinfer(model: ModelProto) -> ModelProto:
    """Drop cached shape annotations and re-infer. Stock exports carry stale batch=1 value_info that the
    CUDA EP's buffer planner trusts and fails on at batch>1 (CPU EP merges leniently). infer_shapes
    serializes to a >2GB-capped string; larger text encoders exceed it, so fall back to a disk round-trip
    with external weights.
    """
    del model.graph.value_info[:]
    for output in model.graph.output:
        output.type.tensor_type.ClearField("shape")
    try:
        return onnx.shape_inference.infer_shapes(model)
    except google.protobuf.message.EncodeError:
        # save_as_external_data externalizes weights in place (clears raw_data); reload the
        # inferred model's own weights to match the exception-free path (which never touches input)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            src, dst = tmp / "m.onnx", tmp / "m.inferred.onnx"
            onnx.save(
                model, src.as_posix(), save_as_external_data=True, all_tensors_to_one_file=True, location="m.onnx.data"
            )
            onnx.shape_inference.infer_shapes_path(src.as_posix(), dst.as_posix())
            inferred = onnx.load(dst.as_posix())
            onnx.load_external_data_for_model(model, tmp_dir)
            return inferred


def wrap(
    backbone: ModelProto,
    pre: OnnxFunction[Any, Any],
    post: OnnxFunction[Any, Any],
    post_io_map: list[tuple[str, str]],
) -> ModelProto:
    """Compose pre -> backbone -> post, then optimize/reinfer/check. onnxscript internals get prefixed
    (names can't collide) but input/output contract names are preserved.
    """
    pre_model, post_model = pre.to_model_proto(), post.to_model_proto()
    pre_model.ir_version = post_model.ir_version = backbone.ir_version
    pre_model = compose.add_prefix(pre_model, "pre_", rename_inputs=False, rename_outputs=False)
    post_model = compose.add_prefix(post_model, "post_", rename_inputs=False, rename_outputs=False)

    pre_io_map = [(o.name, i.name) for o, i in zip(pre_model.graph.output, backbone.graph.input)]
    model = compose.merge_models(pre_model, backbone, io_map=pre_io_map)
    model = compose.merge_models(model, post_model, io_map=post_io_map)

    model = onnxscript.optimizer.optimize(model)
    model = reinfer(model)
    onnx.checker.check_model(model)
    return model
