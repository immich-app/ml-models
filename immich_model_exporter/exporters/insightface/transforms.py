"""Graph transforms that turn stock InsightFace ONNX models into Immich's fused format.

Used two ways: by this package's exporter when building models from the official packs, and
by immich_ml at load time to upgrade legacy cached models in place. The fused contract:

- detection: uint8 BGR NHWC input [batch, 640, 640, 3]; preprocessing and the SCRFD anchor
  decode fold into the graph. Outputs scores [batch, N], boxes [batch, N, 4] and landmarks
  [batch, N, 10] in input pixel space, so only thresholding and NMS remain host-side.
- recognition: takes a loose square crop [batch, 256, 256, 3] plus the detector's five
  landmarks [batch, 5, 2] in crop coordinates. Face alignment (closed-form Umeyama
  similarity + GridSample warp) runs in-graph; embeddings come back L2-normalized.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnx.numpy_helper as nh
import onnxscript.optimizer
from onnx import ModelProto, compose, version_converter
from onnxscript import ir
from onnxscript.rewriter import pattern
from onnxscript.rewriter.pattern import MatchResult
from onnxscript.values import OnnxFunction

from . import _dsl

log = logging.getLogger(__name__)


def transform_detection(model: ModelProto) -> ModelProto:
    graph = model.graph
    if len(graph.output) != 9:
        raise ValueError(f"Expected a 9-output SCRFD keypoint model, got {len(graph.output)} outputs")

    # dynamic batch, fixed 640x640
    dims = graph.input[0].type.tensor_type.shape.dim
    dims[0].ClearField("dim_value")
    dims[0].dim_param = "batch"
    for i in (2, 3):
        dims[i].ClearField("dim_param")
        dims[i].dim_value = _dsl.DET_SIZE

    _fix_fpn_resizes(model)
    model = _fix_head_layout(model)

    model = _reinfer(model)
    model = onnxscript.optimizer.optimize(model)
    model = _reinfer(version_converter.convert_version(model, _dsl.OPSET))

    # map the backbone's outputs onto the decode inputs by (channels, anchor count)
    outputs_by_key = {}
    for output in model.graph.output:
        out_dims = output.type.tensor_type.shape.dim
        outputs_by_key[(out_dims[2].dim_value, out_dims[1].dim_value)] = output.name
    io_map = []
    for stride in _dsl.DET_STRIDES:
        anchors = 2 * (_dsl.DET_SIZE // stride) ** 2
        io_map += [
            (outputs_by_key[(1, anchors)], f"scores{stride}"),
            (outputs_by_key[(4, anchors)], f"boxes{stride}"),
            (outputs_by_key[(10, anchors)], f"kps{stride}"),
        ]
    return _wrap(model, _dsl.det_preprocess, _dsl.det_postprocess, io_map)


def transform_recognition(model: ModelProto) -> ModelProto:
    dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    dim.ClearField("dim_value")
    dim.dim_param = "batch"

    model = _reinfer(model)
    model = onnxscript.optimizer.optimize(model)
    model = _reinfer(version_converter.convert_version(model, _dsl.OPSET))

    io_map = [(model.graph.output[0].name, "embedding_raw")]
    return _wrap(model, _dsl.rec_preprocess, _dsl.l2_normalize, io_map)


def upgrade_detection(model_path: Path) -> None:
    """Upgrade a legacy face detection model file to the fused format, in place."""
    log.info("Upgrading legacy face detection model at %s", model_path)
    onnx.save(transform_detection(onnx.load(model_path)), model_path)


def upgrade_recognition(model_path: Path) -> None:
    """Upgrade a legacy facial recognition model file to the fused format, in place."""
    log.info("Upgrading legacy facial recognition model at %s", model_path)
    onnx.save(transform_recognition(onnx.load(model_path)), model_path)


class BatchSCRFDHead(pattern.RewriteRuleClassBase):
    """Batch the 2020-era SCRFD head, which flattens batch into the anchor dim:
    Transpose(2,3,0,1) [H,W,N,C] followed by Reshape([-1, C]) becomes the modern batched
    layout (N first, batch dim preserved with Reshape's 0-means-copy). Both halves are
    load-bearing: at batch 1 the two layouts flatten identically, so any mistake only
    breaks batch > 1.
    """

    def pattern(self, op: Any, x: Any, shape: Any) -> Any:
        return op.Reshape(op.Transpose(x, perm=[2, 3, 0, 1]), shape)

    def check(self, context: Any, shape: ir.Value, **_: Any) -> MatchResult:  # type: ignore[override]
        result = MatchResult()
        value = shape.const_value
        if value is None:
            return result.fail("Reshape shape is not constant.", shape)
        array = value.numpy()
        if not (array.ndim == 1 and array.size == 2 and int(array[0]) == -1):
            return result.fail("Not the legacy [-1, C] flatten.", shape)
        return result

    def rewrite(self, op: Any, x: ir.Value, shape: ir.Value) -> Any:
        assert shape.const_value is not None  # established by check
        channels = int(shape.const_value.numpy()[1])
        batched_shape = ir.tensor([0, -1, channels], dtype=ir.DataType.INT64)
        return op.Reshape(op.Transpose(x, perm=[0, 2, 3, 1]), op.initializer(batched_shape, name=f"{x.name}_shape"))


_HEAD_RULES = pattern.RewriteRuleSet([BatchSCRFDHead.rule()])


def _fix_head_layout(model: ModelProto) -> ModelProto:
    ir_model = ir.from_proto(model)
    applied = _HEAD_RULES.apply_to_model(ir_model)
    if applied != 9:
        raise ValueError(f"Expected to rewrite 9 SCRFD head Transpose+Reshape pairs, matched {applied}")
    return ir.to_proto(ir_model)


def _fix_fpn_resizes(model: ModelProto) -> None:
    """Replace the FPN upsamples' runtime-computed scales with a constant.

    The Shape/Slice/Concat chains computing them can otherwise constant-fold with a stale
    batch=1 baked in; the dead chains are removed by the optimizer.
    """
    model.graph.initializer.append(nh.from_array(np.array([1, 1, 2, 2], np.float32), "resize_scales_2x"))
    for node in model.graph.node:
        if node.op_type == "Resize":
            mode = next(a.s.decode() for a in node.attribute if a.name == "mode")
            if mode != "nearest":
                raise ValueError(f"Expected nearest Resize in FPN, got {mode}")
            node.input[2] = "resize_scales_2x"
            del node.input[3:]


def _reinfer(model: ModelProto) -> ModelProto:
    """Drop all cached shape annotations and re-infer from the inputs.

    The stock exports carry stale batch=1 dims in value_info; the CPU EP merges these
    leniently but the CUDA EP's buffer planner trusts them and fails at batch > 1.
    """
    del model.graph.value_info[:]
    for output in model.graph.output:
        output.type.tensor_type.ClearField("shape")
    return onnx.shape_inference.infer_shapes(model)


def _wrap(
    backbone: ModelProto,
    pre: OnnxFunction[Any, Any],
    post: OnnxFunction[Any, Any],
    post_io_map: list[tuple[str, str]],
) -> ModelProto:
    """Compose pre -> backbone -> post, then clean up.

    The onnxscript models get prefixed internals so their auto-generated names can't
    collide, but keep their (contract) input/output names.
    """
    pre_model, post_model = pre.to_model_proto(), post.to_model_proto()
    pre_model.ir_version = post_model.ir_version = backbone.ir_version
    pre_model = compose.add_prefix(pre_model, "pre_", rename_inputs=False, rename_outputs=False)
    post_model = compose.add_prefix(post_model, "post_", rename_inputs=False, rename_outputs=False)

    pre_io_map = [(o.name, i.name) for o, i in zip(pre_model.graph.output, backbone.graph.input)]
    model = compose.merge_models(pre_model, backbone, io_map=pre_io_map)
    model = compose.merge_models(model, post_model, io_map=post_io_map)

    model = onnxscript.optimizer.optimize(model)
    model = _reinfer(model)
    onnx.checker.check_model(model)
    return model


__all__ = [
    "transform_detection",
    "transform_recognition",
    "upgrade_detection",
    "upgrade_recognition",
]
