# mypy: ignore-errors
"""onnxscript definitions of the fused face pre/postprocessing subgraphs (eDSL — see
immich_model._dsl for the typing arrangement and the opset-19 rationale).

Face alignment stays OUT of the recognition graph (the host warps the aligned 112x112 crop):
the in-graph Umeyama+GridSample variant measured strictly worse on Apple (CoreML runs the whole
alignment on CPU) and GridSample has no RKNPU implementation.
"""

import numpy as np
import onnx.numpy_helper as nh
from onnx import TensorProto

from .._dsl import FLOAT, OPSET, UINT8, Batch, op, script

__all__ = ["OPSET", "Batch"]  # re-exported for the transforms; silences unused-import checks

DET_SIZE = 640
DET_STRIDES = (8, 16, 32)

REC_SIZE = 112  # aligned crop size the backbone expects

# anchor constants concatenated across strides -> one Mul+Add per output ([16800, C] for
# 640x640 SCRFD, 2 anchors per cell)
_centers, _strides = [], []
for _s in DET_STRIDES:
    _gy, _gx = np.mgrid[: DET_SIZE // _s, : DET_SIZE // _s]
    _grid = np.stack([_gx, _gy], axis=-1).astype(np.float32) * _s
    _centers.append(np.stack([_grid.reshape(-1, 2)] * 2, axis=1).reshape(-1, 2))
    _strides.append(np.full((len(_centers[-1]), 1), _s, np.float32))
_centers_cat, _strides_cat = np.concatenate(_centers), np.concatenate(_strides)
BOX_MUL = nh.from_array(_strides_cat * np.array([-1, -1, 1, 1], np.float32), "box_mul")
BOX_ADD = nh.from_array(np.tile(_centers_cat, 2), "box_add")
KPS_MUL = nh.from_array(_strides_cat, "kps_mul")
KPS_ADD = nh.from_array(np.tile(_centers_cat, 5), "kps_add")


@script(default_opset=op)
def det_preprocess(image: UINT8[Batch, 640, 640, 3]) -> FLOAT[Batch, 3, 640, 640]:
    # normalization scale folds into the first conv (fold_input_scale); only the shift
    # remains, since it doesn't commute with the conv's zero-padding
    return op.Transpose(op.Cast(image, to=TensorProto.FLOAT), perm=[0, 3, 1, 2]) - 127.5


@script(default_opset=op)
def det_postprocess(
    scores8: FLOAT[Batch, 12800, 1],
    scores16: FLOAT[Batch, 3200, 1],
    scores32: FLOAT[Batch, 800, 1],
    boxes8: FLOAT[Batch, 12800, 4],
    boxes16: FLOAT[Batch, 3200, 4],
    boxes32: FLOAT[Batch, 800, 4],
    kps8: FLOAT[Batch, 12800, 10],
    kps16: FLOAT[Batch, 3200, 10],
    kps32: FLOAT[Batch, 800, 10],
) -> tuple[FLOAT[Batch, 16800], FLOAT[Batch, 16800, 4], FLOAT[Batch, 16800, 10]]:
    """SCRFD anchor decode (distance2bbox/distance2kps against constant anchor centers)."""
    scores = op.Reshape(op.Concat(scores8, scores16, scores32, axis=1), [0, 16800])
    boxes = op.Concat(boxes8, boxes16, boxes32, axis=1) * op.Constant(value=BOX_MUL) + op.Constant(value=BOX_ADD)
    kps = op.Concat(kps8, kps16, kps32, axis=1) * op.Constant(value=KPS_MUL) + op.Constant(value=KPS_ADD)
    return scores, boxes, kps


@script(default_opset=op)
def rec_preprocess(image: UINT8[Batch, 112, 112, 3]) -> FLOAT[Batch, 3, 112, 112]:
    """Takes the host-aligned 112x112 crop; same normalization contract as det_preprocess."""
    return op.Transpose(op.Cast(image, to=TensorProto.FLOAT), perm=[0, 3, 1, 2]) - 127.5


@script(default_opset=op)
def l2_normalize(embedding_raw: FLOAT[Batch, 512]) -> FLOAT[Batch, 512]:
    norm = op.ReduceL2(embedding_raw, [1], keepdims=1)
    embedding = embedding_raw / op.Max(norm, op.Constant(value_float=1e-12))
    return embedding
