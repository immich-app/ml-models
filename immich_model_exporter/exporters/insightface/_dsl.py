# mypy: ignore-errors
"""onnxscript definitions of the fused pre/postprocessing subgraphs.

This module is an onnxscript eDSL, not regular Python. Pyright/Pylance check it against
the checker-only façade in _dsl_typing.pyi (see the TYPE_CHECKING import), which types
script mode's tracing semantics. mypy stays exempted file-wide: its type-expression
grammar rejects shaped annotations like `FLOAT[Batch, 512]` outright.

Everything targets opset 19 deliberately; higher opsets break real backends:
- ORT CUDA EP kernel registrations for MaxPool stop at opset 21 (the spec changed at 22),
  which silently pushes the detection stem maxpool to CPU with a device round-trip.
- TensorRT's ONNX parser (classic and RTX) does not recognize GridSample's opset-20 mode
  name "linear" and silently samples nearest-neighbor; opset 19 binds GridSample-16 with
  mode "bilinear". (Fix submitted upstream to onnx/onnx-tensorrt.)
- rknn-toolkit2 rejects models above opset 19 outright.
"""

from typing import TYPE_CHECKING

import numpy as np
import onnx.numpy_helper as nh
from onnx import TensorProto

if TYPE_CHECKING:
    # stub-only module; nothing to import at runtime
    from ._dsl_typing import FLOAT, UINT8, op, script  # pyright: ignore[reportMissingModuleSource]
else:
    from onnxscript import FLOAT, UINT8, script
    from onnxscript import opset19 as op

OPSET = 19

Batch = "batch"

DET_SIZE = 640
DET_STRIDES = (8, 16, 32)

REC_CROP = 256  # loose-crop input size fed to the recognition model
REC_SIZE = 112  # aligned size expected by the backbone

# canonical ArcFace 5-point template for a 112x112 crop
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)

# detection anchor constants, concatenated across strides so the whole decode is one
# Mul + Add per output ([16800, C] tensors for 640x640 SCRFD with 2 anchors per cell)
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

# alignment constants, in the centered unit-scaled frames that keep the least-squares solve
# within fp16 range (TensorRT converts whole engines to fp16)
_template = (ARCFACE_DST - REC_SIZE / 2) / REC_SIZE
TMPL_XY = nh.from_array(_template.reshape(-1, 1), "tmpl_xy")
TMPL_YX = nh.from_array(np.stack([_template[:, 1], -_template[:, 0]], axis=-1).reshape(-1, 1), "tmpl_yx")
TMPL_SUM_X = float(_template[:, 0].sum())
TMPL_SUM_Y = float(_template[:, 1].sum())

_ys, _xs = np.mgrid[:REC_SIZE, :REC_SIZE].astype(np.float32)
TMPL_GRID = nh.from_array(
    np.stack(
        [
            (_xs.ravel() - REC_SIZE / 2) / REC_SIZE,
            (_ys.ravel() - REC_SIZE / 2) / REC_SIZE,
            np.ones(REC_SIZE * REC_SIZE, np.float32),
        ],
        axis=-1,
    )[None],
    "tmpl_grid",
)


@script(default_opset=op)
def det_preprocess(image: UINT8[Batch, 640, 640, 3]) -> FLOAT[Batch, 3, 640, 640]:
    rgb = op.Gather(op.Cast(image, to=TensorProto.FLOAT), [2, 1, 0], axis=3)
    blob = op.Transpose((rgb - 127.5) * (1.0 / 128.0), perm=[0, 3, 1, 2])
    return blob


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
    scores = op.Reshape(op.Concat(scores8, scores16, scores32, axis=1), [0, -1])
    boxes = op.Concat(boxes8, boxes16, boxes32, axis=1) * op.Constant(value=BOX_MUL) + op.Constant(value=BOX_ADD)
    kps = op.Concat(kps8, kps16, kps32, axis=1) * op.Constant(value=KPS_MUL) + op.Constant(value=KPS_ADD)
    return scores, boxes, kps


@script(default_opset=op)
def rec_preprocess(
    image: UINT8[Batch, 256, 256, 3],
    kps: FLOAT[Batch, 5, 2],
) -> FLOAT[Batch, 3, 112, 112]:
    """Estimate the kps -> ArcFace-template similarity transform (closed-form least
    squares) and warp the crop to the aligned 112x112 the backbone expects.

    Closed form for the conformal fit  min sum ||R@k_i + t - d_i||^2  (k=kps, d=template):
      Sx,Sy = sum(k) ; Sxx = sum|k|^2 ; Su,Sv = sum(d) (const)
      Sxu = sum(x*u + y*v) ; Sxv = sum(x*v - y*u)      (constant-vector MatMuls)
      D  = Sxx - (Sx^2 + Sy^2)/n
      a  = (Sxu - (Sx*Su + Sy*Sv)/n)/D ,  b = (Sxv + (Sy*Su - Sx*Sv)/n)/D
      tx = (Su - a*Sx + b*Sy)/n        ,  ty = (Sv - b*Sx - a*Sy)/n
    Solved in centered unit frames so every intermediate stays within fp16 range.
    """
    kps_n = (kps - 128.0) * (1.0 / 256.0)
    kps_flat = op.Reshape(kps_n, [0, -1])

    # scalar sums of the least-squares normal equations, all [batch, 1]
    sxu = op.MatMul(kps_flat, op.Constant(value=TMPL_XY))
    sxv = op.MatMul(kps_flat, op.Constant(value=TMPL_YX))
    sxy = op.ReduceSum(kps_n, [1], keepdims=0)
    sx, sy = sxy[:, 0:1], sxy[:, 1:2]
    sxx = op.ReduceSum(kps_flat * kps_flat, [1], keepdims=1)

    # similarity kps -> template: x' = a*x - b*y + tx, y' = b*x + a*y + ty
    d = sxx - (sx * sx + sy * sy) * 0.2
    a = (sxu - (sx * TMPL_SUM_X + sy * TMPL_SUM_Y) * 0.2) / d
    b = (sxv + (sy * TMPL_SUM_X - sx * TMPL_SUM_Y) * 0.2) / d
    tx = (TMPL_SUM_X - a * sx + b * sy) * 0.2
    ty = (TMPL_SUM_Y - b * sx - a * sy) * 0.2

    # inverse similarity (template -> crop), packed as a [batch, 2, 3] matrix
    inv_scale = 1.0 / (a * a + b * b)
    ia, ib = a * inv_scale, b * inv_scale
    itx = 0.0 - (ia * tx + ib * ty)
    ity = ib * tx - ia * ty
    matrix = op.Reshape(op.Concat(ia, ib, itx, op.Neg(ib), ia, ity, axis=1), [0, 2, 3])

    # push the constant template pixel grid through it; to [-1, 1] with align_corners=0
    src = op.MatMul(op.Constant(value=TMPL_GRID), op.Transpose(matrix, perm=[0, 2, 1]))
    grid = op.Reshape(src * 2.0 + (1.0 / 256.0), [0, 112, 112, 2])

    warped = op.GridSample(
        op.Transpose(op.Cast(image, to=TensorProto.FLOAT), perm=[0, 3, 1, 2]),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=0,
    )
    rgb = op.Gather(warped, [2, 1, 0], axis=1)
    aligned = (rgb - 127.5) * (1.0 / 127.5)
    return aligned


@script(default_opset=op)
def l2_normalize(embedding_raw: FLOAT[Batch, 512]) -> FLOAT[Batch, 512]:
    norm = op.ReduceL2(embedding_raw, [1], keepdims=1)
    embedding = embedding_raw / op.Max(norm, op.Constant(value_float=1e-12))
    return embedding
