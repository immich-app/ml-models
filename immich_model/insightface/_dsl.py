# mypy: ignore-errors
"""onnxscript definitions of the fused face pre/postprocessing subgraphs (eDSL — see immich_model._dsl).
Alignment stays OUT: in-graph GridSample is slower on Apple and has no RKNPU implementation."""

from onnx import TensorProto

from .._dsl import FLOAT, UINT8, Batch, Height, Width, script
from .._dsl import opset19 as op

OPSET = 19

__all__ = ["OPSET", "Batch", "Height", "Width"]  # re-exported for the transforms; silences unused-import checks

DET_STRIDES = (8, 16, 32)

REC_SIZE = 112  # aligned crop size the backbone expects


@script(default_opset=op)
def det_preprocess(image: UINT8[Batch, Height, Width, 3]) -> FLOAT[Batch, 3, Height, Width]:
    # the SCRFD anchor decode stays host-side (layout-marshalling on RKNN), so the graph emits 9 raw heads.
    # The scale folds into the first conv; the shift cannot, that conv's padding reading zeros
    return op.Transpose(op.Cast(image, to=TensorProto.FLOAT), perm=[0, 3, 1, 2]) - 127.5


@script(default_opset=op)
def rec_preprocess(image: UINT8[Batch, 112, 112, 3]) -> FLOAT[Batch, 3, 112, 112]:
    """Takes the host-aligned crop; same normalization contract as det_preprocess."""
    return op.Transpose(op.Cast(image, to=TensorProto.FLOAT), perm=[0, 3, 1, 2]) - 127.5


@script(default_opset=op)
def l2_normalize(embedding_raw: FLOAT[Batch, 512]) -> FLOAT[Batch, 512]:
    # the floor is fp16's smallest normal, not an epsilon: a smaller one rounds to zero in fp16 and
    # silently retires the zero-norm guard. Clip rather than Max because the f16 derivation blocklists Max
    norm = op.ReduceL2(embedding_raw, [1], keepdims=1)
    embedding = embedding_raw / op.Clip(norm, op.Constant(value_float=6.103515625e-05))
    return embedding
