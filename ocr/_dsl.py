# mypy: ignore-errors
"""onnxscript definitions of the fused PP-OCR pre/postprocessing subgraphs (eDSL — see immich_model._dsl).
opset 20 for the `Gelu` the PP-OCRv6 erf chains fuse into; RKNN's graphs are pinned back to 19."""

from onnx import TensorProto

from .._dsl import FLOAT, INT32, UINT8, Batch, Height, Width, script
from .._dsl import opset20 as op

OPSET = 20

__all__ = ["OPSET", "Batch", "Height", "Width"]  # re-exported for the transforms; silences unused-import checks

Seq = "seq"
Classes = "classes"

REC_HEIGHT = 48


@script(default_opset=op)
def det_preprocess(image: UINT8[Batch, Height, Width, 3]) -> FLOAT[Batch, 3, Height, Width]:
    # scale and BGR->RGB flip fold into the first conv; the shift stays, not commuting with its padding
    return op.Transpose(op.Cast(image, to=TensorProto.FLOAT), perm=[0, 3, 1, 2]) - 127.5


@script(default_opset=op)
def det_postprocess(probs_raw: FLOAT[Batch, 1, Height, Width]) -> FLOAT[Batch, Height, Width]:
    probs = op.Squeeze(probs_raw, [1])
    return probs


@script(default_opset=op)
def rec_preprocess(image: UINT8[Batch, 48, Width, 3]) -> FLOAT[Batch, 3, 48, Width]:
    # same normalization contract as det_preprocess
    return op.Transpose(op.Cast(image, to=TensorProto.FLOAT), perm=[0, 3, 1, 2]) - 127.5


@script(default_opset=op)
def rec_postprocess(logits: FLOAT[Batch, Seq, Classes]) -> tuple[INT32[Batch, Seq], FLOAT[Batch, Seq]]:
    """Greedy CTC head over raw logits: per-step argmax index and its probability, so the readback is two
    [batch, seq] tensors rather than the full class map."""
    indices = op.Cast(op.ArgMax(logits, axis=2, keepdims=0), to=TensorProto.INT32)
    max_logits = op.ReduceMax(logits, [2], keepdims=1)
    probs = op.Reciprocal(op.ReduceSum(op.Exp(logits - max_logits), [2], keepdims=0))
    return indices, probs
