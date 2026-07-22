# mypy: ignore-errors
"""onnxscript definitions of the fused pre/postprocessing subgraphs for PP-OCR (eDSL — see
immich_model._dsl for the typing arrangement and the opset-19 rationale)."""

from onnx import TensorProto

from .._dsl import FLOAT, INT32, OPSET, UINT8, Batch, op, script

__all__ = ["OPSET", "Batch"]  # re-exported for the transforms; silences unused-import checks

Height = "height"
Width = "width"
Seq = "seq"
Classes = "classes"

REC_HEIGHT = 48


@script(default_opset=op)
def det_preprocess(image: UINT8[Batch, Height, Width, 3]) -> FLOAT[Batch, 3, Height, Width]:
    # scale (*1/127.5) and BGR->RGB flip fold into the first conv (fold_input_scale); only the shift
    # remains here, since it doesn't commute with the conv's zero-padding
    return op.Transpose(op.Cast(image, to=TensorProto.FLOAT), perm=[0, 3, 1, 2]) - 127.5


@script(default_opset=op)
def det_postprocess(probs_raw: FLOAT[Batch, 1, Height, Width]) -> FLOAT[Batch, Height, Width]:
    """Drop the singleton channel dim of the DBNet probability map."""
    probs = op.Squeeze(probs_raw, [1])
    return probs


@script(default_opset=op)
def rec_preprocess(image: UINT8[Batch, 48, Width, 3]) -> FLOAT[Batch, 3, 48, Width]:
    # same normalization contract as det_preprocess
    return op.Transpose(op.Cast(image, to=TensorProto.FLOAT), perm=[0, 3, 1, 2]) - 127.5


@script(default_opset=op)
def rec_postprocess(logits: FLOAT[Batch, Seq, Classes]) -> tuple[INT32[Batch, Seq], FLOAT[Batch, Seq]]:
    """Greedy CTC head over raw logits: per-step argmax index and its softmax probability. Shrinks the
    readback from [batch, seq, classes] fp32 (~6 MB/crop at 18k classes) to two [batch, seq] tensors.
    The softmax is never materialized (elided in transforms): argmax is softmax-invariant and the
    winning prob is 1 / sum(exp(logits - max))."""
    indices = op.Cast(op.ArgMax(logits, axis=2, keepdims=0), to=TensorProto.INT32)
    max_logits = op.ReduceMax(logits, [2], keepdims=1)
    probs = op.Reciprocal(op.ReduceSum(op.Exp(logits - max_logits), [2], keepdims=0))
    return indices, probs
