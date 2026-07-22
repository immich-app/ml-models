"""Shared prelude for the onnxscript eDSL modules (insightface/_dsl.py, ocr/_dsl.py), authored
in script mode. Pyright checks them against the _dsl_typing.pyi facade (script-mode tracing:
scalars/lists autocast to constant tensors); mypy exempts the eDSL files (its grammar rejects
FLOAT[Batch, 512]).

opset 19 deliberately: ORT CUDA MaxPool kernels stop at opset 21 (spec changed at 22), pushing
a maxpool to CPU; rknn-toolkit2 rejects opset > 19."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # checker facade; no runtime counterpart
    from ._dsl_typing import FLOAT, INT32, UINT8, op, script  # pyright: ignore[reportMissingModuleSource]
else:
    from onnxscript import FLOAT, INT32, UINT8, script
    from onnxscript import opset19 as op

OPSET = 19

Batch = "batch"

__all__ = ["FLOAT", "INT32", "UINT8", "op", "script", "OPSET", "Batch"]
