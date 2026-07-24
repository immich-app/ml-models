"""Shared prelude for the onnxscript eDSL modules. Each package's opset must equal its converted
backbone's; the ceiling is rknn-toolkit2, which rejects opset > 19."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # checker facade; no runtime counterpart
    from ._dsl_typing import FLOAT, INT32, UINT8, script  # pyright: ignore[reportMissingModuleSource]
    from ._dsl_typing import op as opset19  # pyright: ignore[reportMissingModuleSource]
    from ._dsl_typing import op as opset20  # pyright: ignore[reportMissingModuleSource]
else:
    from onnxscript import FLOAT, INT32, UINT8, opset19, opset20, script

Batch = "batch"
Height = "height"
Width = "width"

__all__ = ["FLOAT", "INT32", "UINT8", "opset19", "opset20", "script", "Batch", "Height", "Width"]
