"""InsightFace model transforms and pack exporter.

The transforms are consumed both by this package's export pipeline and by immich_ml, which
applies them at load time to upgrade legacy cached models in place. Keep this module light:
it must be importable with only the base dependencies (onnx, onnxscript, numpy).
"""

from .export import export
from .transforms import (
    transform_detection,
    transform_recognition,
    upgrade_detection,
    upgrade_recognition,
)

__all__ = [
    "export",
    "transform_detection",
    "transform_recognition",
    "upgrade_detection",
    "upgrade_recognition",
]
