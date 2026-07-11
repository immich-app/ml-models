"""Immich model export pipeline.

The InsightFace transforms under `immich_model_exporter.exporters.insightface` are also
consumed by immich_ml at runtime and only need the base dependencies (onnx, onnxscript,
numpy). The CLI and the CLIP export pipeline live behind the `export` extra; import them
via `immich_model_exporter.cli`.
"""
