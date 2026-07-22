"""Export Immich's models to ONNX and rewrite them for on-device runtimes. The core package
stays importable without torch; the exporter CLI and model sources live behind the
export/rknn extras and load lazily."""
