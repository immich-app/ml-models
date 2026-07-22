"""Export Immich's models to ONNX and rewrite them for on-device runtimes. The core package
stays importable without torch; the exporter CLI and model sources live behind the
export/rknn extras and load lazily."""

from typing import Any

__version__ = "0.2.0"


def __getattr__(name: str) -> Any:
    if name == "app":  # the Typer CLI, available with the [export] extra
        from .cli import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
