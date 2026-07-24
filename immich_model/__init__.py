"""Export Immich's models to ONNX and rewrite them for on-device runtimes; the core package stays importable
without torch."""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "app":  # the Typer CLI, available with the [export] extra
        from .cli import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
