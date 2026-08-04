"""Typer parameter aliases shared by the root and per-runtime CLI modules."""

from pathlib import Path
from typing import Annotated

from typer import Argument, Option

ModelName = Annotated[str, Argument(help="Model name; also the per-model output subdirectory name.")]
OutputDir = Annotated[Path, Option(help="Base directory holding per-model output directories.")]
Cache = Annotated[bool, Option(help="Reuse existing outputs instead of regenerating them.")]
