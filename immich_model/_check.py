"""Rebuild the committed renderings and diff them against the tree."""

import difflib
from collections.abc import Iterator
from pathlib import Path

from ._render import graph, plans, rewrites
from .constants import CATALOG, catalog

# a wholly rewritten fixture diffs to thousands of lines; the head of it is the news
ELIDE = 60


def fixtures(root: Path, models: Path | None) -> Iterator[tuple[Path, str]]:
    """Every rendering this tree should produce, driven by the catalog rather than by what the tree holds:
    an export directory is shared across branches and carries models this one does not declare."""
    yield root / "plans.txt", plans()
    declared = catalog()
    for onnx in sorted(models.glob("*/*/model.onnx") if models else []):
        model, submodel = onnx.parent.parent.name, onnx.parent.name
        if model not in declared:
            continue
        yield root / "graphs" / model / f"{submodel}.txt", graph(onnx)
        yield root / "rewrites" / model / f"{submodel}.txt", rewrites(onnx)


def gaps(root: Path) -> list[str]:
    """Models the catalog declares that nothing rendered, and renderings it no longer declares; an export
    job holds one model, so it can see neither."""
    declared = set(catalog())
    problems = []
    for kind in ("graphs", "rewrites"):
        rendered = {d.name for d in (root / kind).glob("*") if d.is_dir()}
        problems += [f"{kind}/{n}: declared in {CATALOG.name} but never rendered" for n in sorted(declared - rendered)]
        problems += [f"{kind}/{n}: rendered but no longer in {CATALOG.name}" for n in sorted(rendered - declared)]
    return problems


def diff(committed: str, rebuilt: str, label: str) -> list[str]:
    return list(
        difflib.unified_diff(
            committed.splitlines(), rebuilt.splitlines(), f"{label} committed", f"{label} rebuilt", lineterm=""
        )
    )
