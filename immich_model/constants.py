from collections.abc import Mapping, Sequence
from enum import Enum
from functools import cache
from pathlib import Path
from typing import NamedTuple


class StrEnum(str, Enum):
    """enum.StrEnum without the 3.11 floor."""

    __str__ = str.__str__


class ModelSource(StrEnum):
    INSIGHTFACE = "insightface"
    MCLIP = "mclip"
    OPENCLIP = "openclip"


class ModelTask(StrEnum):
    FACIAL_RECOGNITION = "facial-recognition"
    SEARCH = "clip"


class ModelFormat(StrEnum):
    ONNX = "onnx"
    RKNN = "rknn"


class SourceMetadata(NamedTuple):
    name: str
    link: str
    type: str


SOURCE_TO_METADATA = {
    ModelSource.MCLIP: SourceMetadata("M-CLIP", "https://huggingface.co/M-CLIP", "CLIP"),
    ModelSource.OPENCLIP: SourceMetadata("OpenCLIP", "https://github.com/mlfoundations/open_clip", "CLIP"),
    ModelSource.INSIGHTFACE: SourceMetadata(
        "InsightFace", "https://github.com/deepinsight/insightface/tree/master", "facial recognition"
    ),
}


SOURCE_TO_TASK = {
    ModelSource.MCLIP: ModelTask.SEARCH,
    ModelSource.OPENCLIP: ModelTask.SEARCH,
    ModelSource.INSIGHTFACE: ModelTask.FACIAL_RECOGNITION,
}


class OrtBackend(StrEnum):
    """ORT execution-provider backends selectable at the CLI."""

    CPU = "cpu"
    COREML = "coreml"
    CUDA = "cuda"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    MIGRAPHX = "migraphx"
    TENSORRT_RTX = "tensorrt-rtx"


class Submodel(StrEnum):
    """The per-model output subdirectories that may hold a model.onnx, in export order."""

    TEXTUAL = "textual"
    VISUAL = "visual"
    DETECTION = "detection"
    RECOGNITION = "recognition"


class Soc(StrEnum):
    RK3566 = "rk3566"
    RK3568 = "rk3568"
    RK3576 = "rk3576"
    RK3588 = "rk3588"


RKNN_SOCS = list(Soc)

SUBMODELS = list(Submodel)

CATALOG = Path(__file__).resolve().parents[1] / "models.yaml"
FIXTURES = CATALOG.parent / "ci"


@cache
def catalog() -> dict[str, ModelTask]:
    """Every model this repo publishes, mapped to the task its declared source implies."""
    import yaml

    entries = yaml.safe_load(CATALOG.read_text())["models"]
    return {entry["name"]: SOURCE_TO_TASK[ModelSource(entry["source"])] for entry in entries}


def task_of(model: str) -> ModelTask:
    """Which family a model belongs to, from the catalog that declares it and never sniffed off a name:
    `detection` is SCRFD in one family and DBNet in the other, and a name rule would answer confidently
    for a model it has never seen, picking a wrong stimulus and a canvas the tower never serves."""
    if (task := catalog().get(model)) is None:
        raise KeyError(f"{model} is not in {CATALOG.name}, so its task is undeclared")
    return task


BATCH = "batch"

# RKNPU compiles one static graph, so the exporter -- not the runtime -- picks the canvases, and one canvas is
# one binary. These mirror the shape grid immich_ml already snaps its inputs to.
CANVASES: dict[tuple[ModelTask, Submodel], tuple[dict[str, int], ...]] = {
    (ModelTask.FACIAL_RECOGNITION, Submodel.DETECTION): ({"height": 640, "width": 640},),
}


class UncoveredDims(RuntimeError):
    """A graph leaves a non-batch dim free that no declared canvas names."""


def declared_canvases(task: ModelTask, submodel: Submodel) -> tuple[dict[str, int], ...]:
    """The grid this (task, submodel) deploys at, without opening any graph: a planner that had to read
    every artifact to decide which cells exist would die on the first absent one."""
    return CANVASES.get((task, submodel), ({},))


def canvases_of(onnx_path: Path) -> list[dict[str, int]]:
    """The grid for a graph in the exporter's own `<model>/<submodel>/model.onnx` layout, the one layout
    that carries its identity in its path because this repo writes it."""
    return canvases(task_of(onnx_path.parents[1].name), Submodel(onnx_path.parent.name), onnx_path)


def canvases(task: ModelTask, submodel: Submodel, onnx_path: Path) -> list[dict[str, int]]:
    """The declared grid restricted to the dims the graph actually leaves free, deduplicated. A free
    non-batch dim no canvas covers raises: any default for it is a silent failure with a plausible
    number."""
    import onnx_ir as ir

    inputs = ir.load(onnx_path).graph.inputs
    free = {str(dim) for inp in inputs for dim in (inp.shape or []) if not isinstance(dim, int)} - {BATCH}
    grid: list[dict[str, int]] = []
    for canvas in declared_canvases(task, submodel):
        pinned = {name: size for name, size in canvas.items() if name in free}
        if pinned not in grid:
            grid.append(pinned)
    if missing := free - set(grid[0]):
        raise UncoveredDims(f"{task}/{submodel}: no canvas for free dim(s) {sorted(missing)}")
    return grid


def canvas_label(canvas: Mapping[str, int]) -> str:
    """One canvas as a filename-safe token, sorted by name so it does not depend on how CANVASES is written."""
    return "_".join(f"{name}{size}" for name, size in sorted(canvas.items()))


def variant_dir(grid: Sequence[Mapping[str, int]], canvas: Mapping[str, int]) -> str:
    """The per-binary subdirectory a canvas compiles into; a lone canvas stays unnamed, and immich_ml
    opens only that flat path."""
    return canvas_label(canvas) if len(grid) > 1 else ""


# glob to delete old UUID blobs when reuploading models
_uuid_char = "[a-fA-F0-9]"
_uuid_glob = _uuid_char * 8 + "-" + _uuid_char * 4 + "-" + _uuid_char * 4 + "-" + _uuid_char * 4 + "-" + _uuid_char * 12
DELETE_PATTERNS = [
    "**/*onnx*",
    "**/*.safetensors",  # the weight sidecar, which `*onnx*` does not name
    "**/Constant*",
    "**/*.weight",
    "**/*.bias",
    "**/*.proj",
    "**/*in_proj_bias",
    "**/*.npy",
    "**/*.latent",
    "**/*.pos_embed",
    f"**/{_uuid_glob}",
]
