import math
from collections.abc import Mapping, Sequence
from enum import Enum
from functools import cache
from itertools import chain
from pathlib import Path
from typing import NamedTuple


class StrEnum(str, Enum):
    """enum.StrEnum without the 3.11 floor."""

    __str__ = str.__str__


class ModelSource(StrEnum):
    INSIGHTFACE = "insightface"
    MCLIP = "mclip"
    OPENCLIP = "openclip"
    PADDLE = "paddle"


class ModelTask(StrEnum):
    FACIAL_RECOGNITION = "facial-recognition"
    OCR = "ocr"
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
    ModelSource.PADDLE: SourceMetadata("PaddleOCR", "https://github.com/PaddlePaddle/PaddleOCR", "OCR"),
}


SOURCE_TO_TASK = {
    ModelSource.MCLIP: ModelTask.SEARCH,
    ModelSource.OPENCLIP: ModelTask.SEARCH,
    ModelSource.INSIGHTFACE: ModelTask.FACIAL_RECOGNITION,
    ModelSource.PADDLE: ModelTask.OCR,
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
    """The family a model belongs to, from the catalog and never sniffed off a name: `detection` is SCRFD
    in one family and DBNet in the other."""
    if (task := catalog().get(model)) is None:
        raise KeyError(f"{model} is not in {CATALOG.name}, so its task is undeclared")
    return task


BATCH = "batch"


class DimSet(NamedTuple):
    """One binary's shapes. `label` names the subdirectory it compiles into, and is only carried where a
    submodel ships more than one set."""

    dims: list[dict[str, int]]
    label: str = ""


class UncoveredDims(RuntimeError):
    """A graph leaves a non-batch dim free that no declared set names."""


def declared_dims(task: ModelTask, submodel: Submodel) -> list[DimSet]:
    """The sets this (task, submodel) deploys at, without opening any graph. RKNPU fixes its shapes at
    compile time, one set being one binary."""
    match task, submodel:
        case ModelTask.OCR, Submodel.DETECTION:
            return [
                DimSet(
                    [
                        {"height": height, "width": width}
                        for height, width in dict.fromkeys(
                            chain.from_iterable(
                                ((math.ceil(ratio * size / 32) * 32, size), (size, math.ceil(ratio * size / 32) * 32))
                                for ratio in (1, 4 / 3, 3 / 2, 16 / 9, 2)
                            )
                        )
                    ],
                    f"res{size}",
                )
                for size in (736, 1088, 1440)
            ]
        case ModelTask.OCR, Submodel.RECOGNITION:
            return [DimSet([{"width": size} for size in (224, 320, 448, 640, 1280, 2048)])]
        case ModelTask.FACIAL_RECOGNITION, Submodel.DETECTION:
            return [DimSet([{"height": 640, "width": 640}])]
        case _:
            return [DimSet([{}])]


def dim_sets_of(onnx_path: Path) -> list[DimSet]:
    """The sets for a graph in the exporter's own `<model>/<submodel>/model.onnx` layout, the one layout
    that carries its identity in its path because this repo writes it."""
    return dim_sets(task_of(onnx_path.parents[1].name), Submodel(onnx_path.parent.name), onnx_path)


def dim_sets(task: ModelTask, submodel: Submodel, onnx_path: Path) -> list[DimSet]:
    """The declared sets restricted to the dims the graph leaves free, deduplicated. A free non-batch dim
    no set covers raises: any default for it is a silent failure with a plausible number."""
    import onnx_ir as ir

    inputs = ir.load(onnx_path).graph.inputs
    free = {str(dim) for inp in inputs for dim in (inp.shape or []) if not isinstance(dim, int)} - {BATCH}
    sets: list[DimSet] = []
    for declared in declared_dims(task, submodel):
        group: list[dict[str, int]] = []
        for dims in declared.dims:
            pinned = {name: size for name, size in dims.items() if name in free}
            if pinned not in group:
                group.append(pinned)
        if all(group != existing.dims for existing in sets):
            sets.append(DimSet(group, declared.label))
    if missing := free - set(sets[0].dims[0]):
        raise UncoveredDims(f"{task}/{submodel}: nothing declared for free dim(s) {sorted(missing)}")
    return sets


def served_dims(task: ModelTask, submodel: Submodel, onnx_path: Path) -> list[dict[str, int]]:
    """Every shape the sets cover, flattened: a bench cell runs one shape, whichever binary serves it."""
    grid: list[dict[str, int]] = []
    for group in dim_sets(task, submodel, onnx_path):
        grid += [dims for dims in group.dims if dims not in grid]
    return grid


def served_dims_of(onnx_path: Path) -> list[dict[str, int]]:
    """`served_dims` for a graph in the exporter's own layout."""
    return served_dims(task_of(onnx_path.parents[1].name), Submodel(onnx_path.parent.name), onnx_path)


def dims_label(dims: Mapping[str, int]) -> str:
    """One shape as a filename-safe token, sorted by name so it does not depend on declaration order."""
    return "_".join(f"{name}{size}" for name, size in sorted(dims.items()))


def max_dims(group: Sequence[Mapping[str, int]]) -> dict[str, int]:
    """The shape a set is sized against. Every other shape in the binary is declared relative to it, and
    it is the only one `eval_perf` and `eval_memory` ever report, so it is also the one worth rendering."""
    return dict(max(group, key=lambda dims: (math.prod(dims.values()), sorted(dims.items()))))


def variant_dir(sets: Sequence[DimSet], dims: Mapping[str, int]) -> str:
    """The per-binary subdirectory a shape compiles into, named for the set that serves it; a lone set
    stays unnamed, and immich_ml opens only that flat path."""
    if len(sets) == 1:
        return ""
    return next(group.label for group in sets if dims in group.dims)


# glob to delete old UUID blobs when reuploading models
_uuid_char = "[a-fA-F0-9]"
_uuid_glob = _uuid_char * 8 + "-" + _uuid_char * 4 + "-" + _uuid_char * 4 + "-" + _uuid_char * 4 + "-" + _uuid_char * 12
DELETE_PATTERNS = [
    "**/*onnx*",
    "**/*rknpu*",
    "**/*rknn*",
    "**/*.safetensors",
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
