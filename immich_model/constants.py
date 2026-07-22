from enum import Enum
from typing import NamedTuple


class StrEnum(str, Enum):
    """enum.StrEnum without the 3.11 floor (requires-python starts at 3.10)."""

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


class Soc(StrEnum):
    RK3566 = "rk3566"
    RK3568 = "rk3568"
    RK3576 = "rk3576"
    RK3588 = "rk3588"


RKNN_SOCS = list(Soc)

# per-model output subdirectories that may hold a model.onnx
SUBMODELS = ["textual", "visual", "detection", "recognition"]


# glob to delete old UUID blobs when reuploading models
_uuid_char = "[a-fA-F0-9]"
_uuid_glob = _uuid_char * 8 + "-" + _uuid_char * 4 + "-" + _uuid_char * 4 + "-" + _uuid_char * 4 + "-" + _uuid_char * 12
DELETE_PATTERNS = [
    "**/*onnx*",
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
