"""Pinned source artifacts (RapidAI ONNX conversions on ModelScope, RapidOCR v3.9.2 tag; the
PP-OCRv5 blobs are byte-identical to the v3.8.0 pins) for the PP-OCR exporters. DET_MODELS keys on
(generation, variant) — detection has no language variants — REC_MODELS on (generation, lang,
variant), lang being the name prefix (unprefixed = ch). Entries pin the structural fold counts the
transforms must reproduce, so a restructured upstream model fails closed."""

from typing import NamedTuple

_REPO = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2"


class DetSource(NamedTuple):
    path: str
    sha256: str
    affine_folds: int

    @property
    def url(self) -> str:
        return f"{_REPO}/{self.path}"


class RecSource(NamedTuple):
    path: str
    sha256: str
    affine_folds: int
    layernorms: int
    shape_domains: int

    @property
    def url(self) -> str:
        return f"{_REPO}/{self.path}"


DET_MODELS = {
    ("PP-OCRv5", "mobile"): DetSource(
        "onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
        "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
        affine_folds=38,
    ),
    ("PP-OCRv5", "server"): DetSource(
        "onnx/PP-OCRv5/det/ch_PP-OCRv5_det_server.onnx",
        "0f8846b1d4bba223a2a2f9d9b44022fbc22cc019051a602b41a7fda9667e4cad",
        affine_folds=0,
    ),
    ("PP-OCRv6", "tiny"): DetSource(
        "onnx/PP-OCRv6/det/PP-OCRv6_det_tiny.onnx",
        "f42c0fbd294d95eac1a550e131b277dac97462c8025fa4b6c3cec1b7894bd3d5",
        affine_folds=0,
    ),
    ("PP-OCRv6", "small"): DetSource(
        "onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx",
        "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
        affine_folds=0,
    ),
    ("PP-OCRv6", "medium"): DetSource(
        "onnx/PP-OCRv6/det/PP-OCRv6_det_medium.onnx",
        "92078b7355007ccfffcd4c8cd441a3afd4538904d06881b29a155e1e679907c2",
        affine_folds=0,
    ),
}

REC_MODELS = {
    ("PP-OCRv5", "ch", "mobile"): RecSource(
        "onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile.onnx",
        "5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5",
        affine_folds=40,
        layernorms=5,
        shape_domains=1,
    ),
    ("PP-OCRv5", "ch", "server"): RecSource(
        "onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_server.onnx",
        "e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a",
        affine_folds=0,
        layernorms=5,
        shape_domains=1,
    ),
    ("PP-OCRv5", "korean", "mobile"): RecSource(
        "onnx/PP-OCRv5/rec/korean_PP-OCRv5_rec_mobile.onnx",
        "cd6e2ea50f6943ca7271eb8c56a877a5a90720b7047fe9c41a2e541a25773c9b",
        affine_folds=40,
        layernorms=5,
        shape_domains=1,
    ),
    ("PP-OCRv5", "latin", "mobile"): RecSource(
        "onnx/PP-OCRv5/rec/latin_PP-OCRv5_rec_mobile.onnx",
        "b20bd37c168a570f583afbc8cd7925603890efbcdc000a59e22c269d160b5f5a",
        affine_folds=40,
        layernorms=5,
        shape_domains=1,
    ),
    ("PP-OCRv5", "eslav", "mobile"): RecSource(
        "onnx/PP-OCRv5/rec/eslav_PP-OCRv5_rec_mobile.onnx",
        "08705d6721849b1347d26187f15a5e362c431963a2a62bfff4feac578c489aab",
        affine_folds=40,
        layernorms=5,
        shape_domains=1,
    ),
    ("PP-OCRv5", "en", "mobile"): RecSource(
        "onnx/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile.onnx",
        "c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8",
        affine_folds=40,
        layernorms=5,
        shape_domains=1,
    ),
    ("PP-OCRv5", "th", "mobile"): RecSource(
        "onnx/PP-OCRv5/rec/th_PP-OCRv5_rec_mobile.onnx",
        "de541dd83161c241ff426f7ecfd602a0ba77d686cf3ab9a6c255ea82fd08006e",
        affine_folds=40,
        layernorms=5,
        shape_domains=1,
    ),
    ("PP-OCRv5", "el", "mobile"): RecSource(
        "onnx/PP-OCRv5/rec/el_PP-OCRv5_rec_mobile.onnx",
        "b4368bccd557123c702b7549fee6cd1e94b581337d1c9b65310f109131542b7f",
        affine_folds=40,
        layernorms=5,
        shape_domains=1,
    ),
    ("PP-OCRv6", "ch", "tiny"): RecSource(
        "onnx/PP-OCRv6/rec/PP-OCRv6_rec_tiny.onnx",
        "e16e242de5937ad92609223f19bc2aff3727ee40b095f996907c24749bad251b",
        affine_folds=0,
        layernorms=0,
        shape_domains=0,
    ),
    ("PP-OCRv6", "ch", "small"): RecSource(
        "onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx",
        "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
        affine_folds=0,
        layernorms=5,
        shape_domains=1,
    ),
    ("PP-OCRv6", "ch", "medium"): RecSource(
        "onnx/PP-OCRv6/rec/PP-OCRv6_rec_medium.onnx",
        "eef444829dbbe18d7fea59a3f6eb75647518d2b3a9568d27c92e42940204894b",
        affine_folds=0,
        layernorms=5,
        shape_domains=1,
    ),
}
