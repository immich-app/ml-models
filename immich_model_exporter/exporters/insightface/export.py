"""Download official InsightFace packs and export the fused detection/recognition models."""

import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import onnx

from ._dsl import ARCFACE_DST, DET_SIZE, REC_CROP, REC_SIZE
from .transforms import transform_detection, transform_recognition

PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/{}.zip"


def export(model_name: str, output_dir: Path, cache: bool = True) -> None:
    det_path = output_dir / "detection" / "model.onnx"
    rec_path = output_dir / "recognition" / "model.onnx"
    if cache and det_path.exists() and rec_path.exists():
        print(f"Models {det_path} and {rec_path} already exist, skipping")
        return

    pack_dir = _download_pack(model_name, output_dir.parent / ".insightface", cache=cache)
    det_src, rec_src = _find_models(pack_dir)

    print(f"Transforming detection model {det_src}")
    det = transform_detection(onnx.load(det_src))
    det_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(det, det_path)
    _smoke_test_detection(det_path)

    print(f"Transforming recognition model {rec_src}")
    rec = transform_recognition(onnx.load(rec_src))
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rec, rec_path)
    _smoke_test_recognition(rec_path)


def _download_pack(model_name: str, cache_dir: Path, cache: bool = True) -> Path:
    pack_dir = cache_dir / model_name
    if cache and pack_dir.is_dir() and any(pack_dir.glob("**/*.onnx")):
        return pack_dir

    zip_path = cache_dir / f"{model_name}.zip"
    if not (cache and zip_path.exists()):
        url = PACK_URL.format(model_name)
        print(f"Downloading {url}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)

    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    with zipfile.ZipFile(zip_path) as f:
        f.extractall(pack_dir)
    return pack_dir


def _find_models(pack_dir: Path) -> tuple[Path, Path]:
    """Identify the detection and recognition models by graph signature.

    Detection is the model with >=5 outputs (9 for the keypoint variants Immich needs);
    recognition has a single output and a square 112x112 input. Other pack members
    (genderage, landmark regressors) don't match either signature.
    """
    det_path = rec_path = None
    for path in sorted(pack_dir.glob("**/*.onnx")):
        model = onnx.load(path, load_external_data=False)
        inputs = model.graph.input
        dims = [d.dim_value for d in inputs[0].type.tensor_type.shape.dim] if len(inputs) == 1 else []
        if len(model.graph.output) >= 5:
            det_path = path
        elif len(model.graph.output) == 1 and len(dims) == 4 and dims[2] == dims[3] == REC_SIZE:
            rec_path = path
    if det_path is None or rec_path is None:
        raise ValueError(f"Could not identify detection/recognition models in {pack_dir}")
    return det_path, rec_path


def _smoke_test_detection(model_path: Path) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    image = np.random.randint(0, 256, (2, DET_SIZE, DET_SIZE, 3)).astype(np.uint8)
    scores, boxes, kps = session.run(None, {"image": image})
    anchors = scores.shape[1]
    if scores.shape != (2, anchors) or boxes.shape != (2, anchors, 4) or kps.shape != (2, anchors, 10):
        raise ValueError(f"Unexpected detection output shapes: {scores.shape}, {boxes.shape}, {kps.shape}")
    if not (0 <= scores.min() and scores.max() <= 1):
        raise ValueError("Detection scores are not post-sigmoid")


def _smoke_test_recognition(model_path: Path) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    image = np.random.randint(0, 256, (2, REC_CROP, REC_CROP, 3)).astype(np.uint8)
    kps = (ARCFACE_DST * (REC_CROP / REC_SIZE))[None].repeat(2, 0).astype(np.float32)
    (embedding,) = session.run(None, {"image": image, "kps": kps})
    if embedding.shape != (2, 512):
        raise ValueError(f"Unexpected embedding shape: {embedding.shape}")
    norms = np.linalg.norm(embedding, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(f"Embeddings are not L2-normalized: {norms}")
