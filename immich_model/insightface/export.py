"""Download official InsightFace packs and export the fused detection/recognition models."""

import shutil
import urllib.request
import zipfile
from pathlib import Path

import onnx

from ..onnx.graph import save_with_external_data
from ._dsl import REC_SIZE
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
    save_with_external_data(det, det_path)

    print(f"Transforming recognition model {rec_src}")
    rec = transform_recognition(onnx.load(rec_src))
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    save_with_external_data(rec, rec_path)


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
    """Identify detection (>=5 outputs, 9 for keypoint variants) vs recognition (1 output, square
    112x112 input) by graph signature; other pack members match neither."""
    det_path = rec_path = None
    # skip macOS AppleDouble sidecars ("._x.onnx"): 4KB of xattr metadata, not protobuf
    for path in sorted(p for p in pack_dir.glob("**/*.onnx") if not p.name.startswith("._")):
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
