"""Download the stock PP-OCR ONNX models and export the fused detection/recognition models."""

import hashlib
import urllib.request
from pathlib import Path

import onnx

from ..onnx.graph import save_with_external_data
from ._sources import DET_MODELS, REC_MODELS, DetSource, RecSource
from .transforms import transform_detection, transform_recognition


def export(model_name: str, output_dir: Path, cache: bool = True) -> None:
    version, lang, variant = _parse_name(model_name)
    det_path = output_dir / "detection" / "model.onnx"
    rec_path = output_dir / "recognition" / "model.onnx"
    charset_path = output_dir / "recognition" / "charset.txt"
    if cache and det_path.exists() and rec_path.exists() and charset_path.exists():
        print(f"Models {det_path} and {rec_path} already exist, skipping")
        return

    cache_dir = output_dir.parent / ".ppocr"
    det_source = DET_MODELS[(version, variant)]
    rec_source = REC_MODELS[(version, lang, variant)]
    det_src = _download(det_source, cache_dir, cache=cache)
    rec_src = _download(rec_source, cache_dir, cache=cache)

    print(f"Transforming detection model {det_src}")
    det = transform_detection(onnx.load(det_src), det_source.affine_folds)
    det_path.parent.mkdir(parents=True, exist_ok=True)
    save_with_external_data(det, det_path)

    print(f"Transforming recognition model {rec_src}")
    rec = onnx.load(rec_src)
    charset_path.parent.mkdir(parents=True, exist_ok=True)
    _write_charset(rec, charset_path)  # before the transform: it consumes the stock metadata
    rec = transform_recognition(rec, rec_source.affine_folds, rec_source.layernorms, rec_source.shape_domains)
    save_with_external_data(rec, rec_path)


def _parse_name(model_name: str) -> tuple[str, str, str]:
    """'LATIN__PP-OCRv5_mobile' -> (PP-OCRv5, latin, mobile); 'PP-OCRv6_small' -> (PP-OCRv6, ch, small)."""
    lang, _, base = model_name.rpartition("__")
    lang = lang.lower() or "ch"
    version, _, variant = base.rpartition("_")
    if (version, lang, variant) not in REC_MODELS:
        raise ValueError(f"No known source for {version!r} language {lang!r} variant {variant!r}")
    return version, lang, variant


def _download(source: DetSource | RecSource, cache_dir: Path, cache: bool = True) -> Path:
    path = cache_dir / source.path.rsplit("/", 1)[-1]
    if not (cache and path.exists()):
        print(f"Downloading {source.url}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(source.url, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source.sha256:
        raise ValueError(f"SHA256 mismatch for {path}: expected {source.sha256}, got {digest}")
    return path


def _write_charset(rec_source: onnx.ModelProto, charset_path: Path) -> None:
    """Extract the charset embedded in the stock model's ONNX metadata into a sidecar file."""
    charset = next((p.value for p in rec_source.metadata_props if p.key == "character"), None)
    if charset is None:
        raise ValueError("Recognition source model has no 'character' metadata to extract")
    chars = charset.splitlines()

    # CTC class count = charset + blank (index 0) + space (last)
    num_classes = rec_source.graph.output[0].type.tensor_type.shape.dim[2].dim_value
    if num_classes and len(chars) != num_classes - 2:
        raise ValueError(f"Charset has {len(chars)} entries but the model predicts {num_classes} classes")

    charset_path.write_text(charset if charset.endswith("\n") else charset + "\n")
