import logging
import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from .util import get_model_path, save_config

log = logging.getLogger(__name__)


@dataclass
class OpenCLIPModelConfig:
    name: str
    pretrained: str

    @cached_property
    def model_config(self) -> dict[str, Any]:
        import open_clip

        config: dict[str, Any] | None = open_clip.get_model_config(self.name)
        if config is None:
            raise ValueError(f"Unknown model {self.name}")
        return config

    @property
    def image_size(self) -> int:
        image_size: int = self.model_config["vision_cfg"]["image_size"]
        return image_size

    @property
    def sequence_length(self) -> int:
        context_length: int = self.model_config["text_cfg"].get("context_length", 77)
        return context_length


def to_onnx(
    model_cfg: OpenCLIPModelConfig,
    opset_version: int,
    output_dir_visual: Path | str | None = None,
    output_dir_textual: Path | str | None = None,
    cache: bool = True,
    force_quick_gelu: bool | None = None,
) -> tuple[Path | None, Path | None]:
    visual_path = None
    textual_path = None
    if output_dir_visual is not None:
        output_dir_visual = Path(output_dir_visual)
        visual_path = get_model_path(output_dir_visual)

    if output_dir_textual is not None:
        output_dir_textual = Path(output_dir_textual)
        textual_path = get_model_path(output_dir_textual)

    if cache and ((textual_path is None or textual_path.exists()) and (visual_path is None or visual_path.exists())):
        print(f"Models {textual_path} and {visual_path} already exist, skipping")
        return visual_path, textual_path

    import open_clip
    from transformers import AutoTokenizer

    model = open_clip.create_model(
        model_cfg.name,
        pretrained=model_cfg.pretrained,
        force_quick_gelu=force_quick_gelu or model_cfg.pretrained == "openai",
        jit=False,
        require_pretrained=True,
    )

    text_vision_cfg = open_clip.get_model_config(model_cfg.name)

    model.eval()
    _stabilize_sinusoids(model)
    for param in model.parameters():
        param.requires_grad_(False)

    if visual_path is not None and output_dir_visual is not None:
        if not cache or not visual_path.exists():
            save_config(
                open_clip.get_model_preprocess_cfg(model),
                output_dir_visual / "preprocess_cfg.json",
            )
            save_config(text_vision_cfg, output_dir_visual.parent / "config.json")
            _export_image_encoder(model, model_cfg, visual_path, opset_version)
        else:
            print(f"Model {visual_path} already exists, skipping")

    if textual_path is not None and output_dir_textual is not None:
        if not cache or not textual_path.exists():
            tokenizer_name = text_vision_cfg["text_cfg"].get("hf_tokenizer_name", "openai/clip-vit-base-patch32")
            AutoTokenizer.from_pretrained(tokenizer_name).save_pretrained(output_dir_textual)
            _export_text_encoder(model, model_cfg, textual_path, opset_version)
        else:
            print(f"Model {textual_path} already exists, skipping")
    return visual_path, textual_path


def _stabilize_sinusoids(model: Any) -> None:
    """Rebuild NLLB's position table, which is the one weight in the catalog computed at init rather than
    read from a checkpoint. transformers builds the phase with float32 exp and multiply before sin/cos,
    none of which are correctly rounded, so the table -- and the export -- comes out per-machine."""
    import math

    import torch
    from transformers.models.m2m_100.modeling_m2m_100 import M2M100SinusoidalPositionalEmbedding

    for module in model.modules():
        if not isinstance(module, M2M100SinusoidalPositionalEmbedding):
            continue
        count, width = module.weights.shape
        half = width // 2
        freq = torch.exp(torch.arange(half, dtype=torch.float64) * -(math.log(10000) / (half - 1)))
        phase = torch.arange(count, dtype=torch.float64).unsqueeze(1) * freq.unsqueeze(0)
        table = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1).view(count, -1)
        if width % 2 == 1:
            table = torch.cat([table, torch.zeros(count, 1, dtype=torch.float64)], dim=1)
        if module.padding_idx is not None:
            table[module.padding_idx, :] = 0
        module.weights = table.to(module.weights.dtype)


def _export_encoder(
    model: Any,
    args: tuple[Any, ...],
    output_path: Path | str,
    opset_version: int,
    input_names: list[str],
    output_names: list[str],
    *,
    fuse_norm: tuple[list[float], list[float]] | None = None,
    rewrite_eot: bool = False,
    tag: str,
) -> None:
    """Export to a raw dynamo graph, then run the ir pipeline over it; the raw is deleted only on success,
    so a crashed export cannot satisfy the cache check."""
    import torch

    output_path = Path(output_path)
    raw_path = output_path.with_name("raw.onnx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        torch.onnx.export(
            model,
            args,
            raw_path.as_posix(),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_shapes={name: {0: "batch"} for name in input_names},  # named dim: no post-hoc rename
            dynamic_axes=None,
        )

    import onnx_ir as ir

    from .._ir import FlushDenormalsPass, save_with_external_data
    from ..transforms import (
        CanonicalizeConstantsPass,
        DevitalizeShapeDomainPass,
        FoldEmbeddingScalePass,
        Fp16TokenEmbeddingPass,
        FuseVisualInputPass,
        StripTorchMetadataPass,
    )

    log.info("exporting %s", tag)
    pipeline = ir.passes.Sequential(
        CanonicalizeConstantsPass(),
        *([FuseVisualInputPass(*fuse_norm)] if fuse_norm is not None else []),
        DevitalizeShapeDomainPass(rewrite_eot=rewrite_eot),
        # last: the fp16 table's Cast must survive verbatim
        Fp16TokenEmbeddingPass(),
        FoldEmbeddingScalePass(),  # after: the scale folds into the table that ships
        StripTorchMetadataPass(),  # dynamo is the only source of it, so this covers every export path
        FlushDenormalsPass(),
    )
    # one ir.load, in-place passes, one ir.save: large weights stay mmap'd, never inlined into protobuf
    fixed = pipeline(ir.load(raw_path.as_posix())).model
    save_with_external_data(fixed, output_path)
    for path in raw_path.parent.glob(f"{raw_path.name}*"):  # raw + any external sidecar
        path.unlink()


def _export_image_encoder(
    model: Any, model_cfg: OpenCLIPModelConfig, output_path: Path | str, opset_version: int
) -> None:
    import open_clip
    import torch

    def encode_image(image: torch.Tensor) -> torch.Tensor:
        output = model.encode_image(image, normalize=True)
        assert isinstance(output, torch.Tensor)
        return output

    model.forward = encode_image
    preprocess = open_clip.get_model_preprocess_cfg(model)

    # batch of 2: torch.export specializes size-1 dims, baking batch=1 into reshape targets
    args = (torch.randn(2, 3, model_cfg.image_size, model_cfg.image_size),)
    _export_encoder(
        model,
        args,
        output_path,
        opset_version,
        ["image"],
        ["image_embedding"],
        fuse_norm=(list(preprocess["mean"]), list(preprocess["std"])),
        tag="visual",
    )


def _export_text_encoder(
    model: Any, model_cfg: OpenCLIPModelConfig, output_path: Path | str, opset_version: int
) -> None:
    import torch

    def encode_text(text: torch.Tensor) -> torch.Tensor:
        output = model.encode_text(text, normalize=True)
        assert isinstance(output, torch.Tensor)
        return output

    model.forward = encode_text

    args = (torch.ones(2, model_cfg.sequence_length, dtype=torch.int32),)
    _export_encoder(
        model, args, output_path, opset_version, ["text"], ["text_embedding"], rewrite_eot=True, tag="textual"
    )
