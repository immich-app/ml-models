from pathlib import Path
from typing import Any

from .openclip import OpenCLIPModelConfig
from .openclip import to_onnx as openclip_to_onnx
from .util import get_model_path

_MCLIP_TO_OPENCLIP = {
    "M-CLIP/XLM-Roberta-Large-Vit-B-32": OpenCLIPModelConfig("ViT-B-32", "openai"),
    "M-CLIP/XLM-Roberta-Large-Vit-B-16Plus": OpenCLIPModelConfig("ViT-B-16-plus-240", "laion400m_e32"),
    "M-CLIP/LABSE-Vit-L-14": OpenCLIPModelConfig("ViT-L-14", "openai"),
    "M-CLIP/XLM-Roberta-Large-Vit-L-14": OpenCLIPModelConfig("ViT-L-14", "openai"),
}


def to_onnx(
    model_name: str,
    opset_version: int,
    output_dir_visual: Path | str,
    output_dir_textual: Path | str,
    cache: bool = True,
) -> tuple[Path, Path]:
    textual_path = get_model_path(output_dir_textual)
    if not cache or not textual_path.exists():
        from transformers import AutoTokenizer

        model = _load_model(model_name)
        AutoTokenizer.from_pretrained(model_name).save_pretrained(output_dir_textual)

        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        _export_text_encoder(model, textual_path, opset_version)
    else:
        print(f"Model {textual_path} already exists, skipping")
    # Keep the original activation since M-CLIP's text encoder was aligned against it
    visual_path, _ = openclip_to_onnx(
        _MCLIP_TO_OPENCLIP[model_name], opset_version, output_dir_visual, cache=cache, force_quick_gelu=False
    )
    assert visual_path is not None, "Visual model export failed"
    return visual_path, textual_path


def _load_model(model_name: str) -> Any:
    # transformers 5 breaks multilingual_clip, so we instantiate the model manually.
    import torch
    from huggingface_hub import hf_hub_download
    from multilingual_clip import Config_MCLIP
    from multilingual_clip.pt_multilingual_clip import MultilingualCLIP

    config = Config_MCLIP.MCLIPConfig.from_pretrained(model_name)
    model = MultilingualCLIP(config)

    weights_path = hf_hub_download(model_name, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    missing, _ = model.load_state_dict(state_dict, strict=False)
    assert not missing, f"Missing weights when loading {model_name}: {missing}"
    return model


def _export_text_encoder(model: Any, output_path: Path | str, opset_version: int) -> None:
    import torch
    from multilingual_clip.pt_multilingual_clip import MultilingualCLIP

    from .openclip import _export_encoder

    def forward(self: MultilingualCLIP, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embs = self.transformer(input_ids, attention_mask)[0]
        embs = (embs * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(dim=1)[:, None]
        embs = self.LinearTransformation(embs)
        return torch.nn.functional.normalize(embs, dim=-1)

    # monkeypatch for tracing
    MultilingualCLIP.forward = forward

    # batch of 2 so torch.export doesn't specialize the size-1 batch into reshape targets;
    # XLM-R pools by masked mean, not an EOT gather, so no rewrite_eot
    args = (torch.ones(2, 77, dtype=torch.int32), torch.ones(2, 77, dtype=torch.int32))
    inputs = ["input_ids", "attention_mask"]
    _export_encoder(model, args, output_path, opset_version, inputs, ["text_embedding"], tag="mclip textual")
