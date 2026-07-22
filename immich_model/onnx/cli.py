from pathlib import Path
from typing import Annotated

from typer import Argument, Option, Typer

from .._cli import Cache, ModelName, OutputDir
from ..constants import SOURCE_TO_METADATA, ModelSource

app = Typer(no_args_is_help=True, help="Export models to ONNX.")


def generate_readme(model_name: str, model_source: ModelSource) -> str:
    name, link, type = SOURCE_TO_METADATA[model_source]
    match model_source:
        case ModelSource.MCLIP:
            tags = ["immich", "clip", "multilingual"]
        case ModelSource.OPENCLIP:
            tags = ["immich", "clip"]
            lowered = model_name.lower()
            if "xlm" in lowered or "nllb" in lowered:
                tags.append("multilingual")
        case ModelSource.INSIGHTFACE:
            tags = ["immich", "facial-recognition"]
        case ModelSource.PADDLE:
            tags = ["immich", "ocr"]
        case _:
            raise ValueError(f"Unsupported model source {model_source}")

    return f"""---
tags:
{" - " + "\n - ".join(tags)}
---
# Model Description

This repo contains ONNX exports for the associated {type} model by {name}. See the [{name}]({link}) repo for more info.

This repo is specifically intended for use with [Immich](https://immich.app/), a self-hosted photo library.
"""


@app.command()
def export(
    model_name: ModelName,
    model_source: Annotated[ModelSource, Argument(help="Upstream source the model comes from.")],
    hf_model_name: Annotated[str | None, Option(help="Hugging Face repo to fetch; defaults to model_name.")] = None,
    output_dir: OutputDir = Path("models"),
    opset: Annotated[int, Option(help="ONNX opset for the exported model.")] = 23,
    cache: Cache = True,
) -> None:
    """Export a model to ONNX (plus tokenizer/config) under <output-dir>/<model-name>."""
    if not hf_model_name:
        hf_model_name = model_name
    output_dir = output_dir / model_name
    match model_source:
        case ModelSource.MCLIP | ModelSource.OPENCLIP:
            from .export import export as export_clip

            output_dir.mkdir(parents=True, exist_ok=True)
            export_clip(hf_model_name, model_source, output_dir, opset=opset, cache=cache)
        case ModelSource.INSIGHTFACE:
            from .. import insightface

            insightface.export(model_name, output_dir, cache=cache)
        case ModelSource.PADDLE:
            from .. import ocr

            ocr.export(model_name, output_dir, cache=cache)
        case _:
            raise ValueError(f"Unsupported model source {model_source}")

    readme_path = output_dir / "README.md"
    if not (cache or readme_path.exists()):
        with open(readme_path, "w") as f:
            f.write(generate_readme(model_name, model_source))
