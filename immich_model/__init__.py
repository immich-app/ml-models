import json
from pathlib import Path
from typing import Annotated

from typer import Argument, Exit, Option, Typer, echo

from .constants import DELETE_PATTERNS, SOURCE_TO_METADATA, ModelFormat, ModelSource

app = Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Export models used by Immich to ONNX and compile them for on-device runtimes.",
)

ModelName = Annotated[str, Argument(help="Model name; also the per-model output subdirectory name.")]
OutputDir = Annotated[Path, Option(help="Base directory holding per-model output directories.")]
Cache = Annotated[bool, Option(help="Reuse existing outputs instead of regenerating them.")]


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
    from . import onnx

    if not hf_model_name:
        hf_model_name = model_name
    output_dir = output_dir / model_name
    match model_source:
        case ModelSource.MCLIP | ModelSource.OPENCLIP:
            output_dir.mkdir(parents=True, exist_ok=True)
            onnx.export(hf_model_name, model_source, output_dir, opset=opset, cache=cache)
        case ModelSource.INSIGHTFACE:
            from huggingface_hub import snapshot_download

            # TODO: start from insightface dump instead of downloading from HF
            snapshot_download(f"immich-app/{hf_model_name}", local_dir=output_dir)
        case _:
            raise ValueError(f"Unsupported model source {model_source}")

    readme_path = output_dir / "README.md"
    if not (cache or readme_path.exists()):
        with open(readme_path, "w") as f:
            f.write(generate_readme(model_name, model_source))


@app.command()
def compile(
    model_name: ModelName,
    input_dir: Annotated[
        Path, Option(help="Base directory holding the ONNX to compile; defaults to --output-dir.")
    ] = Path("models"),
    output_dir: OutputDir = Path("models"),
    cache: Cache = True,
) -> None:
    """Compile an exported ONNX model into a device binary, writing <output-dir>/<model-name>/**/rknpu."""
    from . import rknn

    try:
        rknn.compile(input_dir / model_name, output_dir / model_name, cache=cache)
    except Exception as e:
        echo(f"Failed to compile {model_name} to RKNN: {e}", err=True)
        raise Exit(code=1)


@app.command()
def profile(
    model_name: ModelName,
    model_format: Annotated[ModelFormat, Option("--format", help="Artifact format to profile.")] = ModelFormat.ONNX,
    base_dir: OutputDir = Path("models"),
    soc: Annotated[str, Option(help="RKNN target SoC (only for --format rknn; needs an attached NPU).")] = "rk3588",
    output_path: Annotated[
        Path | None, Option(help="Profile JSON path; defaults to profiling/<model-name>.<format>.json.")
    ] = None,
) -> None:
    """Benchmark an exported model per-node/per-layer, writing a JSON report."""
    model_dir = base_dir / model_name
    match model_format:
        case ModelFormat.ONNX:
            from . import onnx

            result = onnx.profile(model_dir)
        case ModelFormat.RKNN:
            from . import rknn

            result = rknn.profile(model_dir, soc)
        case _:
            raise ValueError(f"Profiling not supported for format {model_format}")

    if output_path is None:
        output_path = Path("profiling") / f"{model_name}.{model_format}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    for sub, data in result["submodels"].items():
        echo(f"  {sub}: {data['summary']}")
    echo(f"wrote {output_path}")


@app.command()
def upload(
    model_name: ModelName,
    input_dir: Annotated[Path, Option(help="Base directory holding the exported model directory.")] = Path("models"),
    hf_model_name: Annotated[str | None, Option(help="Target Hugging Face repo name; defaults to model_name.")] = None,
    hf_organization: Annotated[str, Option(help="Hugging Face organization to upload under.")] = "immich-app",
) -> None:
    """Upload an exported model directory (<input-dir>/<model-name>) to a Hugging Face repo."""
    from huggingface_hub import create_repo, upload_folder
    from tenacity import retry, stop_after_attempt, wait_fixed

    if not hf_model_name:
        hf_model_name = model_name
    model_dir = input_dir / model_name
    repo_id = f"{hf_organization}/{hf_model_name}"

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
    def upload_model() -> None:
        create_repo(repo_id, exist_ok=True)
        upload_folder(
            repo_id=repo_id,
            folder_path=model_dir,
            # remote repo files to be deleted before uploading
            # deletion is in the same commit as the upload, so it's atomic
            delete_patterns=DELETE_PATTERNS,
        )

    upload_model()
