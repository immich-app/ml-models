import json
import resource
from pathlib import Path
from typing import Annotated

from typer import Argument, Exit, Option, Typer, echo

from .exporters.constants import DELETE_PATTERNS, SOURCE_TO_METADATA, ModelSource, ModelTask

app = Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Export models used by Immich to ONNX and compile them for on-device runtimes.",
)

ModelName = Annotated[str, Argument(help="Model name; also the per-model output subdirectory name.")]
OutputDir = Annotated[Path, Option(help="Base directory holding per-model output directories.")]
Cache = Annotated[bool, Option(help="Reuse existing outputs instead of regenerating them.")]


def generate_readme(model_name: str, model_source: ModelSource) -> str:
    (name, link, type) = SOURCE_TO_METADATA[model_source]
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
    hf_model_name: Annotated[
        str | None, Option(help="Hugging Face repo/model to fetch; defaults to model_name.")
    ] = None,
    output_dir: OutputDir = Path("models"),
    cache: Cache = True,
) -> None:
    """Export a model to ONNX (plus tokenizer/config) under <output-dir>/<model-name>."""
    from .exporters.onnx import export as onnx_export

    if not hf_model_name:
        hf_model_name = model_name
    output_dir = output_dir / model_name
    match model_source:
        case ModelSource.MCLIP | ModelSource.OPENCLIP:
            output_dir.mkdir(parents=True, exist_ok=True)
            onnx_export(hf_model_name, model_source, output_dir, cache=cache)
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
def compile(model_name: ModelName, output_dir: OutputDir = Path("models"), cache: Cache = True) -> None:
    """Compile an exported ONNX model into a device binary, reading <output-dir>/<model-name>."""
    from .exporters.rknn import export as rknn_export

    model_dir = output_dir / model_name
    try:
        rknn_export(model_dir, cache=cache)
    except Exception as e:
        echo(f"Failed to compile {model_name} to RKNN: {e}", err=True)
        (model_dir / "rknpu").unlink(missing_ok=True)
        raise Exit(code=1)


@app.command()
def profile(
    model_name: ModelName,
    model_task: Annotated[ModelTask, Argument(help="Task the model performs.")],
    base_dir: OutputDir = Path("models"),
    output_path: Annotated[
        Path | None, Option(help="Profile JSON path; defaults to profiling/<model-name>.json.")
    ] = None,
) -> None:
    """Benchmark an exported model's ONNX Runtime latency and peak memory on CPU."""
    from timeit import timeit

    import numpy as np
    import onnxruntime as ort

    model_dir = base_dir / model_name
    if output_path is None:
        output_path = Path("profiling") / f"{model_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)

    sess_options = ort.SessionOptions()
    sess_options.enable_cpu_mem_arena = False
    providers = ["CPUExecutionProvider"]
    provider_options = [{"arena_extend_strategy": "kSameAsRequested"}]
    match model_task:
        case ModelTask.SEARCH:
            textual = ort.InferenceSession(
                model_dir / "textual" / "model.onnx",
                sess_options=sess_options,
                providers=providers,
                provider_options=provider_options,
            )
            tokens = {node.name: np.random.rand(*node.shape).astype(np.int32) for node in textual.get_inputs()}

            visual = ort.InferenceSession(
                model_dir / "visual" / "model.onnx",
                sess_options=sess_options,
                providers=providers,
                provider_options=provider_options,
            )
            image = {node.name: np.random.rand(*node.shape).astype(np.float32) for node in visual.get_inputs()}

            def predict() -> None:
                textual.run(None, tokens)
                visual.run(None, image)

        case ModelTask.FACIAL_RECOGNITION:
            detection = ort.InferenceSession(
                model_dir / "detection" / "model.onnx",
                sess_options=sess_options,
                providers=providers,
                provider_options=provider_options,
            )
            image = {node.name: np.random.rand(1, 3, 640, 640).astype(np.float32) for node in detection.get_inputs()}

            recognition = ort.InferenceSession(
                model_dir / "recognition" / "model.onnx",
                sess_options=sess_options,
                providers=providers,
                provider_options=provider_options,
            )
            face = {node.name: np.random.rand(1, 3, 112, 112).astype(np.float32) for node in recognition.get_inputs()}

            def predict() -> None:
                detection.run(None, image)
                recognition.run(None, face)

        case _:
            raise ValueError(f"Unsupported model task {model_task}")
    predict()
    ms = timeit(predict, number=100)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    json.dump({"pretrained_model": model_dir.name, "peak_rss": rss, "exec_time_ms": ms}, output_path.open("w"))
    print(f"Model {model_dir.name} took {ms:.2f}ms per iteration using {rss / 1024:.2f}MiB of memory")


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
