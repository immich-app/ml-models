from pathlib import Path
from typing import Annotated

from typer import Option, Typer

from ._cli import ModelName
from .constants import DELETE_PATTERNS
from .onnx.cli import app as onnx_app
from .rknn.cli import app as rknn_app

app = Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Export models used by Immich to ONNX and compile them for on-device runtimes.",
)
app.add_typer(onnx_app, name="onnx")
app.add_typer(rknn_app, name="rknn")


@app.command()
def upload(
    model_name: ModelName,
    hf_branch: Annotated[str, Option(help="Repo branch to upload to.")],
    hf_model_name: Annotated[str | None, Option(help="Target Hugging Face repo name; defaults to model_name.")] = None,
    hf_organization: Annotated[str, Option(help="Hugging Face organization to upload under.")] = "immich-app",
    input_dir: Annotated[Path, Option(help="Base directory holding the exported model directory.")] = Path("models"),
) -> None:
    """Upload an exported model directory (<input-dir>/<model-name>) to a Hugging Face repo."""
    from huggingface_hub import create_branch, create_repo, upload_folder
    from tenacity import retry, stop_after_attempt, wait_fixed

    if not hf_model_name:
        hf_model_name = model_name
    model_dir = input_dir / model_name
    repo_id = f"{hf_organization}/{hf_model_name}"

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
    def upload_model() -> None:
        create_repo(repo_id, exist_ok=True)
        if hf_branch != "main":
            create_branch(repo_id, branch=hf_branch, exist_ok=True)
        upload_folder(
            repo_id=repo_id,
            folder_path=model_dir,
            revision=hf_branch,
            # deleted in the same commit as the upload, so it's atomic
            delete_patterns=DELETE_PATTERNS,
        )

    upload_model()
