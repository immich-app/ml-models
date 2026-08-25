from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

from typer import Argument, Exit, Option, Typer, echo

from ._cli import ModelName
from .constants import DELETE_PATTERNS, FIXTURES
from .onnx.cli import app as onnx_app
from .rknn.cli import app as rknn_app

app = Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Export models used by Immich to ONNX and compile them for on-device runtimes.",
)
app.add_typer(onnx_app, name="onnx")
app.add_typer(rknn_app, name="rknn")


check_app = Typer(no_args_is_help=True, help="Diff the committed CI fixtures against what this tree produces.")
app.add_typer(check_app, name="check")


@check_app.command()
def plan(
    fixtures: Annotated[Path, Option(help="Directory holding the committed renderings.")] = FIXTURES,
    bless: Annotated[bool, Option(help="Rewrite the committed fixtures instead of diffing them.")] = False,
) -> None:
    """Diff the rewrite plans and account for every model the catalog declares."""
    from ._check import gaps, plan_fixture

    root = fixtures.resolve()
    _verify(root, plan_fixture(root), [] if bless else gaps(root), bless)


@check_app.command()
def model(
    models: Annotated[Path, Argument(help="Exported model tree.")],
    fixtures: Annotated[Path, Option(help="Directory holding the committed renderings.")] = FIXTURES,
    bless: Annotated[bool, Option(help="Rewrite the committed renderings instead of diffing them.")] = False,
) -> None:
    """Diff the graph and rewrite renderings of the models present."""
    from ._check import model_fixtures

    root = fixtures.resolve()
    _verify(root, model_fixtures(root, models), [], bless)


def _verify(root: Path, items: Iterable[tuple[Path, str]], problems: Iterable[str], bless: bool) -> None:
    from ._check import ELIDE, diff

    failures = 0
    for problem in problems:
        echo(problem)
        failures += 1
    for path, text in items:
        label = path.relative_to(root.parent)
        if bless:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            echo(f"blessed {label}")
        elif not path.exists():
            echo(f"{label}: no committed rendering; rerun with --bless")
            failures += 1
        elif lines := diff(path.read_text(), text, str(label)):
            echo("\n".join(lines[:ELIDE]))
            if len(lines) > ELIDE:
                echo(f"... and {len(lines) - ELIDE} more diff lines")
            failures += 1
    echo(f"{failures} problem(s)")
    raise Exit(bool(failures))


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
            ignore_patterns=["**/model.json"],
            # deleted in the same commit as the upload, so it's atomic
            delete_patterns=DELETE_PATTERNS,
        )

    upload_model()
