from pathlib import Path
from typing import Annotated

from typer import Exit, Option, Typer, echo

from .._cli import Cache, ModelName, OutputDir
from ..constants import Soc

app = Typer(no_args_is_help=True, help="Compile ONNX exports for RKNPU.")


@app.command()
def compile(
    model_name: ModelName,
    input_dir: Annotated[
        Path, Option(help="Base directory holding the ONNX to compile; defaults to --output-dir.")
    ] = Path("models"),
    output_dir: OutputDir = Path("models"),
    cache: Cache = True,
    soc: Annotated[
        list[Soc] | None,
        Option(help="Target RKNPU SoC(s) to compile for; repeatable. Defaults to all supported SoCs."),
    ] = None,
) -> None:
    """Compile an exported ONNX model into a device binary, writing <output-dir>/<model-name>/**/rknpu."""
    from .compile import compile as compile_model

    try:
        compile_model(input_dir / model_name, output_dir / model_name, cache=cache, socs=soc or None)
    except Exception as e:
        echo(f"Failed to compile {model_name} to RKNN: {e}", err=True)
        raise Exit(code=1)
