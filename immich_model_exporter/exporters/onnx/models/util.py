import json
from pathlib import Path
from typing import Any


def get_model_path(output_dir: Path | str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "model.onnx"


def save_config(config: Any, output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(config, output_path.open("w"))


def _fix_io_name_collisions(model: Any) -> int:
    """Restore SSA form when a tensor name is assigned by multiple nodes.

    The dynamo exporter names tensors after FX nodes, so a user-provided output name
    (e.g. "embedding") can collide with an intermediate (e.g. the token-embedding
    Gather's output). In topological order, each consumer refers to the most recent
    assignment, so rename every non-final producer to a fresh alias and rewrite the
    consumers in between; the final producer keeps the name feeding the graph output.
    """
    from collections import Counter

    graph = model.graph
    producer_counts = Counter(out for node in graph.node for out in node.output)
    duplicated = {o.name for o in graph.output if producer_counts[o.name] >= 2}
    renamed = 0
    for name in duplicated:
        producers = [n for n in graph.node if name in n.output]
        alias: str | None = None
        for node in graph.node:
            for idx, inp in enumerate(node.input):
                if inp == name and alias is not None:
                    node.input[idx] = alias
            if name in node.output:
                if node is producers[-1]:
                    alias = None
                else:
                    renamed += 1
                    alias = f"{name}_ssa_{renamed}"
                    node.output[list(node.output).index(name)] = alias
    return renamed


def infer_shapes(model_path: Path | str) -> None:
    import onnx

    model_path = Path(model_path)
    model = onnx.load(model_path)
    if _fix_io_name_collisions(model):
        onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=True)
    onnx.save(model, model_path, save_as_external_data=True, all_tensors_to_one_file=True)
