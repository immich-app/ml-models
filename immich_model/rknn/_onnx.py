import math
from pathlib import Path
from typing import Any

# tanh-GELU coefficients
_GELU_C0 = math.sqrt(2 / math.pi)
_GELU_C1 = 0.044715


def prepare_for_rknn(onnx_path: Path, work_dir: Path) -> Path:
    """Return an ONNX path that ``rknn.build`` can ingest.

    rknn-toolkit2 rejects opset > 19 and has no NPU `Erf` kernel, so each native `Gelu` node
    is rewritten into its tanh approximation and the opset is pinned to 19.
    """
    import onnx

    model = onnx.load(onnx_path.as_posix())
    opset = max((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), default=0)
    has_gelu = any(node.op_type == "Gelu" for node in model.graph.node)
    if opset <= 19 and not has_gelu:
        return onnx_path

    if has_gelu:
        _decompose_gelu_to_tanh(model)
    _pin_opset(model, 19)

    out_path = work_dir / "model.onnx"
    onnx.save(
        model,
        out_path.as_posix(),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
    )
    return out_path


def _decompose_gelu_to_tanh(model: Any) -> None:
    """Replace every native `Gelu` node with its tanh-approximation subgraph, in place."""
    import numpy as np
    from onnx import helper, numpy_helper

    graph = model.graph
    consts: dict[str, Any] = {}

    def const(name: str, value: float) -> str:
        if name not in consts:
            consts[name] = numpy_helper.from_array(np.array(value, dtype=np.float32), name)
        return name

    new_nodes = []
    for node in graph.node:
        if node.op_type != "Gelu":
            new_nodes.append(node)
            continue
        x, y = node.input[0], node.output[0]
        p = node.name or y
        c0, c1 = const("gelu_c0", _GELU_C0), const("gelu_c1", _GELU_C1)
        half, one = const("gelu_half", 0.5), const("gelu_one", 1.0)
        new_nodes += [
            helper.make_node("Mul", [x, x], [f"{p}_x2"]),
            helper.make_node("Mul", [f"{p}_x2", x], [f"{p}_x3"]),
            helper.make_node("Mul", [f"{p}_x3", c1], [f"{p}_c1x3"]),
            helper.make_node("Add", [x, f"{p}_c1x3"], [f"{p}_inner"]),
            helper.make_node("Mul", [f"{p}_inner", c0], [f"{p}_scaled"]),
            helper.make_node("Tanh", [f"{p}_scaled"], [f"{p}_tanh"]),
            helper.make_node("Add", [f"{p}_tanh", one], [f"{p}_1ptanh"]),
            helper.make_node("Mul", [x, half], [f"{p}_halfx"]),
            helper.make_node("Mul", [f"{p}_halfx", f"{p}_1ptanh"], [y]),
        ]
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(consts.values())


def _pin_opset(model: Any, version: int) -> None:
    """Force the ai.onnx opset to `version`."""
    from onnx import helper

    kept = [opset for opset in model.opset_import if opset.domain not in ("", "ai.onnx")]
    kept.append(helper.make_operatorsetid("", version))
    del model.opset_import[:]
    model.opset_import.extend(kept)
