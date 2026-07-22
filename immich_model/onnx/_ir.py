"""Generic onnx_ir graph-surgery helpers shared across the export transforms."""

from typing import Any

import numpy as np
import onnx_ir as ir


def make_init(graph: ir.Graph, name: str, array: np.ndarray) -> ir.Value:
    tensor = ir.tensor(array, name=name)
    value = ir.Value(name=name, shape=tensor.shape, type=ir.TensorType(tensor.dtype), const_value=tensor)
    graph.register_initializer(value)
    return value


def make_node(
    op_type: str, inputs: list[ir.Value], name: str | None = None, out: str | None = None, **attributes: Any
) -> ir.Node:
    node = ir.node(op_type, inputs=inputs, attributes=attributes or None, num_outputs=1, name=name)
    if out is not None:
        node.outputs[0].name = out
    return node


def const_array(value: ir.Value | None) -> np.ndarray | None:
    if value is None or value.const_value is None:
        return None
    return value.const_value.numpy()


def const_ints(value: ir.Value | None) -> list[int] | None:
    arr = const_array(value)
    if arr is None or arr.dtype.kind not in "iu":
        return None
    return [int(v) for v in arr.reshape(-1)]


def producer_of(value: ir.Value | None, op_type: str) -> ir.Node | None:
    node = value.producer() if value is not None else None
    return node if node is not None and node.op_type == op_type else None


def sole_consumer(value: ir.Value | None, op_type: str) -> ir.Node | None:
    uses = value.uses() if value is not None else ()
    if len(uses) == 1 and uses[0].node.op_type == op_type:
        return uses[0].node
    return None


def single_use(value: ir.Value) -> bool:
    return len(value.uses()) == 1


def clear_cached_annotations(graph: ir.Graph) -> None:
    """Drop cached shape/type annotations (graph-output types kept) before re-inference: post-surgery
    the old-path annotations are stale and strict consumers (ORT session load) reject them."""
    graph_outputs = set(graph.outputs)
    for node in graph:
        for value in node.outputs:
            value.shape = None
            if value not in graph_outputs:
                value.type = None
