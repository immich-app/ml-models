"""Generic onnx_ir graph-surgery helpers shared across the export transforms."""

from pathlib import Path
from typing import Any

import numpy as np
import onnx_ir as ir
import onnx_ir.passes.common as common_passes


def save_with_external_data(model: ir.Model, output_path: Path) -> None:
    """Save weights to a `<stem>.safetensors` sidecar, keeping model.onnx under the protobuf 2GB cap. The
    external-data offsets ir writes point into the safetensors payload, so ORT reads it by offset."""
    for value in model.graph.initializers.values():
        # safetensors keys on the tensor's name, ir reads it back by the value's; a rewrite leaves it unset
        value.const_value.name = value.name
    ir.save_safetensors(model, output_path.as_posix())


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


def pointwise(conv: ir.Node) -> bool:
    """A 1x1 Conv at unit stride/dilation, ungrouped and unpadded: a pure per-pixel matmul."""
    attributes = conv.attributes
    unit = [1, 1]
    # list(): get_ints returns a tuple when present and the default object when absent, so an explicit
    # `strides=[1,1]` would compare unequal to [1,1] and read as non-unit
    return (
        conv.op_type == "Conv"
        and list(attributes.get_ints("kernel_shape", [])) == unit
        and attributes.get_int("group", 1) == 1
        and list(attributes.get_ints("strides", unit)) == unit
        and list(attributes.get_ints("dilations", unit)) == unit
        and not any(attributes.get_ints("pads", [0, 0, 0, 0]))
    )


class CanonicalizeConstantsPass(common_passes.LiftConstantsToInitializersPass):
    """Lift every rule-emitted `Constant` to an initializer, which is what the matchers downstream read."""

    def __init__(self) -> None:
        super().__init__(lift_all_constants=True, size_limit=0)


class FlushDenormalsPass(ir.passes.InPlacePass):
    """Zero fp32 initializer values below 1e-30: a denormal multiply costs a microcode assist on some x86 parts.
    The threshold sits above the subnormal boundary because a 1e-35 weight still makes a subnormal PRODUCT."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        for value in model.graph.initializers.values():
            tensor = value.const_value
            if tensor is None or tensor.dtype != ir.DataType.FLOAT:
                continue
            array = tensor.numpy()
            tiny = (np.abs(array) < 1e-30) & (array != 0.0)
            if not tiny.any():
                continue
            flushed = array.copy()
            flushed[tiny] = 0.0
            value.const_value = ir.tensor(flushed, name=value.name)
            modified = True
        return ir.passes.PassResult(model, modified)


class ReinferShapesPass(ir.passes.InPlacePass):
    """Drop cached shape/type annotations (graph-output types kept) and re-infer. Stale annotations break strict
    consumers -- ORT rejects a Slice off a floated uint8 image still annotated UINT8 -- and every pipeline opens
    its optimizer with this, the optimizer otherwise baking a stale annotation into the nodes it rewrites."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph_outputs = set(model.graph.outputs)
        for node in model.graph:
            for value in node.outputs:
                value.shape = None
                if value not in graph_outputs:
                    value.type = None
        return common_passes.ShapeInferencePass()(model)
