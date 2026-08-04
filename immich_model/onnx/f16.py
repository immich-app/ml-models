"""fp16 derivation over the fp32 graphs `export` writes, as an `ir` pass over lazily-read weights."""

from functools import cache
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
from onnx import defs

from ._ir import save_with_external_data
from .graph import ReinferPass, UnifyDimSymbolsPass

# what fp16 can represent, exclusively: values are clamped rather than rounded to zero or to infinity
SMALLEST_SUBNORMAL = 5.96e-08
LARGEST_FINITE = 65504.0


@cache
def _fp32_only_inputs(op_type: str, domain: str, version: int) -> frozenset[int]:
    """Input slots this op's schema forbids from being fp16, by index, read off the schema rather than
    transcribed -- a hand-written list narrows an op it does not cover into an invalid graph in silence."""
    try:
        schema = defs.get_schema(op_type, version, domain)
    except Exception:  # an op the local onnx has no schema for narrows like any other
        return frozenset()
    allowed = {constraint.type_param_str: set(constraint.allowed_type_strs) for constraint in schema.type_constraints}
    return frozenset(
        index
        for index, formal in enumerate(schema.inputs)
        if "tensor(float16)" not in allowed.get(formal.type_str, {formal.type_str})
    )


def narrow(array: np.ndarray) -> np.ndarray:
    """fp32 -> fp16, preserving sign and finiteness. NaN, the zeros and the infinities pass through."""
    magnitude = np.abs(array)
    clamped = np.where(
        (magnitude > 0) & (magnitude < SMALLEST_SUBNORMAL), np.copysign(SMALLEST_SUBNORMAL, array), array
    )
    clamped = np.where(np.isfinite(array) & (magnitude > LARGEST_FINITE), np.copysign(LARGEST_FINITE, array), clamped)
    return clamped.astype(np.float16)


class NarrowToFloat16Pass(ir.passes.InPlacePass):
    """Narrow every fp32 value to fp16 except the slots the op schemas pin, keeping the graph's own inputs and
    outputs fp32 behind a Cast. Weights are read one at a time so the peak is the largest single tensor, and
    casts are inserted at the producer, so the result is topologically ordered by construction."""

    def __init__(self, keep_io_types: bool = True) -> None:
        super().__init__()
        self.keep_io_types = keep_io_types

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        version = model.opset_imports.get("", 0)
        pinned = {
            node.inputs[index]
            for node in graph
            for index in _fp32_only_inputs(node.op_type, node.domain, node.version or version)
            if index < len(node.inputs) and node.inputs[index] is not None
        }

        for value in graph.initializers.values():
            tensor = value.const_value
            if tensor is None or tensor.dtype != ir.DataType.FLOAT or value in pinned:
                continue
            value.const_value = ir.tensor(narrow(tensor.numpy()), name=value.name)
            value.type = ir.TensorType(ir.DataType.FLOAT16)

        outputs = set(graph.outputs)
        for node in graph:
            # a Cast that targeted fp32 now targets fp16, unless it feeds a pinned slot or IS an output
            if node.op_type == "Cast" and node.attributes.get_int("to", 0) == ir.DataType.FLOAT:
                if not (node.outputs[0] in pinned or node.outputs[0] in outputs):
                    node.attributes["to"] = ir.Attr("to", ir.AttributeType.INT, ir.DataType.FLOAT16)
            # a Constant carries its tensor in an ATTRIBUTE, which no initializer walk reaches, and ORT types
            # the node from it, so setting the output type below alone leaves the node producing fp32
            if node.op_type == "Constant" and not (node.outputs[0] in pinned or node.outputs[0] in outputs):
                tensor = getattr(node.attributes.get("value"), "value", None)
                if tensor is not None and tensor.dtype == ir.DataType.FLOAT:
                    narrowed = ir.tensor(narrow(tensor.numpy()), name=tensor.name)
                    node.attributes["value"] = ir.Attr("value", ir.AttributeType.TENSOR, narrowed)
            for value in node.outputs:
                if value.dtype == ir.DataType.FLOAT and value not in pinned and value not in outputs:
                    value.type = ir.TensorType(ir.DataType.FLOAT16)

        if self.keep_io_types:
            for value in graph.inputs:
                if value.dtype == ir.DataType.FLOAT:
                    self._cast_after_input(graph, value)
            for index, value in enumerate(graph.outputs):
                if value.dtype == ir.DataType.FLOAT:
                    graph.outputs[index] = self._cast_before_output(graph, value)
        else:
            for index, value in enumerate(graph.outputs):
                if value.dtype == ir.DataType.FLOAT:
                    value.type = ir.TensorType(ir.DataType.FLOAT16)

        # a Cast the narrowing retargeted onto its own input type is now a full-tensor copy of nothing
        identities = [
            (node, source)
            for node in graph
            if node.op_type == "Cast"
            and (source := node.inputs[0]) is not None
            and source.dtype == node.attributes.get_int("to", 0)
            and node.outputs[0] not in outputs
        ]
        for node, source in identities:
            node.outputs[0].replace_all_uses_with(source)
            graph.remove(node, safe=True)
        return ir.passes.PassResult(model, True)

    def _cast_after_input(self, graph: ir.Graph, value: ir.Value) -> None:
        """Keep a float graph input fp32 and narrow it for everything downstream."""
        cast = ir.node("Cast", inputs=[value], attributes={"to": ir.DataType.FLOAT16}, name=f"{value.name}/narrow")
        cast.outputs[0].type = ir.TensorType(ir.DataType.FLOAT16)
        cast.outputs[0].shape = value.shape
        value.replace_all_uses_with(cast.outputs[0])
        cast.replace_input_with(0, value)
        graph.insert_before(next(iter(graph)), cast)

    def _cast_before_output(self, graph: ir.Graph, value: ir.Value) -> ir.Value:
        """Widen a graph output back to fp32. The OUTPUT keeps the name and the narrowed value inside takes
        the new one; the other way round reads the same on a diff but renames the graph's output."""
        name = value.name
        value.name = f"{name}_fp16"
        value.type = ir.TensorType(ir.DataType.FLOAT16)
        cast = ir.node("Cast", inputs=[value], attributes={"to": ir.DataType.FLOAT}, name=f"{name}/widen")
        widened = cast.outputs[0]
        widened.name = name
        widened.type = ir.TensorType(ir.DataType.FLOAT)
        widened.shape = value.shape
        producer = value.producer()
        graph.insert_after(producer, cast) if producer is not None else graph.append(cast)
        return widened


def derive(src: Path, dst: Path, outputs_fp16: bool = False) -> None:
    """Convert an fp32 graph to fp16, weights external; the re-inference is what places the casts correctly.
    onnxconverter_common's converter is no substitute, choking on the uint8 input Cast.

    `outputs_fp16` drops keep_io_types: widening is exact, so an fp32 output buys no precision and only moves
    the narrowing onto the host. Never for embeddings -- an fp16 one serialises into pgvector without raising."""
    model = ir.load(src)
    # `ReinferPass` restores the batch symbol but not the names the exporter asserts after its own last
    # inference pass, so the fp16 artifact would declare a different contract from its fp32 sibling.
    named = [
        [dim.value or "" if isinstance(dim, ir.SymbolicDim) else "" for dim in out.shape or ()]
        for out in model.graph.outputs
    ]
    ReinferPass()(model)
    NarrowToFloat16Pass(keep_io_types=not outputs_fp16)(model)

    for output, names in zip(model.graph.outputs, named):
        for axis, name in enumerate(names):
            # a dim inference resolved to a constant is better information than a name
            if name and output.shape is not None and isinstance(output.shape[axis], ir.SymbolicDim):
                output.shape[axis] = ir.SymbolicDim(name)

    # the names land after `ReinferPass` unified, so re-unify to pull the internals onto the asserted name
    UnifyDimSymbolsPass()(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_with_external_data(model, dst)
    onnx.checker.check_model(dst.as_posix())
