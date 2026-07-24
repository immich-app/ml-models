"""Post-export graph surgery for the CLIP encoders. Every transform mutates one lazy `ir.Model` between a
single load and save, so the large token/embedding tables stay mmapped and never inline into a protobuf.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import onnx
import onnx_ir as ir
import onnx_ir.passes.common as common_passes
from onnx import TensorProto, helper
from onnxscript.rewriter import RewritePass, RewriteRuleSet
from onnxscript.rewriter.pattern import MatchResult, OrValue, RewriteRuleClassBase

from ._ir import (
    CanonicalizeConstantsPass,
    ReinferShapesPass,
    const_array,
    const_ints,
    make_init,
    make_node,
    producer_of,
    single_use,
    sole_consumer,
)
from .graph import UnifyDimSymbolsPass
from .lowering import FoldConstantGatherElements, FoldZeroIndexGather

log = logging.getLogger(__name__)


class Probe(NamedTuple):
    shape: tuple[int, ...]
    value: np.ndarray | None  # captured only for small integer tensors (the shape domain)
    dtype: int  # onnx TensorProto elem_type


Probes = dict[str, list[Probe]]

_BROADCAST_OPS = {"Add", "Sub", "Mul", "Div", "Pow"}


class StripTorchMetadataPass(ir.passes.InPlacePass):
    """Drop torch provenance: its absolute paths make one commit export different bytes per checkout."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        stripped = 0
        for holder in (model.graph, *model.graph):
            for key in [k for k in holder.metadata_props if k.startswith("pkg.torch.")]:
                del holder.metadata_props[key]
                stripped += 1
        return ir.passes.PassResult(model, bool(stripped))


def probe_runtime(model: ir.Model) -> Probes:
    """Every tensor's shape (and small int values) at two batches: ground truth past static shape inference.
    Routed through disk because the larger text encoders exceed the 2GB in-memory protobuf cap."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        base = tmp / "base.onnx"
        ir.save(model, base.as_posix(), external_data="base.onnx.data")

        inferred_path = tmp / "inferred.onnx"
        onnx.shape_inference.infer_shapes_path(base.as_posix(), inferred_path.as_posix(), data_prop=True)
        # graph only (weights stay in base.onnx.data, referenced relatively) to read types
        probe = onnx.load(inferred_path.as_posix(), load_external_data=False)
        types = {
            v.name: v.type.tensor_type.elem_type
            for v in list(probe.graph.value_info) + list(probe.graph.output) + list(probe.graph.input)
        }
        existing = {o.name for o in probe.graph.output}
        for node in probe.graph.node:
            for out in node.output:
                if out and out not in existing and types.get(out, TensorProto.UNDEFINED) != TensorProto.UNDEFINED:
                    probe.graph.output.append(helper.make_tensor_value_info(out, types[out], None))
                    existing.add(out)

        path = tmp / "probe.onnx"  # small proto; initializers still point at base.onnx.data
        onnx.save(probe, path.as_posix())
        session = ort.InferenceSession(path.as_posix(), options, providers=["CPUExecutionProvider"])

        rng = np.random.default_rng(0)
        results: Probes = {}
        for batch in (2, 3):
            feed = {}
            for graph_input in session.get_inputs():
                dims = [batch if not isinstance(d, int) else d for d in graph_input.shape]
                if graph_input.type == "tensor(uint8)":
                    feed[graph_input.name] = rng.integers(0, 256, dims, dtype=np.uint8)
                elif graph_input.type in ("tensor(int32)", "tensor(int64)"):
                    int_dtype = np.int32 if graph_input.type == "tensor(int32)" else np.int64
                    feed[graph_input.name] = np.ones(dims, dtype=int_dtype)
                else:
                    feed[graph_input.name] = rng.standard_normal(dims).astype(np.float32)
            names = [o.name for o in session.get_outputs()]
            for name, arr in zip(names, session.run(None, feed)):
                value = arr.copy() if arr.dtype in (np.int32, np.int64) and arr.size <= 64 else None
                elem_type = int(helper.np_dtype_to_tensor_dtype(arr.dtype))
                results.setdefault(name, []).append(Probe(tuple(arr.shape), value, elem_type))

        del session
    return results


class FuseVisualInputPass(ir.passes.InPlacePass):
    """Retype the visual input to uint8 NHWC and fold the normalization into the stem. A padded stem takes the
    scale only: subtracting a mean does not commute with a zero pad, so its shift stays an in-graph Sub."""

    def __init__(self, mean: list[float], std: list[float]) -> None:
        self.mean, self.std = mean, std

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        input_value = graph.inputs[0]
        input_name = input_value.name
        dims = [d if isinstance(d, int) else d.value for d in input_value.shape]
        assert dims[1] == 3, f"expected NCHW visual input, got {dims}"
        size = int(dims[2])

        # captured before the pre nodes are added, which would add uses of their own
        input_uses = [(use.node, use.idx) for use in input_value.uses()]
        all_consumers = input_value.consumers()
        # Shape consumers only read dims, and the NCHW tensor they get repointed to declares the same ones
        consumers = [n for n in all_consumers if n.op_type == "Conv"]
        unexpected = [n.op_type for n in all_consumers if n.op_type not in ("Conv", "Shape")]
        if not consumers or unexpected:
            raise ValueError(f"Cannot fuse visual input: consumed by {unexpected or 'nothing'}")

        scale = 1.0 / (255.0 * np.asarray(self.std, dtype=np.float64))
        shift = np.asarray(self.mean, dtype=np.float64) / np.asarray(self.std, dtype=np.float64)

        def _padded(conv: ir.Node) -> bool:
            if any(conv.attributes.get_ints("pads", [])):
                return True
            return conv.attributes.get_string("auto_pad", "NOTSET") not in ("NOTSET", "VALID")

        fold_shift = not any(_padded(conv) for conv in consumers)

        for conv in consumers:
            w_value = conv.inputs[1]
            weight = w_value.const_value.numpy().astype(np.float64)  # [O, C, kH, kW]
            folded = (weight * scale[None, :, None, None]).astype(np.float32)
            w_value.const_value = ir.tensor(folded, name=w_value.name)
            if not fold_shift:
                continue

            bias_delta = -(weight.sum(axis=(2, 3)) @ shift)
            if len(conv.inputs) > 2 and conv.inputs[2] is not None:
                b_value = conv.inputs[2]
                bias = b_value.const_value.numpy().astype(np.float64) + bias_delta
                b_value.const_value = ir.tensor(bias.astype(np.float32), name=b_value.name)
            else:
                b_value = make_init(graph, f"{conv.name}_fused_bias", bias_delta.astype(np.float32))
                conv.resize_inputs(3)
                conv.replace_input_with(2, b_value)

        cast = make_node("Cast", [input_value], name="pre_cast", out=f"{input_name}_f32", to=int(TensorProto.FLOAT))
        pre = [cast]
        transpose_in = cast.outputs[0]
        if not fold_shift:
            shift_value = make_init(
                graph, f"{input_name}_shift", (255.0 * np.asarray(self.mean, dtype=np.float64)).astype(np.float32)
            )
            sub_node = make_node("Sub", [transpose_in, shift_value], name="pre_shift", out=f"{input_name}_shifted")
            pre.append(sub_node)
            transpose_in = sub_node.outputs[0]
        transpose_node = make_node(
            "Transpose", [transpose_in], name="pre_nhwc_to_nchw", out=f"{input_name}_chw", perm=[0, 3, 1, 2]
        )
        pre.append(transpose_node)
        chw_value = transpose_node.outputs[0]

        for node, idx in input_uses:
            node.replace_input_with(idx, chw_value)
        graph.extend(pre)
        common_passes.TopologicalSortPass()(model)

        input_value.dtype = ir.DataType.UINT8
        input_value.shape = ir.Shape([dims[0], size, size, 3])

        return ReinferShapesPass()(model)


class _EotOneHotSelect(RewriteRuleClassBase):
    """`x[arange(batch), idx]` EOT pooling (GatherND) as a one-hot matmul: exact, with ops every EP claims under
    dynamic batch. GatherND(batch_dims=1) is leaner but the CoreML builder (#28598) needs constant indices."""

    def __init__(self, probes: Probes) -> None:
        super().__init__(remove_nodes=False)
        self._probes = probes

    def pattern(self, op: Any, data: Any, start: Any, limit: Any, delta: Any, index: Any, ax1: Any, ax2: Any) -> Any:
        rng = op.Range(start, limit, delta, _outputs=["rng"])
        cols = op.Concat(op.Unsqueeze(rng, ax1), op.Unsqueeze(index, ax2), _allow_other_attributes=True)
        return op.GatherND(data, cols, _allow_other_attributes=True, _outputs=["out"])

    def check(self, context: Any, data: Any, rng: Any, out: Any, **_: Any) -> MatchResult:
        result = MatchResult()
        if out.producer().attributes.get_int("batch_dims", 0) != 0:
            return result.fail("GatherND has batch_dims")
        for entry in self._probes[rng.name]:
            is_arange = entry.value is not None and np.array_equal(entry.value, np.arange(entry.shape[0]))
            assert is_arange, "Range is not arange(batch)"
        seq_lengths = {entry.shape[1] for entry in self._probes[data.name]}
        assert len(seq_lengths) == 1, f"sequence dim varies across probes: {seq_lengths}"
        return result

    def rewrite(self, op: Any, data: Any, index: Any, out: Any, **_: Any) -> Any:
        base = out.name
        seq = self._probes[data.name][0].shape[1]
        # mixed-dtype MatMul is illegal, so the eye takes data's dtype; a one-hot is exact at any precision
        eye_np_dtype = helper.tensor_dtype_to_np_dtype(self._probes[data.name][0].dtype)
        eye = op.initializer(ir.tensor(np.eye(seq, dtype=eye_np_dtype), name=f"{base}_eye"))
        axes = op.initializer(ir.tensor(np.array([1], dtype=np.int64), name=f"{base}_axes1"))
        onehot = op.Unsqueeze(op.Gather(eye, index, axis=0), axes)
        return op.Squeeze(op.MatMul(onehot, data), axes)


class RewriteEotGatherndPass(RewritePass):
    def __init__(self, probes: Probes) -> None:
        super().__init__([_EotOneHotSelect.rule(probes)])


class ConstantifyPositionIdsPass(ir.passes.InPlacePass):
    """Replace XLM-R's data-dependent ``position_ids`` with the constant arange an all-non-pad input yields: pad
    positions are masked downstream, so only the non-pad ids matter. RKNPU has no int32 ``Equal`` kernel."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        import onnxruntime as ort

        graph = model.graph
        init_names = graph.initializers  # name -> Value
        graph_inputs = set(graph.inputs)

        def is_terminal(value: ir.Value) -> bool:
            return value.is_initializer() or value.name in init_names or value in graph_inputs

        def reaches_cumsum(value: ir.Value | None, seen: set) -> bool:
            if value is None or value.name in seen or is_terminal(value):
                return False
            seen.add(value.name)
            node = value.producer()
            return node is not None and (node.op_type == "CumSum" or any(reaches_cumsum(i, seen) for i in node.inputs))

        # not a re-gather: that is the token-type path downstream, and replacing the root kills the chain
        position_ids: ir.Value | None = None
        for node in graph:
            if node.op_type != "Gather" or node.inputs[0] is None or not is_terminal(node.inputs[0]):
                continue
            table = node.inputs[0].const_value  # metadata only; never materializes the table
            if table is None:
                continue
            idx = node.inputs[1]
            idx_producer = idx.producer() if idx is not None else None
            if (
                len(table.shape) == 2
                and np.issubdtype(table.dtype.numpy(), np.floating)
                and reaches_cumsum(idx, set())
                and (idx_producer is None or idx_producer.op_type not in ("Gather", "GatherElements", "GatherND"))
            ):
                position_ids = idx
                break
        if position_ids is None:
            return ir.passes.PassResult(model, False)

        sub_graph = ir.convenience.extract(graph, list(graph.inputs), [position_ids])
        sub_model = helper.make_model(
            ir.to_proto(sub_graph), opset_imports=[helper.make_opsetid(d, v) for d, v in graph.opset_imports.items()]
        )
        session = ort.InferenceSession(sub_model.SerializeToString(), providers=["CPUExecutionProvider"])
        feed = {}
        for i in graph.inputs:
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
            np_dtype = {ir.DataType.INT32: np.int32, ir.DataType.INT64: np.int64}.get(i.dtype, np.int64)
            feed[i.name] = np.ones(shape, np_dtype) if "mask" in i.name.lower() else np.full(shape, 100, np_dtype)
        pid_dtype = position_ids.dtype if position_ids.dtype is not None else ir.DataType.INT64
        const = session.run([position_ids.name], feed)[0].astype(pid_dtype.numpy())

        replacement = make_init(graph, position_ids.name, const)
        position_ids.replace_all_uses_with(replacement)
        # sweep the retired derivation now: `FoldPadMaskPass` reads the shared `ne` and fails closed on danglers
        common_passes.RemoveUnusedNodesPass()(model)
        return ir.passes.PassResult(model, True)


class _FoldConstantGather(RewriteRuleClassBase):
    """``Gather(const_table, const_indices, axis=0)`` -> the gathered rows as one constant, retiring the position
    rows a context can never reach. Gated on shrinking the table, so a long index cannot inline a copy instead."""

    def pattern(self, op: Any, table: Any, indices: Any) -> Any:
        return op.Gather(table, indices, _outputs=["gathered"])

    def check(self, context: Any, table: Any, indices: Any, gathered: Any) -> MatchResult:
        result = MatchResult()
        if gathered.producer().attributes.get_int("axis", 0) != 0:
            return result.fail("Gather axis is not 0")
        rows = table.const_value  # metadata only: the token table must not materialize here
        index = const_array(indices)
        if rows is None or index is None or len(rows.shape) != 2 or index.dtype.kind not in "iu":
            return result.fail("table or indices is not a constant lookup")
        if index.size >= int(rows.shape[0]):
            return result.fail("the gathered rows would not be smaller than the table")
        return result

    def rewrite(self, op: Any, table: Any, indices: Any, gathered: Any) -> Any:
        return op.Constant(value=ir.tensor(table.const_value.numpy()[const_array(indices)]))


_MASK_ISLAND_OPS = ("Cast", "Unsqueeze", "ReduceSum", "GatherND", "And")


def _reaches(value: ir.Value | None, predicate: Any, budget: int = 32) -> bool:
    """Backward walk to a value satisfying `predicate`, never entering a `Range`: its limit is batch-derived."""
    stack, seen = [value], set()
    while stack and len(seen) < budget:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if predicate(current):
            return True
        node = current.producer()
        if node is not None and node.op_type != "Range":
            stack.extend(node.inputs)
    return False


def _is_batch_range(value: ir.Value) -> bool:
    node = value.producer()
    if node is None or node.op_type != "Range":
        return False
    start, _, delta = (i.const_value for i in node.inputs)
    return start is not None and int(start.numpy()) == 0 and delta is not None and int(delta.numpy()) == 1


def _row_broadcast_gathernd(node: ir.Node, data: ir.Value) -> bool:
    """`GatherND(mask[b,S], idx)` whose index tuples are exactly `(i, j)`: a broadcast dressed up as a gather."""
    if node.attributes.get_int("batch_dims", 0) != 0 or data.shape is None or len(data.shape) != 2:
        return False
    seq = data.shape[1]
    out_dims = node.outputs[0].shape
    if not isinstance(seq, int) or out_dims is None or list(out_dims[1:]) != [1, 1, seq]:
        return False

    def is_arange(value: ir.Value) -> bool:
        const = value.const_value
        arr = const.numpy() if const is not None else None
        return arr is not None and arr.size == seq and np.array_equal(arr.reshape(-1), np.arange(seq, dtype=arr.dtype))

    indices = node.inputs[1]
    return _reaches(indices, _is_batch_range) and _reaches(indices, is_arange)


def _all_true_operand(node: ir.Node, other: ir.Value) -> bool:
    """The `And`'s constant operand carries no masking of its own and stretches the key row no further than
    `[b,1,S,S]`, the shape the rewrite rebuilds by hand."""
    const = next((v.const_value for v in node.inputs if v is not other), None)
    dims = other.shape
    if const is None or const.dtype != ir.DataType.BOOL or len(const.shape) > 4 or dims is None:
        return False
    if any(d not in (1, dims[-1]) for d in const.shape):
        return False
    return bool(const.numpy().all())


class FoldPadMaskPass(ir.passes.InPlacePass):
    """Rebuild the openclip XLM-R/NLLB towers' key-padding mask as one float lookup into a `[V]` keep table, indexed
    by the same `text` the token embedding already gathers, retiring the integer island that fragments them. A lookup
    rather than `Clip(Abs(Cast(text) - pad), 0, 1)` because token ids run past fp16's range. Must follow
    `ConstantifyPositionIdsPass`: the position ids come off the same `ne`, so the root folds only once it is gone."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        ids = [v for v in graph.inputs if v.dtype is not None and v.dtype.is_integer()]

        def fold(root: ir.Value, text: ir.Value, pad_id: int) -> bool:
            if root.shape is None or len(root.shape) != 2 or not isinstance(root.shape[1], int):
                return False
            seq = root.shape[1]
            if seq >= 1 << 24:  # the float token count must stay in fp32's exact-integer range
                return False

            # vocabulary from the token-embedding table this same input indexes; metadata only, never materialized
            table = next(
                (
                    use.node.inputs[0]
                    for use in text.uses()
                    if use.node.op_type == "Gather"
                    and use.idx == 1
                    and use.node.attributes.get_int("axis", 0) == 0
                    and use.node.inputs[0].const_value is not None
                    and len(use.node.inputs[0].shape) == 2
                ),
                None,
            )
            if table is None or not 0 <= pad_id < int(table.shape[0]):
                return False

            island: dict[int, ir.Value] = {id(root): root}
            parent: dict[int, ir.Value] = {}
            float_exits: list[tuple[ir.Node, ir.Value]] = []
            attention_mask: ir.Value | None = None
            # read the compute precision off the exits: any other float dtype is an illegal Attention input
            precisions: set[ir.DataType] = set()
            stack = [root]
            while stack:
                value = stack.pop()
                for use in value.uses():
                    node = use.node
                    if node.op_type == "Cast" and ir.DataType(node.attributes.get_int("to")).is_floating_point():
                        float_exits.append((node, value))
                        precisions.add(ir.DataType(node.attributes.get_int("to")))
                        continue
                    if node.op_type == "Attention" and use.idx == 3:
                        attention_mask = value
                        precisions.add(node.inputs[0].dtype)
                        continue
                    if node.op_type not in _MASK_ISLAND_OPS:
                        return False
                    if node.op_type == "GatherND" and not _row_broadcast_gathernd(node, value):
                        return False
                    if node.op_type == "And" and not _all_true_operand(node, value):
                        return False
                    out = node.outputs[0]
                    if id(out) not in island:
                        island[id(out)], parent[id(out)] = out, value
                        stack.append(out)
            if len(precisions) != 1:
                return False

            np_float = precisions.pop().numpy()
            keep_array = np.ones(int(table.shape[0]), np_float)
            keep_array[pad_id] = 0.0
            keep_init = make_init(graph, f"{text.name}_pad_keep", keep_array)
            keep_node = make_node("Gather", [keep_init, text], name="pad_keep", out=f"{text.name}_keep", axis=0)
            new_nodes = [keep_node]
            mirrored: dict[int, ir.Value] = {id(root): keep_node.outputs[0]}

            def emit(op_type: str, inputs: list[ir.Value], ref: ir.Value, **attributes: Any) -> ir.Value:
                node = make_node(op_type, inputs, name=f"{ref.name}_f", out=f"{ref.name}_f", **attributes)
                new_nodes.append(node)
                return node.outputs[0]

            def const(array: np.ndarray, name: str) -> ir.Value:
                return make_init(graph, f"{text.name}_{name}", array)

            def mirror(value: ir.Value) -> ir.Value:
                """The float equivalent of an island value, built on demand."""
                if id(value) in mirrored:
                    return mirrored[id(value)]
                node = value.producer()
                source = mirror(parent[id(value)])
                if node.op_type in ("Cast", "And"):  # dtype plumbing / an all-true operand: value unchanged
                    result = source
                elif node.op_type == "Unsqueeze":
                    result = emit("Unsqueeze", [source, node.inputs[1]], value)
                elif node.op_type == "ReduceSum":  # <= seq ones: exact in float
                    names = ("keepdims", "noop_with_empty_axes")  # keepdims=0 is meaningful: never drop it
                    attrs = {k: v for k in names if (v := node.attributes.get_int(k)) is not None}
                    result = emit("ReduceSum", [source, node.inputs[1]], value, **attrs)
                else:  # GatherND: the broadcast its index tuples spell out
                    result = emit("Unsqueeze", [source, const(np.array([1, 2], np.int64), "row_axes")], value)
                mirrored[id(value)] = result
                return result

            for cast_node, source in float_exits:
                cast_node.outputs[0].replace_all_uses_with(mirror(source), replace_graph_outputs=True)
            if attention_mask is not None:
                row = mirror(attention_mask)  # [b,1,1,S], 1.0 keep / 0.0 pad
                shifted = emit("Sub", [row, const(np.array(1.0, np_float), "one")], attention_mask)
                additive = make_node(
                    "Mul", [shifted, const(np.array(1.0e4, np_float), "scale")], out=f"{row.name}_bias"
                )
                # ORT CPU refuses to broadcast the mask's query axis, so materialize it
                biased = make_node(
                    "Add", [additive.outputs[0], const(np.zeros((seq, 1), np_float), "q_axis")], out=f"{row.name}_mask"
                )
                new_nodes += [additive, biased]
                for use in list(attention_mask.uses()):
                    use.node.replace_input_with(use.idx, biased.outputs[0])

            graph.extend(new_nodes)
            common_passes.TopologicalSortPass()(model)
            common_passes.RemoveUnusedNodesPass()(model)  # the island, incl. Expands a later pass would materialize
            return True

        for node in list(graph):
            if node.op_type != "Equal":
                continue
            text = next((v for v in node.inputs if any(v is i for i in ids)), None)
            pad = next((v for v in node.inputs if v is not text and const_array(v) is not None), None)
            negate = sole_consumer(node.outputs[0], "Not")
            if text is None or pad is None or const_array(pad).size != 1 or negate is None:
                continue
            if fold(negate.outputs[0], text, int(const_array(pad).reshape(-1)[0])):
                return ir.passes.PassResult(model, True)
        return ir.passes.PassResult(model, False)


class _ConstantifyReshapeTarget(RewriteRuleClassBase):
    """Replace a batch-derived Reshape target with a constant pinned from the OUTPUT's probed shape at two batches
    (agreeing dims literal, the varying one -> -1). The is_initializer guard is what terminates the rule."""

    def __init__(self, probes: Probes) -> None:
        super().__init__(remove_nodes=False)
        self._probes = probes

    def pattern(self, op: Any, data: Any, target: Any) -> Any:
        return op.Reshape(data, target, _allow_other_attributes=True, _outputs=["out"])

    def check(self, context: Any, target: Any, out: Any, **_: Any) -> MatchResult:
        result = MatchResult()
        if target.is_initializer():
            return result.fail("target is already constant")
        entries = self._probes.get(out.name)
        if entries is None:
            return result.fail("no probe for the reshape output")
        shapes = [np.asarray(entry.shape, dtype=np.int64) for entry in entries]
        if len(np.nonzero(shapes[0] != shapes[1])[0]) > 1:
            return result.fail("more than one batch-varying dim")
        return result

    def rewrite(self, op: Any, data: Any, out: Any, **_: Any) -> Any:
        shapes = [np.asarray(entry.shape, dtype=np.int64) for entry in self._probes[out.name]]
        target = shapes[0]
        target[np.nonzero(shapes[0] != shapes[1])[0]] = -1
        init = op.initializer(ir.tensor(target, name=f"{out.name}_target"))
        return op.Reshape(data, init, **out.producer().attributes)


class ConstantifyReshapeTargetsPass(RewritePass):
    """`_ConstantifyReshapeTarget`, with every dynamic Reshape target required to have been pinned."""

    def __init__(self, probes: Probes) -> None:
        super().__init__([_ConstantifyReshapeTarget.rule(probes)])

    def ensures(self, model: ir.Model) -> None:
        unresolved = [
            node.name for node in model.graph if node.op_type == "Reshape" and not node.inputs[1].is_initializer()
        ]
        if unresolved:
            raise ir.passes.PostconditionError(f"Reshape targets not resolvable from probes: {unresolved}")


class EliminateDynamicExpandsPass(ir.passes.InPlacePass):
    """Remove Expand nodes with batch-derived target shapes: rewire to `data` where the consumer broadcasts it back
    anyway, else `data + static_zeros + batch_col`, which leaves only batch symbolic so runtime compilers accept it."""

    def __init__(self, probes: Probes) -> None:
        self.probes = probes

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        probes = self.probes
        graph = model.graph
        initializers = dict(graph.initializers)  # name -> ir.Value, snapshot (matches proto)

        def shape_at(value: ir.Value | None, probe_idx: int) -> tuple[int, ...] | None:
            if value is None:
                return None
            if value.name in initializers:
                return tuple(int(d) for d in initializers[value.name].shape)
            entries = probes.get(value.name)
            return entries[probe_idx].shape if entries else None

        def dtype_of(value: ir.Value) -> int:
            if value.name in initializers:
                return int(initializers[value.name].dtype)
            if value.name in probes:
                return probes[value.name][0].dtype
            raise ValueError(f"Unknown dtype for Expand data {value.name!r}")

        batch_input = next((i for i in graph.inputs if not isinstance(i.shape[0], int)), None)
        probe_batches = (
            tuple(int(p.shape[0]) for p in probes[batch_input.name])
            if batch_input is not None and batch_input.name in probes
            else None
        )

        batch_seed: list[ir.Value] = []  # shared across expands

        def batch_zeros_1d() -> ir.Value:
            """A `[batch]` rank-1 float zero column tied to the runtime batch dim. Built by slicing rather than
            reducing: a global reduction overflows fp16 to inf*0=NaN and the ANE compiler cannot lower it."""
            if batch_seed:
                return batch_seed[0]
            if batch_input is None:
                raise ValueError("No batch-carrying graph input to seed Expand materialization")
            src = batch_input.name
            nonbatch = list(range(1, len(batch_input.shape)))
            # TensorRT cannot Slice a uint8 tensor, so slice the fused visual input's existing float Cast instead
            cast_of_input = next((use.node for use in batch_input.uses() if use.node.op_type == "Cast"), None)
            reuse_float = batch_input.dtype == ir.DataType.UINT8 and cast_of_input is not None
            slice_src = cast_of_input.outputs[0] if reuse_float else batch_input
            starts = make_init(graph, f"{src}_ez_starts", np.zeros(len(nonbatch), dtype=np.int64))
            ends = make_init(graph, f"{src}_ez_ends", np.ones(len(nonbatch), dtype=np.int64))
            axes = make_init(graph, f"{src}_ez_axes", np.array(nonbatch, dtype=np.int64))
            flat = make_init(graph, f"{src}_ez_flat", np.array([-1], dtype=np.int64))
            zero = make_init(graph, f"{src}_ez_zero", np.zeros([], dtype=np.float32))
            slice_node = make_node("Slice", [slice_src, starts, ends, axes], name="ez_slice", out=f"{src}_ez_s")
            seed_nodes = [slice_node]
            flat_src = slice_node.outputs[0]
            if not reuse_float:
                cast_node = make_node("Cast", [flat_src], name="ez_cast", out=f"{src}_ez_f", to=TensorProto.FLOAT)
                seed_nodes.append(cast_node)
                flat_src = cast_node.outputs[0]
            reshape_node = make_node("Reshape", [flat_src, flat], name="ez_flatten", out=f"{src}_ez_r")
            mul_node = make_node("Mul", [reshape_node.outputs[0], zero], name="ez_mul", out=f"{src}_ez")
            seed_nodes += [reshape_node, mul_node]
            graph.extend(seed_nodes)
            batch_seed.append(mul_node.outputs[0])
            return batch_seed[0]

        eliminated = 0
        for node in list(graph):
            if node.op_type != "Expand" or node.inputs[1].is_initializer():
                continue
            data, out = node.inputs[0], node.outputs[0]
            consumers = out.consumers()

            def broadcast_invariant() -> bool:
                for consumer in consumers:
                    if consumer.op_type not in _BROADCAST_OPS:
                        return False
                    others = [i for i in consumer.inputs if i is not out]
                    for p in range(len(probes[out.name])):
                        shapes = [shape_at(o, p) for o in others]
                        d_shape = shape_at(data, p)
                        if d_shape is None or any(s is None for s in shapes):
                            return False
                        try:
                            with_data = np.broadcast_shapes(d_shape, *shapes)
                            with_expand = np.broadcast_shapes(probes[out.name][p].shape, *shapes)
                        except ValueError:
                            return False
                        if with_data != with_expand:
                            return False
                return True

            if broadcast_invariant():
                out.replace_all_uses_with(data)
                graph.remove(node, safe=True)
                eliminated += 1
                continue

            shapes = [np.asarray(entry.shape, dtype=np.int64) for entry in probes[out.name]]
            varying = np.nonzero(shapes[0] != shapes[1])[0]
            if len(varying) != 1:
                raise ValueError(f"Expand {node.name} has {len(varying)} batch-varying axes, expected 1: {shapes}")
            rank = len(shapes[0])
            axis = int(varying[0])
            if probe_batches is not None and (int(shapes[0][axis]), int(shapes[1][axis])) != probe_batches:
                raise ValueError(
                    f"Expand {node.name}: dynamic axis {axis} varies as "
                    f"{(int(shapes[0][axis]), int(shapes[1][axis]))}, a multiple of batch {probe_batches}; "
                    "materialization from a [batch] seed would mis-size it"
                )
            data_dtype = dtype_of(data)
            # ONNX Add has no bool kernel, so bool data round-trips through int32
            compute_dtype = TensorProto.INT32 if data_dtype == TensorProto.BOOL else data_dtype
            np_compute = helper.tensor_dtype_to_np_dtype(compute_dtype)

            static_shape = shapes[0].copy()  # target shape with the batch axis pinned to 1
            static_shape[axis] = 1
            static_iv = make_init(graph, f"{out.name}_static0", np.zeros(static_shape, dtype=np_compute))

            seed = batch_zeros_1d()  # [batch], float32
            new_nodes: list[ir.Node] = []

            data_in = data
            if data_dtype == TensorProto.BOOL:
                cast = make_node(
                    "Cast", [data], name=f"{node.name}_data_cast", out=f"{out.name}_data_i", to=compute_dtype
                )
                data_in = cast.outputs[0]
                new_nodes.append(cast)

            col = seed
            if rank > 1:
                unsqueeze_axes = [a for a in range(rank) if a != axis]
                ua_iv = make_init(graph, f"{out.name}_ua", np.array(unsqueeze_axes, dtype=np.int64))
                unsq = make_node("Unsqueeze", [col, ua_iv], name=f"{node.name}_col", out=f"{out.name}_col")
                col = unsq.outputs[0]
                new_nodes.append(unsq)
            if compute_dtype != TensorProto.FLOAT:
                castc = make_node(
                    "Cast", [col], name=f"{node.name}_col_cast", out=f"{out.name}_col_c", to=compute_dtype
                )
                col = castc.outputs[0]
                new_nodes.append(castc)

            add_static = make_node(
                "Add", [data_in, static_iv], name=f"{node.name}_bcast_static", out=f"{out.name}_static"
            )
            add_bcast = make_node("Add", [add_static.outputs[0], col], name=f"{node.name}_bcast")
            new_nodes += [add_static, add_bcast]

            if data_dtype == TensorProto.BOOL:
                add_bcast.outputs[0].name = f"{out.name}_pre"
                out_cast = make_node("Cast", [add_bcast.outputs[0]], name=f"{node.name}_out_cast", to=TensorProto.BOOL)
                final = out_cast.outputs[0]
                new_nodes.append(out_cast)
            else:
                final = add_bcast.outputs[0]

            out.replace_all_uses_with(final)
            graph.remove(node, safe=True)
            final.name = out.name  # inherit the Expand output name (safe: `out` is now orphaned)
            for n in new_nodes:
                graph.append(n)
            eliminated += 1

        common_passes.TopologicalSortPass()(model)
        log.info("Eliminated %d dynamic Expand nodes", eliminated)
        return ir.passes.PassResult(model, bool(eliminated))


class _FuseClassTokenPrepend(RewriteRuleClassBase):
    """Fuse the ViT class-token prepend and positional-embedding add into a static `Pad` plus one constant, retiring
    the class-token Expand and the batch-tied zero column it needs. The second pattern arm is the shape
    `EliminateDynamicExpandsPass` leaves, so either order fuses."""

    def pattern(
        self, op: Any, class_embedding: Any, expand_shape: Any, patches: Any, positional: Any, zeros: Any, col: Any
    ) -> Any:
        expanded = OrValue(
            [op.Expand(class_embedding, expand_shape), op.Add(op.Add(class_embedding, zeros), col)],
            name="prepended",
        )
        return op.Add(op.Concat(expanded, patches, axis=1), positional)

    def check(
        self, context: Any, class_embedding: Any, expand_shape: Any, positional: Any, zeros: Any = None, **_: Any
    ) -> MatchResult:
        result = MatchResult()
        cls, pos = class_embedding.const_value, positional.const_value
        if cls is None or pos is None:
            return result.fail("class/positional not constant")
        if cls.numpy().reshape(-1).shape[0] != pos.numpy().shape[-1]:
            return result.fail("class width != positional width")
        if expand_shape is None and (const_array(zeros) is None or np.any(const_array(zeros))):
            return result.fail("the materialized prepend does not widen against a broadcast-neutral zero")
        return result

    def rewrite(self, op: Any, class_embedding: Any, patches: Any, positional: Any, **_: Any) -> Any:
        pos_arr = positional.const_value.numpy()
        cls_arr = class_embedding.const_value.numpy()
        w = pos_arr.shape[-1]
        seq = pos_arr.size // w
        cp = pos_arr.astype(np.float32).reshape(seq, w).copy()
        cp[0] += cls_arr.reshape(-1).astype(np.float32)  # class + pos[0] folded in fp32
        cp = cp.reshape(1, seq, w).astype(pos_arr.dtype)
        pads = op.Constant(value=ir.tensor(np.array([0, 1, 0, 0, 0, 0], np.int64)))
        pad_value = op.Constant(value=ir.tensor(np.array(0, pos_arr.dtype)))
        padded = op.Pad(patches, pads, pad_value)
        return op.Add(padded, op.Constant(value=ir.tensor(cp)))


class _ScalarGatherToSlice(RewriteRuleClassBase):
    """``Gather(data, scalar_index)`` -> ``Slice`` + ``Squeeze``: the CoreML EP rejects a scalar-index Gather unless
    data is fully static. After DCE only: a shape-domain scalar would leave a rank-0 Squeeze MIL refuses to build."""

    def pattern(self, op: Any, data: Any, index: Any) -> Any:
        return op.Gather(data, index, _outputs=["gathered"])

    def check(self, context: Any, data: Any, index: Any, gathered: Any) -> MatchResult:
        result = MatchResult()
        value = index.const_value
        if value is None or value.numpy().ndim != 0:
            return result.fail("Gather index is not a scalar constant")
        return result

    def rewrite(self, op: Any, data: Any, index: Any, gathered: Any) -> Any:
        i = int(index.const_value.numpy())
        axis = gathered.producer().attributes.get_int("axis", 0)
        axis_const = op.Constant(value=ir.tensor(np.array([axis], np.int64)))
        dims = data.shape
        if dims is not None and isinstance(dims[axis], int) and dims[axis] == 1:
            return op.Squeeze(data, axis_const)  # the axis has one element: the Slice is a no-op copy
        end = i + 1 if i != -1 else np.iinfo(np.int64).max  # a -1 index slices to the end
        sliced = op.Slice(
            data,
            op.Constant(value=ir.tensor(np.array([i], np.int64))),
            op.Constant(value=ir.tensor(np.array([end], np.int64))),
            axis_const,
        )
        return op.Squeeze(sliced, axis_const)


class _ScalarGatherToSlicePass(RewritePass):
    def __init__(self) -> None:
        super().__init__([_ScalarGatherToSlice.rule()])

    def requires(self, model: ir.Model) -> None:
        # lowering first would strand `_SelectBeforeLayerNorm`, which roots on the same Gather: correct but slower
        stuck = sum(
            1
            for node in model.graph
            if node.op_type == "Gather"
            and (index := const_array(node.inputs[1])) is not None
            and index.ndim == 0
            and producer_of(node.inputs[0], "LayerNormalization") is not None
        )
        if stuck:
            raise ir.passes.PreconditionError(f"{stuck} select(s) still to hoist over their LayerNormalization")


class _SelectBeforeLayerNorm(RewriteRuleClassBase):
    """``Gather(LayerNormalization(x), scalar)`` -> ``LayerNormalization(Gather(x))``: last-axis LN normalizes each
    token independently, so the select commutes bit-exactly. Ordered ahead of ``_ScalarGatherToSlice``."""

    def pattern(self, op: Any, x: Any, scale: Any, bias: Any, index: Any) -> Any:
        ln = op.LayerNormalization(x, scale, bias, _outputs=["ln"])
        return op.Gather(ln, index, _outputs=["sel"])

    def check(self, context: Any, x: Any, scale: Any, bias: Any, index: Any, ln: Any, sel: Any) -> MatchResult:
        result = MatchResult()
        if index.const_value is None or index.const_value.numpy().ndim != 0:
            return result.fail("Gather index is not a scalar constant")
        if x.shape is None:
            return result.fail("LayerNormalization input rank unknown")
        rank = len(x.shape)
        ln_axis = ln.producer().attributes.get_int("axis", -1) % rank
        if ln_axis != rank - 1:
            return result.fail("LayerNormalization spans more than the last axis")
        if sel.producer().attributes.get_int("axis", 0) % rank == ln_axis:
            return result.fail("Gather selects along the normalized axis")
        return result

    def rewrite(self, op: Any, x: Any, scale: Any, bias: Any, index: Any, ln: Any, sel: Any) -> Any:
        rank = len(x.shape)
        attrs = ln.producer().attributes
        selected = op.Gather(x, index, axis=sel.producer().attributes.get_int("axis", 0))
        return op.LayerNormalization(
            selected,
            scale,
            bias,
            axis=(attrs.get_int("axis", -1) % rank) - rank,
            epsilon=attrs.get_float("epsilon", 1e-5),
            stash_type=attrs.get_int("stash_type", 1),
        )


def _is_onehot_selector(value: ir.Value | None) -> bool:
    """The ``Unsqueeze(Gather(eye, index))`` row picker ``_EotOneHotSelect`` emits; pinning the identity
    table keeps ordinary ``MatMul(x, weight)`` from passing for a select."""
    unsqueeze = producer_of(value, "Unsqueeze")
    gather = producer_of(unsqueeze.inputs[0], "Gather") if unsqueeze is not None else None
    table = const_array(gather.inputs[0]) if gather is not None else None
    if table is None or table.ndim != 2 or table.shape[0] != table.shape[1]:
        return False
    return bool(np.array_equal(table, np.eye(table.shape[0], dtype=table.dtype)))


class _EotSelectBeforeLayerNorm(RewriteRuleClassBase):
    """``MatMul(one_hot, LN(x))`` -> ``LN(MatMul(one_hot, x))``: the EOT select commutes with a last-axis norm."""

    def pattern(self, op: Any, selector: Any, x: Any, scale: Any, bias: Any) -> Any:
        ln = op.LayerNormalization(x, scale, bias, _outputs=["ln"])
        return op.MatMul(selector, ln, _outputs=["mm"])

    def check(self, context: Any, selector: Any, x: Any, scale: Any, bias: Any, ln: Any, mm: Any) -> MatchResult:
        result = MatchResult()
        if not _is_onehot_selector(selector):
            return result.fail("selector is not the EOT Unsqueeze(Gather(eye)) one-hot chain")
        if x.shape is None:
            return result.fail("LayerNormalization input rank unknown")
        rank = len(x.shape)
        if ln.producer().attributes.get_int("axis", -1) % rank != rank - 1:
            return result.fail("LayerNormalization spans more than the last axis")
        return result

    def rewrite(self, op: Any, selector: Any, x: Any, scale: Any, bias: Any, ln: Any, mm: Any) -> Any:
        attrs = ln.producer().attributes
        return op.LayerNormalization(
            op.MatMul(selector, x),
            scale,
            bias,
            axis=attrs.get_int("axis", -1),
            epsilon=attrs.get_float("epsilon", 1e-5),
            stash_type=attrs.get_int("stash_type", 1),
        )


class _IdentityAveragePool(RewriteRuleClassBase):
    """``AveragePool`` with kernel 1, stride 1 and no pad is an identity copy; ModifiedResNet's layer1 stamps one."""

    def pattern(self, op: Any, x: Any) -> Any:
        return op.AveragePool(x, _outputs=["pooled"])

    def check(self, context: Any, x: Any, pooled: Any) -> MatchResult:
        result = MatchResult()
        attrs = pooled.producer().attributes
        kernel = attrs.get_ints("kernel_shape")
        if kernel is None or any(k != 1 for k in kernel):
            return result.fail("kernel is not 1x..x1")
        if any(s != 1 for s in attrs.get_ints("strides", [])):
            return result.fail("stride is not 1")
        if any(attrs.get_ints("pads", [])):
            return result.fail("padded")
        if any(d != 1 for d in attrs.get_ints("dilations", [])):
            return result.fail("dilated")
        if attrs.get_string("auto_pad", "NOTSET") not in ("", "NOTSET"):
            return result.fail("auto_pad set")
        return result

    def rewrite(self, op: Any, x: Any, pooled: Any) -> Any:
        return op.Identity(x)


class _BroadcastMaskRebuild(RewriteRuleClassBase):
    """XLM-R's ``attention_mask`` -> ``[batch,1,S,S]`` rebuild via a runtime GatherND index table is just a broadcast
    of ``mask[b, j]``. Replacing it kills the graph's only data-dependent shape, the island that fragments CoreML."""

    def pattern(self, op: Any, tril: Any, mask: Any, indices: Any) -> Any:
        gathered = op.GatherND(op.Cast(mask, to=int(ir.DataType.BOOL)), indices, _outputs=["gathered"])
        return op.And(tril, gathered, _outputs=["anded"])

    def check(self, context: Any, tril: Any, mask: Any, indices: Any, gathered: Any, anded: Any) -> MatchResult:
        result = MatchResult()
        if mask.producer() is not None or mask.shape is None or len(mask.shape) != 2:
            return result.fail("mask is not a rank-2 graph input")
        seq = mask.shape[1]
        if not isinstance(seq, int):
            return result.fail("mask sequence length is not static")
        tril_const = tril.const_value
        if tril_const is None or tuple(tril_const.shape) != (1, 1, seq, 1):
            return result.fail("And operand is not the [1,1,S,1] constant mask")
        if gathered.producer().attributes.get_int("batch_dims", 0) != 0:
            return result.fail("GatherND has batch_dims != 0")

        range_ok = arange_ok = False
        seen: set[int] = set()
        stack = [indices]
        while stack and len(seen) < 32:
            value = stack.pop()
            const = value.const_value
            if (
                const is not None
                and const.size == seq
                and np.array_equal(const.numpy().reshape(-1), np.arange(seq, dtype=const.numpy().dtype))
            ):
                arange_ok = True
                continue
            node = value.producer()
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            if node.op_type == "Range":
                start, _, delta = (i.const_value for i in node.inputs)
                if start is not None and int(start.numpy()) == 0 and delta is not None and int(delta.numpy()) == 1:
                    range_ok = True
                continue  # the Range's limit is batch-derived by construction; don't walk it
            stack.extend(i for i in node.inputs if i is not None)
        if not range_ok:
            return result.fail("no Range(0, ., 1) batch coordinate in the index construction")
        if not arange_ok:
            return result.fail("no constant arange(S) token coordinate in the index construction")
        return result

    def rewrite(self, op: Any, tril: Any, mask: Any, indices: Any, gathered: Any, anded: Any) -> Any:
        unsqueezed = op.Unsqueeze(
            op.Cast(mask, to=int(ir.DataType.BOOL)),
            op.Constant(value=ir.tensor(np.array([1, 2], np.int64))),
        )
        return op.And(tril, unsqueezed)


class _AdditivePadMask(RewriteRuleClassBase):
    """XLM-R's boolean ``Attention`` mask -> the equivalent float additive key bias (0 keeps, -1e4 removes, softmax
    underflowing to exactly 0), because the bool plumbing strands an integer island on float-only accelerators. The
    trailing zero-Add materializes the query axis: ORT CPU's ``Attention`` enforces ``mask.shape[-2] == q_len``."""

    def pattern(self, op: Any, tril: Any, mask: Any, axes: Any) -> Any:
        unsqueezed = op.Unsqueeze(op.Cast(mask, to=int(ir.DataType.BOOL)), axes)
        return op.And(tril, unsqueezed, _outputs=["anded"])

    def check(self, context: Any, tril: Any, mask: Any, axes: Any, anded: Any) -> MatchResult:
        result = MatchResult()
        if mask.producer() is not None or mask.shape is None or len(mask.shape) != 2:
            return result.fail("mask is not a rank-2 graph input")
        seq = mask.shape[1]
        if not isinstance(seq, int):
            return result.fail("mask sequence length is not static")
        if axes.const_value is None or list(axes.const_value.numpy().reshape(-1)) != [1, 2]:
            return result.fail("Unsqueeze axes are not [1, 2]")
        tril_const = tril.const_value
        if tril_const is None or tril_const.dtype != ir.DataType.BOOL:
            return result.fail("And operand is not a bool constant")
        if any(d not in (1, seq) for d in tril_const.shape) or len(tril_const.shape) > 4:
            return result.fail("And operand does not broadcast over [b,1,S,S]")
        if not tril_const.numpy().all():
            return result.fail("And operand is not all-True: it carries masking of its own")
        uses = anded.uses()
        if not uses or any(use.node.op_type != "Attention" or use.idx != 3 for use in uses):
            return result.fail("mask feeds a non-Attention consumer")
        return result

    def rewrite(self, op: Any, tril: Any, mask: Any, axes: Any, anded: Any) -> Any:
        seq = mask.shape[1]
        shifted = op.Sub(
            op.Cast(mask, to=int(ir.DataType.FLOAT)),
            op.Constant(value=ir.tensor(np.array(1.0, np.float32))),
        )
        additive = op.Mul(shifted, op.Constant(value=ir.tensor(np.array(1.0e4, np.float32))))
        keys = op.Unsqueeze(additive, op.Constant(value=ir.tensor(np.array([1, 2], np.int64))))
        return op.Add(keys, op.Constant(value=ir.tensor(np.zeros((seq, 1), np.float32))))


class _FloatMaskCount(RewriteRuleClassBase):
    """The mean-pool token count takes an int64 detour for a sum of 0/1 values fp32 counts exactly; count in float
    instead, leaving the post-embedding graph float-only. The Unsqueeze arm is optional, the spelling being
    family-dependent."""

    def pattern(self, op: Any, mask: Any, axes: Any, unsqueeze_axes: Any) -> Any:
        total = op.ReduceSum(op.Cast(mask, to=int(ir.DataType.INT64)), axes, _outputs=["total"])
        # exactly one arm binds; the other leaves `unsqueeze_axes` None
        kept = OrValue([op.Unsqueeze(total, unsqueeze_axes), total])
        return op.Cast(kept, to=int(ir.DataType.FLOAT))

    def check(self, context: Any, mask: Any, axes: Any, unsqueeze_axes: Any, total: Any) -> MatchResult:
        result = MatchResult()
        if mask.dtype not in (ir.DataType.INT32, ir.DataType.INT64, ir.DataType.BOOL):
            return result.fail("mask is not an integer/bool tensor")
        dims = mask.shape
        if dims is None or any(not isinstance(d, int) for d in dims[1:]):
            return result.fail("mask shape is not static beyond batch")
        if int(np.prod([d for d in dims[1:]])) >= 1 << 24:
            return result.fail("count could exceed fp32 exact-integer range")
        return result

    def rewrite(self, op: Any, mask: Any, axes: Any, unsqueeze_axes: Any, total: Any) -> Any:
        attrs = total.producer().attributes
        kwargs = {k: attrs.get_int(k) for k in ("keepdims", "noop_with_empty_axes") if attrs.get(k) is not None}
        counted = op.ReduceSum(op.Cast(mask, to=int(ir.DataType.FLOAT)), axes, **kwargs)
        return counted if unsqueeze_axes is None else op.Unsqueeze(counted, unsqueeze_axes)


class _FoldConstantAttnQuery(RewriteRuleClassBase):
    """SigLIP's MAP-head query projection runs entirely on constants, recomputing the same ``[1,H,1,d]`` every
    inference; fold it into one initializer plus an Add against the zero column that alone carries batch. Matches
    only after `batch_zeros_1d` has materialized that seed, which the pattern pins."""

    def pattern(self, op: Any, latent: Any, zeros: Any, seed: Any, ua: Any, weight: Any, bias: Any, target: Any) -> Any:
        query = op.Add(op.Add(latent, zeros), op.Unsqueeze(seed, ua))
        projected = op.Add(op.MatMul(query, weight), bias)
        return op.Transpose(op.Reshape(projected, target), _outputs=["q_heads"])

    def check(
        self,
        context: Any,
        latent: Any,
        zeros: Any,
        seed: Any,
        ua: Any,
        weight: Any,
        bias: Any,
        target: Any,
        q_heads: Any,
    ) -> MatchResult:
        result = MatchResult()
        if any(v.const_value is None for v in (latent, zeros, weight, bias, ua, target)):
            return result.fail("query-branch operand is not constant")
        lat = latent.const_value.numpy()
        if lat.ndim < 2 or any(d != 1 for d in lat.shape[:-1]) or not np.issubdtype(lat.dtype, np.floating):
            return result.fail("latent is not a single float query token")
        z = zeros.const_value.numpy()
        neutral = z.ndim == lat.ndim and all(d == 1 for d in z.shape[:-1]) and z.shape[-1] in (1, lat.shape[-1])
        if not neutral or np.any(z):
            return result.fail("Expand-materialization operand is not a broadcast-neutral zero")
        w = weight.const_value.numpy()
        if w.ndim != 2 or w.shape[0] != lat.shape[-1] or bias.const_value.numpy().size != w.shape[1]:
            return result.fail("projection is not latent-width MatMul + bias")

        def zero_scalar(v: ir.Value | None) -> bool:
            if v is None or v.const_value is None:
                return False
            arr = v.const_value.numpy()
            return arr.size == 1 and not np.any(arr)

        mul = seed.producer()
        if mul is None or mul.op_type != "Mul" or not any(zero_scalar(v) for v in mul.inputs):
            return result.fail("column seed is not the batch_zeros_1d Mul-by-zero output")
        if list(ua.const_value.numpy().reshape(-1)) != list(range(1, lat.ndim)):
            return result.fail("column axes do not place batch leading at the latent rank")
        t = target.const_value.numpy()
        if t.ndim != 1 or t[0] != -1 or np.count_nonzero(t == -1) != 1 or np.any(t == 0):
            return result.fail("Reshape target is not [-1, heads...] with batch leading")
        if np.prod(t[1:]) != w.shape[1]:
            return result.fail("Reshape target size does not match the projection width")
        perm = q_heads.producer().attributes.get_ints("perm")
        if perm is None or sorted(perm) != list(range(len(t))):
            return result.fail("Transpose perm is not a full permutation of the packed rank")
        return result

    def rewrite(
        self, op: Any, latent: Any, zeros: Any, seed: Any, ua: Any, weight: Any, bias: Any, target: Any, q_heads: Any
    ) -> Any:
        lat = latent.const_value.numpy()
        q = lat.astype(np.float64).reshape(1, -1) @ weight.const_value.numpy().astype(
            np.float64
        ) + bias.const_value.numpy().astype(np.float64).reshape(-1)
        t = [int(d) for d in target.const_value.numpy()]
        perm = list(q_heads.producer().attributes.get_ints("perm"))
        qc = q.reshape([1 if d == -1 else d for d in t]).transpose(perm).astype(lat.dtype)
        col_axes = np.array([a for a in range(len(perm)) if a != perm.index(0)], np.int64)
        col = op.Unsqueeze(seed, op.Constant(value=ir.tensor(col_axes)))
        return op.Add(op.Constant(value=ir.tensor(qc)), col)


class PruneAttnpoolDeadQueriesPass(ir.passes.InPlacePass):
    """Restructure CLIP's ResNet attention-pool to compute only the query it keeps, the projections being per-row.
    Equivalent but not bit-exact: the q and out-projection GEMMs run at different row counts."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        rewritten = 0

        def producer_chain(value: ir.Value, ops: list[str]) -> list[ir.Node] | None:
            nodes = []
            for op_type in ops:
                node = value.producer()
                if node is None or node.op_type != op_type:
                    return None
                nodes.append(node)
                value = node.inputs[0]
            return nodes

        def new_target(node: ir.Node, values: list[int], name: str) -> None:
            """Point a Reshape at a FRESH target initializer -- the old one may be shared with k/v -- and update the
            cached shape, which ``_ScalarGatherToSlice`` reads before the next re-inference."""
            target = make_init(graph, name, np.array(values, np.int64))
            node.replace_input_with(1, target)
            old_dims = node.outputs[0].shape
            if old_dims is not None and len(old_dims) == len(values):
                node.outputs[0].shape = ir.Shape([v if v != -1 else old_dims[i] for i, v in enumerate(values)])

        for attention in [n for n in graph if n.op_type == "Attention"]:
            q_chain = producer_chain(attention.inputs[0], ["Reshape", "Transpose", "Reshape", "Add", "MatMul"])
            if q_chain is None:
                continue
            q_pack, q_perm, q_unpack, _, q_matmul = q_chain
            pack_target, unpack_target = const_ints(q_pack.inputs[1]), const_ints(q_unpack.inputs[1])
            perm = q_perm.attributes.get_ints("perm")
            if pack_target is None or unpack_target is None or perm is None or list(perm) != [1, 0, 2]:
                continue
            if len(unpack_target) != 3 or len(pack_target) != 4 or unpack_target[0] != pack_target[2]:
                continue
            seq = unpack_target[0]
            if not isinstance(seq, int) or seq <= 1:
                continue

            # the trailing Gather is lowered later, by then to a bare Squeeze: its axis has one row
            out_perm = sole_consumer(attention.outputs[0], "Transpose")
            if out_perm is None or list(out_perm.attributes.get_ints("perm", [])) != [2, 0, 1, 3]:
                continue
            out_flat = sole_consumer(out_perm.outputs[0], "Reshape")
            gemm = sole_consumer(out_flat.outputs[0], "Gemm") if out_flat is not None else None
            out_unflat = sole_consumer(gemm.outputs[0], "Reshape") if gemm is not None else None
            token_gather = sole_consumer(out_unflat.outputs[0], "Gather") if out_unflat is not None else None
            if token_gather is None:
                continue
            unflat_target = const_ints(out_unflat.inputs[1])
            if unflat_target is None or len(unflat_target) != 3 or unflat_target[0] != seq:
                continue
            index = token_gather.inputs[1].const_value
            if token_gather.attributes.get_int("axis", 0) != 0:
                continue
            if index is None or index.numpy().ndim != 0 or int(index.numpy()) != 0:
                continue

            row = make_node(  # the pre-projection tensor is [S, batch, E]
                "Slice",
                [
                    q_matmul.inputs[0],
                    make_init(graph, f"{attention.name}_q_row_start", np.array([0], np.int64)),
                    make_init(graph, f"{attention.name}_q_row_end", np.array([1], np.int64)),
                    make_init(graph, f"{attention.name}_q_row_axis", np.array([0], np.int64)),
                ],
                out=f"{attention.name}_q_row",
            )
            graph.insert_before(q_matmul, [row])
            q_matmul.replace_input_with(0, row.outputs[0])
            new_target(q_unpack, [1, unpack_target[1], unpack_target[2]], f"{attention.name}_q_unpack_1")
            new_target(q_pack, [pack_target[0], pack_target[1], 1, pack_target[3]], f"{attention.name}_q_pack_1")
            new_target(out_unflat, [1, unflat_target[1], unflat_target[2]], f"{attention.name}_out_1")
            rewritten += 1
        log.info("Pruned %d attention-pool dead query set(s)", rewritten)
        return ir.passes.PassResult(model, bool(rewritten))


class RestructureAttention3dPass(ir.passes.InPlacePass):
    """Collapse the exported per-head attention plumbing into batch-first 3D ``Attention``. ``q_num_heads`` and
    ``kv_num_heads`` are always emitted because OpenVINO hard-fails without them. The V-projection bias folds into
    the out-proj bias, exact because softmax rows sum to one, and its block is zeroed."""

    def requires(self, model: ir.Model) -> None:
        # `fold_v_bias` writes `const_value`, which a `Constant` node does not serialize: the fold would vanish
        if stuck := [node.outputs[0].name for node in model.graph if node.op_type == "Constant"]:
            raise ir.passes.PreconditionError(f"{len(stuck)} constant(s) are not initializers: {stuck[:3]}")

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph

        def perm_of(node: ir.Node) -> list[int] | None:
            perm = node.attributes.get_ints("perm")
            return list(perm) if perm is not None else None

        def axes_of(node: ir.Node) -> list[int] | None:
            return const_ints(node.inputs[1]) if len(node.inputs) > 1 else None

        def slice_start(node: ir.Node, length: int) -> int | None:
            """The i of a unit-width ``Slice(i:i+1, axis 0)``, else None."""
            params = [const_ints(node.inputs[i]) if len(node.inputs) > i else None for i in (1, 2, 3, 4)]
            starts, ends, axes, steps = params
            if starts is None or len(starts) != 1 or ends != [starts[0] + 1] or axes != [0]:
                return None
            if steps not in (None, [1]) or not 0 <= starts[0] < length:
                return None
            return starts[0]

        def bias_add(node: ir.Node | None, width: int) -> tuple[ir.Value, ir.Value] | None:
            """Split an ``Add(x, bias)`` into (data input, single-use constant [width] bias)."""
            if node is None:
                return None
            bias = next((v for v in node.inputs if const_array(v) is not None), None)
            data = next((v for v in node.inputs if v is not bias), None)
            if bias is None or data is None or bias.const_value.size != width or not single_use(bias):
                return None
            return data, bias

        def set_heads(att: ir.Node, heads: int) -> None:
            att.attributes["q_num_heads"] = ir.AttrInt64("q_num_heads", heads)
            att.attributes["kv_num_heads"] = ir.AttrInt64("kv_num_heads", heads)

        def fold_v_bias(packed_bias: ir.Value, out_weight: np.ndarray, out_bias: ir.Value) -> None:
            """``bo' = bo + b_v @ Wo`` in fp64, then zero the trailing (V) block of the packed bias."""
            packed = packed_bias.const_value.numpy()
            width = out_weight.shape[0]
            folded = packed.copy()
            folded[-width:] = 0
            bo = out_bias.const_value.numpy()
            bo_new = bo.astype(np.float64) + packed[-width:].astype(np.float64) @ out_weight.astype(np.float64)
            packed_bias.const_value = ir.tensor(folded, name=packed_bias.name)
            out_bias.const_value = ir.tensor(bo_new.astype(bo.dtype), name=out_bias.name)

        class Packed(NamedTuple):
            """A packed-projection unbind: MatMul+Add -> (motif-specific shuffle) -> n branches."""

            x: ir.Value  # [B,S,D], batch-first
            weight: ir.Value  # [D, n*D]
            bias: ir.Value  # [n*D]
            add_out: ir.Value
            heads: int
            width: int
            seq: int

        def match_seqfirst_packed(att: ir.Node) -> dict[str, Any] | None:
            """open_clip resblock: seq-first packed QKV in, flattened Gemm out."""
            shared: ir.Node | None = None
            pack_target = unpack_target = None
            for i in range(3):
                pack = producer_of(att.inputs[i], "Reshape")  # [-1,H,S,dh]
                if pack is None or not single_use(pack.outputs[0]):
                    return None
                head_tr = producer_of(pack.inputs[0], "Transpose")
                if head_tr is None or perm_of(head_tr) != [1, 0, 2] or not single_use(head_tr.outputs[0]):
                    return None
                unpack = producer_of(head_tr.inputs[0], "Reshape")  # [S,-1,dh]
                if unpack is None or not single_use(unpack.outputs[0]):
                    return None
                if pack_target is None:
                    pack_target, unpack_target = const_ints(pack.inputs[1]), const_ints(unpack.inputs[1])
                elif const_ints(pack.inputs[1]) != pack_target or const_ints(unpack.inputs[1]) != unpack_target:
                    return None
                branch = unbind_branch(unpack.inputs[0])
                if branch is None or branch[0] != i:
                    return None
                if i == 0:
                    shared = branch[1]
                elif branch[1] is not shared:
                    return None
            if (
                shared is None
                or shared.op_type != "Squeeze"
                or axes_of(shared) not in ([-2], [3])
                or len(shared.outputs[0].uses()) != 3
            ):
                return None
            tr5 = producer_of(shared.inputs[0], "Transpose")
            if tr5 is None or perm_of(tr5) != [3, 1, 2, 0, 4] or not single_use(tr5.outputs[0]):
                return None
            unsqueeze = producer_of(tr5.inputs[0], "Unsqueeze")
            if unsqueeze is None or axes_of(unsqueeze) != [0] or not single_use(unsqueeze.outputs[0]):
                return None
            packed_reshape = producer_of(unsqueeze.inputs[0], "Reshape")  # [S,-1,3,D]
            if packed_reshape is None or not single_use(packed_reshape.outputs[0]):
                return None
            packed_target = const_ints(packed_reshape.inputs[1])
            if packed_target is None or len(packed_target) != 4 or packed_target[1:3] != [-1, 3]:
                return None
            seq, width = packed_target[0], packed_target[3]
            if pack_target is None or unpack_target is None or len(pack_target) != 4 or len(unpack_target) != 3:
                return None
            heads, head_dim = pack_target[1], pack_target[3]
            if pack_target != [-1, heads, seq, head_dim] or unpack_target != [seq, -1, head_dim]:
                return None
            if seq <= 0 or heads <= 0 or heads * head_dim != width:
                return None
            # the projection output must feed this cluster alone: its bias is about to be mutated
            if not single_use(packed_reshape.inputs[0]):
                return None
            packed = bias_add(producer_of(packed_reshape.inputs[0], "Add"), 3 * width)
            if packed is None or not single_use(packed[0]):
                return None
            mm_out, bias = packed
            matmul = producer_of(mm_out, "MatMul")
            if matmul is None:
                return None
            weight = matmul.inputs[1]
            w_arr = const_array(weight)
            if w_arr is None or w_arr.shape != (width, 3 * width):
                return None
            pre_tr = producer_of(matmul.inputs[0], "Transpose")
            if pre_tr is None or perm_of(pre_tr) != [1, 0, 2] or not single_use(pre_tr.outputs[0]):
                return None

            out_tr = sole_consumer(att.outputs[0], "Transpose")
            if out_tr is None or perm_of(out_tr) != [2, 0, 1, 3]:
                return None
            out_flat = sole_consumer(out_tr.outputs[0], "Reshape")
            if out_flat is None or const_ints(out_flat.inputs[1]) != [-1, width]:
                return None
            gemm = sole_consumer(out_flat.outputs[0], "Gemm")
            if gemm is None or len(gemm.inputs) != 3:
                return None
            attrs = gemm.attributes
            trans_b = attrs.get_int("transB", 0)
            if (
                attrs.get_float("alpha", 1.0) != 1.0
                or attrs.get_float("beta", 1.0) != 1.0
                or attrs.get_int("transA", 0)
            ):
                return None
            wo, bo = gemm.inputs[1], gemm.inputs[2]
            wo_arr = const_array(wo)
            bo_arr = const_array(bo)
            if wo_arr is None or wo_arr.shape != (width, width) or not single_use(wo):
                return None
            if bo_arr is None or bo_arr.size != width or not single_use(bo):
                return None
            out_unflat = sole_consumer(gemm.outputs[0], "Reshape")
            if out_unflat is None or const_ints(out_unflat.inputs[1]) != [seq, -1, width]:
                return None
            out_tr2 = sole_consumer(out_unflat.outputs[0], "Transpose")
            if out_tr2 is None or perm_of(out_tr2) != [1, 0, 2]:
                return None
            return {
                # add_out is the seq-first projection output: unusable batch-first, replaced on rewrite
                "packed": Packed(pre_tr.inputs[0], weight, bias, packed_reshape.inputs[0], heads, width, seq),
                "wo_t": wo_arr.T if trans_b else wo_arr,  # y = x @ wo_t orientation, fp-exact
                "bo": bo,
                "final": out_tr2.outputs[0],
            }

        def unbind_branch(value: ir.Value | None) -> tuple[int, ir.Node] | None:
            """A ``Squeeze(Slice(shared, i:i+1, axis 0), [0])`` unbind branch -> (i, shared node)."""
            squeeze = producer_of(value, "Squeeze")
            if squeeze is None or axes_of(squeeze) != [0] or not single_use(squeeze.outputs[0]):
                return None
            unbind = producer_of(squeeze.inputs[0], "Slice")
            if unbind is None or not single_use(unbind.outputs[0]):
                return None
            start = slice_start(unbind, 3)
            source = unbind.inputs[0].producer()
            if start is None or source is None:
                return None
            return start, source

        def match_packed_shuffle(shared: ir.Node, n: int) -> Packed | None:
            """The shared ``Transpose(2,0,3,1,4)(Reshape[-1,S,n,H,dh](MatMul+Add))`` unbind source."""
            if shared.op_type != "Transpose" or perm_of(shared) != [2, 0, 3, 1, 4]:
                return None
            if len(shared.outputs[0].uses()) != n:
                return None
            reshape = producer_of(shared.inputs[0], "Reshape")
            if reshape is None or not single_use(reshape.outputs[0]):
                return None
            target = const_ints(reshape.inputs[1])
            if target is None or len(target) != 5 or target[0] != -1 or target[2] != n:
                return None
            seq, heads, head_dim = target[1], target[3], target[4]
            width = heads * head_dim
            if seq <= 0 or heads <= 0 or head_dim <= 0:
                return None
            # the projection output must feed this cluster alone: its bias is about to be mutated
            if not single_use(reshape.inputs[0]):
                return None
            packed = bias_add(producer_of(reshape.inputs[0], "Add"), n * width)
            if packed is None or not single_use(packed[0]):
                return None
            mm_out, bias = packed
            matmul = producer_of(mm_out, "MatMul")
            if matmul is None:
                return None
            w_arr = const_array(matmul.inputs[1])
            if w_arr is None or w_arr.shape != (width, n * width):
                return None
            return Packed(matmul.inputs[0], matmul.inputs[1], bias, reshape.inputs[0], heads, width, seq)

        def match_batchfirst_out(att: ir.Node, seq: int, width: int) -> dict[str, Any] | None:
            """Batch-first out side: Transpose(0,2,1,3) -> Reshape[-1,S,D] -> MatMul + Add."""
            out_tr = sole_consumer(att.outputs[0], "Transpose")
            if out_tr is None or perm_of(out_tr) != [0, 2, 1, 3]:
                return None
            out_reshape = sole_consumer(out_tr.outputs[0], "Reshape")
            if out_reshape is None or const_ints(out_reshape.inputs[1]) != [-1, seq, width]:
                return None
            out_mm = sole_consumer(out_reshape.outputs[0], "MatMul")
            if out_mm is None:
                return None
            wo_arr = const_array(out_mm.inputs[1])
            if wo_arr is None or wo_arr.shape != (width, width):
                return None
            folded = bias_add(sole_consumer(out_mm.outputs[0], "Add"), width)
            if folded is None:
                return None
            return {"final": out_reshape.outputs[0], "wo": wo_arr, "bo": folded[1]}

        def match_batchfirst_packed(att: ir.Node) -> dict[str, Any] | None:
            """timm block: batch-first packed QKV unbind; all three inputs slice one shared shuffle."""
            branches = [unbind_branch(att.inputs[i]) for i in range(3)]
            if any(b is None for b in branches) or [b[0] for b in branches] != [0, 1, 2]:  # type: ignore[index]
                return None
            sources = {id(b[1]) for b in branches}  # type: ignore[index]
            if len(sources) != 1:
                return None
            packed = match_packed_shuffle(branches[0][1], 3)  # type: ignore[index]
            if packed is None:
                return None
            out = match_batchfirst_out(att, packed.seq, packed.width)
            if out is None:
                return None
            return {"packed": packed, **out}

        def match_attnpool(att: ir.Node) -> dict[str, Any] | None:
            """SigLIP MAP head: folded-constant query + batch-first packed KV unbind (2-way)."""
            branches = [unbind_branch(att.inputs[i]) for i in (1, 2)]
            if any(b is None for b in branches) or [b[0] for b in branches] != [0, 1]:  # type: ignore[index]
                return None
            if branches[0][1] is not branches[1][1]:  # type: ignore[index]
                return None
            packed = match_packed_shuffle(branches[0][1], 2)  # type: ignore[index]
            if packed is None:
                return None
            # the folded constant query `_FoldConstantAttnQuery` leaves
            q_add = producer_of(att.inputs[0], "Add")
            if q_add is None or not single_use(q_add.outputs[0]):
                return None
            query = next((v for v in q_add.inputs if const_array(v) is not None), None)
            col = next((v for v in q_add.inputs if v is not query), None)
            q_arr = const_array(query)
            if q_arr is None or q_arr.shape != (1, packed.heads, 1, packed.width // packed.heads):
                return None
            unsqueeze = producer_of(col, "Unsqueeze")
            if unsqueeze is None or axes_of(unsqueeze) != [1, 2, 3] or not single_use(unsqueeze.outputs[0]):
                return None
            out = match_batchfirst_out(att, 1, packed.width)
            if out is None:
                return None
            return {"packed": packed, "query": query, "seed": unsqueeze.inputs[0], **out}

        def match_separate(att: ir.Node) -> dict[str, Any] | None:
            """HF-style separate q/k/v projections feeding per-head Reshape+Transpose (XLM-R)."""
            sources = []
            shape = None
            v_bias = None
            for i in range(3):
                head_tr = producer_of(att.inputs[i], "Transpose")
                if head_tr is None or perm_of(head_tr) != [0, 2, 1, 3] or not single_use(head_tr.outputs[0]):
                    return None
                reshape = producer_of(head_tr.inputs[0], "Reshape")
                if reshape is None or not single_use(reshape.outputs[0]):
                    return None
                target = const_ints(reshape.inputs[1])
                if target is None or len(target) != 4 or target[0] != -1 or min(target[1:]) <= 0:
                    return None
                if shape is None:
                    shape = target
                elif target != shape:
                    return None
                projected = bias_add(reshape.inputs[0].producer(), target[2] * target[3])
                if projected is None or producer_of(projected[0], "MatMul") is None:
                    return None
                # V's bias folds into the out-proj's, so take the projection ahead of its Add; nothing else may read it
                if i == 2:
                    if not single_use(reshape.inputs[0]):
                        return None
                    sources.append(projected[0])
                    v_bias = projected[1]
                else:
                    sources.append(reshape.inputs[0])
            assert shape is not None
            out = match_batchfirst_out(att, shape[1], shape[2] * shape[3])
            if out is None:
                return None
            return {"sources": sources, "heads": shape[2], "v_bias": v_bias, **out}

        split_sizes: dict[tuple[int, int], ir.Value] = {}  # (n, width) -> shared Split sizes initializer

        def emit_split(base: str, packed: Packed, n: int) -> list[ir.Value]:
            sizes = split_sizes.get((n, packed.width))
            if sizes is None:
                sizes = make_init(graph, f"attn3d_split_{n}x{packed.width}", np.full(n, packed.width, np.int64))
                split_sizes[(n, packed.width)] = sizes
            split = ir.node("Split", inputs=[packed.add_out, sizes], attributes={"axis": -1}, num_outputs=n)
            split.name = f"{base}_qkv_split"
            for out, tag in zip(split.outputs, ("q", "k", "v")[3 - n :]):
                out.name = f"{base}_{tag}"
            graph.append(split)
            return list(split.outputs)

        rewritten = 0
        for att in [n for n in graph if n.op_type == "Attention"]:
            if len(att.outputs) != 1 or len(att.inputs) < 3 or len(att.inputs) > 4:
                continue
            if any(att.inputs[i] is None for i in range(3)):
                continue
            base = att.name or att.outputs[0].name

            if (m := match_seqfirst_packed(att)) is not None:
                packed: Packed = m["packed"]
                fold_v_bias(packed.bias, m["wo_t"], m["bo"])
                matmul = make_node("MatMul", [packed.x, packed.weight], name=f"{base}_qkv_mm", out=f"{base}_qkv_mm_out")
                add = make_node("Add", [matmul.outputs[0], packed.bias], name=f"{base}_qkv_bias", out=f"{base}_qkv")
                graph.extend([matmul, add])
                qkv = emit_split(base, packed._replace(add_out=add.outputs[0]), 3)
                for i in range(3):
                    att.replace_input_with(i, qkv[i])
                set_heads(att, packed.heads)
                wo_t = make_init(graph, f"{base}_wo_t", np.ascontiguousarray(m["wo_t"]))
                out_mm = make_node("MatMul", [att.outputs[0], wo_t], name=f"{base}_out_mm", out=f"{base}_out_mm_out")
                out_add = make_node("Add", [out_mm.outputs[0], m["bo"]], name=f"{base}_out_bias", out=f"{base}_out")
                graph.extend([out_mm, out_add])
                m["final"].replace_all_uses_with(out_add.outputs[0], replace_graph_outputs=True)
            elif (m := match_batchfirst_packed(att)) is not None:
                packed = m["packed"]
                fold_v_bias(packed.bias, m["wo"], m["bo"])
                qkv = emit_split(base, packed, 3)
                for i in range(3):
                    att.replace_input_with(i, qkv[i])
                set_heads(att, packed.heads)
                m["final"].replace_all_uses_with(att.outputs[0], replace_graph_outputs=True)
            elif (m := match_attnpool(att)) is not None:
                packed = m["packed"]
                fold_v_bias(packed.bias, m["wo"], m["bo"])
                kv = emit_split(base, packed, 2)
                q_arr = const_array(m["query"]).transpose(0, 2, 1, 3).reshape(1, 1, packed.width)
                query = make_init(graph, f"{base}_q3", np.ascontiguousarray(q_arr))
                axes = make_init(graph, f"{base}_q_col_axes", np.array([1, 2], np.int64))
                unsqueeze = make_node("Unsqueeze", [m["seed"], axes], name=f"{base}_q_col", out=f"{base}_q_col_out")
                q_add = make_node("Add", [query, unsqueeze.outputs[0]], name=f"{base}_q_bcast", out=f"{base}_q")
                graph.extend([unsqueeze, q_add])
                att.replace_input_with(0, q_add.outputs[0])
                att.replace_input_with(1, kv[0])
                att.replace_input_with(2, kv[1])
                set_heads(att, packed.heads)
                m["final"].replace_all_uses_with(att.outputs[0], replace_graph_outputs=True)
            elif (m := match_separate(att)) is not None:
                fold_v_bias(m["v_bias"], m["wo"], m["bo"])
                for i in range(3):
                    att.replace_input_with(i, m["sources"][i])
                set_heads(att, m["heads"])
                m["final"].replace_all_uses_with(att.outputs[0], replace_graph_outputs=True)
            else:
                continue
            att.outputs[0].shape = None  # now [B,S,D]; stale per-head annotation must not survive
            rewritten += 1

        if rewritten:
            common_passes.TopologicalSortPass()(model)
            common_passes.RemoveUnusedNodesPass()(model)
        log.info("Restructured %d attention site(s) into 3D Attention", rewritten)
        return ir.passes.PassResult(model, bool(rewritten))


class _FlipCausalAttention(RewriteRuleClassBase):
    """Replace a constant causal ``Attention`` mask with ``is_causal=1``, exact here (q_len == kv_len). Terminates
    without a guard: the replacement drops the mask input the pattern requires."""

    def pattern(self, op: Any, q: Any, k: Any, v: Any, mask: Any) -> Any:
        return op.Attention(q, k, v, mask, _allow_other_attributes=True, _outputs=["attn"])

    def check(self, context: Any, mask: Any, attn: Any, **_: Any) -> MatchResult:
        result = MatchResult()
        if attn.producer().attributes.get_int("is_causal", 0) != 0:
            return result.fail("the Attention is already causal")
        array = const_array(mask)
        if array is None or not np.issubdtype(array.dtype, np.floating) or array.ndim < 2:
            return result.fail("the mask is not a float constant")
        seq = array.shape[-1]
        if seq < 2 or array.shape[-2] != seq or array.size != seq * seq:
            return result.fail("the mask is not a broadcastable square [.., S, S] tensor")
        square = array.reshape(seq, seq).astype(np.float64)
        on_or_below = np.tril(np.ones((seq, seq), np.bool_))
        if not (np.all(square[on_or_below] == 0.0) and np.all(square[~on_or_below] <= -1.0e4)):
            return result.fail("the mask is not the lower-triangular additive one")
        return result

    def rewrite(self, op: Any, q: Any, k: Any, v: Any, mask: Any, attn: Any, **_: Any) -> Any:
        return op.Attention(q, k, v, **{**attn.producer().attributes, "is_causal": 1})


_ROW_WISE_UNARY = {"Gelu", "Relu", "Sigmoid"}
_ROW_WISE_BINARY = {"Add", "Div", "Mul", "Sub"}


def _find_pooling_select(graph: ir.Graph) -> tuple[ir.Node, ir.Value, int] | None:
    """The tower's single-token pooling select -- EOT one-hot ``MatMul`` or ``Slice(axis=1)`` -> node, source, S."""
    for node in graph:
        if node.op_type == "MatMul" and _is_onehot_selector(node.inputs[0]):
            data = node.inputs[1]
        elif node.op_type == "Slice" and len(node.inputs) >= 4 and const_ints(node.inputs[3]) == [1]:
            data = node.inputs[0]
        else:
            continue
        dims, picked = data.shape, node.outputs[0].shape
        if dims is not None and len(dims) == 3 and isinstance(dims[1], int) and dims[1] > 1 and picked[1] == 1:
            return node, data, dims[1]
    return None


class HoistPoolingSelectPass(ir.passes.InPlacePass):
    """Push the single-token pooling select back through the last transformer block, so its tail runs at sequence
    length 1. Sound because that tail is row-wise, which a masked or causal ``Attention`` is not -- its bias is indexed
    by the query position, so those towers stop at the attention output. Runs last: needs the 3D ``Attention``."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        found = _find_pooling_select(graph)
        if found is None:
            return ir.passes.PassResult(model, False)
        select, root, seq = found

        def rows(value: ir.Value | None) -> bool:
            """`value` carries one entry per token on axis 1, so the select applies to it."""
            dims = value.shape if value is not None else None
            return dims is not None and len(dims) == 3 and dims[1] == seq

        def broadcasts(value: ir.Value) -> bool:
            """A per-feature operand holding the same value on every token."""
            dims = value.shape
            return dims is not None and (len(dims) < 3 or (len(dims) == 3 and dims[1] == 1))

        def row_wise(node: ir.Node) -> bool:
            if node.op_type in _ROW_WISE_UNARY:
                return True
            if node.op_type in _ROW_WISE_BINARY:
                return all(rows(v) or broadcasts(v) for v in node.inputs)
            if node.op_type == "LayerNormalization":
                return node.attributes.get_int("axis", -1) in (-1, 2)
            if node.op_type == "MatMul":  # rank-2 rhs is a per-row projection; a rank-3 rhs mixes tokens
                weight = node.inputs[1]
                return rows(node.inputs[0]) and weight.shape is not None and len(weight.shape) == 2
            return node.op_type == "Attention" and len(node.inputs) == 3 and not node.attributes.get_int("is_causal", 0)

        region: list[ir.Node] = []
        inside = {id(select)}
        need: dict[int, ir.Value] = {id(root): root}
        grew = True
        while grew:  # fixed point: a diamond's second arm can admit a producer an earlier sweep deferred
            grew = False
            for value in list(need.values()):
                node = value.producer()
                if node is None or id(node) in inside or not row_wise(node):
                    continue
                # dropping rows is only sound when nothing outside the region reads the full-length result
                if any(o.is_graph_output() or any(id(u.node) not in inside for u in o.uses()) for o in node.outputs):
                    continue
                inside.add(id(node))
                region.append(node)
                grew = True
                for inp in node.inputs[:1] if node.op_type == "Attention" else node.inputs:
                    if rows(inp):
                        need.setdefault(id(inp), inp)

        frontier = [v for v in need.values() if v.producer() is None or id(v.producer()) not in inside]
        assert any(n.op_type == "Attention" for n in region) or any(
            producer_of(v, "Attention") is not None for v in frontier
        ), f"pooling-select hoist stopped short of the last Attention: {[n.op_type for n in region]}"

        picked: dict[int, ir.Value] = {}
        for value in frontier:
            inputs = [select.inputs[0], value] if select.op_type == "MatMul" else [value, *select.inputs[1:]]
            node = make_node(select.op_type, inputs, name=f"pool_hoist_{value.name}", out=f"{value.name}_pooled")
            node.outputs[0].shape = ir.Shape([value.shape[0], 1, value.shape[2]])
            node.outputs[0].type = value.type
            picked[id(value)] = node.outputs[0]
            graph.append(node)

        for node in region:
            for index, inp in enumerate(node.inputs):
                if inp is not None and id(inp) in picked:
                    node.replace_input_with(index, picked[id(inp)])
            for out in node.outputs:
                if rows(out):
                    out.shape = ir.Shape([out.shape[0], 1, out.shape[2]])

        select.outputs[0].replace_all_uses_with(root)  # root now carries exactly the row the select picked
        graph.remove(select, safe=True)
        common_passes.TopologicalSortPass()(model)
        log.info("Hoisted the pooling select back over %d node(s)", len(region))
        return ir.passes.PassResult(model, True)


def _assert_fp16_safe(array: np.ndarray, name: str) -> None:
    """Export-time guard for the fp16 table: nothing may overflow fp16, and no LIVE row may lose relative L2 norm to
    the round-trip. Damage alone is not a defect, so only rows above a norm floor are read."""
    over = int(np.count_nonzero(np.abs(array) > 65504.0))
    if over:
        raise ValueError(f"{name}: {over} values exceed the fp16 max of 65504")
    # blocked: a whole-table round-trip would treble the table's transient footprint
    blocks = [array[start : start + 16384] for start in range(0, len(array), 16384)]
    norms = np.concatenate([np.linalg.norm(block, axis=1) for block in blocks])
    errors = np.concatenate([np.linalg.norm(block - block.astype(np.float16), axis=1) for block in blocks])
    live = norms >= 1e-2 * float(np.median(norms))
    damaged = int(np.count_nonzero((errors > 1e-3 * np.maximum(norms, 1e-30)) & live))
    if damaged:
        raise ValueError(f"{name}: {damaged} live rows lose more than 1e-3 of their L2 norm to fp16")


def _cast_source(value: ir.Value) -> ir.Value:
    """Walk back through dtype/no-op casts to the value actually being indexed with."""
    while (producer := value.producer()) is not None and producer.op_type in ("Cast", "Identity"):
        source = producer.inputs[0]
        if source is None:
            break
        value = source
    return value


class _Fp16TokenEmbedding(RewriteRuleClassBase):
    """Store the token-embedding table as fp16, casting the gathered rows back to fp32 so compute is untouched: the
    table dominates a textual export's size. GATHER-FIRST IS LOAD-BEARING -- the mirror shape `Cast(table) -> Gather`
    constant-folds the whole table back to fp32 at session init and cannot be stopped. The fp32 check terminates it."""

    def pattern(self, op: Any, table: Any, indices: Any) -> Any:
        return op.Gather(table, indices, _outputs=["gathered"])

    def check(self, context: Any, table: Any, indices: Any, gathered: Any) -> MatchResult:
        result = MatchResult()
        if gathered.producer().attributes.get_int("axis", 0) != 0:
            return result.fail("Gather axis is not 0")
        if not table.is_initializer() or table.const_value is None or table.dtype != ir.DataType.FLOAT:
            return result.fail("table is not an fp32 initializer")
        if table.shape is None or len(table.shape) != 2 or gathered.is_graph_output():
            return result.fail("table is not a rank-2 lookup feeding the graph")
        ids = {v for v in context.model.graph.inputs if v.dtype is not None and v.dtype.is_integer()}
        if _cast_source(indices) not in ids:
            return result.fail("indices are not (a cast of) an integer graph input")
        return result

    def rewrite(self, op: Any, table: Any, indices: Any, gathered: Any) -> Any:
        array = table.const_value.numpy()
        _assert_fp16_safe(array, table.name)  # raises: a rejected table must abort the export, not skip
        fp16 = op.initializer(ir.tensor(array.astype(np.float16), name=f"{table.name}_fp16"))
        return op.Cast(op.Gather(fp16, indices, axis=0), to=ir.DataType.FLOAT)


class Fp16TokenEmbeddingPass(ir.passes.Sequential):
    """`_Fp16TokenEmbedding`, sweeping the fp32 table it retired: the rule emits a fresh initializer beside it."""

    def __init__(self) -> None:
        super().__init__(RewritePass([_Fp16TokenEmbedding.rule()]), common_passes.RemoveUnusedNodesPass())


class _FoldEmbeddingScale(RewriteRuleClassBase):
    """Fold NLLB's ``embed_scale`` into the token-embedding table, dropping a full ``[B,S,D]`` Mul. Bit-exact only for
    a power-of-two scale on a table with fp16 headroom, both checked. Runs after ``Fp16TokenEmbeddingPass`` so the
    shift lands on the table that ships, and overwrites that initializer in place, being the last transform."""

    def pattern(self, op: Any, table: Any, indices: Any, scale: Any) -> Any:
        rows = op.Gather(table, indices, _allow_other_attributes=True, _outputs=["rows"])
        return op.Mul(OrValue([op.Cast(rows, _outputs=["widened"]), rows], name="looked_up"), scale)

    def check(self, context: Any, table: Any, indices: Any, scale: Any, **_: Any) -> MatchResult:
        result = MatchResult()
        factor = const_array(scale)
        if factor is None or factor.size != 1 or abs(np.frexp(float(factor.reshape(-1)[0]))[0]) != 0.5:
            return result.fail("the scale is not a single power-of-two constant")
        if indices not in {v for v in context.model.graph.inputs if v.dtype is not None and v.dtype.is_integer()}:
            return result.fail("the lookup is not indexed by an integer graph input")
        weights = table.const_value  # metadata first; the table materializes only past the gate
        if weights is None or len(weights.shape) != 2:
            return result.fail("the table is not a rank-2 constant")
        if float(np.abs(weights.numpy()).max()) * float(factor.reshape(-1)[0]) > 65504.0:
            return result.fail("the scaled table would leave fp16 range")
        return result

    def rewrite(self, op: Any, table: Any, indices: Any, scale: Any, rows: Any, widened: Any = None, **_: Any) -> Any:
        array, factor = table.const_value.numpy(), float(const_array(scale).reshape(-1)[0])
        table.const_value = ir.tensor(array * array.dtype.type(factor), name=table.name)
        looked_up = op.Gather(table, indices, **rows.producer().attributes)
        return looked_up if widened is None else op.Cast(looked_up, **widened.producer().attributes)


class FoldEmbeddingScalePass(RewritePass):
    def __init__(self) -> None:
        # commute: nothing pins which side of the Mul the exporter writes the scale on
        super().__init__(RewriteRuleSet([_FoldEmbeddingScale.rule()], commute=True))

    def requires(self, model: ir.Model) -> None:
        ids = {v for v in model.graph.inputs if v.dtype is not None and v.dtype.is_integer()}
        stuck = [
            node.inputs[0].name
            for node in model.graph
            if node.op_type == "Gather"
            and _cast_source(node.inputs[1]) in ids
            and node.inputs[0].dtype == ir.DataType.FLOAT
            and node.inputs[0].shape is not None
            and len(node.inputs[0].shape) == 2
        ]
        if stuck:
            raise ir.passes.PreconditionError(f"token table(s) {stuck} are still fp32: the fp16 pass runs first")


class _FloatMaskConsumersPass(RewritePass):
    """Leave nothing integer downstream of an explicit `attention_mask` input: rebuild the broadcast, hand
    `Attention` the additive form, count the mean-pool divisor in float. The mclip counterpart to `FoldPadMaskPass`."""

    def __init__(self) -> None:
        super().__init__([_BroadcastMaskRebuild.rule(), _AdditivePadMask.rule(), _FloatMaskCount.rule()])

    def ensures(self, model: ir.Model) -> None:
        """Mask counts `_FloatMaskCount` had to floatify and did not; blind to everything downstream of the reduction.
        A post-condition rather than a count pin: a tower reaching here with none did so via `FoldPadMaskPass`."""
        integers = (int(ir.DataType.INT32), int(ir.DataType.INT64))
        stuck = 0
        for node in model.graph:
            if node.op_type != "ReduceSum" or not node.inputs or node.inputs[0] is None:
                continue
            cast = node.inputs[0].producer()
            if cast is not None and cast.op_type == "Cast" and cast.attributes.get_int("to", 0) in integers:
                stuck += 1
        if stuck:
            raise ir.passes.PostconditionError(f"{stuck} integer mask count(s) survived _FloatMaskCount")


class DevitalizeShapeDomainPass(ir.passes.InPlacePass):
    """A dynamo-exported encoder's whole edit list, in order. One pass rather than pipeline entries because the probe
    it opens with needs the final input contract."""

    def __init__(self, *, rewrite_eot: bool = False) -> None:
        self.rewrite_eot = rewrite_eot

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        probes = probe_runtime(model)
        stages: list[ir.passes.PassBase] = []
        if self.rewrite_eot:
            stages.append(RewriteEotGatherndPass(probes))
        stages += [
            RewritePass([_FuseClassTokenPrepend.rule()]),
            ConstantifyPositionIdsPass(),
            # after the position ids and before the Expands, which would otherwise materialize the mask island
            FoldPadMaskPass(),
            # before the Expand pass materializes the token-type index into a batch-dependent chain
            RewritePass([FoldZeroIndexGather.rule()]),
            EliminateDynamicExpandsPass(probes),
            # both lookups are compile-time by now, and their chains drop into the DCE below
            RewritePass([FoldConstantGatherElements.rule(), _FoldConstantGather.rule()]),
            ConstantifyReshapeTargetsPass(probes),
            _FloatMaskConsumersPass(),
            RewritePass([_IdentityAveragePool.rule()]),
            common_passes.IdentityEliminationPass(),
            PruneAttnpoolDeadQueriesPass(),
            # after the Expands and the Reshape targets: materialized-Expand + const Reshape target
            RewritePass([_FoldConstantAttnQuery.rule()]),
            common_passes.RemoveUnusedNodesPass(),  # drop the now-dead Shape chains
            RewritePass([_SelectBeforeLayerNorm.rule(), _EotSelectBeforeLayerNorm.rule()]),
            _ScalarGatherToSlicePass(),
            CanonicalizeConstantsPass(),
            common_passes.DeduplicateInitializersPass(),
            # after the lift: the V-bias fold rewrites a bias tensor in place, which only an initializer ships
            RestructureAttention3dPass(),
            # after the 3D restructure: constant causal masks -> is_causal, their [1,1,S,S] initializers dead
            RewritePass([_FlipCausalAttention.rule()]),
            common_passes.RemoveUnusedNodesPass(),
            ReinferShapesPass(),  # surgery leaves stale annotations on every rewritten path
            # last: reads the re-inferred [B,S,D] annotations and the 3D Attention, and maintains the shapes
            # it changes itself rather than paying a second whole-graph inference
            HoistPoolingSelectPass(),
            # after every shape any pass declares is final
            UnifyDimSymbolsPass(),
        ]
        return ir.passes.Sequential(*stages)(model)

    def ensures(self, model: ir.Model) -> None:
        """Every tower that builds a padding mask retires its GatherND, by family. Op type and nothing else:
        `_AdditivePadMask` can only match the And `_BroadcastMaskRebuild` emits, so a miss in the first silently
        disables the second and both counts read 0."""
        if survivors := sum(1 for node in model.graph if node.op_type == "GatherND"):
            raise ir.passes.PostconditionError(f"{survivors} GatherND mask rebuild(s) survived the collapse")
