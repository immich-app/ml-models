"""Post-export graph surgery for the CLIP encoders: kill the batch-dependent shape domain so runtime
EP compilers (CoreML/RKNN/TensorRT) can constant-fold it, fuse the visual uint8 NHWC input contract,
and collapse per-head attention into 3D `Attention`.

ir throughout: one `ir.load`, every transform mutates the same lazy `ir.Model`, one `ir.save` — the
large token/embedding tables stay memory-mapped, never inlined into a protobuf.
"""

import tempfile
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import onnx
import onnx_ir as ir
import onnx_ir.passes.common as common_passes
from onnx import TensorProto, helper
from onnxscript.rewriter import RewriteRuleSet
from onnxscript.rewriter.pattern import MatchResult, RewriteRuleClassBase

from .lowering import _FoldConstantGatherElements


class Probe(NamedTuple):
    """One tensor's observed runtime state at a given batch size."""

    shape: tuple[int, ...]
    value: np.ndarray | None  # captured only for small integer tensors (the shape domain)
    dtype: int  # onnx TensorProto elem_type


# per-tensor observations across the probed batch sizes, keyed by tensor name
Probes = dict[str, list[Probe]]

_BROADCAST_OPS = {"Add", "Sub", "Mul", "Div", "Pow"}


def _make_init(graph: ir.Graph, name: str, array: np.ndarray) -> ir.Value:
    tensor = ir.tensor(array, name=name)
    value = ir.Value(name=name, shape=tensor.shape, type=ir.TensorType(tensor.dtype), const_value=tensor)
    graph.register_initializer(value)
    return value


def _node(
    op_type: str, inputs: list[ir.Value], name: str | None = None, out: str | None = None, **attributes: Any
) -> ir.Node:
    node = ir.node(op_type, inputs=inputs, attributes=attributes or None, num_outputs=1, name=name)
    if out is not None:
        node.outputs[0].name = out
    return node


def _const_array(value: ir.Value | None) -> np.ndarray | None:
    if value is None or value.const_value is None:
        return None
    return value.const_value.numpy()


def _const_ints(value: ir.Value | None) -> list[int] | None:
    arr = _const_array(value)
    if arr is None or arr.dtype.kind not in "iu":
        return None
    return [int(v) for v in arr.reshape(-1)]


def _producer_of(value: ir.Value | None, op_type: str) -> ir.Node | None:
    node = value.producer() if value is not None else None
    return node if node is not None and node.op_type == op_type else None


def _sole_consumer(value: ir.Value | None, op_type: str) -> ir.Node | None:
    uses = value.uses() if value is not None else ()
    if len(uses) == 1 and uses[0].node.op_type == op_type:
        return uses[0].node
    return None


def _single_use(value: ir.Value) -> bool:
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


def probe_runtime(model: ir.Model) -> Probes:
    """Run on CPU at batches 2,3 and record every tensor's shape (and value, for small int tensors —
    the shape domain). Ground truth past static shape inference. Routes through disk (save external ->
    infer_shapes_path -> run from path): the larger text encoders exceed the 2GB in-memory protobuf cap.
    `ir.save(external_data=...)` writes a fresh sidecar without mutating the in-memory model, so no
    restore is needed."""
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


def canonicalize_constants(model: ir.Model) -> None:
    """Convert every Constant node to a graph initializer, in place. CoreML's EP builder only
    recognizes initializers as constants, so Constant-fed Reshape/Gather/Slice get rejected without this."""
    common_passes.LiftConstantsToInitializersPass(lift_all_constants=True, size_limit=0)(model)


def fuse_visual_input(model: ir.Model, mean: list[float], std: list[float]) -> ir.Model:
    graph = model.graph
    input_value = graph.inputs[0]
    input_name = input_value.name
    dims = [d if isinstance(d, int) else d.value for d in input_value.shape]
    assert dims[1] == 3, f"expected NCHW visual input, got {dims}"
    size = int(dims[2])

    # (node, input-index) pairs to repoint later — captured before we add the pre nodes
    input_uses = [(use.node, use.idx) for use in input_value.uses()]
    all_consumers = input_value.consumers()
    # Shape consumers only read dims; repointing them to the NCHW tensor preserves the declared shape
    consumers = [n for n in all_consumers if n.op_type == "Conv"]
    unexpected = [n.op_type for n in all_consumers if n.op_type not in ("Conv", "Shape")]
    if not consumers or unexpected:
        raise ValueError(f"Cannot fuse visual input: consumed by {unexpected or 'nothing'}")

    scale = 1.0 / (255.0 * np.asarray(std, dtype=np.float64))
    shift = np.asarray(mean, dtype=np.float64) / np.asarray(std, dtype=np.float64)

    def _padded(conv: ir.Node) -> bool:
        if any(conv.attributes.get_ints("pads", [])):
            return True
        return conv.attributes.get_string("auto_pad", "NOTSET") not in ("NOTSET", "VALID")

    # ViT patch convs unpadded -> whole normalization folds into weights+bias. ResNet stems pad: the
    # shift doesn't commute with a zero pad, the scale does (0*s=0) -> fold scale only, subtract
    # 255*mean in-graph (pad zeros in that shifted domain match the model's normalized-zero pad values).
    fold_shift = not any(_padded(conv) for conv in consumers)

    for conv in consumers:
        w_value = conv.inputs[1]
        weight = w_value.const_value.numpy().astype(np.float64)  # [O, C, kH, kW]
        folded = (weight * scale[None, :, None, None]).astype(np.float32)
        w_value.const_value = ir.tensor(folded, name=w_value.name)
        if not fold_shift:
            continue

        bias_delta = -(weight.sum(axis=(2, 3)) @ shift)  # conv of the constant shift, per out channel
        if len(conv.inputs) > 2 and conv.inputs[2] is not None:
            b_value = conv.inputs[2]
            bias = b_value.const_value.numpy().astype(np.float64) + bias_delta
            b_value.const_value = ir.tensor(bias.astype(np.float32), name=b_value.name)
        else:
            b_value = _make_init(graph, f"{conv.name}_fused_bias", bias_delta.astype(np.float32))
            conv.resize_inputs(3)
            conv.replace_input_with(2, b_value)

    # prepend uint8 NHWC -> float NCHW: Cast (+ Sub when the shift stayed in-graph) + Transpose
    cast_node = _node("Cast", [input_value], name="pre_cast", out=f"{input_name}_f32", to=int(TensorProto.FLOAT))
    pre = [cast_node]
    transpose_in = cast_node.outputs[0]
    if not fold_shift:
        # [3] broadcasts over NHWC's trailing channel axis
        shift_value = _make_init(
            graph, f"{input_name}_shift", (255.0 * np.asarray(mean, dtype=np.float64)).astype(np.float32)
        )
        sub_node = _node("Sub", [transpose_in, shift_value], name="pre_shift", out=f"{input_name}_shifted")
        pre.append(sub_node)
        transpose_in = sub_node.outputs[0]
    transpose_node = _node(
        "Transpose", [transpose_in], name="pre_nhwc_to_nchw", out=f"{input_name}_chw", perm=[0, 3, 1, 2]
    )
    pre.append(transpose_node)
    chw_value = transpose_node.outputs[0]

    for node, idx in input_uses:  # repoint the original Conv/Shape consumers onto the NCHW tensor
        node.replace_input_with(idx, chw_value)
    graph.extend(pre)
    graph.sort()

    # switch to uint8 NHWC, keeping the source's leading dim (batch for dynamo exports, literal 1 for
    # legacy caches upgraded through this same transform)
    input_value.dtype = ir.DataType.UINT8
    input_value.shape = ir.Shape([dims[0], size, size, 3])

    for out in graph.outputs:  # reinfer: drop stale output shapes, re-derive
        out.shape = None
    return common_passes.ShapeInferencePass()(model).model


class _EotOneHotSelect(RewriteRuleClassBase):
    """Rewrite `x[arange(batch), idx]` EOT pooling (GatherND) as a one-hot matmul
    `(eye[S][idx])[b,1,S] @ x[b,S,D]` — exact selection with ops every EP claims under dynamic batch;
    the Range/GatherND machinery dies in DCE (remove_nodes=False). Overhead noise-level (RK3588
    48us=0.12%). Leaner GatherND(batch_dims=1) stays out: CoreML builder (#28598) needs constant
    indices + batch_dims=0, splitting the partition right before the final LN."""

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
        # share data's dtype (mixed-type MatMul is illegal on fp16/bf16); one-hot is exact at any precision
        eye_np_dtype = helper.tensor_dtype_to_np_dtype(self._probes[data.name][0].dtype)
        eye = op.initializer(ir.tensor(np.eye(seq, dtype=eye_np_dtype), name=f"{base}_eye"))
        axes = op.initializer(ir.tensor(np.array([1], dtype=np.int64), name=f"{base}_axes1"))
        onehot = op.Unsqueeze(op.Gather(eye, index, axis=0), axes)
        return op.Squeeze(op.MatMul(onehot, data), axes)


def rewrite_eot_gathernd(model: ir.Model, probes: Probes) -> int:
    return RewriteRuleSet([_EotOneHotSelect.rule(probes)]).apply_to_model(model)


def constantify_position_ids(model: ir.Model) -> int:
    """Replace XLM-R's data-dependent ``position_ids`` with the equivalent constant arange, in place.
    Pad positions are masked out everywhere downstream, so only the non-pad ids matter and those are a
    fixed ``arange(pad+1, pad+1+seq)``; evaluating the weight-free subgraph on an all-non-pad input
    yields it, bit-exact. Kills the Equal/Not/CumSum chain (RKNPU has no int32 ``Equal`` kernel; CoreML
    fragments it). No-op without the pattern (openclip text, LaBSE). Returns count replaced."""
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

    # position-embedding lookup: 2D float table indexed by a CumSum-derived tensor whose producer is
    # arithmetic, not a re-gather (that's the token-type path downstream; replacing the root kills the chain)
    position_ids: ir.Value | None = None
    for node in graph:
        if node.op_type != "Gather" or node.inputs[0] is None or not is_terminal(node.inputs[0]):
            continue
        table = node.inputs[0].const_value  # metadata only; never materializes the big token table
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
        return 0

    # eval the weight-free position_ids subgraph at all-non-pad (extract walks backward; big table untouched)
    sub_graph = ir.convenience.extract(graph, list(graph.inputs), [position_ids])
    sub_model = helper.make_model(
        ir.to_proto(sub_graph), opset_imports=[helper.make_opsetid(d, v) for d, v in graph.opset_imports.items()]
    )
    session = ort.InferenceSession(sub_model.SerializeToString(), providers=["CPUExecutionProvider"])
    feed = {}
    for i in graph.inputs:
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        np_dtype = {ir.DataType.INT32: np.int32, ir.DataType.INT64: np.int64}.get(i.dtype, np.int64)
        # all-non-pad: every position valid (mask all-ones) and no token equal to the pad id
        feed[i.name] = np.ones(shape, np_dtype) if "mask" in i.name.lower() else np.full(shape, 100, np_dtype)
    pid_dtype = position_ids.dtype if position_ids.dtype is not None else ir.DataType.INT64
    const = session.run([position_ids.name], feed)[0].astype(pid_dtype.numpy())

    producer = position_ids.producer()
    replacement = _make_init(graph, position_ids.name, const)
    position_ids.replace_all_uses_with(replacement)
    graph.remove(producer, safe=True)
    return 1


class _ConstantifyReshapeTarget(RewriteRuleClassBase):
    """Replace a batch-derived Reshape target with a constant initializer, pinned from the OUTPUT's
    probed shape at two batches (agreeing dims literal, the batch-varying dim -> -1). Termination hinges
    on the replacement being an initializer: the is_initializer guard stops the rule re-firing on its
    re-emitted Reshape (which inherits the output name, so the probe lookup still hits)."""

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


def constantify_reshape_targets(model: ir.Model, probes: Probes) -> int:
    """Replace batch-derived Reshape targets with constant initializers, in place. Fail-closed: any
    dynamic-target Reshape not pinnable from the probes is collected and raised."""
    rewritten = RewriteRuleSet([_ConstantifyReshapeTarget.rule(probes)]).apply_to_model(model)
    unresolved = [
        node.name for node in model.graph if node.op_type == "Reshape" and not node.inputs[1].is_initializer()
    ]
    if unresolved:
        raise ValueError(f"Reshape targets not resolvable from probes: {unresolved}")
    return rewritten


def eliminate_dynamic_expands(model: ir.Model, probes: Probes) -> int:
    """Remove Expand nodes with batch-derived target shapes, in place. Two cases by consumer:
    broadcast-native consumers with a provably unchanged result at both probed batches -> drop the
    Expand, rewire to `data`; a consumer needing the materialized shape -> `data + static_zeros +
    batch_col`, where static_zeros (target shape, batch axis pinned to 1) widens every non-batch axis
    and batch_col (`[..,batch,..,1]`) widens the batch axis. Only batch is symbolic, so every EP
    compiles it — same result as the Expand without a dynamic-dim op runtime compilers reject."""
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

    # the batch sizes the probe actually ran at, read off a batch-carrying input's two shapes
    batch_input = next((i for i in graph.inputs if not isinstance(i.shape[0], int)), None)
    probe_batches = (
        tuple(int(p.shape[0]) for p in probes[batch_input.name])
        if batch_input is not None and batch_input.name in probes
        else None
    )

    batch_seed: list[ir.Value] = []  # lazily built [batch] rank-1 zero column, shared across expands

    def batch_zeros_1d() -> ir.Value:
        """A `[batch]` rank-1 float zero column tied to the runtime batch dim. Slice one element off
        each non-batch axis (reduction-free: a global reduction overflows fp16 — image sum ~2e7 ->
        inf*0=NaN — and the ANE compiler can't lower it), flatten to `[batch]`, zero it. Built once, shared."""
        if batch_seed:
            return batch_seed[0]
        if batch_input is None:
            raise ValueError("No batch-carrying graph input to seed Expand materialization")
        src = batch_input.name
        nonbatch = list(range(1, len(batch_input.shape)))
        starts = _make_init(graph, f"{src}_ez_starts", np.zeros(len(nonbatch), dtype=np.int64))
        ends = _make_init(graph, f"{src}_ez_ends", np.ones(len(nonbatch), dtype=np.int64))
        axes = _make_init(graph, f"{src}_ez_axes", np.array(nonbatch, dtype=np.int64))
        flat = _make_init(graph, f"{src}_ez_flat", np.array([-1], dtype=np.int64))
        zero = _make_init(graph, f"{src}_ez_zero", np.zeros([], dtype=np.float32))
        # final order after sort is Slice -> Cast -> Reshape -> Mul
        slice_node = _node("Slice", [batch_input, starts, ends, axes], name="ez_slice", out=f"{src}_ez_s")
        cast_node = _node("Cast", [slice_node.outputs[0]], name="ez_cast", out=f"{src}_ez_f", to=TensorProto.FLOAT)
        reshape_node = _node("Reshape", [cast_node.outputs[0], flat], name="ez_flatten", out=f"{src}_ez_r")
        mul_node = _node("Mul", [reshape_node.outputs[0], zero], name="ez_mul", out=f"{src}_ez")
        graph.extend([slice_node, cast_node, reshape_node, mul_node])
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
        # column seed is exactly [batch], so the widened axis must be batch itself, not a multiple
        # (e.g. batch*heads) — else silently mis-sized. Fail loud.
        if probe_batches is not None and (int(shapes[0][axis]), int(shapes[1][axis])) != probe_batches:
            raise ValueError(
                f"Expand {node.name}: dynamic axis {axis} varies as "
                f"{(int(shapes[0][axis]), int(shapes[1][axis]))}, a multiple of batch {probe_batches}; "
                "materialization from a [batch] seed would mis-size it"
            )
        data_dtype = dtype_of(data)
        # ONNX Add has no bool kernel -> round-trip bool data through int32 (0/1-exact); other dtypes add directly
        compute_dtype = TensorProto.INT32 if data_dtype == TensorProto.BOOL else data_dtype
        np_compute = helper.tensor_dtype_to_np_dtype(compute_dtype)

        static_shape = shapes[0].copy()  # target shape with the batch axis pinned to 1
        static_shape[axis] = 1
        static_iv = _make_init(graph, f"{out.name}_static0", np.zeros(static_shape, dtype=np_compute))

        seed = batch_zeros_1d()  # [batch], float32
        new_nodes: list[ir.Node] = []

        data_in = data
        if data_dtype == TensorProto.BOOL:
            cast = _node("Cast", [data], name=f"{node.name}_data_cast", out=f"{out.name}_data_i", to=compute_dtype)
            data_in = cast.outputs[0]
            new_nodes.append(cast)

        col = seed
        if rank > 1:  # reshape the [batch] seed to [..,batch,..,1] with batch at `axis`
            unsqueeze_axes = [a for a in range(rank) if a != axis]
            ua_iv = _make_init(graph, f"{out.name}_ua", np.array(unsqueeze_axes, dtype=np.int64))
            unsq = _node("Unsqueeze", [col, ua_iv], name=f"{node.name}_col", out=f"{out.name}_col")
            col = unsq.outputs[0]
            new_nodes.append(unsq)
        if compute_dtype != TensorProto.FLOAT:
            castc = _node("Cast", [col], name=f"{node.name}_col_cast", out=f"{out.name}_col_c", to=compute_dtype)
            col = castc.outputs[0]
            new_nodes.append(castc)

        # data + static_zeros widens every non-batch axis; + batch_col widens the batch axis
        add_static = _node("Add", [data_in, static_iv], name=f"{node.name}_bcast_static", out=f"{out.name}_static")
        add_bcast = _node("Add", [add_static.outputs[0], col], name=f"{node.name}_bcast")
        new_nodes += [add_static, add_bcast]

        if data_dtype == TensorProto.BOOL:
            add_bcast.outputs[0].name = f"{out.name}_pre"
            out_cast = _node("Cast", [add_bcast.outputs[0]], name=f"{node.name}_out_cast", to=TensorProto.BOOL)
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

    graph.sort()  # nodes appended out of order above; restore topological order for serialization
    return eliminated


class FuseClassTokenPrepend(RewriteRuleClassBase):
    """Fuse the ViT class-token prepend + positional-embedding add into `Pad` + one constant. Left
    alone the class-token Expand gets materialized into a zero column whose seed reduces the whole image
    to `[batch]` — overflows fp16 (image sum ~2e7 -> inf*0=NaN) and the ANE can't lower it. Instead
    prepend a zero row with static `Pad` and add `cp` with `cp[0]=class+pos[0]`, `cp[1:]=pos[1:]`
    (associativity of the two constant adds), fully static. Runs before `eliminate_dynamic_expands` so
    the Expand is consumed here. Class-token ViTs only (SigLIP attn-pool + text encoders untouched)."""

    def pattern(self, op: Any, class_embedding: Any, expand_shape: Any, patches: Any, positional: Any) -> Any:
        expanded = op.Expand(class_embedding, expand_shape)
        return op.Add(op.Concat(expanded, patches, axis=1), positional)

    def check(
        self, context: Any, class_embedding: Any, expand_shape: Any, patches: Any, positional: Any
    ) -> MatchResult:
        result = MatchResult()
        cls, pos = class_embedding.const_value, positional.const_value
        if cls is None or pos is None:
            return result.fail("class/positional not constant")
        if cls.numpy().reshape(-1).shape[0] != pos.numpy().shape[-1]:
            return result.fail("class width != positional width")
        return result

    def rewrite(self, op: Any, class_embedding: Any, expand_shape: Any, patches: Any, positional: Any) -> Any:
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
    """``Gather(data, scalar_index)`` -> ``Slice`` + ``Squeeze``: CoreML EP rejects scalar-index Gather
    unless data is fully static (a dynamic batch rules that out). Applied after DCE so only data-domain
    gathers remain — a shape-domain scalar would become a rank-0 Squeeze that MIL refuses to compile."""

    def pattern(self, op: Any, data: Any, index: Any) -> Any:
        return op.Gather(data, index, _outputs=["gathered"])

    def check(self, context: Any, data: Any, index: Any, gathered: Any) -> MatchResult:
        result = MatchResult()
        value = index.const_value
        if value is None or value.numpy().ndim != 0:  # only a rank-0 (scalar) constant index
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


class _SelectBeforeLayerNorm(RewriteRuleClassBase):
    """``Gather(LayerNormalization(x), scalar)`` -> ``LayerNormalization(Gather(x))``: last-axis LN
    normalizes each token independently, so selecting one token commutes bit-exactly (the 76 discarded
    positions are pure waste). Must run before ``_ScalarGatherToSlice``, which consumes the Gather."""

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


class _EotSelectBeforeLayerNorm(RewriteRuleClassBase):
    """``MatMul(one_hot, LayerNormalization(x))`` -> ``LayerNormalization(MatMul(one_hot, x))``: the EOT
    one-hot select commutes bit-exactly with last-axis ``ln_final``. The check pins the selector to the
    identity-table Gather so MLP ``MatMul(ln, weight)`` nodes can't match."""

    def pattern(self, op: Any, selector: Any, x: Any, scale: Any, bias: Any) -> Any:
        ln = op.LayerNormalization(x, scale, bias, _outputs=["ln"])
        return op.MatMul(selector, ln, _outputs=["mm"])

    def check(self, context: Any, selector: Any, x: Any, scale: Any, bias: Any, ln: Any, mm: Any) -> MatchResult:
        result = MatchResult()
        unsqueeze = selector.producer()
        if unsqueeze is None or unsqueeze.op_type != "Unsqueeze":
            return result.fail("selector is not the EOT Unsqueeze(Gather(eye)) chain")
        gather = unsqueeze.inputs[0].producer() if unsqueeze.inputs[0] is not None else None
        if gather is None or gather.op_type != "Gather":
            return result.fail("selector is not the EOT Unsqueeze(Gather(eye)) chain")
        table = gather.inputs[0].const_value
        if table is None or len(table.shape) != 2 or table.shape[0] != table.shape[1]:
            return result.fail("selector table is not a square constant")
        if not np.array_equal(table.numpy(), np.eye(table.shape[0], dtype=table.numpy().dtype)):
            return result.fail("selector table is not the identity (not an exact one-hot select)")
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
    """``AveragePool`` with kernel 1 / stride 1 / no pad is an identity copy — ModifiedResNet's layer1
    downsample stamps one, so RN50-class visuals ship a full-tensor no-op (~0.5% CPU; every EP pays it).
    Rewritten to ``Identity``, folded by ``IdentityEliminationPass``."""

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
    """XLM-R's ``attention_mask`` -> ``[batch,1,S,S]`` rebuild via a runtime GatherND index table is
    just a broadcast of ``mask[b, j]`` (index tuples are exactly ``(i, j)``). Replaced with
    ``And(tril, Unsqueeze(Cast(mask, bool), [1, 2]))``, killing the graph's only data-dependent-shape op
    (``Range`` over batch) + ``GatherND`` — the island that fragments the mclip text encoder on CoreML.
    Check pins the ``(i, j)`` index construction (``Range(0, ·, 1)`` batch coord + constant ``arange(S)``
    token coord); else fails closed."""

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

        # walk the index construction: Range(0, ., 1) batch coord + constant arange(S) token coord
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
    """XLM-R's boolean ``Attention`` mask -> the equivalent float additive key bias. After
    ``_BroadcastMaskRebuild`` the mask is ``And(all_true, Unsqueeze(Cast(attention_mask, bool), [1,2]))``;
    the const operand carries no masking (XLM-R is bidirectional), so it's just the key-padding row and
    the bool plumbing strands an integer island on float-only accelerators (CoreML splits the tower
    there). Additive form feeds a ``[b,1,S,S]`` float bias: 0 keeps, -1e4 removes (softmax underflows to
    exactly 0; bit-exact for 0/1 masks on ORT CPU). The trailing zero-Add materializes the query axis —
    ORT CPU ``Attention`` enforces ``mask.shape[-2] == q_len`` instead of broadcasting (attention_helper.h).
    Fails closed unless the And operand is an all-True bool const, the mask a rank-2 input, and every
    consumer an ``Attention`` mask input."""

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
        # [b,1,1,S] + zeros[S,1] -> [b,1,S,S]: ORT CPU refuses to broadcast the mask's q axis
        return op.Add(keys, op.Constant(value=ir.tensor(np.zeros((seq, 1), np.float32))))


class _FloatMaskCount(RewriteRuleClassBase):
    """Mean-pool token count takes an int64 detour for a sum of 0/1 values fp32 counts exactly (int
    exact to 2^24). Count in float, dropping the int64 ops, leaving the post-embedding graph float-only."""

    def pattern(self, op: Any, mask: Any, axes: Any, unsqueeze_axes: Any) -> Any:
        total = op.ReduceSum(op.Cast(mask, to=int(ir.DataType.INT64)), axes, _outputs=["total"])
        return op.Cast(op.Unsqueeze(total, unsqueeze_axes), to=int(ir.DataType.FLOAT))

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
        return op.Unsqueeze(counted, unsqueeze_axes)


class _FoldConstantAttnQuery(RewriteRuleClassBase):
    """SigLIP's MAP-head query projection runs entirely on constants (batch-materialized latent,
    ``MatMul``+bias, ``Reshape``+``Transpose`` — six ops recomputing the same ``[1,H,1,d]`` every
    inference). Fold (fp64) into one initializer + a single Add against the zero column re-axed to the
    packed rank, which alone carries batch. Fails closed unless every operand but the column is constant,
    the Expand zeros exactly zero, the seed is ``batch_zeros_1d``'s ``Mul``-by-zero output, and the batch
    axis leads the Reshape target (the ResNet attnpool's data-derived queries fail here)."""

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


def prune_attnpool_dead_queries(model: ir.Model) -> int:
    """Restructure CLIP's ResNet attention-pool to compute only the query it keeps. The head runs full
    50-query attention then ``Slice[0:1]`` discards 49 (~5% of RN50 FLOPs); slicing the pre-projection
    tensor to row 0 first is mathematically identical (per-row projections). Numerically equivalent, not
    bit-exact: the q/proj GEMMs run at different row counts. Matches only the ResNet attnpool cluster
    (SigLIP2's MAP head has a constant query and cannot match). Returns clusters rewritten.
    """
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
        """Point a Reshape at a fresh target initializer (the old one may be shared with k/v) and update
        the cached shape so the pre-reinference no-op-Slice path of ``_ScalarGatherToSlice`` sees the
        single-row form."""
        target = _make_init(graph, name, np.array(values, np.int64))
        node.replace_input_with(1, target)
        old_dims = node.outputs[0].shape
        if old_dims is not None and len(old_dims) == len(values):
            node.outputs[0].shape = ir.Shape([v if v != -1 else old_dims[i] for i, v in enumerate(values)])

    for attention in [n for n in graph if n.op_type == "Attention"]:
        # q branch: Reshape[-1,H,S,d] <- Transpose[1,0,2] <- Reshape[S,-1,d] <- Add <- MatMul
        q_chain = producer_chain(attention.inputs[0], ["Reshape", "Transpose", "Reshape", "Add", "MatMul"])
        if q_chain is None:
            continue
        q_pack, q_perm, q_unpack, _, q_matmul = q_chain
        pack_target, unpack_target = _const_ints(q_pack.inputs[1]), _const_ints(q_unpack.inputs[1])
        perm = q_perm.attributes.get_ints("perm")
        if pack_target is None or unpack_target is None or perm is None or list(perm) != [1, 0, 2]:
            continue
        if len(unpack_target) != 3 or len(pack_target) != 4 or unpack_target[0] != pack_target[2]:
            continue
        seq = unpack_target[0]
        if not isinstance(seq, int) or seq <= 1:
            continue

        # downstream: Transpose[2,0,1,3] -> Reshape[-1,E] -> Gemm -> Reshape[S,-1,D] -> Gather(0 @ axis 0),
        # lowered to Slice+Squeeze later (by then the axis has one row -> bare Squeeze via no-op-Slice path)
        out_perm = _sole_consumer(attention.outputs[0], "Transpose")
        if out_perm is None or list(out_perm.attributes.get_ints("perm", [])) != [2, 0, 1, 3]:
            continue
        out_flat = _sole_consumer(out_perm.outputs[0], "Reshape")
        gemm = _sole_consumer(out_flat.outputs[0], "Gemm") if out_flat is not None else None
        out_unflat = _sole_consumer(gemm.outputs[0], "Reshape") if gemm is not None else None
        token_gather = _sole_consumer(out_unflat.outputs[0], "Gather") if out_unflat is not None else None
        if token_gather is None:
            continue
        unflat_target = _const_ints(out_unflat.inputs[1])
        if unflat_target is None or len(unflat_target) != 3 or unflat_target[0] != seq:
            continue
        index = token_gather.inputs[1].const_value
        if token_gather.attributes.get_int("axis", 0) != 0:
            continue
        if index is None or index.numpy().ndim != 0 or int(index.numpy()) != 0:
            continue

        # slice the [S, batch, E] pre-projection tensor to row 0 for the q path only
        row = _node(
            "Slice",
            [
                q_matmul.inputs[0],
                _make_init(graph, f"{attention.name}_q_row_start", np.array([0], np.int64)),
                _make_init(graph, f"{attention.name}_q_row_end", np.array([1], np.int64)),
                _make_init(graph, f"{attention.name}_q_row_axis", np.array([0], np.int64)),
            ],
            out=f"{attention.name}_q_row",
        )
        graph.insert_before(q_matmul, [row])
        q_matmul.replace_input_with(0, row.outputs[0])
        new_target(q_unpack, [1, unpack_target[1], unpack_target[2]], f"{attention.name}_q_unpack_1")
        new_target(q_pack, [pack_target[0], pack_target[1], 1, pack_target[3]], f"{attention.name}_q_pack_1")
        new_target(out_unflat, [1, unflat_target[1], unflat_target[2]], f"{attention.name}_out_1")
        rewritten += 1
    return rewritten


def restructure_attention_3d(model: ir.Model) -> int:
    """Collapse the exported per-head attention plumbing into batch-first 3D ``Attention`` (opset-23
    q/k/v ``[B,S,D]`` + ``q_num_heads``/``kv_num_heads``, both always emitted — OpenVINO hard-fails
    without them). Three motifs: open_clip seq-first packed QKV (out-proj ``Gemm`` -> ``MatMul``, weight
    pre-transposed at export, fp-exact), timm batch-first packed QKV (incl. SigLIP2 MAP-head kv, whose
    folded constant query is re-laid-out ``[1,H,1,dh]`` -> ``[1,1,D]``), and separate q/k/v projections
    (XLM-R). For the packed motifs the V-projection bias folds through into the out-proj bias
    (``bo' = bo + b_v @ Wo^T``, fp64; softmax rows sum to one so a constant added to V passes verbatim)
    and the V bias block is zeroed. Masks untouched; ``scale``/``is_causal``/``softcap`` carry over.
    Non-matching sites (ResNet attnpool's data-derived queries) left as-is. Returns sites restructured."""
    graph = model.graph

    def perm_of(node: ir.Node) -> list[int] | None:
        perm = node.attributes.get_ints("perm")
        return list(perm) if perm is not None else None

    def axes_of(node: ir.Node) -> list[int] | None:
        return _const_ints(node.inputs[1]) if len(node.inputs) > 1 else None

    def slice_start(node: ir.Node, length: int) -> int | None:
        """The i of a unit-width ``Slice(i:i+1, axis 0)``, else None."""
        params = [_const_ints(node.inputs[i]) if len(node.inputs) > i else None for i in (1, 2, 3, 4)]
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
        bias = next((v for v in node.inputs if _const_array(v) is not None), None)
        data = next((v for v in node.inputs if v is not bias), None)
        if bias is None or data is None or bias.const_value.size != width or not _single_use(bias):
            return None
        return data, bias

    def set_heads(att: ir.Node, heads: int) -> None:
        att.attributes["q_num_heads"] = ir.AttrInt64("q_num_heads", heads)
        att.attributes["kv_num_heads"] = ir.AttrInt64("kv_num_heads", heads)

    def fold_v_bias(packed_bias: ir.Value, out_weight: np.ndarray, out_bias: ir.Value) -> None:
        """``bo' = bo + b_v @ Wo`` in fp64 (``out_weight`` in y = x @ W orientation), then zero
        the trailing (V) block of the packed bias. Exact through softmax: rows sum to one."""
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

        x: ir.Value  # batch-first [B,S,D] projection input
        weight: ir.Value  # [D, n*D]
        bias: ir.Value  # [n*D]
        add_out: ir.Value  # the packed projection output
        heads: int
        width: int  # D
        seq: int

    def match_seqfirst_packed(att: ir.Node) -> dict[str, Any] | None:
        """open_clip resblock: seq-first packed QKV in, flattened Gemm out."""
        shared: ir.Node | None = None
        pack_target = unpack_target = None
        for i in range(3):
            pack = _producer_of(att.inputs[i], "Reshape")  # [-1,H,S,dh]
            if pack is None or not _single_use(pack.outputs[0]):
                return None
            head_tr = _producer_of(pack.inputs[0], "Transpose")
            if head_tr is None or perm_of(head_tr) != [1, 0, 2] or not _single_use(head_tr.outputs[0]):
                return None
            unpack = _producer_of(head_tr.inputs[0], "Reshape")  # [S,-1,dh]
            if unpack is None or not _single_use(unpack.outputs[0]):
                return None
            if pack_target is None:
                pack_target, unpack_target = _const_ints(pack.inputs[1]), _const_ints(unpack.inputs[1])
            elif _const_ints(pack.inputs[1]) != pack_target or _const_ints(unpack.inputs[1]) != unpack_target:
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
        tr5 = _producer_of(shared.inputs[0], "Transpose")
        if tr5 is None or perm_of(tr5) != [3, 1, 2, 0, 4] or not _single_use(tr5.outputs[0]):
            return None
        unsqueeze = _producer_of(tr5.inputs[0], "Unsqueeze")
        if unsqueeze is None or axes_of(unsqueeze) != [0] or not _single_use(unsqueeze.outputs[0]):
            return None
        packed_reshape = _producer_of(unsqueeze.inputs[0], "Reshape")  # [S,-1,3,D]
        if packed_reshape is None or not _single_use(packed_reshape.outputs[0]):
            return None
        packed_target = _const_ints(packed_reshape.inputs[1])
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
        if not _single_use(packed_reshape.inputs[0]):
            return None
        packed = bias_add(_producer_of(packed_reshape.inputs[0], "Add"), 3 * width)
        if packed is None or not _single_use(packed[0]):
            return None
        mm_out, bias = packed
        matmul = _producer_of(mm_out, "MatMul")
        if matmul is None:
            return None
        weight = matmul.inputs[1]
        w_arr = _const_array(weight)
        if w_arr is None or w_arr.shape != (width, 3 * width):
            return None
        pre_tr = _producer_of(matmul.inputs[0], "Transpose")
        if pre_tr is None or perm_of(pre_tr) != [1, 0, 2] or not _single_use(pre_tr.outputs[0]):
            return None

        # out side: Transpose(2,0,1,3) -> Reshape[-1,D] -> Gemm -> Reshape[S,-1,D] -> Transpose(1,0,2)
        out_tr = _sole_consumer(att.outputs[0], "Transpose")
        if out_tr is None or perm_of(out_tr) != [2, 0, 1, 3]:
            return None
        out_flat = _sole_consumer(out_tr.outputs[0], "Reshape")
        if out_flat is None or _const_ints(out_flat.inputs[1]) != [-1, width]:
            return None
        gemm = _sole_consumer(out_flat.outputs[0], "Gemm")
        if gemm is None or len(gemm.inputs) != 3:
            return None
        attrs = gemm.attributes
        trans_b = attrs.get_int("transB", 0)
        if attrs.get_float("alpha", 1.0) != 1.0 or attrs.get_float("beta", 1.0) != 1.0 or attrs.get_int("transA", 0):
            return None
        wo, bo = gemm.inputs[1], gemm.inputs[2]
        wo_arr = _const_array(wo)
        bo_arr = _const_array(bo)
        if wo_arr is None or wo_arr.shape != (width, width) or not _single_use(wo):
            return None
        if bo_arr is None or bo_arr.size != width or not _single_use(bo):
            return None
        out_unflat = _sole_consumer(gemm.outputs[0], "Reshape")
        if out_unflat is None or _const_ints(out_unflat.inputs[1]) != [seq, -1, width]:
            return None
        out_tr2 = _sole_consumer(out_unflat.outputs[0], "Transpose")
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
        squeeze = _producer_of(value, "Squeeze")
        if squeeze is None or axes_of(squeeze) != [0] or not _single_use(squeeze.outputs[0]):
            return None
        unbind = _producer_of(squeeze.inputs[0], "Slice")
        if unbind is None or not _single_use(unbind.outputs[0]):
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
        reshape = _producer_of(shared.inputs[0], "Reshape")
        if reshape is None or not _single_use(reshape.outputs[0]):
            return None
        target = _const_ints(reshape.inputs[1])
        if target is None or len(target) != 5 or target[0] != -1 or target[2] != n:
            return None
        seq, heads, head_dim = target[1], target[3], target[4]
        width = heads * head_dim
        if seq <= 0 or heads <= 0 or head_dim <= 0:
            return None
        # the projection output must feed this cluster alone: its bias is about to be mutated
        if not _single_use(reshape.inputs[0]):
            return None
        packed = bias_add(_producer_of(reshape.inputs[0], "Add"), n * width)
        if packed is None or not _single_use(packed[0]):
            return None
        mm_out, bias = packed
        matmul = _producer_of(mm_out, "MatMul")
        if matmul is None:
            return None
        w_arr = _const_array(matmul.inputs[1])
        if w_arr is None or w_arr.shape != (width, n * width):
            return None
        return Packed(matmul.inputs[0], matmul.inputs[1], bias, reshape.inputs[0], heads, width, seq)

    def match_batchfirst_out(att: ir.Node, seq: int, width: int) -> dict[str, Any] | None:
        """Batch-first out side: Transpose(0,2,1,3) -> Reshape[-1,S,D] -> MatMul + Add."""
        out_tr = _sole_consumer(att.outputs[0], "Transpose")
        if out_tr is None or perm_of(out_tr) != [0, 2, 1, 3]:
            return None
        out_reshape = _sole_consumer(out_tr.outputs[0], "Reshape")
        if out_reshape is None or _const_ints(out_reshape.inputs[1]) != [-1, seq, width]:
            return None
        out_mm = _sole_consumer(out_reshape.outputs[0], "MatMul")
        if out_mm is None:
            return None
        wo_arr = _const_array(out_mm.inputs[1])
        if wo_arr is None or wo_arr.shape != (width, width):
            return None
        folded = bias_add(_sole_consumer(out_mm.outputs[0], "Add"), width)
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
        # q: Add(const [1,H,1,dh], Unsqueeze(batch_zeros_1d seed, [1,2,3])) from _FoldConstantAttnQuery
        q_add = _producer_of(att.inputs[0], "Add")
        if q_add is None or not _single_use(q_add.outputs[0]):
            return None
        query = next((v for v in q_add.inputs if _const_array(v) is not None), None)
        col = next((v for v in q_add.inputs if v is not query), None)
        q_arr = _const_array(query)
        if q_arr is None or q_arr.shape != (1, packed.heads, 1, packed.width // packed.heads):
            return None
        unsqueeze = _producer_of(col, "Unsqueeze")
        if unsqueeze is None or axes_of(unsqueeze) != [1, 2, 3] or not _single_use(unsqueeze.outputs[0]):
            return None
        out = match_batchfirst_out(att, 1, packed.width)
        if out is None:
            return None
        return {"packed": packed, "query": query, "seed": unsqueeze.inputs[0], **out}

    def match_separate(att: ir.Node) -> dict[str, Any] | None:
        """HF-style separate q/k/v projections feeding per-head Reshape+Transpose (XLM-R)."""
        sources = []
        shape = None
        for i in range(3):
            head_tr = _producer_of(att.inputs[i], "Transpose")
            if head_tr is None or perm_of(head_tr) != [0, 2, 1, 3] or not _single_use(head_tr.outputs[0]):
                return None
            reshape = _producer_of(head_tr.inputs[0], "Reshape")
            if reshape is None or not _single_use(reshape.outputs[0]):
                return None
            target = _const_ints(reshape.inputs[1])
            if target is None or len(target) != 4 or target[0] != -1 or min(target[1:]) <= 0:
                return None
            if shape is None:
                shape = target
            elif target != shape:
                return None
            projected = bias_add(reshape.inputs[0].producer(), target[2] * target[3])
            if projected is None or _producer_of(projected[0], "MatMul") is None:
                return None
            sources.append(reshape.inputs[0])
        assert shape is not None
        seq, heads, head_dim = shape[1], shape[2], shape[3]
        out_tr = _sole_consumer(att.outputs[0], "Transpose")
        if out_tr is None or perm_of(out_tr) != [0, 2, 1, 3]:
            return None
        out_reshape = _sole_consumer(out_tr.outputs[0], "Reshape")
        if out_reshape is None or _const_ints(out_reshape.inputs[1]) != [-1, seq, heads * head_dim]:
            return None
        return {"sources": sources, "heads": heads, "final": out_reshape.outputs[0]}

    split_sizes: dict[tuple[int, int], ir.Value] = {}  # (n, width) -> shared Split sizes initializer

    def emit_split(base: str, packed: Packed, n: int) -> list[ir.Value]:
        sizes = split_sizes.get((n, packed.width))
        if sizes is None:
            sizes = _make_init(graph, f"attn3d_split_{n}x{packed.width}", np.full(n, packed.width, np.int64))
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
            matmul = _node("MatMul", [packed.x, packed.weight], name=f"{base}_qkv_mm", out=f"{base}_qkv_mm_out")
            add = _node("Add", [matmul.outputs[0], packed.bias], name=f"{base}_qkv_bias", out=f"{base}_qkv")
            graph.extend([matmul, add])
            qkv = emit_split(base, packed._replace(add_out=add.outputs[0]), 3)
            for i in range(3):
                att.replace_input_with(i, qkv[i])
            set_heads(att, packed.heads)
            wo_t = _make_init(graph, f"{base}_wo_t", np.ascontiguousarray(m["wo_t"]))
            out_mm = _node("MatMul", [att.outputs[0], wo_t], name=f"{base}_out_mm", out=f"{base}_out_mm_out")
            out_add = _node("Add", [out_mm.outputs[0], m["bo"]], name=f"{base}_out_bias", out=f"{base}_out")
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
            q_arr = _const_array(m["query"]).transpose(0, 2, 1, 3).reshape(1, 1, packed.width)
            query = _make_init(graph, f"{base}_q3", np.ascontiguousarray(q_arr))
            axes = _make_init(graph, f"{base}_q_col_axes", np.array([1, 2], np.int64))
            unsqueeze = _node("Unsqueeze", [m["seed"], axes], name=f"{base}_q_col", out=f"{base}_q_col_out")
            q_add = _node("Add", [query, unsqueeze.outputs[0]], name=f"{base}_q_bcast", out=f"{base}_q")
            graph.extend([unsqueeze, q_add])
            att.replace_input_with(0, q_add.outputs[0])
            att.replace_input_with(1, kv[0])
            att.replace_input_with(2, kv[1])
            set_heads(att, packed.heads)
            m["final"].replace_all_uses_with(att.outputs[0], replace_graph_outputs=True)
        elif (m := match_separate(att)) is not None:
            for i in range(3):
                att.replace_input_with(i, m["sources"][i])
            set_heads(att, m["heads"])
            m["final"].replace_all_uses_with(att.outputs[0], replace_graph_outputs=True)
        else:
            continue
        att.outputs[0].shape = None  # now [B,S,D]; stale per-head annotation must not survive
        rewritten += 1

    if rewritten:
        graph.sort()
        common_passes.RemoveUnusedNodesPass()(model)
    return rewritten


def flip_causal_attention(model: ir.Model) -> int:
    """Replace a constant causal ``Attention`` mask with ``is_causal=1``, in place. Opset-23 ``is_causal``
    expresses the ``[1,1,S,S]`` lower-triangular additive mask exactly (q_len == kv_len == S here),
    dropping the mask input and its initializer. Verified numerically (exact 0 on/below diagonal, <= -1e4
    above — both underflow softmax to 0); anything else, incl. data-dependent padding masks, is left
    untouched. Bit-exact on ORT CPU. Returns nodes flipped."""
    graph = model.graph
    flipped = 0
    for node in graph:
        if node.op_type != "Attention" or len(node.inputs) != 4 or node.inputs[3] is None:
            continue
        if node.attributes.get_int("is_causal", 0) != 0:
            continue
        mask_const = node.inputs[3].const_value
        if mask_const is None:
            continue
        arr = mask_const.numpy()
        if not np.issubdtype(arr.dtype, np.floating) or arr.ndim < 2:
            continue
        seq = arr.shape[-1]
        if seq < 2 or arr.shape[-2] != seq or arr.size != seq * seq:
            continue  # not a broadcastable square [.., S, S] mask
        square = arr.reshape(seq, seq).astype(np.float64)
        on_or_below = np.tril(np.ones((seq, seq), np.bool_))
        if not (np.all(square[on_or_below] == 0.0) and np.all(square[~on_or_below] <= -1.0e4)):
            continue
        node.resize_inputs(3)  # drop the mask input; the initializer goes dead
        node.attributes["is_causal"] = ir.AttrInt64("is_causal", 1)
        flipped += 1
    return flipped


def prune_unused_initializers(model: ir.Model) -> int:
    """Drop initializers no node references (rewrite leftovers). ORT strips these at load with a warning;
    converters that don't (RKNN) ship the dead bytes."""
    graph = model.graph
    inputs = set(graph.inputs)
    dead = [
        name
        for name, value in graph.initializers.items()
        if value not in inputs and next(iter(value.uses()), None) is None
    ]
    for name in dead:
        del graph.initializers[name]
    return len(dead)


_CLASS_TOKEN_RULES = RewriteRuleSet([FuseClassTokenPrepend.rule()])
_CONST_GATHER_RULES = RewriteRuleSet([_FoldConstantGatherElements.rule()])
_SCALAR_GATHER_RULES = RewriteRuleSet([_ScalarGatherToSlice.rule()])
_CLEANUP_RULES = RewriteRuleSet([_BroadcastMaskRebuild.rule(), _FloatMaskCount.rule(), _IdentityAveragePool.rule()])
# applied after _CLEANUP_RULES: matches the And form _BroadcastMaskRebuild just emitted
_ADDITIVE_MASK_RULES = RewriteRuleSet([_AdditivePadMask.rule()])
_LN_COMMUTE_RULES = RewriteRuleSet([_SelectBeforeLayerNorm.rule(), _EotSelectBeforeLayerNorm.rule()])
_ATTNPOOL_QUERY_RULES = RewriteRuleSet([_FoldConstantAttnQuery.rule()])


def devitalize_shape_domain(model: ir.Model, *, rewrite_eot: bool = False) -> tuple[ir.Model, dict[str, int]]:
    """Run the full shape-domain-elimination pipeline on a dynamo-exported encoder. `fuse_visual_input`
    + `canonicalize_constants` must already have run — the probe needs the final input contract. Operates
    on the lazy `ir.Model` (weights stay external); returns the re-inferred model + per-stage rewrite counts."""
    probes = probe_runtime(model)
    counts: dict[str, int] = {}
    if rewrite_eot:
        counts["eot"] = rewrite_eot_gathernd(model, probes)
    # runs before eliminate_dynamic_expands so the class-token Expand is consumed here
    counts["class_token"] = _CLASS_TOKEN_RULES.apply_to_model(model)
    counts["position_ids"] = constantify_position_ids(model)
    counts["expands"] = eliminate_dynamic_expands(model, probes)
    # both GatherElements operands are now constant (constant position-id indices over the
    # batch-materialized token-type row): fold the lookup, dropping its materialization chain into DCE below
    counts["gather_elements"] = _CONST_GATHER_RULES.apply_to_model(model)
    counts["reshapes"] = constantify_reshape_targets(model, probes)
    # mask-rebuild collapse, float token count, no-op AvgPool -> Identity (folded right after)
    counts["cleanup"] = _CLEANUP_RULES.apply_to_model(model)
    # the bool key mask _BroadcastMaskRebuild rebuilt -> float additive bias (kills the bool island)
    counts["additive_masks"] = _ADDITIVE_MASK_RULES.apply_to_model(model)
    common_passes.IdentityEliminationPass()(model)
    counts["attnpool_queries"] = prune_attnpool_dead_queries(model)
    # runs after eliminate_dynamic_expands/constantify_reshape_targets: materialized-Expand + const Reshape target
    counts["attnpool_const_query"] = _ATTNPOOL_QUERY_RULES.apply_to_model(model)

    before = len(model.graph)  # DCE: drop the now-dead Shape chains
    common_passes.RemoveUnusedNodesPass()(model)
    counts["dead"] = before - len(model.graph)

    # move single-token selects ahead of their LayerNormalization before the Gathers are lowered
    counts["ln_commute"] = _LN_COMMUTE_RULES.apply_to_model(model)
    counts["gathers"] = _SCALAR_GATHER_RULES.apply_to_model(model)
    # size_limit=0: lift EVERY rule-emitted Constant (incl. 1-element Slice params + Pad amounts); the
    # default size_limit=16 leaves those as Constant nodes the CoreML EP rejects (see canonicalize_constants)
    common_passes.LiftConstantsToInitializersPass(size_limit=0)(model)
    common_passes.DeduplicateInitializersPass()(model)
    # after the lift: the MAP-head query constant from _FoldConstantAttnQuery must be an initializer for
    # the attnpool motif to read it
    counts["attention_3d"] = restructure_attention_3d(model)
    # after the 3D restructure: constant causal masks -> is_causal; retired [1,1,S,S] masks fall to the prune below
    counts["causal"] = flip_causal_attention(model)
    counts["pruned_inits"] = prune_unused_initializers(model)

    # surgery leaves stale annotations on rewritten paths; clear all and re-derive. A partial merge would
    # either fail or ship stale value_info (which the CUDA planner trusts).
    clear_cached_annotations(model.graph)
    model = common_passes.ShapeInferencePass()(model).model
    return model, counts
