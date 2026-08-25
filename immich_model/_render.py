"""Canonical renderings CI can diff: the per-target rewrite plan, and a graph as text."""

import hashlib
import re
import tempfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import onnx_ir as ir

from .constants import dim_sets_of, max_dims
from .rknn._onnx import rknn_config
from .rknn.compile import _pin
from .runtime import (
    REGISTRY,
    REWRITE_SET_VERSION,
    RKNPU,
    RewriteContext,
    apply_rewrites,
    plan_rewrites,
)

# what onnx_ir's NameAuthority hands a node or value the exporter left unnamed. Both counters run off
# construction order, so any edit upstream of a site renumbers every site after it.
_RULE_TAG = "pkg.onnxscript.rewriter.rule_name"
_GENERATED_NODE = re.compile(r"node_\w+_\d+")
_GENERATED_VALUE = re.compile(r"val_\d+")


def plans() -> str:
    """Target -> the rewrites runtime.REGISTRY plans for it, in the order they run."""
    lines = [f"REWRITE_SET_VERSION {REWRITE_SET_VERSION}"]
    for target in _targets():
        # ort_version reaches the digest, never a gate, so any value renders the same plan
        lines += ["", target, *(f"    {name}" for name in plan_rewrites(RewriteContext(target, ())).names)]
    return "\n".join(lines) + "\n"


def _targets() -> list[str]:
    return sorted({target for rewrite in REGISTRY for target in rewrite.gates})


def graph(path: Path) -> str:
    model = ir.load(path)
    # off the graph, so their decls leave the value_info block the weight list below replaces
    weights = [model.graph.initializers.pop(name) for name in list(model.graph.initializers)]
    _renumber(model, weights)
    _normalize(model)
    return _unfold(ir.to_onnx_text(model, exclude_initializers=True)) + "\nweights:\n" + "".join(_weights(weights))


def _normalize(model: ir.Model) -> None:
    """Drops what describes the exporting toolchain rather than the graph: the producer version uv.lock
    already pins, and an attribute order that varies by platform."""
    model.producer_version = None
    for node in ir.traversal.RecursiveGraphIterator(model.graph):
        ordered = sorted(node.attributes.items())
        node.attributes.clear()
        node.attributes.update(ordered)


def _renumber(model: ir.Model, weights: list[ir.Value]) -> None:
    values = [*model.graph.inputs, *weights]
    for node in ir.traversal.RecursiveGraphIterator(model.graph):
        if node.name is not None and _GENERATED_NODE.fullmatch(node.name):
            node.name = None  # the printer omits the `[name]` prefix entirely
        values += node.outputs
    # ⚠ NOT clearable: serialize_node_into appends the name to node_proto.input and rejects None. Every
    # holder of a generated name is renumbered here, weights included -- torch names 44 of ViT-B-32
    # visual's initializers `val_<N>`, so renumbering the node outputs alone aliases them.
    generated = (value for value in values if value.name is not None and _GENERATED_VALUE.fullmatch(value.name))
    for position, value in enumerate(generated):
        value.name = f"val_{position}"


def _bytes(tensor: ir.TensorProtocol) -> bytes:
    """Released once read: onnx_ir keeps every external weight mmapped, and one re-read per target
    exhausts the descriptor limit."""
    raw = tensor.tobytes()
    if isinstance(tensor, ir.ExternalTensor):
        tensor.release()
    return raw


def _weights(weights: list[ir.Value]) -> list[str]:
    """Sorted, so a weight moving in the graph does not diff and a weight changing does."""
    return sorted(
        f"{value.name} {tensor.dtype.name}{tensor.shape} {hashlib.sha256(_bytes(tensor)).hexdigest()[:12]}\n"
        for value in weights
        if (tensor := value.const_value) is not None
    )


def _unfold(text: str) -> str:
    """The printer puts every value_info on one line, and a 13 KB line is not a reviewable diff. Sorted
    for the same reason the weights are: the body already carries the order."""
    head, block, body = re.split(r"\n +<(.*)>\n(?=\{$)", text, maxsplit=1, flags=re.MULTILINE)
    declared = sorted(block.split(", "), key=lambda declaration: declaration.rsplit(" ", 1)[-1])
    return "\n".join([head, "   <", *(f"      {d}" for d in declared), "   >", body, ""])


def rewrites(path: Path) -> str:
    """Per target: what runtime.apply_rewrites did to this graph, as aggregates.

    Op counts and the weight digest are exact and mechanism-free; the rule tally comes from the stamps
    onnxscript leaves, which count REPLACEMENT NODES rather than applications and comma-join a node
    several rules touched. A rule that fail-closes on a family's spelling shows as its op delta simply
    not appearing."""
    base = _census(path, path.parent)[0]
    blocks = []
    with tempfile.TemporaryDirectory() as work:
        for target in _targets():
            plan = plan_rewrites(RewriteContext(target, (1, 26, 0)))
            for label, source, out_dir in _sources(target, path, Path(work)):
                # as rknn.compile writes it: its plan materializes the weights, and a tower inlined back
                # into the proto no longer serializes
                standalone = target == RKNPU
                out = apply_rewrites(source, plan, out_dir=out_dir, standalone=standalone)
                if out == source:
                    blocks.append(f"{label}\n    unchanged\n")
                    continue
                ops, nodes, digest, stamps, wiring = _census(out, out.parent if standalone else source.parent)
                delta = {op: ops[op] - base.get(op, 0) for op in set(ops) | set(base)}
                # total order: ties on the count are broken by set iteration otherwise, which varies per process
                ordered = sorted(delta.items(), key=lambda item: (-abs(item[1]), item[0]))
                lines = [f"    {count:+d} {op}" for op, count in ordered if count]
                lines += [f"    {nodes} nodes, weights {digest}"]
                # a property of what the plan PRODUCED, so it renders beside the digest rather than the label
                if config := rknn_config(out):
                    lines += [f"    config {config}"]
                lines += [f"    {rule} x{count} {wiring[rule]}" for rule, count in sorted(stamps.items())]
                if unclaimed := wiring.get(""):
                    lines += [f"    unstamped x{nodes - sum(stamps.values())} {unclaimed}"]
                blocks.append(f"{label}\n" + "\n".join(lines) + "\n")
    return "\n".join(blocks)


def _sources(target: str, path: Path, work: Path) -> Iterator[tuple[str, Path, Path]]:
    """The graphs a target's plan is rendered against and where each is written: the export itself for an EP,
    and for RKNPU one pinned shape per binary."""
    if target != RKNPU:
        yield target, path, work
        return
    for index, group in enumerate(dim_sets_of(path)):
        dims = max_dims(group.dims)
        out_dir = work / f"rknn{index}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pinned = _pin(path, out_dir, dims) if dims else path
        shape = " ".join(f"{name}={size}" for name, size in sorted(dims.items())) or "static"
        yield f"{target} {shape}", pinned, out_dir


def _signature(node: ir.Node) -> str:
    """The op with the attributes it carries: counting op types alone cannot tell a rule that emitted the
    right op from one that emitted it with the wrong axis or perm."""
    rendered = []
    for name, attr in sorted(node.attributes.items()):
        if attr.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS):
            value = "<graph>"  # walked as its own nodes, so its body is already counted
        elif attr.type in (ir.AttributeType.TENSOR, ir.AttributeType.TENSORS):
            value = "<tensor>"  # the bytes reach the weight digest; the shape is on the value_info
        else:
            value = str(attr.value)
        rendered.append(f"{name}={value}")
    return " ".join([node.op_type, *rendered])


def _fingerprint(node: ir.Node) -> str:
    """A node's operands named by what produced them, which is the one thing a per-node summary cannot
    carry: swapped operands leave op, attributes, count and weights all untouched. Named by producer
    rather than by position so an edit elsewhere in the graph does not move it."""
    operands = []
    for value in node.inputs:
        if value is None:
            operands.append("_")
        elif (producer := value.producer()) is None:
            operands.append(value.name or "?")  # a graph input or an initializer, whose name is real
        else:
            operands.append(f"{producer.op_type}.{producer.outputs.index(value)}")
    return f"{_signature(node)}({','.join(operands)})"


def _wiring(nodes: list[ir.Node]) -> dict[str, str]:
    """One digest per rule, over the nodes that rule stamped, and one for what no rule claims. Split
    because a single graph-wide digest cannot distinguish the change you meant from one riding along with
    it -- both only say that something moved, and blessing then takes both."""
    buckets: dict[str, list[str]] = {}
    for node in nodes:
        buckets.setdefault(node.metadata_props.get(_RULE_TAG, ""), []).append(_fingerprint(node))
    # sorted: a bucket answers for its own nodes, never for where they sit among everyone else's
    return {tag: hashlib.sha256("\n".join(sorted(prints)).encode()).hexdigest()[:8] for tag, prints in buckets.items()}


def _census(path: Path, weights: Path) -> tuple[Counter, int, str, Counter, str]:
    model = ir.load(path)
    ir.external_data.set_base_dir(model.graph, weights)  # a rewritten graph points at the source sidecar
    nodes = list(ir.traversal.RecursiveGraphIterator(model.graph))
    stamps = Counter(tag for node in nodes if (tag := node.metadata_props.get(_RULE_TAG)))
    digest = hashlib.sha256()
    for name in sorted(model.graph.initializers):
        if (tensor := model.graph.initializers[name].const_value) is not None:
            digest.update(_bytes(tensor))
    return Counter(_signature(node) for node in nodes), len(nodes), digest.hexdigest()[:12], stamps, _wiring(nodes)
