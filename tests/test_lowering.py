import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

from immich_model.onnx import lowering
from immich_model.runtime import RewriteContext, plan_rewrites

SHAPE = [1, 8, 6, 5]


def mean_model(axes: list[int], keepdims: int = 1, shape: list[int] | None = SHAPE, behind: str | None = None) -> bytes:
    """x -> [behind ->] ReduceMean(axes, keepdims) -> y, at opset 18 where the axes are an input, as exported."""
    nodes = []
    source = "x"
    if behind is not None:  # an op in front leaves the mean's operand with no shape unless inference gives it one
        nodes.append(helper.make_node(behind, ["x"], ["h"]))
        source = "h"
    nodes.append(helper.make_node("ReduceMean", [source, "axes"], ["y"], keepdims=keepdims))
    graph = helper.make_graph(
        nodes,
        "mean",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [helper.make_tensor("axes", TensorProto.INT64, [len(axes)], axes)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]).SerializeToString()


def rewritten(proto: bytes) -> tuple[bool, onnx.ModelProto]:
    model = ir.from_proto(onnx.load_from_string(proto))
    result = lowering.SpatialMeanAsGlobalPoolPass()(model)
    return result.modified, ir.to_proto(result.model)


def ops(model: onnx.ModelProto) -> list[str]:
    return [node.op_type for node in model.graph.node]


def run(model: onnx.ModelProto | bytes, x: np.ndarray) -> np.ndarray:
    (y,) = ReferenceEvaluator(model).run(None, {"x": x})
    return np.asarray(y)


@pytest.mark.parametrize("axes", [[2, 3], [3, 2], [-2, -1], [2, -1]])
def test_spatial_mean_is_a_global_pool_of_the_same_value(axes: list[int]) -> None:
    proto = mean_model(axes)
    x = np.random.default_rng(0).standard_normal(SHAPE).astype(np.float32)

    modified, model = rewritten(proto)

    assert modified and ops(model) == ["GlobalAveragePool"]
    assert model.graph.output[0].name == "y"
    np.testing.assert_allclose(run(model, x), run(proto, x), rtol=1e-6, atol=1e-7)
    assert run(model, x).shape == (1, 8, 1, 1)


@pytest.mark.parametrize("shape", [[1, 4, 7], [2, 4, 3, 5, 6]])
def test_any_rank_past_the_channels_pools_the_same(shape: list[int]) -> None:
    proto = mean_model(list(range(2, len(shape))), shape=shape)
    x = np.random.default_rng(1).standard_normal(shape).astype(np.float32)

    modified, model = rewritten(proto)

    assert modified and ops(model) == ["GlobalAveragePool"]
    np.testing.assert_allclose(run(model, x), run(proto, x), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("axes", "keepdims", "shape", "behind"),
    [
        ([2, 3], 0, SHAPE, None),  # drops the reduced axes, which the pool keeps
        ([3], 1, SHAPE, None),  # one spatial axis of two
        ([1, 2, 3], 1, SHAPE, None),  # the channels too
        ([0, 2, 3], 1, SHAPE, None),  # the batch too
        ([1], 1, SHAPE, None),  # the channels alone
        ([2, 3], 1, None, None),  # a rank no one knows
        ([2, 3], 1, SHAPE, "Relu"),  # an operand of no recorded shape
    ],
)
def test_other_means_are_left_alone(
    axes: list[int], keepdims: int, shape: list[int] | None, behind: str | None
) -> None:
    modified, model = rewritten(mean_model(axes, keepdims, shape, behind))

    assert not modified and ops(model)[-1] == "ReduceMean"


def test_axes_not_known_until_run_are_left_alone() -> None:
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["x", "axes"], ["y"], keepdims=1)],
        "mean",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, SHAPE),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [2]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]).SerializeToString()

    modified, model = rewritten(proto)

    assert not modified and ops(model) == ["ReduceMean"]


def test_planned_for_cuda_alone() -> None:
    planned = {
        target: "spatial_mean_as_global_pool" in plan_rewrites(RewriteContext(target, (1, 26, 0))).names
        for target in ("CUDAExecutionProvider", "CPUExecutionProvider", "NvTensorRTRTXExecutionProvider")
    }

    assert planned == {
        "CUDAExecutionProvider": True,
        "CPUExecutionProvider": False,
        "NvTensorRTRTXExecutionProvider": False,
    }


def test_cuda_fp16_mean_over_a_large_map_stays_finite() -> None:
    """The bug the rewrite is for: the CUDA EP's fp16 ReduceMean sums in fp16, so a mean over more than ~40k pixels
    of values around 1.5 passes 65504 and comes back inf, where cuDNN's pooling sums in fp32."""
    ort = pytest.importorskip("onnxruntime")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("no CUDA execution provider")
    shape = [1, 16, 480, 360]
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["x", "axes"], ["y"], keepdims=1)],
        "mean",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, None)],
        [helper.make_tensor("axes", TensorProto.INT64, [2], [2, 3])],
    )
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]).SerializeToString()
    x = np.full(shape, 1.5, dtype=np.float16)
    modified, model = rewritten(proto)
    assert modified

    def on_cuda(model: onnx.ModelProto | bytes) -> np.ndarray:
        blob = model if isinstance(model, bytes) else model.SerializeToString()
        session = ort.InferenceSession(blob, providers=["CUDAExecutionProvider"])
        if session.get_providers()[0] != "CUDAExecutionProvider":
            pytest.skip("CUDA execution provider failed to load")
        return session.run(None, {"x": x})[0]

    pooled = on_cuda(model)

    # the mean as exported is not asserted on: it is the EP's to fix, and a release that does retires the gate
    assert np.isfinite(pooled).all() and np.allclose(pooled.astype(np.float32), 1.5, atol=1e-3)
