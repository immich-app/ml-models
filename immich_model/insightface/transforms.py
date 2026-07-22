"""Graph transforms turning stock InsightFace ONNX models into Immich's fused format.

Also consumed by immich_ml, which applies them to legacy cached models at load time. Inputs
become uint8 RGB NHWC; preprocessing, the SCRFD anchor decode, and L2-normalization fold into
the graph; face alignment stays host-side (see _dsl.py).
"""

from typing import Any

import numpy as np
import onnx
import onnx.numpy_helper as nh
import onnxscript.optimizer
from onnx import ModelProto, version_converter
from onnxscript import ir
from onnxscript.rewriter.pattern import RewriteRuleSet
from onnxscript.rewriter.rules.common import fuse_batchnorm_into_conv_rule, fuse_batchnorm_into_gemm_rule

from ..onnx.graph import fold_input_scale, reinfer, wrap
from . import _dsl

_BN_RULES = RewriteRuleSet([fuse_batchnorm_into_gemm_rule, fuse_batchnorm_into_conv_rule])


def transform_detection(model: ModelProto) -> ModelProto:
    graph = model.graph
    if len(graph.output) != 9:
        raise ValueError(f"Expected a 9-output SCRFD keypoint model, got {len(graph.output)} outputs")

    dims = graph.input[0].type.tensor_type.shape.dim
    dims[0].ClearField("dim_value")
    dims[0].dim_param = "batch"
    for i in (2, 3):
        dims[i].ClearField("dim_param")
        dims[i].dim_value = _dsl.DET_SIZE

    fold_input_scale(model, scale=1.0 / 128.0)

    _fix_fpn_resizes(model)
    _merge_heads(model)

    model = _finalize(model)

    # map the backbone's outputs onto the decode inputs by (channels, anchor count)
    outputs_by_key = {}
    for output in model.graph.output:
        out_dims = output.type.tensor_type.shape.dim
        outputs_by_key[(out_dims[2].dim_value, out_dims[1].dim_value)] = output.name
    io_map = []
    for stride in _dsl.DET_STRIDES:
        anchors = 2 * (_dsl.DET_SIZE // stride) ** 2
        io_map += [
            (outputs_by_key[(1, anchors)], f"scores{stride}"),
            (outputs_by_key[(4, anchors)], f"boxes{stride}"),
            (outputs_by_key[(10, anchors)], f"kps{stride}"),
        ]
    return wrap(model, _dsl.det_preprocess, _dsl.det_postprocess, io_map)


def transform_recognition(model: ModelProto) -> ModelProto:
    dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    dim.ClearField("dim_value")
    dim.dim_param = "batch"

    # ArcFace tail Gemm -> features-BN folds into the Gemm (exact affine); IResNet residual
    # BNs follow Adds not convs, so they stay untouched (statistics are fp16-safe)
    ir_model = ir.from_proto(model)
    _BN_RULES.apply_to_model(ir_model)
    model = ir.to_proto(ir_model)

    fold_input_scale(model, scale=1.0 / 127.5)

    model = _finalize(model)

    io_map = [(model.graph.output[0].name, "embedding_raw")]
    return wrap(model, _dsl.rec_preprocess, _dsl.l2_normalize, io_map)


def _finalize(model: ModelProto) -> ModelProto:
    model = onnxscript.optimizer.optimize(reinfer(model))
    return reinfer(version_converter.convert_version(model, _dsl.OPSET))


def _merge_heads(model: ModelProto) -> None:
    """Merge each SCRFD stride's three head convs (cls/reg/kps, 2/8/20ch) into one 30ch Conv +
    batched Transpose/Reshape/Split/Reshape, in place; also subsumes the legacy-layout rewrite.

    reg-scale Mul folds into the merged weights (fp64 then cast, matching ORT ConvMulFusion):
    numeric_equiv, decoded boxes ~1e-4 off in 640px space at cos 1.0. cls/kps stay bit-exact bar
    conv ULP. Ordering holds because channels are anchor-major and Reshape's 0 keeps the batch dim.
    """
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    producers = {out: node for node in graph.node for out in node.output}

    # walk each head output back to its Conv, recording channels/sigmoid/reg-scale
    branches: dict[str, list[dict[str, Any]]] = {}
    stale: list[str] = []
    for output in graph.output:
        cursor, channels, has_sigmoid, scale = output.name, None, False, None
        while True:
            node = producers[cursor]
            stale.append(node.name)
            if node.op_type == "Conv":
                branch = {"conv": node, "output": output.name, "channels": channels, "scale": scale}
                branches.setdefault(node.input[0], []).append(branch)
                break
            if node.op_type == "Sigmoid":
                has_sigmoid, cursor = True, node.input[0]
            elif node.op_type == "Reshape":
                channels, cursor = int(nh.to_array(inits[node.input[1]])[-1]), node.input[0]
            elif node.op_type == "Transpose":
                cursor = node.input[0]
            elif node.op_type == "Mul":
                scale_name = next(i for i in node.input if i in inits)
                scale, cursor = float(nh.to_array(inits[scale_name])), next(i for i in node.input if i != scale_name)
            else:
                raise ValueError(f"Unexpected {node.op_type} in SCRFD head branch")
        if channels == 1 and not has_sigmoid:  # the walk only exits at a Conv
            raise ValueError("SCRFD cls branch is missing its Sigmoid")

    merged: list[Any] = []
    for stride, (feature, group) in enumerate(branches.items()):
        by_channels = {branch["channels"]: branch for branch in group}
        if sorted(by_channels) != [1, 4, 10]:
            raise ValueError(f"Expected cls/reg/kps (1/4/10 channels) per stride, got {sorted(by_channels)}")
        cls, reg, kps = by_channels[1], by_channels[4], by_channels[10]

        weights, biases = [], []
        for branch in (cls, reg, kps):
            conv = branch["conv"]
            weight = nh.to_array(inits[conv.input[1]])
            bias = nh.to_array(inits[conv.input[2]]) if len(conv.input) > 2 else np.zeros(weight.shape[0], weight.dtype)
            if branch["scale"] is not None:  # pre-fold the reg scale, fp64 then cast back
                weight = (weight.astype(np.float64) * branch["scale"]).astype(weight.dtype)
                bias = (bias.astype(np.float64) * branch["scale"]).astype(bias.dtype)
            weights.append(weight)
            biases.append(bias)
        splits = [weight.shape[0] for weight in weights]  # [2, 8, 20]

        name = f"scrfd_head{stride}"
        conv_out, cols, cls_pre = f"{name}_conv", f"{name}_cols", f"{name}_cls_logits"
        parts = [f"{name}_cls", f"{name}_reg", f"{name}_kps"]
        graph.initializer.extend(
            [
                nh.from_array(np.concatenate(weights, axis=0), f"{name}_weight"),
                nh.from_array(np.concatenate(biases, axis=0), f"{name}_bias"),
                nh.from_array(np.array([0, -1, sum(splits)], np.int64), f"{name}_flat_shape"),
                nh.from_array(np.array([0, -1, cls["channels"]], np.int64), f"{name}_cls_shape"),
                nh.from_array(np.array([0, -1, reg["channels"]], np.int64), f"{name}_reg_shape"),
                nh.from_array(np.array([0, -1, kps["channels"]], np.int64), f"{name}_kps_shape"),
            ]
        )
        node = onnx.helper.make_node
        conv = node("Conv", [feature, f"{name}_weight", f"{name}_bias"], [conv_out], name=f"{name}_conv")
        conv.attribute.extend(cls["conv"].attribute)  # kernel 3x3 / pad 1 / stride 1, shared across the triple
        merged += [
            conv,
            node("Transpose", [conv_out], [f"{name}_nhwc"], name=f"{name}_transpose", perm=[0, 2, 3, 1]),
            node("Reshape", [f"{name}_nhwc", f"{name}_flat_shape"], [cols], name=f"{name}_reshape"),
            node("Split", [cols], parts, name=f"{name}_split", axis=2, split=splits),
            node("Reshape", [parts[0], f"{name}_cls_shape"], [cls_pre], name=f"{name}_cls_reshape"),
            node("Sigmoid", [cls_pre], [cls["output"]], name=f"{name}_sigmoid"),
            node("Reshape", [parts[1], f"{name}_reg_shape"], [reg["output"]], name=f"{name}_reg_reshape"),
            node("Reshape", [parts[2], f"{name}_kps_shape"], [kps["output"]], name=f"{name}_kps_reshape"),
        ]

    stale_set = set(stale)
    kept = [node for node in graph.node if node.name not in stale_set]
    del graph.node[:]
    graph.node.extend(kept + merged)  # merged nodes only read the (already-produced) feature maps


def _fix_fpn_resizes(model: ModelProto) -> None:
    """Replace the FPN upsamples' runtime-computed scales with a constant: the Shape/Slice/Concat
    chains otherwise constant-fold with a stale batch=1 baked in (dead chains get optimized away)."""
    model.graph.initializer.append(nh.from_array(np.array([1, 1, 2, 2], np.float32), "resize_scales_2x"))
    for node in model.graph.node:
        if node.op_type == "Resize":
            mode = next(a.s.decode() for a in node.attribute if a.name == "mode")
            if mode != "nearest":
                raise ValueError(f"Expected nearest Resize in FPN, got {mode}")
            node.input[2] = "resize_scales_2x"
            del node.input[3:]


__all__ = ["transform_detection", "transform_recognition"]
