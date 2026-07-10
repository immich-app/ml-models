"""InsightFace pack exporter.

Downloads an official InsightFace model pack (buffalo_*/antelopev2) and re-exports the
detection and recognition models for Immich:

- dynamic batch axis, fixed spatial dims (detection 640x640, recognition 112x112 post-warp)
- uint8 BGR NHWC image inputs; cast/channel-swap/normalization folded into the graph
- detection: SCRFD anchor decode folded into the graph. Outputs (in order):
  scores [batch, 16800], boxes [batch, 16800, 4], kps [batch, 16800, 10], all in
  640x640 pixel space, post-sigmoid scores. Thresholding/NMS stay host-side (data-dependent
  shapes break NPU toolchains).
- recognition: takes a loose square face crop [batch, 256, 256, 3] plus the detector's five
  landmarks [batch, 5, 2] in crop coordinates. The similarity alignment (closed-form Umeyama
  least-squares + GridSample warp) runs in-graph; embeddings are L2-normalized.

The pre/postprocessing subgraphs are written as onnxscript functions and composed around the
patched backbone with onnx.compose.

Both models target opset 19 deliberately; higher opsets break real backends:
- ORT CUDA EP kernel registrations for MaxPool stop at opset 21 (the spec changed at 22),
  which silently pushes the detection stem maxpool to CPU with a device round-trip.
- TensorRT's ONNX parser does not recognize GridSample's opset-20 mode name "linear" and
  silently samples nearest-neighbor; opset 19 binds GridSample-16 with mode "bilinear".
- rknn-toolkit2 rejects models above opset 19 outright.
Opset 19 has full ORT CUDA EP kernel coverage for every op in these graphs (verified by
profiling for CPU-fallback nodes).
"""

import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnx.numpy_helper as nh
import onnxscript.optimizer
from onnx import ModelProto, compose, version_converter
from onnxscript import FLOAT, UINT8, script
from onnxscript import opset19 as op
from onnxscript.values import OnnxFunction

PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/{}.zip"

OPSET = 19

DET_SIZE = 640
DET_STRIDES = (8, 16, 32)

REC_CROP = 256  # loose-crop input size fed to the recognition model
REC_SIZE = 112  # aligned size expected by the backbone

# canonical ArcFace 5-point template for a 112x112 crop
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)

BGR2RGB = nh.from_array(np.array([2, 1, 0], np.int64), "bgr2rgb")

# detection anchor constants, concatenated across strides so the whole decode is one
# Mul + Add per output ([16800, C] tensors for 640x640 SCRFD with 2 anchors per cell)
_centers, _strides = [], []
for _s in DET_STRIDES:
    _grid = np.stack(np.mgrid[: DET_SIZE // _s, : DET_SIZE // _s][::-1], axis=-1).astype(np.float32) * _s
    _centers.append(np.stack([_grid.reshape(-1, 2)] * 2, axis=1).reshape(-1, 2))
    _strides.append(np.full((len(_centers[-1]), 1), _s, np.float32))
_centers_cat, _strides_cat = np.concatenate(_centers), np.concatenate(_strides)
BOX_MUL = nh.from_array(_strides_cat * np.array([-1, -1, 1, 1], np.float32), "box_mul")
BOX_ADD = nh.from_array(np.tile(_centers_cat, 2), "box_add")
KPS_MUL = nh.from_array(_strides_cat, "kps_mul")
KPS_ADD = nh.from_array(np.tile(_centers_cat, 5), "kps_add")

# alignment constants, in the centered unit-scaled frames that keep the least-squares solve
# within fp16 range (TensorRT converts the whole engine to fp16)
_template = (ARCFACE_DST - REC_SIZE / 2) / REC_SIZE
TMPL_XY = nh.from_array(_template.reshape(-1, 1), "tmpl_xy")
TMPL_YX = nh.from_array(np.stack([_template[:, 1], -_template[:, 0]], axis=-1).reshape(-1, 1), "tmpl_yx")
TMPL_SUM_X = float(_template[:, 0].sum())
TMPL_SUM_Y = float(_template[:, 1].sum())

_ys, _xs = np.mgrid[:REC_SIZE, :REC_SIZE].astype(np.float32)
TMPL_GRID = nh.from_array(
    np.stack(
        [
            (_xs.ravel() - REC_SIZE / 2) / REC_SIZE,
            (_ys.ravel() - REC_SIZE / 2) / REC_SIZE,
            np.ones(REC_SIZE * REC_SIZE, np.float32),
        ],
        axis=-1,
    )[None],
    "tmpl_grid",
)


@script(default_opset=op)
def det_preprocess(image: UINT8["batch", 640, 640, 3]) -> FLOAT["batch", 3, 640, 640]:  # noqa: F821
    rgb = op.Gather(op.Cast(image, to=1), op.Constant(value=BGR2RGB), axis=3)
    blob = op.Transpose((rgb - 127.5) * (1.0 / 128.0), perm=[0, 3, 1, 2])
    return blob


@script(default_opset=op)
def det_postprocess(
    scores8: FLOAT["batch", 12800, 1],  # noqa: F821
    scores16: FLOAT["batch", 3200, 1],  # noqa: F821
    scores32: FLOAT["batch", 800, 1],  # noqa: F821
    boxes8: FLOAT["batch", 12800, 4],  # noqa: F821
    boxes16: FLOAT["batch", 3200, 4],  # noqa: F821
    boxes32: FLOAT["batch", 800, 4],  # noqa: F821
    kps8: FLOAT["batch", 12800, 10],  # noqa: F821
    kps16: FLOAT["batch", 3200, 10],  # noqa: F821
    kps32: FLOAT["batch", 800, 10],  # noqa: F821
) -> tuple[FLOAT["batch", 16800], FLOAT["batch", 16800, 4], FLOAT["batch", 16800, 10]]:  # noqa: F821
    """SCRFD anchor decode (distance2bbox/distance2kps against constant anchor centers)."""
    scores = op.Reshape(op.Concat(scores8, scores16, scores32, axis=1), op.Constant(value_ints=[0, -1]))
    boxes = op.Concat(boxes8, boxes16, boxes32, axis=1) * op.Constant(value=BOX_MUL) + op.Constant(value=BOX_ADD)
    kps = op.Concat(kps8, kps16, kps32, axis=1) * op.Constant(value=KPS_MUL) + op.Constant(value=KPS_ADD)
    return scores, boxes, kps


@script(default_opset=op)
def rec_preprocess(
    image: UINT8["batch", 256, 256, 3],  # noqa: F821
    kps: FLOAT["batch", 5, 2],  # noqa: F821
) -> FLOAT["batch", 3, 112, 112]:  # noqa: F821
    """Estimate the kps -> ArcFace-template similarity transform (closed-form least
    squares) and warp the crop to the aligned 112x112 the backbone expects."""
    kps_n = (kps - 128.0) * (1.0 / 256.0)
    kps_flat = op.Reshape(kps_n, op.Constant(value_ints=[0, -1]))

    # scalar sums of the least-squares normal equations, all [batch, 1]
    sxu = op.MatMul(kps_flat, op.Constant(value=TMPL_XY))
    sxv = op.MatMul(kps_flat, op.Constant(value=TMPL_YX))
    sxy = op.ReduceSum(kps_n, op.Constant(value_ints=[1]), keepdims=0)
    sx, sy = sxy[:, 0:1], sxy[:, 1:2]
    sxx = op.ReduceSum(kps_flat * kps_flat, op.Constant(value_ints=[1]), keepdims=1)

    # similarity kps -> template: x' = a*x - b*y + tx, y' = b*x + a*y + ty
    d = sxx - (sx * sx + sy * sy) * 0.2
    a = (sxu - (sx * TMPL_SUM_X + sy * TMPL_SUM_Y) * 0.2) / d
    b = (sxv + (sy * TMPL_SUM_X - sx * TMPL_SUM_Y) * 0.2) / d
    tx = (TMPL_SUM_X - a * sx + b * sy) * 0.2
    ty = (TMPL_SUM_Y - b * sx - a * sy) * 0.2

    # inverse similarity (template -> crop), packed as a [batch, 2, 3] matrix
    inv_scale = 1.0 / (a * a + b * b)
    ia, ib = a * inv_scale, b * inv_scale
    itx = 0.0 - (ia * tx + ib * ty)
    ity = ib * tx - ia * ty
    matrix = op.Reshape(op.Concat(ia, ib, itx, op.Neg(ib), ia, ity, axis=1), op.Constant(value_ints=[0, 2, 3]))

    # push the constant template pixel grid through it; to [-1, 1] with align_corners=0
    src = op.MatMul(op.Constant(value=TMPL_GRID), op.Transpose(matrix, perm=[0, 2, 1]))
    grid = op.Reshape(src * 2.0 + (1.0 / 256.0), op.Constant(value_ints=[0, 112, 112, 2]))

    warped = op.GridSample(
        op.Transpose(op.Cast(image, to=1), perm=[0, 3, 1, 2]),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=0,
    )
    rgb = op.Gather(warped, op.Constant(value=BGR2RGB), axis=1)
    aligned = (rgb - 127.5) * (1.0 / 127.5)
    return aligned


@script(default_opset=op)
def l2_normalize(embedding_raw: FLOAT["batch", 512]) -> FLOAT["batch", 512]:  # noqa: F821
    norm = op.ReduceL2(embedding_raw, op.Constant(value_ints=[1]), keepdims=1)
    embedding = embedding_raw / op.Max(norm, op.Constant(value_float=1e-12))
    return embedding


def export(model_name: str, output_dir: Path, cache: bool = True) -> None:
    det_path = output_dir / "detection" / "model.onnx"
    rec_path = output_dir / "recognition" / "model.onnx"
    if cache and det_path.exists() and rec_path.exists():
        print(f"Models {det_path} and {rec_path} already exist, skipping")
        return

    pack_dir = _download_pack(model_name, output_dir.parent / ".insightface", cache=cache)
    det_src, rec_src = _find_models(pack_dir)

    print(f"Transforming detection model {det_src}")
    det = _transform_detection(onnx.load(det_src))
    det_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(det, det_path)
    _smoke_test_detection(det_path)

    print(f"Transforming recognition model {rec_src}")
    rec = _transform_recognition(onnx.load(rec_src))
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rec, rec_path)
    _smoke_test_recognition(rec_path)


def _download_pack(model_name: str, cache_dir: Path, cache: bool = True) -> Path:
    pack_dir = cache_dir / model_name
    if cache and pack_dir.is_dir() and any(pack_dir.glob("**/*.onnx")):
        return pack_dir

    zip_path = cache_dir / f"{model_name}.zip"
    if not (cache and zip_path.exists()):
        url = PACK_URL.format(model_name)
        print(f"Downloading {url}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)

    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    with zipfile.ZipFile(zip_path) as f:
        f.extractall(pack_dir)
    return pack_dir


def _find_models(pack_dir: Path) -> tuple[Path, Path]:
    """Identify the detection and recognition models by graph signature.

    Detection is the model with >=5 outputs (9 for the keypoint variants Immich needs);
    recognition has a single output and a square 112x112 input. Other pack members
    (genderage, landmark regressors) don't match either signature.
    """
    det_path = rec_path = None
    for path in sorted(pack_dir.glob("**/*.onnx")):
        model = onnx.load(path, load_external_data=False)
        inputs = model.graph.input
        dims = [d.dim_value for d in inputs[0].type.tensor_type.shape.dim] if len(inputs) == 1 else []
        if len(model.graph.output) >= 5:
            det_path = path
        elif len(model.graph.output) == 1 and len(dims) == 4 and dims[2] == dims[3] == REC_SIZE:
            rec_path = path
    if det_path is None or rec_path is None:
        raise ValueError(f"Could not identify detection/recognition models in {pack_dir}")
    return det_path, rec_path


def _reinfer(model: ModelProto) -> ModelProto:
    """Drop all cached shape annotations and re-infer from the inputs.

    The stock exports carry stale batch=1 dims in value_info; the CPU EP merges these
    leniently but the CUDA EP's buffer planner trusts them and fails at batch > 1.
    """
    del model.graph.value_info[:]
    for output in model.graph.output:
        output.type.tensor_type.ClearField("shape")
    return onnx.shape_inference.infer_shapes(model)


def _wrap(
    backbone: ModelProto, pre: OnnxFunction, post: OnnxFunction, post_io_map: list[tuple[str, str]]
) -> ModelProto:
    """Compose pre -> backbone -> post, then clean up.

    The onnxscript models get prefixed internals so their auto-generated names can't
    collide, but keep their (contract) input/output names.
    """
    pre_model, post_model = pre.to_model_proto(), post.to_model_proto()
    pre_model.ir_version = post_model.ir_version = backbone.ir_version
    pre_model = compose.add_prefix(pre_model, "pre_", rename_inputs=False, rename_outputs=False)
    post_model = compose.add_prefix(post_model, "post_", rename_inputs=False, rename_outputs=False)

    pre_io_map = [(o.name, i.name) for o, i in zip(pre_model.graph.output, backbone.graph.input)]
    model = compose.merge_models(pre_model, backbone, io_map=pre_io_map)
    model = compose.merge_models(model, post_model, io_map=post_io_map)

    model = onnxscript.optimizer.optimize(model)
    model = _reinfer(model)
    onnx.checker.check_model(model)
    return model


def _transform_detection(model: ModelProto) -> ModelProto:
    graph = model.graph
    if len(graph.output) != 9:
        raise ValueError(f"Expected a 9-output SCRFD keypoint model, got {len(graph.output)} outputs")

    # dynamic batch, fixed 640x640
    dims = graph.input[0].type.tensor_type.shape.dim
    dims[0].ClearField("dim_value")
    dims[0].dim_param = "batch"
    for i in (2, 3):
        dims[i].ClearField("dim_param")
        dims[i].dim_value = DET_SIZE

    # The 2020-era head flattens batch into the anchor dim: Transpose(2,3,0,1) [H,W,N,C]
    # followed by Reshape([-1, C]). Patch to the modern batched layout (N first, batch
    # dim preserved with Reshape's 0-means-copy). Both patches are load-bearing: at batch 1
    # the two layouts flatten identically, so any mistake only breaks batch > 1.
    initializers = {i.name: i for i in graph.initializer}
    graph.initializer.append(nh.from_array(np.array([1, 1, 2, 2], np.float32), "resize_scales_2x"))
    patched_transposes = patched_reshapes = 0
    for node in graph.node:
        if node.op_type == "Transpose":
            perm = next(a for a in node.attribute if a.name == "perm")
            if list(perm.ints) == [2, 3, 0, 1]:
                perm.ints[:] = [0, 2, 3, 1]
                patched_transposes += 1
        elif node.op_type == "Reshape" and node.input[1] in initializers:
            shape = nh.to_array(initializers[node.input[1]])
            if shape.ndim == 1 and len(shape) == 2 and shape[0] == -1:
                new_shape = np.array([0, -1, int(shape[1])], np.int64)
                initializers[node.input[1]].CopyFrom(nh.from_array(new_shape, node.input[1]))
                patched_reshapes += 1
        elif node.op_type == "Resize":
            # FPN top-down 2x nearest upsample: replace the runtime-computed scales
            # (a Shape/Slice/Concat chain that can otherwise fold with batch=1 baked in)
            # with a constant; DCE removes the dead chain.
            mode = next(a.s.decode() for a in node.attribute if a.name == "mode")
            if mode != "nearest":
                raise ValueError(f"Expected nearest Resize in FPN, got {mode}")
            node.input[2] = "resize_scales_2x"
            del node.input[3:]
    if patched_transposes != 9 or patched_reshapes < 3:
        raise ValueError(f"Unexpected head layout: {patched_transposes} transposes, {patched_reshapes} reshapes")

    model = _reinfer(model)
    model = onnxscript.optimizer.optimize(model)
    model = _reinfer(version_converter.convert_version(model, OPSET))

    # map the backbone's outputs onto the decode inputs by (channels, anchor count)
    outputs_by_key = {}
    for output in model.graph.output:
        dims = output.type.tensor_type.shape.dim
        outputs_by_key[(dims[2].dim_value, dims[1].dim_value)] = output.name
    io_map = []
    for stride in DET_STRIDES:
        anchors = 2 * (DET_SIZE // stride) ** 2
        io_map += [
            (outputs_by_key[(1, anchors)], f"scores{stride}"),
            (outputs_by_key[(4, anchors)], f"boxes{stride}"),
            (outputs_by_key[(10, anchors)], f"kps{stride}"),
        ]
    return _wrap(model, det_preprocess, det_postprocess, io_map)


def _transform_recognition(model: ModelProto) -> ModelProto:
    dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    dim.ClearField("dim_value")
    dim.dim_param = "batch"

    model = _reinfer(model)
    model = onnxscript.optimizer.optimize(model)
    model = _reinfer(version_converter.convert_version(model, OPSET))

    io_map = [(model.graph.output[0].name, "embedding_raw")]
    return _wrap(model, rec_preprocess, l2_normalize, io_map)


def _smoke_test_detection(model_path: Path) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    image = np.random.randint(0, 255, (2, DET_SIZE, DET_SIZE, 3), np.uint8)
    scores, boxes, kps = session.run(None, {"image": image})
    anchors = scores.shape[1]
    if scores.shape != (2, anchors) or boxes.shape != (2, anchors, 4) or kps.shape != (2, anchors, 10):
        raise ValueError(f"Unexpected detection output shapes: {scores.shape}, {boxes.shape}, {kps.shape}")
    if not (0 <= scores.min() and scores.max() <= 1):
        raise ValueError("Detection scores are not post-sigmoid")


def _smoke_test_recognition(model_path: Path) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    image = np.random.randint(0, 255, (2, REC_CROP, REC_CROP, 3), np.uint8)
    kps = (ARCFACE_DST * (REC_CROP / REC_SIZE))[None].repeat(2, 0).astype(np.float32)
    (embedding,) = session.run(None, {"image": image, "kps": kps})
    if embedding.shape != (2, 512):
        raise ValueError(f"Unexpected embedding shape: {embedding.shape}")
    norms = np.linalg.norm(embedding, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(f"Embeddings are not L2-normalized: {norms}")
