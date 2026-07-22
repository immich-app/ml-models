import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# per-model subdirectories that hold a model.onnx
SUBMODELS = ["textual", "visual", "detection", "recognition"]


def profile(model_dir: Path, runs: int = 20, provider: str = "CPUExecutionProvider") -> dict[str, Any]:
    """Profile every ONNX submodel via onnxruntime, returning per-op-node costs.

    Uses onnxruntime's built-in profiler (enable_profiling), which records a
    kernel-time trace per node; costs are aggregated by op type.
    """
    import numpy as np
    import onnxruntime as ort

    subs = [s for s in SUBMODELS if (model_dir / s / "model.onnx").is_file()]
    if not subs:
        raise RuntimeError(f"No ONNX model found under {model_dir}")

    result: dict[str, Any] = {"model": model_dir.name, "format": "onnx", "provider": provider, "submodels": {}}
    for sub in subs:
        so = ort.SessionOptions()
        so.enable_profiling = True
        sess = ort.InferenceSession((model_dir / sub / "model.onnx").as_posix(), sess_options=so, providers=[provider])

        feeds = {i.name: _rand_input(i, np) for i in sess.get_inputs()}
        for _ in range(runs):  # profiling records every run; the first (cold) run is averaged in
            sess.run(None, feeds)
        events = json.load(open(sess.end_profiling()))

        per_op: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
        for e in events:
            if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
                op = e["args"].get("op_name", "?")
                per_op[op][0] += 1
                per_op[op][1] += e["dur"]

        ranked = sorted(per_op.items(), key=lambda kv: -kv[1][1])
        ops = [{"op": op, "count": int(n // runs), "us_per_run": round(us / runs, 1)} for op, (n, us) in ranked]
        mean_ms = round(sum(us for _, us in per_op.values()) / runs / 1000, 3)
        hottest = f", hottest {ops[0]['op']} ({ops[0]['us_per_run']}us)" if ops else ""
        result["submodels"][sub] = {"summary": f"{mean_ms} ms/run{hottest}", "mean_ms": mean_ms, "ops": ops}

    return result


def _rand_input(node: Any, np: Any) -> Any:
    shape = [d if isinstance(d, int) else 1 for d in node.shape]
    if "int" in node.type:  # token ids etc. — values don't matter for latency
        return np.zeros(shape, dtype=np.int64 if "int64" in node.type else np.int32)
    return np.random.rand(*shape).astype(np.float32)
