from pathlib import Path
from typing import Any

# per-model subdirectories that hold rknpu/<soc>/model.rknn
SUBMODELS = ["textual", "visual", "detection", "recognition"]


def profile(model_dir: Path, soc: str = "rk3588") -> dict[str, Any]:
    """Profile every RKNN submodel on an attached NPU via eval_perf (per-op NPU/CPU costs)."""
    from rknn.api import RKNN

    subs = [s for s in SUBMODELS if (model_dir / s / "rknpu" / soc / "model.rknn").is_file()]
    if not subs:
        raise RuntimeError(f"No RKNN model for {soc} found under {model_dir}")

    result: dict[str, Any] = {"model": model_dir.name, "format": "rknn", "soc": soc, "submodels": {}}
    for sub in subs:
        rknn = RKNN(verbose=False)
        try:
            if rknn.load_rknn((model_dir / sub / "rknpu" / soc / "model.rknn").as_posix()) != 0:
                raise RuntimeError(f"load_rknn failed for {sub}")
            if rknn.init_runtime(target=soc, perf_debug=True) != 0:
                raise RuntimeError(f"init_runtime failed for {sub} (is a {soc} NPU attached?)")
            result["submodels"][sub] = _parse_perf(rknn.eval_perf(is_print=False))
        finally:
            rknn.release()
    return result


def _parse_perf(report: str) -> dict[str, Any]:
    """Parse eval_perf's per-op-type summary table + CPU/NPU totals out of its report string."""
    ops: list[dict[str, Any]] = []
    totals = {"cpu_us": 0, "npu_us": 0, "total_us": 0}
    in_table = False
    for line in report.splitlines():
        s = line.strip()
        if s.startswith("OpType") and "CallNumber" in s:
            in_table = True
            continue
        if not in_table or not s or s.startswith("-"):
            continue
        parts = s.split()
        if parts[0] == "Total":  # Total <cpu> <gpu> <npu> <total>
            nums = [int(p) for p in parts[1:] if p.lstrip("-").isdigit()]
            if len(nums) >= 4:
                totals = {"cpu_us": nums[0], "npu_us": nums[2], "total_us": nums[3]}
            break
        if len(parts) >= 6 and parts[1].isdigit():  # <op> <calls> <cpu> <gpu> <npu> <total> <ratio%>
            ops.append(
                {
                    "op": parts[0],
                    "calls": int(parts[1]),
                    "cpu_us": int(parts[2]),
                    "npu_us": int(parts[4]),
                    "total_us": int(parts[5]),
                }
            )

    total = totals["total_us"] or 1
    cpu_pct, npu_pct = round(100 * totals["cpu_us"] / total), round(100 * totals["npu_us"] / total)
    return {
        "summary": f"{totals['total_us'] / 1000:.1f} ms/frame (CPU {cpu_pct}%, NPU {npu_pct}%)",
        "total_ms": round(totals["total_us"] / 1000, 2),
        "cpu_pct": cpu_pct,
        "npu_pct": npu_pct,
        "ops": ops,
    }
