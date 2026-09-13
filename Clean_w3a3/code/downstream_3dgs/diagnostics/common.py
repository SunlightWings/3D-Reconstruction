"""Shared helpers for the 15 downstream-3DGS diagnostic gate tests.

This module reuses the maintained pipeline code instead of reimplementing it:
  - code/downstream_3dgs/run_downstream_validation.py ("dv")   -- adapter, camera math, 3DGS invocation, metrics
  - code/analyze_full_dataset_disagreement.py         ("dis")  -- GT decoding, Sim(3), disagreement stats

Nothing here duplicates the frozen 40-scene run; all diagnostics either recompute
on outputs that already exist on disk, or train small new control variants into
a separate output tree so the existing results/ and /var/tmp/.../downstream_validation_v1
are never touched.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
REPO = SCRIPT_PATH.parents[3]
DOWNSTREAM_DIR = REPO / "code" / "downstream_3dgs"
RESULTS_DIR = REPO / "results" / "downstream_3dgs"
DIAG_RESULTS_DIR = RESULTS_DIR / "diagnostics"
FIGURES_DIR = DIAG_RESULTS_DIR / "figures"
STATE_PATH = DIAG_RESULTS_DIR / "state.json"
REPORT_PATH = RESULTS_DIR / "DIAGNOSTIC_RESULTS.md"

# Separate heavy-output tree: new control variants never touch the frozen v1 run.
DIAG_HEAVY_ROOT = Path("/var/tmp/poli22wo/quantsplat/downstream_diagnostics_v1")

DIAG_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DIAG_HEAVY_ROOT.mkdir(parents=True, exist_ok=True)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


dv = _load_module(DOWNSTREAM_DIR / "run_downstream_validation.py", "diag_dv")
dis = _load_module(REPO / "code" / "analyze_full_dataset_disagreement.py", "diag_dis")
pad_transform = dis.import_pad_transform(dv.REPO)

# Eight scenes spanning categories and both signs of the Full-minus-Quant PSNR delta,
# used for every test that requires *new* 3DGS training (too expensive to run on all 40
# scenes in a single background job). All CPU-only / reuse-existing-render tests below
# still run on the full 40-scene set.
EXPENSIVE_SCENES: List[str] = [
    "apple/110_13051_23361",
    "ball/123_14363_28981",
    "bowl/70_5792_13401",
    "broccoli/412_56288_108844",
    "hydrant/167_18184_34441",
    "remote/350_36761_68623",
    "teddybear/187_20215_38541",
    "toaster/372_41229_82130",
]


def log(msg: str = "") -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")
    tmp.replace(path)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def all_scenes() -> List[str]:
    agg = read_json(RESULTS_DIR / "aggregate_metrics.json")
    return [f"{r['category']}/{r['sequence']}" for r in agg["scenes"]]


def load_per_scene_csv() -> Dict[Tuple[str, str], Dict[str, float]]:
    path = REPO / "results" / "per_scene.csv"
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["category"], row["sequence"])
            parsed: Dict[str, float] = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (TypeError, ValueError):
                    parsed[k] = v  # type: ignore[assignment]
            out[key] = parsed
    return out


def load_aggregate_row(category: str, sequence: str) -> Dict[str, Any]:
    agg = read_json(RESULTS_DIR / "aggregate_metrics.json")
    for r in agg["scenes"]:
        if r["category"] == category and r["sequence"] == sequence:
            return r
    raise KeyError(f"{category}/{sequence} not in aggregate_metrics.json")


_FROZEN_MANIFEST = None


def frozen_manifest_scene(category: str, sequence: str) -> Dict[str, Any]:
    global _FROZEN_MANIFEST
    if _FROZEN_MANIFEST is None:
        _FROZEN_MANIFEST = read_json(DOWNSTREAM_DIR.parent / "frozen_dataset_manifest.json")
    for s in _FROZEN_MANIFEST["scenes"]:
        if s["category"] == category and s["sequence"] == sequence:
            return s
    raise KeyError(f"{category}/{sequence} not in frozen_dataset_manifest.json")


def scene_manifest(category: str, sequence: str):
    manifest_path = REPO / "datasets" / "frozen_dataset_manifest.json"
    manifest = read_json(manifest_path)

    for scene in manifest["scenes"]:
        if scene["category"] == category and scene["sequence"] == sequence:
            # Match the downstream contract:
            # use the first valid six-primary-view group as input,
            # and all remaining usable frames as held-out.
            groups = scene.get("groups", [])

            valid_groups = [
                g for g in groups
                if len(g.get("frame_numbers", [])) == 6
                and len(g.get("primary_frame_numbers", [])) == 6
                and not g.get("context_only_frame_numbers", [])
            ]

            if not valid_groups:
                raise RuntimeError(
                    f"No valid six-primary-view group for {category}/{sequence}"
                )

            valid_groups.sort(key=lambda g: g["group_id"])
            group = valid_groups[0]

            input_frames = list(map(int, group["frame_numbers"]))

            usable = [
                int(r["frame_number"])
                for r in scene["usable_frames"]
            ]

            heldout_frames = [
                f for f in usable
                if f not in input_frames
            ]

            return {
                "category": category,
                "sequence": sequence,
                "group_id": group["group_id"],
                "input_frames": input_frames,
                "heldout_frames": heldout_frames,
            }

    raise RuntimeError(
        f"Scene not found in frozen manifest: {category}/{sequence}"
    )

def scene_root(category: str, sequence: str) -> Path:
    return dv.OUT_ROOT / category / sequence


def split_scene(name: str) -> Tuple[str, str]:
    category, sequence = name.split("/", 1)
    return category, sequence


# ---------------------------------------------------------------------------
# Persistent state so the orchestrator is resumable and the markdown report can
# be regenerated after every single test finishes.
# ---------------------------------------------------------------------------

def load_state() -> Dict[str, Any]:
    if STATE_PATH.is_file():
        return read_json(STATE_PATH)
    return {"tests": {}, "updated_at": None}


def save_state(state: Dict[str, Any]) -> None:
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(STATE_PATH, state)


def test_status(test_id: str) -> str:
    return load_state()["tests"].get(test_id, {}).get("status", "pending")


def record_test(test_id: str, status: str, data: Any = None, error: str = None) -> None:
    state = load_state()
    state["tests"][test_id] = {
        "status": status,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data": data,
        "error": error,
    }
    save_state(state)


def evaluate_renders(
    gt_paths: List[Path],
    render_paths: List[Path],
    masks: List[np.ndarray],
    with_ssim: bool = True,
) -> Dict[str, Any]:
    """PSNR (+optional SSIM) between matched GT/render pairs under per-frame masks."""
    from PIL import Image

    if not (len(gt_paths) == len(render_paths) == len(masks)):
        raise ValueError("gt_paths, render_paths, masks length mismatch")

    def read_rgb(path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

    rows = []
    for gt_p, rd_p, mask in zip(gt_paths, render_paths, masks):
        if not mask.any():
            continue
        gt = read_rgb(gt_p)
        rd = read_rgb(rd_p)
        row = {"gt": str(gt_p), "render": str(rd_p), "psnr": dv.psnr(rd, gt, mask)}
        if with_ssim:
            try:
                row["ssim"] = dv.ssim_value(rd, gt, mask)
            except Exception:
                row["ssim"] = None
        rows.append(row)
    finite_psnr = [r["psnr"] for r in rows if np.isfinite(r["psnr"])]
    out = {
        "n_frames": len(rows),
        "mean_psnr": float(np.mean(finite_psnr)) if finite_psnr else float("nan"),
        "median_psnr": float(np.median(finite_psnr)) if finite_psnr else float("nan"),
        "per_frame": rows,
    }
    if with_ssim:
        ssim_vals = [r["ssim"] for r in rows if r.get("ssim") is not None]
        out["mean_ssim"] = float(np.mean(ssim_vals)) if ssim_vals else float("nan")
    return out


def run(cmd, cwd=None, env=None) -> None:
    dv.run(cmd, cwd=cwd, env=env)


def run_gate_test(test_id: str, fn: Callable[[], Any], force: bool = False) -> None:
    if not force and test_status(test_id) == "done":
        log(f"SKIP {test_id} (already done)")
        return
    log(f"START {test_id}")
    t0 = time.time()
    try:
        data = fn()
        dt = time.time() - t0
        log(f"DONE  {test_id} in {dt/60:.1f} min")
        record_test(test_id, "done", data=data)
    except Exception as e:  # noqa: BLE001 - a broken test must not kill the run
        dt = time.time() - t0
        tb = traceback.format_exc()
        log(f"FAILED {test_id} after {dt/60:.1f} min: {e}\n{tb}")
        record_test(test_id, "failed", error=f"{e}\n{tb}")
    # Regenerate the markdown report after every single test so progress is visible live.
    try:
        diag_dir = Path(__file__).resolve().parent
        if str(diag_dir) not in sys.path:
            sys.path.insert(0, str(diag_dir))
        import render_report  # noqa: PLC0415

        render_report.render()
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: report render failed: {e}")
