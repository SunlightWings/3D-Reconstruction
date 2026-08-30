#!/usr/bin/env python3
"""Restart-safe 40-scene Full-VGGT vs W4A4 -> 3DGS downstream validation.

This is an orchestration wrapper around run_downstream_validation.py.

Atomic scene policy
-------------------
A scene is considered complete only after BOTH Full and W4A4 3DGS training,
held-out rendering, and PSNR/SSIM/LPIPS evaluation finish successfully.
Only then is a persistent SCENE_DONE.json marker written.

At the next launch:
  * scenes with a valid SCENE_DONE.json are skipped;
  * any scene that has heavy /var/tmp output but no done marker is treated as
    interrupted and that scene directory is deleted before it is rerun;
  * completed scene metrics/renders are copied to persistent home storage so a
    workstation reboot does not erase the experimental record.

The runner discovers the frozen 40-scene set from the already-validated Full
prediction metadata. It reuses the existing Full/W4A4 NPZ predictions and does
NOT rerun VGGT.

The experiment uses one deterministic six-primary-view group per scene and nine
held-out CO3D known-camera views, matching the controlled downstream protocol.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "run_downstream_validation.py"

# Keep the already-completed 3-scene pilot and extend the same output tree.
HEAVY_ROOT = Path("/var/tmp/luli38se/quantsplat/downstream_validation_v1")

# Small, reboot-persistent record of every completed scene.
STATE_ROOT = Path("/home/utn/luli38se/cv/downstream_full_dataset_v1_state")
COMPLETED_ROOT = STATE_ROOT / "completed"
LOG_ROOT = STATE_ROOT / "logs"

EXPECTED_SCENES = 40


class CleanDailyStop(RuntimeError):
    """Intentional stop near midnight; not an experiment failure."""


def load_base():
    if not BASE_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Missing base runner: {BASE_SCRIPT}\n"
            "Place run_full_dataset_downstream.py beside run_downstream_validation.py."
        )
    spec = importlib.util.spec_from_file_location("downstream_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["downstream_base"] = module
    spec.loader.exec_module(module)
    module.OUT_ROOT = HEAVY_ROOT
    return module


base = load_base()


def log(msg: str = "") -> None:
    print(msg, flush=True)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def scene_key(category: str, sequence: str) -> str:
    return f"{category}/{sequence}"


def persistent_scene_dir(category: str, sequence: str) -> Path:
    return COMPLETED_ROOT / category / sequence


def done_path(category: str, sequence: str) -> Path:
    return persistent_scene_dir(category, sequence) / "SCENE_DONE.json"


def heavy_scene_dir(category: str, sequence: str) -> Path:
    return HEAVY_ROOT / category / sequence


def discover_scenes() -> List[Tuple[str, str]]:
    """Discover the frozen dataset from Full prediction metadata."""
    pairs = set()
    for p in base.PRED_ROOT.rglob("full_meta.json"):
        try:
            m = read_json(p)
        except Exception:
            continue
        category = m.get("category")
        sequence = m.get("sequence")
        if category and sequence:
            pairs.add((str(category), str(sequence)))
    scenes = sorted(pairs)
    if len(scenes) != EXPECTED_SCENES:
        raise RuntimeError(
            f"Expected {EXPECTED_SCENES} frozen scenes, discovered {len(scenes)}"
        )
    return scenes


def complete_heavy_scene(category: str, sequence: str, iterations: int, heldout_views: int) -> bool:
    """Recognize a completed scene, including the existing 3-scene pilot."""
    root = heavy_scene_dir(category, sequence)
    metrics = root / "metrics.json"
    manifest = root / "scene_manifest.json"
    full_ply = root / "models/full/point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    quant_ply = root / "models/w4a4/point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    full_renders = root / "heldout/full/renders"
    quant_renders = root / "heldout/w4a4/renders"
    if not all(p.is_file() for p in (metrics, manifest, full_ply, quant_ply)):
        return False
    if len(list(full_renders.glob("*.png"))) != heldout_views:
        return False
    if len(list(quant_renders.glob("*.png"))) != heldout_views:
        return False
    try:
        d = read_json(metrics)
        if int(d["summary"]["n_views"]) != heldout_views:
            return False
    except Exception:
        return False
    return True


def persist_completed_scene(category: str, sequence: str, iterations: int, heldout_views: int) -> dict:
    """Copy the small scientific record to persistent storage, then atomically mark DONE."""
    src = heavy_scene_dir(category, sequence)
    if not complete_heavy_scene(category, sequence, iterations, heldout_views):
        raise RuntimeError(f"Cannot persist incomplete scene {category}/{sequence}")

    dst = persistent_scene_dir(category, sequence)
    dst.mkdir(parents=True, exist_ok=True)

    # Core provenance + metrics.
    for name in ("metrics.json", "metrics.csv", "scene_manifest.json"):
        s = src / name
        if s.is_file():
            shutil.copy2(s, dst / name)

    # Large renders/models remain in /var/tmp.
    # Persistent home storage keeps only compact scientific records:
    # metrics, provenance manifest, and SCENE_DONE marker.
    metrics = read_json(dst / "metrics.json")
    manifest = read_json(dst / "scene_manifest.json")
    done = {
        "status": "SCENE_DONE",
        "category": category,
        "sequence": sequence,
        "completed_at": datetime.now().astimezone().isoformat(),
        "iterations": iterations,
        "heldout_views": heldout_views,
        "group_id": manifest.get("group_id"),
        "input_frames": manifest.get("input_frames"),
        "heldout_frames": manifest.get("heldout_frames"),
        "full_prediction_sha256": manifest.get("full_prediction_sha256_actual"),
        "w4a4_prediction_sha256": manifest.get("w4a4_prediction_sha256_actual"),
        "metrics_summary": metrics["summary"],
        "heavy_scene_root": str(src),
        "persistent_scene_root": str(dst),
    }
    # Marker is written LAST: its existence means the entire scene is committed.
    write_json(done_path(category, sequence), done)
    return done


def valid_done(category: str, sequence: str, iterations: int, heldout_views: int) -> bool:
    p = done_path(category, sequence)
    if not p.is_file():
        return False
    try:
        d = read_json(p)
        return (
            d.get("status") == "SCENE_DONE"
            and d.get("category") == category
            and d.get("sequence") == sequence
            and int(d.get("iterations")) == iterations
            and int(d.get("heldout_views")) == heldout_views
            and (p.parent / "metrics.json").is_file()
            and (p.parent / "scene_manifest.json").is_file()
        )
    except Exception:
        return False


def bootstrap_existing_completed(scenes: Sequence[Tuple[str, str]], iterations: int, heldout_views: int) -> int:
    """Import already-finished pilot scenes into the persistent DONE ledger."""
    count = 0
    for category, sequence in scenes:
        if valid_done(category, sequence, iterations, heldout_views):
            continue
        if complete_heavy_scene(category, sequence, iterations, heldout_views):
            log(f"BOOTSTRAP DONE: {category}/{sequence}")
            persist_completed_scene(category, sequence, iterations, heldout_views)
            count += 1
    return count


def clean_interrupted_scenes(scenes: Sequence[Tuple[str, str]], iterations: int, heldout_views: int) -> List[str]:
    """Delete only heavy scene directories that do not have a valid persistent DONE marker."""
    removed = []
    for category, sequence in scenes:
        if valid_done(category, sequence, iterations, heldout_views):
            continue
        root = heavy_scene_dir(category, sequence)
        if root.exists():
            log(f"INTERRUPTED/INCOMPLETE -> deleting only: {root}")
            shutil.rmtree(root)
            removed.append(scene_key(category, sequence))
    return removed


def minutes_until_midnight() -> float:
    now = datetime.now().astimezone()
    tomorrow = (now + timedelta(days=1)).date()
    midnight = datetime.combine(tomorrow, datetime.min.time(), tzinfo=now.tzinfo)
    return (midnight - now).total_seconds() / 60.0


def make_deadline_gpu_wait(original_wait, *, abort_wait_minutes: float):
    """Wrap the proven GPU gate so it refuses to begin a new GPU stage too near midnight."""
    def wait(poll_seconds: int, stable_checks: int) -> None:
        stable = 0
        while True:
            remaining = minutes_until_midnight()
            if remaining < abort_wait_minutes:
                raise CleanDailyStop(
                    f"Only {remaining:.1f} minutes remain until midnight; "
                    f"not starting another GPU stage (threshold={abort_wait_minutes:.1f}m)."
                )
            procs = base.gpu_compute_processes()
            if not procs:
                stable += 1
                if stable >= stable_checks:
                    log(
                        f"GPU gate: free for {stable_checks} consecutive checks -> proceed "
                        f"({remaining:.1f} min until midnight)"
                    )
                    return
                log(f"GPU gate: no compute process ({stable}/{stable_checks}); rechecking...")
            else:
                stable = 0
                desc = ", ".join(f"pid={pid} mem={mem}MiB" for pid, mem in procs)
                log(
                    f"GPU gate: busy ({desc}); waiting {poll_seconds}s. "
                    f"No process will be killed. {remaining:.1f} min until midnight."
                )
            time.sleep(poll_seconds)
    return wait


def result_row(category: str, sequence: str, result: dict) -> dict:
    s = result["summary"]
    return {
        "category": category,
        "sequence": sequence,
        "n_views": int(s["n_views"]),
        "full_psnr": float(s["full_mean_psnr_content"]),
        "quant_psnr": float(s["quant_mean_psnr_content"]),
        "full_minus_quant_psnr": float(s["full_minus_quant_psnr_content"]),
        "full_ssim": float(s["full_mean_ssim_content"]),
        "quant_ssim": float(s["quant_mean_ssim_content"]),
        "full_minus_quant_ssim": float(s["full_minus_quant_ssim_content"]),
        "full_lpips": float(s["full_mean_lpips_content"]),
        "quant_lpips": float(s["quant_mean_lpips_content"]),
        "quant_minus_full_lpips": float(s["quant_minus_full_lpips_content"]),
    }


def rebuild_progress(scenes: Sequence[Tuple[str, str]], iterations: int, heldout_views: int) -> dict:
    rows = []
    for category, sequence in scenes:
        if not valid_done(category, sequence, iterations, heldout_views):
            continue
        result = read_json(persistent_scene_dir(category, sequence) / "metrics.json")
        rows.append(result_row(category, sequence, result))

    rows.sort(key=lambda x: (x["category"], x["sequence"]))
    done_keys = {scene_key(r["category"], r["sequence"]) for r in rows}
    remaining = [scene_key(c, s) for c, s in scenes if scene_key(c, s) not in done_keys]

    progress = {
        "status": "COMPLETE" if len(rows) == len(scenes) else "IN_PROGRESS",
        "updated_at": datetime.now().astimezone().isoformat(),
        "total_scenes": len(scenes),
        "completed_scenes": len(rows),
        "remaining_scenes": len(remaining),
        "completed": sorted(done_keys),
        "remaining": remaining,
    }
    write_json(STATE_ROOT / "progress.json", progress)

    if rows:
        with (STATE_ROOT / "aggregate_partial.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        agg = {
            **progress,
            "scenes": rows,
            "mean_full_psnr": float(np.mean([r["full_psnr"] for r in rows])),
            "mean_quant_psnr": float(np.mean([r["quant_psnr"] for r in rows])),
            "mean_full_minus_quant_psnr": float(np.mean([r["full_minus_quant_psnr"] for r in rows])),
            "median_full_minus_quant_psnr": float(np.median([r["full_minus_quant_psnr"] for r in rows])),
            "mean_full_ssim": float(np.mean([r["full_ssim"] for r in rows])),
            "mean_quant_ssim": float(np.mean([r["quant_ssim"] for r in rows])),
            "mean_full_minus_quant_ssim": float(np.mean([r["full_minus_quant_ssim"] for r in rows])),
            "median_full_minus_quant_ssim": float(np.median([r["full_minus_quant_ssim"] for r in rows])),
            "mean_full_lpips": float(np.mean([r["full_lpips"] for r in rows])),
            "mean_quant_lpips": float(np.mean([r["quant_lpips"] for r in rows])),
            "mean_quant_minus_full_lpips": float(np.mean([r["quant_minus_full_lpips"] for r in rows])),
            "median_quant_minus_full_lpips": float(np.median([r["quant_minus_full_lpips"] for r in rows])),
            "scenes_full_better_psnr": int(sum(r["full_minus_quant_psnr"] > 0 for r in rows)),
            "scenes_full_better_ssim": int(sum(r["full_minus_quant_ssim"] > 0 for r in rows)),
            "scenes_full_better_lpips": int(sum(r["quant_minus_full_lpips"] > 0 for r in rows)),
        }
        write_json(STATE_ROOT / "aggregate_partial.json", agg)
        if len(rows) == len(scenes):
            write_json(STATE_ROOT / "aggregate_metrics.json", agg)
            shutil.copy2(STATE_ROOT / "aggregate_partial.csv", STATE_ROOT / "aggregate_metrics.csv")

    return progress


def print_status(progress: dict) -> None:
    log("=" * 72)
    log("FULL DATASET DOWNSTREAM STATUS")
    log("=" * 72)
    log(f"Completed: {progress['completed_scenes']}/{progress['total_scenes']}")
    log(f"Remaining: {progress['remaining_scenes']}")
    log(f"Status:    {progress['status']}")
    log(f"State:     {STATE_ROOT}")
    if progress["remaining"]:
        log("Next unfinished scenes:")
        for x in progress["remaining"][:8]:
            log(f"  {x}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--heldout-views", type=int, default=9)
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--gpu-poll-seconds", type=int, default=10)
    p.add_argument("--gpu-stable-checks", type=int, default=1)
    p.add_argument(
        "--min-start-minutes",
        type=float,
        default=90.0,
        help="Do not start a new scene with fewer than this many minutes until midnight.",
    )
    p.add_argument(
        "--abort-wait-minutes",
        type=float,
        default=45.0,
        help="If GPU waiting reaches this close to midnight, stop cleanly; current incomplete scene is reset next launch.",
    )
    p.add_argument("--status", action="store_true", help="Print completion state and exit.")
    p.add_argument("--prepare-only", action="store_true", help="Prepare one next scene CPU-only, then exit.")
    return p.parse_args()


def run_one_scene(category: str, sequence: str, args: argparse.Namespace) -> dict:
    name = scene_key(category, sequence)
    root = heavy_scene_dir(category, sequence)

    # Atomicity: at entry, no unfinished artifacts are allowed to survive.
    if root.exists() and not valid_done(category, sequence, args.iterations, args.heldout_views):
        log(f"RESET incomplete scene before rerun: {root}")
        shutil.rmtree(root)

    log("\n" + "=" * 72)
    log(f"SCENE START {name}")
    log("=" * 72)

    p = base.prepare_scene(category, sequence, args.stride, args.heldout_views)
    m = p["manifest"]
    log(
        f"Prepared {name}: group={m['group_id']} input={m['input_frames']} "
        f"heldout={m['heldout_frames']} points={m['initial_point_count_each']}"
    )

    if args.prepare_only:
        raise CleanDailyStop("--prepare-only requested after preparing the next unfinished scene")

    full_model = root / "models/full"
    quant_model = root / "models/w4a4"

    base.train_3dgs(
        p["full_source"], full_model, args.iterations,
        args.gpu_poll_seconds, args.gpu_stable_checks,
    )
    base.train_3dgs(
        p["quant_source"], quant_model, args.iterations,
        args.gpu_poll_seconds, args.gpu_stable_checks,
    )
    full_renders = base.render_heldout(
        p["full_heldout_source"], full_model, root / "heldout/full",
        args.iterations, args.heldout_views,
        args.gpu_poll_seconds, args.gpu_stable_checks,
    )
    quant_renders = base.render_heldout(
        p["quant_heldout_source"], quant_model, root / "heldout/w4a4",
        args.iterations, args.heldout_views,
        args.gpu_poll_seconds, args.gpu_stable_checks,
    )

    result = base.evaluate_scene(
        root,
        p["heldout_frames"],
        p["content_masks"],
        full_renders,
        quant_renders,
        p["heldout_image_dir"],
    )
    s = result["summary"]
    log(
        f"RESULT {name}: Full-Quant PSNR={s['full_minus_quant_psnr_content']:+.3f} dB, "
        f"SSIM={s['full_minus_quant_ssim_content']:+.5f}, "
        f"Quant-Full LPIPS={s['quant_minus_full_lpips_content']:+.5f}"
    )

    persist_completed_scene(category, sequence, args.iterations, args.heldout_views)
    log(f"SCENE COMMITTED: {name}")
    return result


def main() -> None:
    args = parse_args()
    if args.stride <= 0 or args.heldout_views <= 0 or args.iterations <= 0:
        raise RuntimeError("stride, heldout-views, and iterations must be positive")

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    COMPLETED_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    HEAVY_ROOT.mkdir(parents=True, exist_ok=True)

    # Proven environment/preflight from the existing runner.
    base.preflight()
    scenes = discover_scenes()

    boot = bootstrap_existing_completed(scenes, args.iterations, args.heldout_views)
    if boot:
        log(f"Imported {boot} already-completed scene(s) into persistent state.")

    removed = clean_interrupted_scenes(scenes, args.iterations, args.heldout_views)
    if removed:
        log(f"Cleaned {len(removed)} incomplete scene(s) from /var/tmp.")

    progress = rebuild_progress(scenes, args.iterations, args.heldout_views)
    print_status(progress)
    if args.status or progress["status"] == "COMPLETE":
        return

    # Use the existing safe shared-GPU gate, augmented with a midnight stop.
    base.wait_for_gpu = make_deadline_gpu_wait(
        base.wait_for_gpu,
        abort_wait_minutes=args.abort_wait_minutes,
    )

    try:
        for category, sequence in scenes:
            if valid_done(category, sequence, args.iterations, args.heldout_views):
                continue

            remaining = minutes_until_midnight()
            if remaining < args.min_start_minutes:
                raise CleanDailyStop(
                    f"Only {remaining:.1f} minutes remain until midnight. "
                    f"Threshold to start a new scene is {args.min_start_minutes:.1f} minutes."
                )

            run_one_scene(category, sequence, args)
            progress = rebuild_progress(scenes, args.iterations, args.heldout_views)
            print_status(progress)

    except CleanDailyStop as e:
        log("\n" + "=" * 72)
        log("CLEAN DAILY STOP")
        log("=" * 72)
        log(str(e))
        progress = rebuild_progress(scenes, args.iterations, args.heldout_views)
        print_status(progress)
        log("Rerun the same command after the workstation is available again.")
        return

    progress = rebuild_progress(scenes, args.iterations, args.heldout_views)
    log("\n" + "=" * 72)
    log("FULL 40-SCENE DOWNSTREAM VALIDATION COMPLETE")
    log("=" * 72)
    print_status(progress)
    log(f"Final aggregate: {STATE_ROOT / 'aggregate_metrics.json'}")


if __name__ == "__main__":
    main()