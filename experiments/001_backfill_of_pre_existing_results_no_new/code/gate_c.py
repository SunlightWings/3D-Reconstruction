"""Gate C -- does the geometry prior matter here at all?

C1 random-initialisation control, C2 densification disabled, C3 iteration sweep.
All three require new 3DGS training and therefore run on the 8-scene
common.EXPENSIVE_SCENES subset rather than the full 40 scenes.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

dv = common.dv
dis = common.dis

RNG_SEED = 20260907


def _train_3dgs_extra(source: Path, model: Path, iterations: int, extra_args: List[str],
                       required_iterations: List[int]) -> None:
    plys = [model / "point_cloud" / f"iteration_{it}" / "point_cloud.ply" for it in required_iterations]
    if all(p.is_file() for p in plys):
        common.log(f"3DGS train: resume skip, found all checkpoints in {model}")
        return
    if model.exists():
        shutil.rmtree(model)
    dv.wait_for_gpu(60, 2)
    model.parent.mkdir(parents=True, exist_ok=True)
    common.run(
        [str(dv.GS_PY), "train.py", "-s", str(source), "-m", str(model),
         "--iterations", str(iterations), "--data_device", "cpu", "--resolution", "1", *extra_args],
        cwd=dv.GS_ROOT, env=dv.gs_env(),
    )
    for p in plys:
        if not p.is_file():
            raise RuntimeError(f"3DGS training completed but checkpoint missing: {p}")


# ---------------------------------------------------------------------------
# C1: random-initialisation control. Cameras are GT-derived and IDENTICAL
# across full / w4a4 / random arms; only the initial point cloud differs.
# Full and W4A4 predicted points are mapped into the shared GT frame with the
# inverse of the same Sim(3) already fit by the main pipeline.
# ---------------------------------------------------------------------------

def _build_c1_arm(category: str, sequence: str, arm: str, stride: int, rng: np.random.Generator) -> Dict[str, Any]:
    manifest = common.scene_manifest(category, sequence)
    records = dv.load_annotations(category, sequence)
    frames = manifest["input_frames"]
    heldout_frames = manifest["heldout_frames"]
    root = common.DIAG_HEAVY_ROOT / category / sequence / f"c1_{arm}"
    train_root, heldout_root = root / "train", root / "heldout"
    ok = root / "PREPARED.ok"
    scene_common = common.scene_root(category, sequence) / "common"
    train_names = [f"image_{i}.png" for i in range(1, len(frames) + 1)]
    heldout_names = [f"frame{f:06d}.png" for f in heldout_frames]

    if ok.is_file():
        return {"train_source": train_root, "heldout_source": heldout_root,
                "model": common.DIAG_HEAVY_ROOT / category / sequence / "models" / f"c1_{arm}",
                "heldout_frames": heldout_frames}

    if root.exists():
        shutil.rmtree(root)
    train_root.mkdir(parents=True, exist_ok=True)
    heldout_root.mkdir(parents=True, exist_ok=True)
    os.symlink(scene_common / "train_images", train_root / "images", target_is_directory=True)
    os.symlink(scene_common / "heldout_images", heldout_root / "images", target_is_directory=True)

    E_gt = np.stack([dv.co3d_to_opencv_camera(records[f])[0] for f in frames])
    K_gt = np.stack([dv.original_to_518_affine(records[f]) @ dv.co3d_to_opencv_camera(records[f])[1] for f in frames])
    E_held = np.stack([dv.co3d_to_opencv_camera(records[f])[0] for f in heldout_frames])
    K_held = np.stack([dv.original_to_518_affine(records[f]) @ dv.co3d_to_opencv_camera(records[f])[1] for f in heldout_frames])

    train_tensor = dv.preprocess_images([dv.resolve_image_path(records[f]) for f in frames])

    if arm in ("full", "w4a4"):
        group_dir, _fm, _wm = dv.find_group(category, sequence)
        arrays = dv.load_npz(group_dir / f"{arm}.npz")
        pts_variant, colors = dv.sampled_points_and_colors(arrays["world_points_from_depth"], train_tensor, stride)
        align = manifest["camera_alignment"][arm]
        s, A, b = align["scale"], np.array(align["rotation"]), np.array(align["translation"])
        points = dis.pred_to_gt(pts_variant.astype(np.float64), s, A, b).astype(np.float32)
    elif arm == "random":
        full_align = manifest["camera_alignment"]["full"]
        group_dir, _fm, _wm = dv.find_group(category, sequence)
        arrays = dv.load_npz(group_dir / "full.npz")
        pts_variant, colors = dv.sampled_points_and_colors(arrays["world_points_from_depth"], train_tensor, stride)
        s, A, b = full_align["scale"], np.array(full_align["rotation"]), np.array(full_align["translation"])
        full_points_gt = dis.pred_to_gt(pts_variant.astype(np.float64), s, A, b)
        lo, hi = full_points_gt.min(axis=0), full_points_gt.max(axis=0)
        points = rng.uniform(lo, hi, size=full_points_gt.shape).astype(np.float32)
    else:
        raise ValueError(arm)

    dv.write_colmap_text_model(train_root, E_gt, K_gt, train_names, points, colors)
    (heldout_root / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    dv.write_colmap_text_model(heldout_root, E_held, K_held, heldout_names, points[:10], colors[:10])
    shutil.copy2(train_root / "sparse" / "0" / "points3D.txt", heldout_root / "sparse" / "0" / "points3D.txt")
    ok.write_text(f"PASS point_count={len(points)}\n", encoding="utf-8")
    return {"train_source": train_root, "heldout_source": heldout_root,
            "model": common.DIAG_HEAVY_ROOT / category / sequence / "models" / f"c1_{arm}",
            "heldout_frames": heldout_frames}


def test_c1_random_init_control() -> Dict[str, Any]:
    rng = np.random.default_rng(RNG_SEED)
    per_scene = []
    for name in common.EXPENSIVE_SCENES:
        category, sequence = common.split_scene(name)
        row = {"scene": name, "arms": {}}
        for arm in ("random", "full", "w4a4"):
            common.log(f"C1 {name}/{arm}: preparing + training")
            built = _build_c1_arm(category, sequence, arm, stride=4, rng=rng)
            dv.train_3dgs(built["train_source"], built["model"], common.TRAIN_ITERATIONS, 60, 2)
            out_dir = common.DIAG_HEAVY_ROOT / category / sequence / "heldout" / f"c1_{arm}"
            renders_dir = dv.render_heldout(built["heldout_source"], built["model"], out_dir, common.TRAIN_ITERATIONS,
                                             len(built["heldout_frames"]), 60, 2)
            records = dv.load_annotations(category, sequence)
            masks = [dv.content_mask_from_record(records[f]) for f in built["heldout_frames"]]
            gt_dir = common.scene_root(category, sequence) / "common" / "heldout_images"
            gt_paths = [gt_dir / f"frame{f:06d}.png" for f in built["heldout_frames"]]
            result = common.evaluate_renders(gt_paths, sorted(renders_dir.glob("*.png")), masks, with_ssim=False)
            row["arms"][arm] = result["mean_psnr"]
            common.log(f"C1 {name}/{arm}: PSNR = {result['mean_psnr']:.2f} dB")
        row["ordering_holds"] = row["arms"]["random"] < row["arms"]["w4a4"] <= row["arms"]["full"] + 1e-6 or \
                                 row["arms"]["random"] < row["arms"]["full"]
        per_scene.append(row)

    n_flat = sum(1 for r in per_scene if abs(r["arms"]["random"] - r["arms"]["full"]) < 0.5)
    return {
        "description": "GT-derived cameras held identical across arms; only the initial point cloud varies (random / w4a4-mapped-to-GT / full-mapped-to-GT).",
        "expect": "random < quant <= full with a visible margin",
        "scenes_tested": common.EXPENSIVE_SCENES,
        "n_scenes_where_random_almost_equals_full_lt_0.5db": n_flat,
        "mean_random_psnr": float(np.mean([r["arms"]["random"] for r in per_scene])),
        "mean_full_psnr_gt_frame": float(np.mean([r["arms"]["full"] for r in per_scene])),
        "mean_w4a4_psnr_gt_frame": float(np.mean([r["arms"]["w4a4"] for r in per_scene])),
        "per_scene": per_scene,
    }


# ---------------------------------------------------------------------------
# C2: densification disabled -- isolates the initial-point-cloud contribution
# by disabling adaptive density control on the EXISTING full/w4a4 sources.
# ---------------------------------------------------------------------------

def test_c2_densification_disabled() -> Dict[str, Any]:
    per_scene = []
    for name in common.EXPENSIVE_SCENES:
        category, sequence = common.split_scene(name)
        manifest = common.scene_manifest(category, sequence)
        heldout_frames = manifest["heldout_frames"]
        records = dv.load_annotations(category, sequence)
        masks = [dv.content_mask_from_record(records[f]) for f in heldout_frames]
        gt_dir = common.scene_root(category, sequence) / "common" / "heldout_images"
        gt_paths = [gt_dir / f"frame{f:06d}.png" for f in heldout_frames]

        row = {"scene": name}
        for variant in ("full", "w4a4"):
            source = Path(manifest[f"{variant}_train_source"])
            heldout_source = Path(manifest[f"{variant}_heldout_source"])
            model = common.DIAG_HEAVY_ROOT / category / sequence / "models" / f"c2_{variant}_densify_off"
            common.log(f"C2 {name}/{variant}: training with --densify_until_iter 0")
            _train_3dgs_extra(source, model, common.TRAIN_ITERATIONS, ["--densify_until_iter", "0"], [common.TRAIN_ITERATIONS])
            out_dir = common.DIAG_HEAVY_ROOT / category / sequence / "heldout" / f"c2_{variant}"
            renders_dir = dv.render_heldout(heldout_source, model, out_dir, common.TRAIN_ITERATIONS, len(heldout_frames), 60, 2)
            result = common.evaluate_renders(gt_paths, sorted(renders_dir.glob("*.png")), masks, with_ssim=False)
            row[f"{variant}_psnr"] = result["mean_psnr"]
        row["full_minus_quant"] = row["full_psnr"] - row["w4a4_psnr"]
        agg = common.load_aggregate_row(category, sequence)
        row["baseline_full_minus_quant_with_densify"] = agg["full_minus_quant_psnr"]
        per_scene.append(row)
        common.log(f"C2 {name}: densify-off gap={row['full_minus_quant']:+.3f} dB vs baseline gap={agg['full_minus_quant_psnr']:+.3f} dB")

    return {
        "description": "Same sources as the main run, but --densify_until_iter 0: isolates the geometry prior by disabling adaptive density control.",
        "expect": "the full-vs-quant gap grows, possibly by a lot",
        "scenes_tested": common.EXPENSIVE_SCENES,
        "mean_gap_densify_off": float(np.mean([r["full_minus_quant"] for r in per_scene])),
        "mean_gap_baseline_with_densify": float(np.mean([r["baseline_full_minus_quant_with_densify"] for r in per_scene])),
        "per_scene": per_scene,
    }


# ---------------------------------------------------------------------------
# C3: iteration sweep -- tests whether the full-vs-quant gap shrinks
# monotonically with more iterations (the "3DGS self-heals" hypothesis).
# ---------------------------------------------------------------------------

SWEEP_ITERS = [500, 1000, 3000, 7000, 15000, 30000]


def test_c3_iteration_sweep() -> Dict[str, Any]:
    per_scene = []
    for name in common.EXPENSIVE_SCENES:
        category, sequence = common.split_scene(name)
        manifest = common.scene_manifest(category, sequence)
        heldout_frames = manifest["heldout_frames"]
        records = dv.load_annotations(category, sequence)
        masks = [dv.content_mask_from_record(records[f]) for f in heldout_frames]
        gt_dir = common.scene_root(category, sequence) / "common" / "heldout_images"
        gt_paths = [gt_dir / f"frame{f:06d}.png" for f in heldout_frames]

        models = {}
        for variant in ("full", "w4a4"):
            source = Path(manifest[f"{variant}_train_source"])
            model = common.DIAG_HEAVY_ROOT / category / sequence / "models" / f"c3_{variant}_sweep"
            common.log(f"C3 {name}/{variant}: training with fine-grained checkpoints {SWEEP_ITERS}")
            # argparse's --save_iterations/--test_iterations use nargs="+": a repeated flag
            # RESETS the list rather than accumulating, so all values must follow one flag.
            extra = (
                ["--save_iterations", *[str(it) for it in SWEEP_ITERS]]
                + ["--test_iterations", *[str(it) for it in SWEEP_ITERS]]
            )
            _train_3dgs_extra(source, model, 30000, extra, SWEEP_ITERS)
            models[variant] = model

        curve = []
        for it in SWEEP_ITERS:
            point = {"iteration": it}
            for variant in ("full", "w4a4"):
                heldout_source = Path(manifest[f"{variant}_heldout_source"])
                out_dir = common.DIAG_HEAVY_ROOT / category / sequence / "heldout" / f"c3_{variant}_iter{it}"
                renders_dir = dv.render_heldout(heldout_source, models[variant], out_dir, it, len(heldout_frames), 60, 2)
                result = common.evaluate_renders(gt_paths, sorted(renders_dir.glob("*.png")), masks, with_ssim=False)
                point[f"{variant}_psnr"] = result["mean_psnr"]
            point["gap"] = point["full_psnr"] - point["w4a4_psnr"]
            curve.append(point)
            common.log(f"C3 {name} @ {it}: full={point['full_psnr']:.2f} dB, w4a4={point['w4a4_psnr']:.2f} dB, gap={point['gap']:+.3f} dB")
        per_scene.append({"scene": name, "curve": curve})

    mean_gap_by_iter = {
        it: float(np.mean([next(p["gap"] for p in s["curve"] if p["iteration"] == it) for s in per_scene]))
        for it in SWEEP_ITERS
    }
    early, late = mean_gap_by_iter[SWEEP_ITERS[0]], mean_gap_by_iter[SWEEP_ITERS[-1]]
    return {
        "description": "Full-vs-quant PSNR gap at increasing iteration counts, on identical sources to the main run.",
        "expect": "a gap at 500-3000 that closes by 30k; a flat line from 500 onward would indicate a Gate A failure rather than robustness",
        "scenes_tested": common.EXPENSIVE_SCENES,
        "mean_gap_by_iteration": mean_gap_by_iter,
        "gap_shrinks_with_iterations": late < early,
        "per_scene": per_scene,
    }


TESTS = [
    ("C1", test_c1_random_init_control),
    ("C2", test_c2_densification_disabled),
    ("C3", test_c3_iteration_sweep),
]
