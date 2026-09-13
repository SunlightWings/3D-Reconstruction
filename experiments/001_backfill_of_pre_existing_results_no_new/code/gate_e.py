"""Gate E -- geometry evidence, not photometry.

E1 Chamfer distance of final Gaussians, E2 rendered depth error on held-out
views, E3 floater/opacity audit. E1 and E3 are CPU-only and reuse the already
-trained point clouds across all 40 scenes; E2 needs one GPU render pass
(reusing the already-trained models, no retraining) also across all 40 scenes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from plyfile import PlyData
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

dv = common.dv
dis = common.dis

TAU_FRACTION_OF_RADIUS = 0.02  # Chamfer / floater distance threshold, as a fraction of scene radius


def _read_gaussians(path: Path):
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float64)
    raw_opacity = np.asarray(v["opacity"], dtype=np.float64)
    opacity = 1.0 / (1.0 + np.exp(-raw_opacity))  # GraphDECO stores opacity pre-sigmoid
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(opacity)
    return xyz[finite], opacity[finite]


def _gaussians_in_gt_frame(category: str, sequence: str, variant: str):
    manifest = common.scene_manifest(category, sequence)
    ply_path = common.scene_root(category, sequence) / "models" / variant / "point_cloud" / f"iteration_{common.TRAIN_ITERATIONS}" / "point_cloud.ply"
    xyz, opacity = _read_gaussians(ply_path)
    align = manifest["camera_alignment"][variant]
    s, A, b = align["scale"], np.array(align["rotation"]), np.array(align["translation"])
    xyz_gt = dis.pred_to_gt(xyz, s, A, b)
    return xyz_gt, opacity


def test_e1_chamfer_distance() -> Dict[str, Any]:
    per_scene_csv = common.load_per_scene_csv()
    per_scene = []
    for name in common.all_scenes():
        category, sequence = common.split_scene(name)
        radius = per_scene_csv[(category, sequence)]["scene_radius"]
        tau = TAU_FRACTION_OF_RADIUS * radius
        gt_entry = common.frozen_manifest_scene(category, sequence)
        gt_points = dis.read_point_cloud(Path(gt_entry["point_cloud_path"]))
        gt_tree = cKDTree(gt_points)
        gt_sub = gt_points if len(gt_points) <= 20000 else gt_points[np.random.default_rng(0).choice(len(gt_points), 20000, replace=False)]

        row = {"scene": name, "tau": tau, "variants": {}}
        for variant in ("full", "w4a4"):
            xyz_gt, opacity = _gaussians_in_gt_frame(category, sequence, variant)
            tree = cKDTree(xyz_gt)
            acc_dist, _ = gt_tree.query(xyz_gt, k=1, workers=-1)  # gaussian -> nearest GT point (accuracy)
            comp_dist, _ = tree.query(gt_sub, k=1, workers=-1)  # GT -> nearest gaussian (completeness)
            precision = float(np.mean(acc_dist < tau))
            recall = float(np.mean(comp_dist < tau))
            fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            row["variants"][variant] = {
                "n_gaussians": int(len(xyz_gt)),
                "accuracy_mean": float(acc_dist.mean()),
                "accuracy_median": float(np.median(acc_dist)),
                "completeness_mean": float(comp_dist.mean()),
                "precision_at_tau": precision,
                "recall_at_tau": recall,
                "fscore_at_tau": fscore,
                "_acc_dist_cache_for_e3": acc_dist,
                "_opacity_cache_for_e3": opacity,
            }
        row["gap_fscore_full_minus_quant"] = row["variants"]["full"]["fscore_at_tau"] - row["variants"]["w4a4"]["fscore_at_tau"]
        per_scene.append(row)
        common.log(
            f"E1 {name}: full F={row['variants']['full']['fscore_at_tau']:.3f}, "
            f"w4a4 F={row['variants']['w4a4']['fscore_at_tau']:.3f}"
        )

    # Cache raw per-gaussian arrays for E3 on local disk (too big for the JSON state file).
    cache_dir = common.DIAG_RESULTS_DIR / "e1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for row in per_scene:
        category, sequence = common.split_scene(row["scene"])
        for variant in ("full", "w4a4"):
            v = row["variants"][variant]
            np.savez(
                cache_dir / f"{category}__{sequence}__{variant}.npz",
                acc_dist=v.pop("_acc_dist_cache_for_e3"),
                opacity=v.pop("_opacity_cache_for_e3"),
                tau=row["tau"],
            )

    full_f = [r["variants"]["full"]["fscore_at_tau"] for r in per_scene]
    quant_f = [r["variants"]["w4a4"]["fscore_at_tau"] for r in per_scene]
    return {
        "description": f"Chamfer accuracy/completeness/F-score of the final Gaussian centres vs the dense CO3D GT point cloud, at tau={TAU_FRACTION_OF_RADIUS}*scene_radius.",
        "expect": "a full-vs-quant gap here even where PSNR shows none",
        "n_scenes": len(per_scene),
        "mean_fscore_full": float(np.mean(full_f)),
        "mean_fscore_w4a4": float(np.mean(quant_f)),
        "mean_gap_fscore": float(np.mean(full_f) - np.mean(quant_f)),
        "per_scene": per_scene,
    }


def test_e3_floater_opacity_audit() -> Dict[str, Any]:
    cache_dir = common.DIAG_RESULTS_DIR / "e1_cache"
    if not cache_dir.is_dir():
        raise RuntimeError("E3 requires E1 to have run first (reads its cached per-gaussian arrays)")
    per_scene = []
    for name in common.all_scenes():
        category, sequence = common.split_scene(name)
        row = {"scene": name, "variants": {}}
        for variant in ("full", "w4a4"):
            npz_path = cache_dir / f"{category}__{sequence}__{variant}.npz"
            if not npz_path.is_file():
                continue
            data = np.load(npz_path)
            acc_dist, opacity, tau = data["acc_dist"], data["opacity"], float(data["tau"])
            is_floater = acc_dist > tau
            total_opacity = float(opacity.sum())
            floater_opacity_mass = float(opacity[is_floater].sum())
            row["variants"][variant] = {
                "n_gaussians": int(len(opacity)),
                "n_floaters": int(is_floater.sum()),
                "floater_fraction_by_count": float(is_floater.mean()),
                "floater_opacity_mass_fraction": floater_opacity_mass / total_opacity if total_opacity > 0 else float("nan"),
                "mean_opacity": float(opacity.mean()),
                "median_opacity": float(np.median(opacity)),
                "fraction_opacity_gt_0.5": float(np.mean(opacity > 0.5)),
            }
        per_scene.append(row)
        common.log(
            f"E3 {name}: full floater-mass={row['variants']['full']['floater_opacity_mass_fraction']:.4f}, "
            f"w4a4 floater-mass={row['variants']['w4a4']['floater_opacity_mass_fraction']:.4f}"
        )

    def mean_of(key: str, variant: str) -> float:
        return float(np.mean([r["variants"][variant][key] for r in per_scene if variant in r["variants"]]))

    return {
        "description": "Opacity-weighted floater mass and Gaussian-count comparison, full vs w4a4, at the same tau used in E1.",
        "expect": "more opacity-weighted floater mass in the quantized arm; a count difference supports an efficiency framing even if quality matches",
        "n_scenes": len(per_scene),
        "mean_floater_mass_full": mean_of("floater_opacity_mass_fraction", "full"),
        "mean_floater_mass_w4a4": mean_of("floater_opacity_mass_fraction", "w4a4"),
        "mean_n_gaussians_full": mean_of("n_gaussians", "full"),
        "mean_n_gaussians_w4a4": mean_of("n_gaussians", "w4a4"),
        "per_scene": per_scene,
    }


def _render_depth(heldout_source: Path, model: Path, out_dir: Path, iterations: int, n_views: int) -> Path:
    depth_dir = out_dir / "depth"
    if depth_dir.is_dir() and len(list(depth_dir.glob("*.npy"))) == n_views:
        return depth_dir
    dv.wait_for_gpu(60, 2)
    script = Path(__file__).resolve().parent / "gs_render_depth.py"
    common.run(
        [str(dv.GS_PY), str(script), "-m", str(model), "-s", str(heldout_source),
         "--iteration", str(iterations), "--out", str(out_dir)],
        cwd=dv.GS_ROOT, env=dv.gs_env(),
    )
    npys = sorted(depth_dir.glob("*.npy"))
    if len(npys) != n_views:
        raise RuntimeError(f"Expected {n_views} depth maps, found {len(npys)} in {depth_dir}")
    return depth_dir


def test_e2_rendered_depth_error() -> Dict[str, Any]:
    per_scene_csv = common.load_per_scene_csv()
    per_scene = []
    for name in common.all_scenes():
        category, sequence = common.split_scene(name)
        manifest = common.scene_manifest(category, sequence)
        heldout_frames = manifest["heldout_frames"]
        records = dv.load_annotations(category, sequence)
        radius = per_scene_csv[(category, sequence)]["scene_radius"]

        row = {"scene": name, "variants": {}}
        for variant in ("full", "w4a4"):
            heldout_source = Path(manifest[f"{variant}_heldout_source"])
            model = common.scene_root(category, sequence) / "models" / variant
            out_dir = common.DIAG_HEAVY_ROOT / category / sequence / "heldout_depth" / variant
            depth_dir = _render_depth(heldout_source, model, out_dir, common.TRAIN_ITERATIONS, len(heldout_frames))
            s = manifest["camera_alignment"][variant]["scale"]

            band_errs, interior_errs, all_errs = [], [], []
            for idx, f in enumerate(heldout_frames):
                depth_variant = np.load(depth_dir / f"{idx:05d}.npy").squeeze()
                depth_gt_equiv = depth_variant / s
                gt = dis.decode_gt_frame(records[f], dv.CO3D_ROOT, common.pad_transform)
                gt_depth, valid, fg = gt["depth"], gt["valid"], gt["fg"] >= 0.5
                err = np.abs(depth_gt_equiv - gt_depth) / radius
                eroded2 = binary_erosion(fg, iterations=2)
                band = valid & fg & ~eroded2
                interior = valid & fg & eroded2
                if band.any():
                    band_errs.append(float(err[band].mean()))
                if interior.any():
                    interior_errs.append(float(err[interior].mean()))
                if valid.any():
                    all_errs.append(float(err[valid].mean()))
            row["variants"][variant] = {
                "mean_depth_error_scene_radii": float(np.mean(all_errs)) if all_errs else float("nan"),
                "boundary_band_depth_error_scene_radii": float(np.mean(band_errs)) if band_errs else float("nan"),
                "interior_depth_error_scene_radii": float(np.mean(interior_errs)) if interior_errs else float("nan"),
            }
            row["variants"][variant]["boundary_enrichment"] = (
                row["variants"][variant]["boundary_band_depth_error_scene_radii"]
                / row["variants"][variant]["interior_depth_error_scene_radii"]
                if row["variants"][variant]["interior_depth_error_scene_radii"] not in (0, float("nan"))
                else float("nan")
            )
        per_scene.append(row)
        common.log(
            f"E2 {name}: full depth err={row['variants']['full']['mean_depth_error_scene_radii']:.4f}, "
            f"w4a4 depth err={row['variants']['w4a4']['mean_depth_error_scene_radii']:.4f}"
        )

    def mean_of(key, variant):
        return float(np.mean([r["variants"][variant][key] for r in per_scene]))

    return {
        "description": "Rendered-depth error on held-out views vs GT depth (normalized by scene radius), full vs w4a4, split by object-boundary band vs interior.",
        "expect": "depth error concentrated in the same boundary regions the region-characterization analysis flagged",
        "n_scenes": len(per_scene),
        "mean_depth_error_full": mean_of("mean_depth_error_scene_radii", "full"),
        "mean_depth_error_w4a4": mean_of("mean_depth_error_scene_radii", "w4a4"),
        "mean_boundary_enrichment_full": mean_of("boundary_enrichment", "full"),
        "mean_boundary_enrichment_w4a4": mean_of("boundary_enrichment", "w4a4"),
        "per_scene": per_scene,
    }


TESTS = [
    ("E1", test_e1_chamfer_distance),
    ("E3", test_e3_floater_opacity_audit),
    ("E2", test_e2_rendered_depth_error),
]
