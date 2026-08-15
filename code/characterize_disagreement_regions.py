#!/usr/bin/env python3
"""Characterize where Full-vs-W4A4 disagreement and quantization excess error occur.

CPU-only follow-up to analyze_full_dataset_disagreement.py. It reuses the frozen
40-scene manifest and the already-completed/validated 1330 Full + 1330 W4A4
predictions. No VGGT model or CUDA initialization occurs.

The analysis measures overlapping, explicitly defined region cues:
  * object_boundary: 2-pixel foreground boundary band
  * rgb_edge: top-decile grayscale gradient magnitude
  * corner: top-decile positive Harris corner response
  * low_texture: bottom quintile local grayscale standard deviation
  * high_texture: top quintile local grayscale standard deviation
  * thin_structure: morphology-based narrow/protrusion heuristic
  * high_depth_gradient: top-decile valid-neighborhood GT log-depth gradient
  * smooth_flat: eroded object interior + low texture + low GT depth gradient

For each cue, the primary outputs are:
  coverage                P(cue)
  top disagreement share  P(cue | top-10% disagreement)
  top excess share        P(cue | top-10% excess error)
  enrichment ratios       conditional share / baseline coverage
  excess-error mass share sum(excess in cue) / sum(excess globally)
  mean-excess enrichment  mean(excess in cue) / mean(excess globally)

Cues overlap by design, so their percentages are not expected to sum to 100%.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    binary_opening,
    gaussian_filter,
    sobel,
    uniform_filter,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import analyze_full_dataset_disagreement as base  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPECTED_MANIFEST_SHA = base.EXPECTED_MANIFEST_SHA
EXPECTED_SCENES = base.EXPECTED_SCENES
EXPECTED_GROUPS = base.EXPECTED_GROUPS
EXPECTED_PRIMARY_FRAMES = base.EXPECTED_PRIMARY_FRAMES
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260815

CUES = (
    "object_boundary",
    "rgb_edge",
    "corner",
    "low_texture",
    "high_texture",
    "thin_structure",
    "high_depth_gradient",
    "smooth_flat",
)

CUE_LABELS = {
    "object_boundary": "Object boundary",
    "rgb_edge": "RGB edge",
    "corner": "Harris corner",
    "low_texture": "Low texture",
    "high_texture": "High texture",
    "thin_structure": "Thin/protrusion heuristic",
    "high_depth_gradient": "High GT depth gradient",
    "smooth_flat": "Smooth/flat interior",
}

CUE_DEFINITIONS = {
    "object_boundary": "Valid foreground pixels removed by a 2-pixel binary erosion.",
    "rgb_edge": "Top 10% grayscale Sobel-gradient magnitude among valid foreground pixels in each frame.",
    "corner": "Top 10% positive Harris corner response among valid foreground pixels in each frame (k=0.04, Gaussian sigma=1.5).",
    "low_texture": "Bottom 20% local grayscale standard deviation (9x9 window) among valid foreground pixels in each frame.",
    "high_texture": "Top 20% local grayscale standard deviation (9x9 window) among valid foreground pixels in each frame.",
    "thin_structure": "Valid pixels surviving 2-pixel erosion but removed by radius-4 disk binary opening; a morphology heuristic for narrow protrusions/structures, not a semantic label.",
    "high_depth_gradient": "Top 10% GT log-depth gradient magnitude, using only pixels whose left/right/up/down depth neighbors are all valid.",
    "smooth_flat": "2-pixel-eroded object interior that is low-texture and lies in the bottom 50% valid GT log-depth gradient magnitude.",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("/var/tmp/luli38se/quantsplat/disagreement_dataset_v1"),
    )
    p.add_argument(
        "--co3d-root",
        type=Path,
        default=Path("/var/tmp/luli38se/co3d_single_all"),
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/home/utn/luli38se/cv/3D-Reconstruction"),
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--verify-file-sha",
        action="store_true",
        help="Re-hash all 2660 prediction NPZ files. Normally unnecessary because the completed strict dataset analysis already validated them.",
    )
    return p.parse_args()


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else float("nan")


def finite_quantile(x: np.ndarray, q: float) -> float:
    v = np.asarray(x)[np.isfinite(x)]
    if len(v) == 0:
        return float("nan")
    return float(np.quantile(v, q))


def disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def grayscale(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    return (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]).astype(np.float32)


def robust_depth_gradient(depth: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference log-depth gradient requiring all four neighbors valid."""
    logd = np.full(depth.shape, np.nan, dtype=np.float32)
    good = valid & np.isfinite(depth) & (depth > 0)
    logd[good] = np.log(np.maximum(depth[good], 1e-12))
    gx = np.full(depth.shape, np.nan, dtype=np.float32)
    gy = np.full(depth.shape, np.nan, dtype=np.float32)

    vx = np.zeros_like(good)
    vy = np.zeros_like(good)
    vx[:, 1:-1] = good[:, :-2] & good[:, 1:-1] & good[:, 2:]
    vy[1:-1, :] = good[:-2, :] & good[1:-1, :] & good[2:, :]
    gx_mid = 0.5 * (logd[:, 2:] - logd[:, :-2])
    gy_mid = 0.5 * (logd[2:, :] - logd[:-2, :])
    gx[:, 1:-1] = np.where(vx[:, 1:-1], gx_mid, np.nan)
    gy[1:-1, :] = np.where(vy[1:-1, :], gy_mid, np.nan)
    gv = good & np.isfinite(gx) & np.isfinite(gy)
    grad = np.full(depth.shape, np.nan, dtype=np.float32)
    grad[gv] = np.hypot(gx[gv], gy[gv])
    return grad, gv


def cue_masks(rgb: np.ndarray, gt: dict[str, Any], valid: np.ndarray) -> dict[str, np.ndarray]:
    fg = gt["fg"] >= 0.5
    valid = valid & fg
    gray = grayscale(rgb)

    # Object boundary / interior.
    eroded2 = binary_erosion(fg, iterations=2)
    object_boundary = valid & ~eroded2

    # Image gradient cue.
    gx = sobel(gray, axis=1, mode="reflect") / 8.0
    gy = sobel(gray, axis=0, mode="reflect") / 8.0
    rgb_grad = np.hypot(gx, gy)
    edge_thr = finite_quantile(rgb_grad[valid], 0.90)
    rgb_edge = valid & np.isfinite(rgb_grad) & (rgb_grad >= edge_thr)

    # Harris corner cue.
    sxx = gaussian_filter(gx * gx, sigma=1.5, mode="reflect")
    syy = gaussian_filter(gy * gy, sigma=1.5, mode="reflect")
    sxy = gaussian_filter(gx * gy, sigma=1.5, mode="reflect")
    harris = sxx * syy - sxy * sxy - 0.04 * (sxx + syy) ** 2
    positive = valid & np.isfinite(harris) & (harris > 0)
    corner = np.zeros_like(valid)
    if positive.any():
        thr = finite_quantile(harris[positive], 0.90)
        corner = positive & (harris >= thr)

    # Local texture cue.
    mu = uniform_filter(gray, size=9, mode="reflect")
    mu2 = uniform_filter(gray * gray, size=9, mode="reflect")
    tex = np.sqrt(np.maximum(mu2 - mu * mu, 0.0))
    q20 = finite_quantile(tex[valid], 0.20)
    q80 = finite_quantile(tex[valid], 0.80)
    low_texture = valid & (tex <= q20)
    high_texture = valid & (tex >= q80)

    # Narrow/protrusion heuristic. Exclude the ordinary 2-pixel boundary band.
    opened4 = binary_opening(fg, structure=disk(4))
    thin_structure = valid & eroded2 & ~opened4

    # Robust GT depth-gradient cue.
    dgrad, dvalid = robust_depth_gradient(gt["depth"], gt["valid"] & fg)
    high_depth_gradient = np.zeros_like(valid)
    low_depth_gradient = np.zeros_like(valid)
    dv = valid & dvalid
    if dv.any():
        q90d = finite_quantile(dgrad[dv], 0.90)
        q50d = finite_quantile(dgrad[dv], 0.50)
        high_depth_gradient = dv & (dgrad >= q90d)
        low_depth_gradient = dv & (dgrad <= q50d)

    smooth_flat = valid & eroded2 & low_texture & low_depth_gradient

    return {
        "object_boundary": object_boundary,
        "rgb_edge": rgb_edge,
        "corner": corner,
        "low_texture": low_texture,
        "high_texture": high_texture,
        "thin_structure": thin_structure,
        "high_depth_gradient": high_depth_gradient,
        "smooth_flat": smooth_flat,
    }


def blank_accumulator() -> dict[str, Any]:
    return {
        "valid_count": 0,
        "top_d_count": 0,
        "top_ex_count": 0,
        "global_excess_sum": 0.0,
        "global_disagreement_sum": 0.0,
        "cues": {
            cue: {
                "cue_count": 0,
                "cue_top_d_count": 0,
                "cue_top_ex_count": 0,
                "cue_excess_sum": 0.0,
                "cue_disagreement_sum": 0.0,
            }
            for cue in CUES
        },
    }


def add_frame(
    acc: dict[str, Any],
    valid: np.ndarray,
    disagreement: np.ndarray,
    excess: np.ndarray,
    masks: dict[str, np.ndarray],
) -> None:
    flat = np.flatnonzero(valid.reshape(-1))
    if len(flat) == 0:
        return
    dv = disagreement.reshape(-1)[flat]
    ev = excess.reshape(-1)[flat]
    high_d_local = base.exact_top_mask(dv, 0.10)
    high_ex_local = base.exact_top_mask(ev, 0.10)
    high_d = np.zeros(valid.size, dtype=bool)
    high_ex = np.zeros(valid.size, dtype=bool)
    high_d[flat] = high_d_local
    high_ex[flat] = high_ex_local
    high_d = high_d.reshape(valid.shape)
    high_ex = high_ex.reshape(valid.shape)

    acc["valid_count"] += int(valid.sum())
    acc["top_d_count"] += int(high_d.sum())
    acc["top_ex_count"] += int(high_ex.sum())
    acc["global_excess_sum"] += float(np.sum(excess[valid]))
    acc["global_disagreement_sum"] += float(np.sum(disagreement[valid]))

    for cue in CUES:
        m = valid & masks[cue]
        c = acc["cues"][cue]
        c["cue_count"] += int(m.sum())
        c["cue_top_d_count"] += int((m & high_d).sum())
        c["cue_top_ex_count"] += int((m & high_ex).sum())
        c["cue_excess_sum"] += float(np.sum(excess[m])) if m.any() else 0.0
        c["cue_disagreement_sum"] += float(np.sum(disagreement[m])) if m.any() else 0.0


def finalize_accumulator(acc: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "valid_count": int(acc["valid_count"]),
        "top_d_count": int(acc["top_d_count"]),
        "top_ex_count": int(acc["top_ex_count"]),
        "global_excess_sum": float(acc["global_excess_sum"]),
        "global_disagreement_sum": float(acc["global_disagreement_sum"]),
        "cues": {},
    }
    n = acc["valid_count"]
    topd = acc["top_d_count"]
    tope = acc["top_ex_count"]
    global_mean_ex = safe_div(acc["global_excess_sum"], n)
    global_mean_d = safe_div(acc["global_disagreement_sum"], n)
    for cue in CUES:
        c = acc["cues"][cue]
        coverage = safe_div(c["cue_count"], n)
        topd_share = safe_div(c["cue_top_d_count"], topd)
        tope_share = safe_div(c["cue_top_ex_count"], tope)
        mean_ex = safe_div(c["cue_excess_sum"], c["cue_count"])
        mean_d = safe_div(c["cue_disagreement_sum"], c["cue_count"])
        out["cues"][cue] = {
            **{k: int(v) if k.endswith("count") else float(v) for k, v in c.items()},
            "coverage": coverage,
            "top_disagreement_share": topd_share,
            "top_excess_share": tope_share,
            "top_disagreement_enrichment": safe_div(topd_share, coverage),
            "top_excess_enrichment": safe_div(tope_share, coverage),
            "excess_mass_share": safe_div(c["cue_excess_sum"], acc["global_excess_sum"]),
            "disagreement_mass_share": safe_div(c["cue_disagreement_sum"], acc["global_disagreement_sum"]),
            "mean_excess_enrichment": safe_div(mean_ex, global_mean_ex),
            "mean_disagreement_enrichment": safe_div(mean_d, global_mean_d),
        }
    return out


def load_scene_radius_map(experiment_root: Path) -> dict[tuple[str, str], float]:
    path = experiment_root / "run/results/per_scene.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected completed dataset-wide per-scene result at {path}. "
            "Run analyze_full_dataset_disagreement.py first."
        )
    out: dict[tuple[str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(row["category"], row["sequence"])] = float(row["scene_radius"])
    return out


def process_scene(job: dict[str, Any]) -> dict[str, Any]:
    scene = job["scene"]
    exp = Path(job["experiment_root"])
    co3d = Path(job["co3d_root"])
    repo = Path(job["repo_root"])
    verify_file_sha = bool(job["verify_file_sha"])
    radius = float(job["scene_radius"])

    cat = str(scene["category"])
    seq = str(scene.get("sequence") or scene.get("sequence_name"))
    pred_root = exp / "run/predictions"
    pad_transform = base.import_pad_transform(repo)

    frame_records = base.load_jgz(co3d / cat / "frame_annotations.jgz")
    records = {
        int(r["frame_number"]): r
        for r in frame_records
        if r.get("sequence_name") == seq
    }
    acc = blank_accumulator()
    frames_analyzed = 0
    groups_analyzed = 0

    for group in scene["groups"]:
        full, quant, _, _ = base.load_prediction_pair(
            pred_root, scene, group, skip_file_sha=not verify_file_sha
        )
        frames = [int(x) for x in group["frame_numbers"]]
        Egt = np.stack([base.co3d_to_opencv(records[f])[0] for f in frames])
        Ef = full["extrinsic"].astype(np.float64)
        Eq = quant["extrinsic"].astype(np.float64)
        sf, Af, bf = base.umeyama(base.centers(Egt), base.centers(Ef))
        sq, Aq, bq = base.umeyama(base.centers(Egt), base.centers(Eq))
        PF = base.pred_to_gt(full["world_points_from_depth"].astype(np.float64), sf, Af, bf)
        PQ = base.pred_to_gt(quant["world_points_from_depth"].astype(np.float64), sq, Aq, bq)

        for frame in [int(x) for x in group["primary_frame_numbers"]]:
            i = frames.index(frame)
            gt = base.decode_gt_frame(records[frame], co3d, pad_transform)
            Pgt = gt["world"].astype(np.float64)
            valid = (
                gt["valid"]
                & np.isfinite(Pgt).all(axis=2)
                & np.isfinite(PF[i]).all(axis=2)
                & np.isfinite(PQ[i]).all(axis=2)
            )
            if not valid.any():
                raise RuntimeError(f"No valid pixels: {cat}/{seq}/frame {frame}")

            ferr = np.linalg.norm(PF[i] - Pgt, axis=2) / radius
            qerr = np.linalg.norm(PQ[i] - Pgt, axis=2) / radius
            disagreement = np.linalg.norm(PQ[i] - PF[i], axis=2) / radius
            excess = np.maximum(qerr - ferr, 0.0)
            rgb = base.rgb_518(gt["image_path"], gt["transform"])
            masks = cue_masks(rgb, gt, valid)
            add_frame(acc, valid, disagreement, excess, masks)
            frames_analyzed += 1

        groups_analyzed += 1

    out = finalize_accumulator(acc)
    out.update({
        "category": cat,
        "sequence": seq,
        "frames_analyzed": frames_analyzed,
        "groups_analyzed": groups_analyzed,
        "scene_radius": radius,
    })
    return out


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [float("nan"), float("nan")]
    n = len(values)
    stats = np.empty(BOOTSTRAP_RESAMPLES, np.float64)
    for i in range(BOOTSTRAP_RESAMPLES):
        stats[i] = np.median(values[rng.integers(0, n, size=n)])
    return [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))]


def summarize(per_scene: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Pooled raw accumulator reconstructed from scene-level raw totals.
    pooled = blank_accumulator()
    for s in per_scene:
        pooled["valid_count"] += s["valid_count"]
        pooled["top_d_count"] += s["top_d_count"]
        pooled["top_ex_count"] += s["top_ex_count"]
        pooled["global_excess_sum"] += s["global_excess_sum"]
        pooled["global_disagreement_sum"] += s["global_disagreement_sum"]
        for cue in CUES:
            for key in pooled["cues"][cue]:
                pooled["cues"][cue][key] += s["cues"][cue][key]
    pooled_final = finalize_accumulator(pooled)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for cue in CUES:
        scene_metrics = [s["cues"][cue] for s in per_scene]
        def arr(key: str) -> np.ndarray:
            return np.asarray([m[key] for m in scene_metrics], np.float64)
        ex_en = arr("top_excess_enrichment")
        d_en = arr("top_disagreement_enrichment")
        mean_ex = arr("mean_excess_enrichment")
        pooled_c = pooled_final["cues"][cue]
        d_ci = bootstrap_median_ci(d_en, rng)
        ex_ci = bootstrap_median_ci(ex_en, rng)
        mean_ex_ci = bootstrap_median_ci(mean_ex, rng)
        row = {
            "cue": cue,
            "label": CUE_LABELS[cue],
            "definition": CUE_DEFINITIONS[cue],
            "median_scene_coverage": float(np.nanmedian(arr("coverage"))),
            "median_scene_top_disagreement_share": float(np.nanmedian(arr("top_disagreement_share"))),
            "median_scene_top_disagreement_enrichment": float(np.nanmedian(d_en)),
            "top_disagreement_enrichment_ci95_low": d_ci[0],
            "top_disagreement_enrichment_ci95_high": d_ci[1],
            "fraction_scenes_top_disagreement_enrichment_gt1": float(np.mean(d_en[np.isfinite(d_en)] > 1.0)),
            "median_scene_top_excess_share": float(np.nanmedian(arr("top_excess_share"))),
            "median_scene_top_excess_enrichment": float(np.nanmedian(ex_en)),
            "top_excess_enrichment_ci95_low": ex_ci[0],
            "top_excess_enrichment_ci95_high": ex_ci[1],
            "fraction_scenes_top_excess_enrichment_gt1": float(np.mean(ex_en[np.isfinite(ex_en)] > 1.0)),
            "median_scene_excess_mass_share": float(np.nanmedian(arr("excess_mass_share"))),
            "median_scene_mean_excess_enrichment": float(np.nanmedian(mean_ex)),
            "mean_excess_enrichment_ci95_low": mean_ex_ci[0],
            "mean_excess_enrichment_ci95_high": mean_ex_ci[1],
            "pooled_coverage": pooled_c["coverage"],
            "pooled_top_disagreement_share": pooled_c["top_disagreement_share"],
            "pooled_top_disagreement_enrichment": pooled_c["top_disagreement_enrichment"],
            "pooled_top_excess_share": pooled_c["top_excess_share"],
            "pooled_top_excess_enrichment": pooled_c["top_excess_enrichment"],
            "pooled_excess_mass_share": pooled_c["excess_mass_share"],
            "pooled_mean_excess_enrichment": pooled_c["mean_excess_enrichment"],
        }
        rows.append(row)

    ranked_ex = sorted(rows, key=lambda r: r["median_scene_top_excess_enrichment"], reverse=True)
    ranked_d = sorted(rows, key=lambda r: r["median_scene_top_disagreement_enrichment"], reverse=True)
    consistent_ex = [
        r for r in ranked_ex
        if r["top_excess_enrichment_ci95_low"] > 1.0
        and r["fraction_scenes_top_excess_enrichment_gt1"] >= 0.60
    ]
    consistent_d = [
        r for r in ranked_d
        if r["top_disagreement_enrichment_ci95_low"] > 1.0
        and r["fraction_scenes_top_disagreement_enrichment_gt1"] >= 0.60
    ]

    summary = {
        "status": "PASS",
        "scenes": len(per_scene),
        "primary_frames": int(sum(s["frames_analyzed"] for s in per_scene)),
        "groups": int(sum(s["groups_analyzed"] for s in per_scene)),
        "cues_are_overlapping": True,
        "top_region_fraction": 0.10,
        "consistent_high_excess_cues": [
            {
                "cue": r["cue"],
                "label": r["label"],
                "median_enrichment": r["median_scene_top_excess_enrichment"],
                "ci95": [r["top_excess_enrichment_ci95_low"], r["top_excess_enrichment_ci95_high"]],
                "fraction_scenes_gt1": r["fraction_scenes_top_excess_enrichment_gt1"],
            }
            for r in consistent_ex
        ],
        "consistent_high_disagreement_cues": [
            {
                "cue": r["cue"],
                "label": r["label"],
                "median_enrichment": r["median_scene_top_disagreement_enrichment"],
                "ci95": [r["top_disagreement_enrichment_ci95_low"], r["top_disagreement_enrichment_ci95_high"]],
                "fraction_scenes_gt1": r["fraction_scenes_top_disagreement_enrichment_gt1"],
            }
            for r in consistent_d
        ],
        "ranked_by_high_excess_enrichment": [r["cue"] for r in ranked_ex],
        "ranked_by_high_disagreement_enrichment": [r["cue"] for r in ranked_d],
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def flatten_scene_rows(per_scene: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for s in per_scene:
        for cue in CUES:
            m = s["cues"][cue]
            rows.append({
                "category": s["category"],
                "sequence": s["sequence"],
                "frames_analyzed": s["frames_analyzed"],
                "groups_analyzed": s["groups_analyzed"],
                "cue": cue,
                "label": CUE_LABELS[cue],
                "coverage": m["coverage"],
                "top_disagreement_share": m["top_disagreement_share"],
                "top_disagreement_enrichment": m["top_disagreement_enrichment"],
                "top_excess_share": m["top_excess_share"],
                "top_excess_enrichment": m["top_excess_enrichment"],
                "excess_mass_share": m["excess_mass_share"],
                "mean_excess_enrichment": m["mean_excess_enrichment"],
            })
    return rows


def make_plots(outdir: Path, rows: list[dict[str, Any]]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: r["median_scene_top_excess_enrichment"])
    labels = [r["label"] for r in ordered]
    vals = np.array([r["median_scene_top_excess_enrichment"] for r in ordered])
    lo = np.array([r["top_excess_enrichment_ci95_low"] for r in ordered])
    hi = np.array([r["top_excess_enrichment_ci95_high"] for r in ordered])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(vals, y, xerr=np.vstack([vals - lo, hi - vals]), fmt="o", capsize=3)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Median scene enrichment in top-10% excess-error pixels")
    ax.set_title("Where quantization-induced excess error concentrates")
    fig.tight_layout()
    fig.savefig(outdir / "high_excess_enrichment.png", dpi=180)
    plt.close(fig)

    ordered = sorted(rows, key=lambda r: r["median_scene_top_disagreement_enrichment"])
    labels = [r["label"] for r in ordered]
    vals = np.array([r["median_scene_top_disagreement_enrichment"] for r in ordered])
    lo = np.array([r["top_disagreement_enrichment_ci95_low"] for r in ordered])
    hi = np.array([r["top_disagreement_enrichment_ci95_high"] for r in ordered])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(vals, y, xerr=np.vstack([vals - lo, hi - vals]), fmt="o", capsize=3)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Median scene enrichment in top-10% Full-W4A4 disagreement pixels")
    ax.set_title("Where Full-W4A4 geometric disagreement concentrates")
    fig.tight_layout()
    fig.savefig(outdir / "high_disagreement_enrichment.png", dpi=180)
    plt.close(fig)

    # Coverage vs error-region share: points above diagonal are enriched.
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in rows:
        x = 100 * r["median_scene_coverage"]
        yv = 100 * r["median_scene_top_excess_share"]
        ax.scatter([x], [yv])
        ax.annotate(r["label"], (x, yv), xytext=(4, 4), textcoords="offset points", fontsize=8)
    lim = max([100 * r["median_scene_coverage"] for r in rows] + [100 * r["median_scene_top_excess_share"] for r in rows]) * 1.1
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Region coverage among valid pixels (%)")
    ax.set_ylabel("Share of top-10% excess-error pixels in region (%)")
    ax.set_title("Region prevalence vs quantization-damage prevalence")
    fig.tight_layout()
    fig.savefig(outdir / "coverage_vs_high_excess_share.png", dpi=180)
    plt.close(fig)


def pct(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{100*x:.1f}%"


def write_report(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    ranked = sorted(rows, key=lambda r: r["median_scene_top_excess_enrichment"], reverse=True)
    lines = [
        "# Full-vs-W4A4 Disagreement Region Characterization",
        "",
        "## Main question",
        "",
        "Which simple visual/geometric region types are disproportionately associated with the already-validated Full-vs-W4A4 geometric disagreement and quantization-induced excess geometric error?",
        "",
        f"This CPU-only analysis reuses **{summary['scenes']} scenes**, **{summary['primary_frames']} primary frames**, and **{summary['groups']} completed six-view prediction groups**. It does not rerun Full VGGT or W4A4.",
        "",
        "## How to read the table",
        "",
        "`Coverage` is the fraction of all valid GT foreground pixels belonging to the cue. `Top-excess share` is the fraction of the top 10% quantization-excess-error pixels belonging to that cue. `Enrichment = top-excess share / coverage`; values above 1 mean the cue is over-represented in high quantization damage. Cues overlap, so percentages do **not** sum to 100%.",
        "",
        "## Dataset-wide region result",
        "",
        "| Cue | Median coverage | Median top-excess share | Median excess enrichment | 95% scene-bootstrap CI | Scenes >1x | Median disagreement enrichment |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ranked:
        lines.append(
            f"| {r['label']} | {pct(r['median_scene_coverage'])} | {pct(r['median_scene_top_excess_share'])} | "
            f"{r['median_scene_top_excess_enrichment']:.2f}x | "
            f"[{r['top_excess_enrichment_ci95_low']:.2f}, {r['top_excess_enrichment_ci95_high']:.2f}] | "
            f"{pct(r['fraction_scenes_top_excess_enrichment_gt1'])} | "
            f"{r['median_scene_top_disagreement_enrichment']:.2f}x |"
        )

    lines += [
        "",
        "## Consistent cues",
        "",
    ]
    if summary["consistent_high_excess_cues"]:
        lines.append("Cues whose median high-excess enrichment has a 95% scene-bootstrap CI entirely above 1.0 and is above 1.0 in at least 60% of scenes:")
        lines.append("")
        for x in summary["consistent_high_excess_cues"]:
            lines.append(
                f"- **{x['label']}**: median {x['median_enrichment']:.2f}x, 95% CI [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}], above 1x in {pct(x['fraction_scenes_gt1'])} of scenes."
            )
    else:
        lines.append("No tested simple cue met the predeclared consistency rule for high excess error. This would support the interpretation that no single handcrafted image-space cue explains the quantization failures dataset-wide.")

    lines += [
        "",
        "## Cue definitions",
        "",
    ]
    for cue in CUES:
        lines.append(f"- **{CUE_LABELS[cue]}**: {CUE_DEFINITIONS[cue]}")

    lines += [
        "",
        "## Interpretation constraint",
        "",
        "This analysis is descriptive/associational. It identifies where disagreement and excess error are enriched; it does not establish that an edge, corner, texture state, or morphology cue causes the quantization failure.",
        "",
        "Because cues overlap, do not interpret the region percentages as a partition of all errors. The primary comparison is each cue's enrichment relative to its own baseline pixel coverage.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    exp = args.experiment_root
    manifest_path = exp / "frozen_dataset_manifest.json"
    actual_sha = base.sha256_file(manifest_path)
    if actual_sha != EXPECTED_MANIFEST_SHA:
        raise RuntimeError(f"Frozen manifest SHA mismatch: {actual_sha}")
    manifest = json.loads(manifest_path.read_text())
    scenes = manifest["scenes"]
    groups = sum(len(s["groups"]) for s in scenes)
    primary = sum(len(g["primary_frame_numbers"]) for s in scenes for g in s["groups"])
    if len(scenes) != EXPECTED_SCENES or groups != EXPECTED_GROUPS or primary != EXPECTED_PRIMARY_FRAMES:
        raise RuntimeError(
            f"Frozen denominator mismatch: scenes={len(scenes)} groups={groups} primary={primary}"
        )

    radius_map = load_scene_radius_map(exp)
    for s in scenes:
        key = (str(s["category"]), str(s.get("sequence") or s.get("sequence_name")))
        if key not in radius_map:
            raise RuntimeError(f"Missing scene radius from completed analysis: {key}")

    out = exp / "run/region_characterization"
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "progress.json"
    started = time.time()

    jobs = []
    for i, scene in enumerate(scenes):
        key = (str(scene["category"]), str(scene.get("sequence") or scene.get("sequence_name")))
        jobs.append({
            "index": i,
            "scene": scene,
            "scene_radius": radius_map[key],
            "experiment_root": str(exp),
            "co3d_root": str(args.co3d_root),
            "repo_root": str(args.repo_root),
            "verify_file_sha": bool(args.verify_file_sha),
        })

    results_by_index: dict[int, dict[str, Any]] = {}
    workers = max(1, min(int(args.workers), len(jobs)))
    if workers == 1:
        iterable = []
        for job in jobs:
            r = process_scene(job)
            results_by_index[job["index"]] = r
            print(
                f"[{len(results_by_index):02d}/{len(jobs)}] {r['category']}/{r['sequence']} frames={r['frames_analyzed']}",
                flush=True,
            )
            base.write_json_atomic(progress, {
                "status": "RUNNING",
                "scenes_completed": len(results_by_index),
                "scenes_total": len(jobs),
                "elapsed_seconds": time.time() - started,
            })
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            future_to_job = {ex.submit(process_scene, job): job for job in jobs}
            for fut in as_completed(future_to_job):
                job = future_to_job[fut]
                r = fut.result()
                results_by_index[job["index"]] = r
                print(
                    f"[{len(results_by_index):02d}/{len(jobs)}] {r['category']}/{r['sequence']} frames={r['frames_analyzed']}",
                    flush=True,
                )
                base.write_json_atomic(progress, {
                    "status": "RUNNING",
                    "scenes_completed": len(results_by_index),
                    "scenes_total": len(jobs),
                    "elapsed_seconds": time.time() - started,
                })

    per_scene = [results_by_index[i] for i in range(len(jobs))]
    if sum(r["frames_analyzed"] for r in per_scene) != EXPECTED_PRIMARY_FRAMES:
        raise RuntimeError("Primary-frame denominator changed during region characterization")
    if sum(r["groups_analyzed"] for r in per_scene) != EXPECTED_GROUPS:
        raise RuntimeError("Group denominator changed during region characterization")

    rows, summary = summarize(per_scene)
    summary.update({
        "manifest_sha256": actual_sha,
        "workers": workers,
        "prediction_file_sha_reverified": bool(args.verify_file_sha),
        "elapsed_seconds": time.time() - started,
        "cue_definitions": CUE_DEFINITIONS,
    })

    base.write_json_atomic(out / "dataset_region_summary.json", summary)
    write_csv(out / "dataset_region_summary.csv", rows)
    write_csv(out / "per_scene_region_summary.csv", flatten_scene_rows(per_scene))
    base.write_json_atomic(out / "per_scene_region_summary.json", per_scene)
    make_plots(out / "figures", rows)
    write_report(out / "REGION_CHARACTERIZATION.md", rows, summary)
    base.write_json_atomic(progress, {
        "status": "PASS",
        "scenes_completed": len(per_scene),
        "scenes_total": len(per_scene),
        "primary_frames": EXPECTED_PRIMARY_FRAMES,
        "groups": EXPECTED_GROUPS,
        "elapsed_seconds": time.time() - started,
    })

    print("\n===== REGION CHARACTERIZATION COMPLETE =====")
    print(json.dumps({
        "status": "PASS",
        "scenes": summary["scenes"],
        "primary_frames": summary["primary_frames"],
        "groups": summary["groups"],
        "consistent_high_excess_cues": summary["consistent_high_excess_cues"],
        "consistent_high_disagreement_cues": summary["consistent_high_disagreement_cues"],
        "summary": str(out / "dataset_region_summary.json"),
        "report": str(out / "REGION_CHARACTERIZATION.md"),
        "elapsed_seconds": summary["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
    