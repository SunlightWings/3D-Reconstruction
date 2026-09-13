#!/usr/bin/env python3
"""Score existing 3DGS held-out renders with the project's exact evaluation protocol.

This is adapted to the current project layout and mirrors render_scoring/score_renders.py:

- headline metrics are RAW PSNR / SSIM / LPIPS
- foreground is the PRIMARY mask
- content is the SECONDARY mask
- PSNR uses the exact binary mask
- SSIM and LPIPS use the mask bounding box
- LPIPS uses AlexNet on CPU, inputs in [-1, 1]
- paired statistics are over SCENES, not individual held-out views
- optional exposure correction uses one per-image, per-channel gain+bias fit with the
  same near-constant-channel guard as the reference scorer

Expected existing project layout per scene:

    <scene_root>/common/heldout_images/frameXXXXXX.png
    <scene_root>/heldout/full_it7000/renders/*.png
    <scene_root>/heldout/w4a4_it7000/renders/*.png
    <scene_root>/heldout/w3a3_it7000/renders/*.png

The script does NOT train or render anything. It only scores renders that already exist.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Project imports / paths
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
REPO = SCRIPT_PATH.parents[2]

CODE_DIR = REPO / "code"
DIAGNOSTICS_DIR = CODE_DIR / "downstream_3dgs" / "diagnostics"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(DIAGNOSTICS_DIR))

import common  # noqa: E402

dv = common.dv

FROZEN = REPO / "datasets" / "frozen_dataset_manifest.json"
DEFAULT_OUT_DIR = Path("/var/tmp/poli22wo/quantsplat/oracle_sweep/results")

RAW_METRICS = ("psnr", "ssim", "lpips")

_LPIPS_NET = None


# ---------------------------------------------------------------------------
# Scene selection
# ---------------------------------------------------------------------------

def frozen_scenes() -> list[str]:
    manifest = json.loads(FROZEN.read_text())
    return [
        f"{scene['category']}/{scene['sequence']}"
        for scene in manifest["scenes"]
    ]


def selected_scenes(which: str) -> list[str]:
    if which == "subset":
        return list(common.EXPENSIVE_SCENES)
    return frozen_scenes()


# ---------------------------------------------------------------------------
# Metric helpers: match render_scoring/score_renders.py
# ---------------------------------------------------------------------------

def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def bbox_from_mask(mask: np.ndarray) -> tuple[slice, slice]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise RuntimeError("empty mask")
    return (
        slice(int(ys.min()), int(ys.max()) + 1),
        slice(int(xs.min()), int(xs.max()) + 1),
    )


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Masked per-pixel PSNR."""
    mse = float(((a - b) ** 2)[mask].mean())
    if mse <= 1e-15:
        return float("inf")
    return -10.0 * math.log10(mse)


def ssim_value(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """SSIM on the tight bounding box of the mask."""
    from skimage.metrics import structural_similarity

    ys, xs = bbox_from_mask(mask)
    return float(
        structural_similarity(
            a[ys, xs],
            b[ys, xs],
            channel_axis=2,
            data_range=1.0,
        )
    )


def lpips_value(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """AlexNet LPIPS on CPU, evaluated on the mask bounding box."""
    global _LPIPS_NET

    import torch

    if _LPIPS_NET is None:
        import lpips as lpips_lib

        _LPIPS_NET = lpips_lib.LPIPS(net="alex").cpu().eval()

    ys, xs = bbox_from_mask(mask)

    ta = (
        torch.from_numpy(np.ascontiguousarray(a[ys, xs]))
        .permute(2, 0, 1)[None]
        .float()
        * 2.0
        - 1.0
    )
    tb = (
        torch.from_numpy(np.ascontiguousarray(b[ys, xs]))
        .permute(2, 0, 1)[None]
        .float()
        * 2.0
        - 1.0
    )

    with torch.no_grad():
        return float(_LPIPS_NET(ta, tb).item())


def gain_bias(render: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-channel least-squares gain+bias fitted on masked pixels.

    The near-constant-channel guard is intentional and matches the supplied scorer.
    """
    out = render.copy()

    for c in range(3):
        x = render[..., c][mask]
        y = gt[..., c][mask]

        if len(x) < 10 or float(np.var(x)) < 1e-12:
            continue

        design = np.stack([x, np.ones_like(x)], axis=1)
        sol, *_ = np.linalg.lstsq(design, y, rcond=None)

        out[..., c] = np.clip(
            render[..., c] * sol[0] + sol[1],
            0.0,
            1.0,
        )

    return out


def paired_stats(deltas: list[float]) -> dict:
    """Paired t-test over scenes."""
    from scipy import stats

    d = np.asarray(
        [x for x in deltas if np.isfinite(x)],
        dtype=np.float64,
    )
    n = len(d)

    if n == 0:
        return {"n": 0}

    mean = float(d.mean())
    out = {
        "n": n,
        "mean_delta": mean,
        "scenes_positive": int((d > 0).sum()),
    }

    if n >= 2:
        sd = float(d.std(ddof=1))
        se = sd / math.sqrt(n)
        test = stats.ttest_1samp(d, 0.0)
        crit = float(stats.t.ppf(0.975, n - 1))

        out.update(
            p=float(test.pvalue),
            sd=sd,
            ci95=[
                float(mean - crit * se),
                float(mean + crit * se),
            ],
            mde_80pct_power=float(2.802 * sd / math.sqrt(n)),
        )

    return out


# ---------------------------------------------------------------------------
# Scene scoring
# ---------------------------------------------------------------------------

def masks_for_region(
    region: str,
    heldout_frames: list[int],
    records: dict,
) -> list[np.ndarray]:
    if region == "foreground":
        return [
            dv.foreground_mask_from_record(records[f])
            for f in heldout_frames
        ]

    if region == "content":
        return [
            dv.content_mask_from_record(records[f])
            for f in heldout_frames
        ]

    raise ValueError(region)


def score_arm_region(
    gt_paths: list[Path],
    render_paths: list[Path],
    masks: list[np.ndarray],
    *,
    use_lpips: bool,
    exposure: bool,
) -> dict[str, float]:
    metric_fns = {
        "psnr": psnr,
        "ssim": ssim_value,
    }
    if use_lpips:
        metric_fns["lpips"] = lpips_value

    acc: dict[str, list[float]] = {
        name: []
        for name in metric_fns
    }

    if exposure:
        for name in metric_fns:
            acc[f"{name}_exposure_corrected"] = []

    for gt_path, render_path, mask in zip(gt_paths, render_paths, masks):
        if not mask.any():
            continue

        gt = load_rgb(gt_path)
        render = load_rgb(render_path)

        if gt.shape != render.shape:
            raise RuntimeError(
                f"shape mismatch: gt {gt_path} {gt.shape}, "
                f"render {render_path} {render.shape}"
            )

        for metric_name, fn in metric_fns.items():
            acc[metric_name].append(
                fn(render, gt, mask)
            )

        if exposure:
            corrected = gain_bias(render, gt, mask)

            for metric_name, fn in metric_fns.items():
                acc[f"{metric_name}_exposure_corrected"].append(
                    fn(corrected, gt, mask)
                )

    return {
        key: float(np.mean(values))
        for key, values in acc.items()
        if values
    }


def score_scene(
    scene_name: str,
    arms: list[str],
    iteration: int,
    regions: list[str],
    *,
    use_lpips: bool,
    exposure: bool,
) -> dict:
    cat, seq = common.split_scene(scene_name)

    scene_root = common.scene_root(cat, seq)
    records = dv.load_annotations(cat, seq)

    # Use the GT images that actually belong to the existing downstream render bundle.
    # This deliberately avoids depending on scene_manifest(), because this machine's
    # scene manifests were reconstructed during the handoff. The validated standalone
    # scorer also pairs GT and renders by sorted order.
    gt_dir = scene_root / "common" / "heldout_images"
    gt_paths = sorted(gt_dir.glob("frame*.png"))

    if not gt_paths:
        raise RuntimeError(
            f"{scene_name}: no held-out GT images found in {gt_dir}"
        )

    try:
        heldout_frames = [
            int(path.stem.removeprefix("frame"))
            for path in gt_paths
        ]
    except ValueError as exc:
        raise RuntimeError(
            f"{scene_name}: unexpected GT filename in {gt_dir}; "
            "expected frameXXXXXX.png"
        ) from exc

    row = {
        "scene": scene_name,
        "n_views": len(gt_paths),
    }

    masks_by_region = {
        region: masks_for_region(
            region,
            heldout_frames,
            records,
        )
        for region in regions
    }

    for arm in arms:
        render_dir = (
            scene_root
            / "heldout"
            / f"{arm}_it{iteration}"
            / "renders"
        )

        render_paths = sorted(render_dir.glob("*.png"))

        if len(render_paths) != len(gt_paths):
            print(
                f"  skip {scene_name}/{arm}: "
                f"{len(render_paths)} renders vs {len(gt_paths)} GT "
                f"({render_dir})",
                flush=True,
            )
            continue

        for region in regions:
            scores = score_arm_region(
                gt_paths,
                render_paths,
                masks_by_region[region],
                use_lpips=use_lpips,
                exposure=exposure,
            )

            for metric, value in scores.items():
                row[f"{arm}_{region}_{metric}"] = value

    return row


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------

def aggregate(
    rows: list[dict],
    arms: list[str],
    regions: list[str],
    *,
    ref: str,
    use_lpips: bool,
    exposure: bool,
) -> dict:
    metrics = ["psnr", "ssim"]
    if use_lpips:
        metrics.append("lpips")

    keys = list(metrics)
    if exposure:
        keys += [
            f"{metric}_exposure_corrected"
            for metric in metrics
        ]

    summary = {
        "n_scenes_requested": len(rows),
        "ref": ref,
        "arms": {},
        "pairs": {},
        "per_scene": rows,
    }

    for arm in arms:
        arm_summary = {}

        for region in regions:
            region_summary = {}

            for metric in keys:
                key = f"{arm}_{region}_{metric}"
                vals = [
                    row[key]
                    for row in rows
                    if key in row
                ]

                if vals:
                    region_summary[metric] = float(
                        np.mean(vals)
                    )

            if region_summary:
                arm_summary[region] = {
                    "n_scenes": sum(
                        f"{arm}_{region}_psnr" in row
                        for row in rows
                    ),
                    **region_summary,
                }

        if arm_summary:
            summary["arms"][arm] = arm_summary

    if ref in arms:
        for arm in arms:
            if arm == ref:
                continue

            for region in regions:
                for metric in keys:
                    ref_key = f"{ref}_{region}_{metric}"
                    arm_key = f"{arm}_{region}_{metric}"

                    deltas = [
                        row[ref_key] - row[arm_key]
                        for row in rows
                        if ref_key in row and arm_key in row
                    ]

                    if len(deltas) >= 2:
                        pair_key = (
                            f"{ref}_minus_{arm}"
                            f"__{region}"
                            f"__{metric}"
                        )
                        summary["pairs"][pair_key] = paired_stats(
                            deltas
                        )

    return summary


def print_summary(
    summary: dict,
    arms: list[str],
    regions: list[str],
    *,
    ref: str,
    use_lpips: bool,
    exposure: bool,
) -> None:
    metrics = ["psnr", "ssim"]
    if use_lpips:
        metrics.append("lpips")

    print("\n" + "=" * 100)
    print("RAW METRICS — use these as the headline results")

    for region in regions:
        print(f"\nMASK = {region}")

        header = (
            f"{'arm':12s}"
            f"{'n':>5s}"
            f"{'PSNR':>12s}"
            f"{'SSIM':>12s}"
        )
        if use_lpips:
            header += f"{'LPIPS':>12s}"

        print(header)

        for arm in arms:
            block = (
                summary["arms"]
                .get(arm, {})
                .get(region)
            )
            if not block:
                continue

            line = (
                f"{arm:12s}"
                f"{block['n_scenes']:5d}"
                f"{block['psnr']:12.4f}"
                f"{block['ssim']:12.4f}"
            )
            if use_lpips:
                line += f"{block['lpips']:12.4f}"

            print(line)

    print(
        f"\nPaired tests over scenes: {ref} minus arm. "
        f"For PSNR/SSIM, positive means {ref} is better."
    )
    if use_lpips:
        print(
            f"For LPIPS, negative means {ref} is better "
            f"(lower LPIPS is better)."
        )

    for key, stats in summary["pairs"].items():
        # Raw only in the primary console section.
        if "_exposure_corrected" in key:
            continue

        n = stats.get("n", 0)
        if n < 2:
            continue

        p = stats.get("p", float("nan"))
        ci = stats.get("ci95", [float("nan"), float("nan")])
        star = "*" if np.isfinite(p) and p < 0.05 else " "

        print(
            f"  {key:55s} "
            f"mean={stats['mean_delta']:+.4f} "
            f"p={p:.4g}{star} "
            f"CI=[{ci[0]:+.4f},{ci[1]:+.4f}] "
            f"{stats['scenes_positive']}/{n}"
        )

    if exposure:
        print(
            "\nExposure-corrected metrics were also saved to JSON. "
            "Treat them as supplementary; the supplied scorer says to cite the RAW columns."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score existing 3DGS renders using the project's validated metric definitions."
    )

    parser.add_argument(
        "--scenes",
        choices=["subset", "all"],
        default="subset",
        help="subset = the same 8 EXPENSIVE_SCENES used before; all = frozen 40 scenes",
    )

    parser.add_argument(
        "--arms",
        nargs="+",
        default=["full", "w4a4", "w3a3"],
        help="arms to score, e.g. full w4a4 w3a3",
    )

    parser.add_argument(
        "--ref",
        default="full",
        help="reference arm used for paired tests",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=7000,
    )

    parser.add_argument(
        "--regions",
        nargs="+",
        choices=["foreground", "content"],
        default=["foreground", "content"],
        help="foreground is primary; content is secondary",
    )

    parser.add_argument(
        "--exposure",
        action="store_true",
        help="also compute exposure-corrected PSNR/SSIM/LPIPS (supplementary)",
    )

    parser.add_argument(
        "--no-lpips",
        action="store_true",
        help="skip LPIPS",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    scenes = selected_scenes(args.scenes)

    print(
        f"scenes={len(scenes)} ({args.scenes})  "
        f"iteration={args.iterations}  "
        f"arms={args.arms}"
    )

    rows = []

    for i, scene_name in enumerate(scenes, 1):
        print(
            f"\n[{i}/{len(scenes)}] {scene_name}",
            flush=True,
        )

        row = score_scene(
            scene_name,
            args.arms,
            args.iterations,
            args.regions,
            use_lpips=not args.no_lpips,
            exposure=args.exposure,
        )

        rows.append(row)

        available = [
            arm
            for arm in args.arms
            if any(
                key.startswith(f"{arm}_")
                for key in row
            )
        ]

        print(
            f"  scored arms: {', '.join(available) if available else 'none'}",
            flush=True,
        )

    summary = aggregate(
        rows,
        args.arms,
        args.regions,
        ref=args.ref,
        use_lpips=not args.no_lpips,
        exposure=args.exposure,
    )

    summary["scenes"] = args.scenes
    summary["iterations"] = args.iterations
    summary["regions"] = args.regions
    summary["protocol"] = {
        "headline": "raw",
        "primary_mask": "foreground",
        "secondary_mask": "content",
        "psnr_support": "exact mask",
        "ssim_support": "mask bounding box",
        "lpips_support": "mask bounding box",
        "lpips_backbone": "alex",
        "statistics_unit": "scene",
        "pair_direction": f"{args.ref} minus comparison arm",
    }

    if args.out is None:
        DEFAULT_OUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )
        out_path = (
            DEFAULT_OUT_DIR
            / f"score_3dgs_{args.scenes}_it{args.iterations}.json"
        )
    else:
        out_path = args.out
        out_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    out_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print_summary(
        summary,
        args.arms,
        args.regions,
        ref=args.ref,
        use_lpips=not args.no_lpips,
        exposure=args.exposure,
    )

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
