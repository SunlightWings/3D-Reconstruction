"""Run downstream 3DGS for a quantized VGGT arm.

Reuses the existing downstream helpers for:
- source construction
- Sim(3) alignment
- 3DGS training
- held-out rendering
- PSNR / SSIM / LPIPS
- exposure-corrected PSNR
- paired statistics

The existing Full and W4A4 models are reused.
The selected quantized variant (for example W3A3) is prepared and trained here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Paths
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

OUT = Path("/var/tmp/poli22wo/quantsplat/oracle_sweep/results")

ARCHIVE_RENDER_ROOT = Path("/var/tmp/poli22wo/renders7k_8scenes")


def exact_heldout_frames(cat: str, seq: str) -> list[int]:
    """
    Reproduce the exact held-out-view selection used by the original
    Full/W4A4 downstream pipeline.

    For the original 8-scene archive, use the archived GT frame IDs
    directly. For all other scenes, apply the original deterministic
    choose_heldout_frames() logic verbatim.
    """

    # ---------------------------------------------------------------
    # Original 8-scene subset: use the known archived evaluation views
    # ---------------------------------------------------------------

    gt_dir = ARCHIVE_RENDER_ROOT / cat / seq / "gt"

    if gt_dir.is_dir():
        frames = [
            int(p.stem.replace("frame", ""))
            for p in sorted(gt_dir.glob("frame*.png"))
        ]

        if len(frames) != 9:
            raise RuntimeError(
                f"{cat}/{seq}: expected 9 archived held-out frames, "
                f"found {len(frames)}: {frames}"
            )

        return frames

    # ---------------------------------------------------------------
    # Remaining scenes: reproduce original downstream selection
    # ---------------------------------------------------------------

    records = dv.load_annotations(cat, seq)

    # This is the same six-view group used for VGGT inference.
    _group_dir, full_meta, _w4_meta = dv.find_group(cat, seq)

    input_frames = list(
        map(int, full_meta["frame_numbers"])
    )

    candidates = []
    input_set = set(input_frames)

    for frame, record in sorted(records.items()):
        if frame in input_set:
            continue

        if not isinstance(
            record.get("viewpoint"),
            dict,
        ):
            continue

        if not record.get(
            "image",
            {},
        ).get("path"):
            continue

        frame_type = str(
            record.get(
                "meta",
                {},
            ).get(
                "frame_type",
                "",
            )
        )

        if frame_type and "known" not in frame_type:
            continue

        candidates.append(frame)

    if len(candidates) < 9:
        raise RuntimeError(
            f"{cat}/{seq}: only {len(candidates)} "
            f"valid held-out candidates; need 9"
        )

    idx = np.linspace(
        0,
        len(candidates) - 1,
        9,
    )

    picked = sorted(
        {
            candidates[int(round(x))]
            for x in idx
        }
    )

    # Original deterministic duplicate fallback.
    if len(picked) < 9:
        for frame in candidates:
            if frame not in picked:
                picked.append(frame)

                if len(picked) == 9:
                    break

        picked.sort()

    return picked[:9]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def frozen_scenes() -> list[str]:
    """Return the exact 40 scenes from the frozen manifest."""
    manifest = json.loads(FROZEN.read_text())

    return [
        f"{scene['category']}/{scene['sequence']}"
        for scene in manifest["scenes"]
    ]


# ---------------------------------------------------------------------------
# Source construction
# ---------------------------------------------------------------------------

def build_variant_sources(
    cat: str,
    seq: str,
    variant: str,
    stride: int = 4,
):
    """Build train + held-out COLMAP sources for one quantized variant."""

    scene_root = common.scene_root(cat, seq)
    man = common.scene_manifest(cat, seq)

    frames = man["input_frames"]
    heldout = exact_heldout_frames(cat, seq)

    recs = dv.load_annotations(cat, seq)

    group_dir, _full_meta, _w4_meta = dv.find_group(cat, seq)

    npz_path = group_dir / f"{variant}.npz"

    if not npz_path.is_file():
        raise SystemExit(
            f"{cat}/{seq}: missing {npz_path}\n"
            f"Run run_w2a4_inference.py --variant {variant} first."
        )

    arrays = dv.load_npz(npz_path)

    dv.require_prediction_contract(
        arrays,
        variant.upper(),
    )

    common_train = scene_root / "common" / "train_images"
    common_heldout = scene_root / "common" / "heldout_images"

    train_names = [
        f"image_{i}.png"
        for i in range(1, len(frames) + 1)
    ]

    train_tensor = dv.preprocess_images(
        [
            dv.resolve_image_path(recs[f])
            for f in frames
        ]
    )

    # Materialize the shared 518x518 image caches.
    # The original downstream run had these already on disk; on this machine
    # only the generated sources/models were transferred.
    from PIL import Image

    def save_batch(batch, names, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, name in enumerate(names):
            out_path = out_dir / name

            if out_path.is_file():
                continue

            img = batch[i]

            if hasattr(img, "detach"):
                img = img.detach().cpu().numpy()

            # CHW -> HWC
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))

            img = np.clip(img, 0.0, 1.0)
            img = (img * 255.0).round().astype(np.uint8)

            Image.fromarray(img).save(out_path)

    save_batch(
        train_tensor,
        train_names,
        common_train,
    )

    heldout_names = [
        f"frame{f:06d}.png"
        for f in heldout
    ]

    heldout_tensor = dv.preprocess_images(
        [
            dv.resolve_image_path(recs[f])
            for f in heldout
        ]
    )

    save_batch(
        heldout_tensor,
        heldout_names,
        common_heldout,
    )

    train_source = dv.build_train_source(
        scene_root,
        variant,
        arrays,
        train_tensor,
        train_names,
        common_train,
        stride,
    )

    # ---------------------------------------------------------------
    # Fit GT -> predicted Sim(3) using the six input cameras.
    # ---------------------------------------------------------------

    E_gt = np.stack(
        [
            dv.co3d_to_opencv_camera(recs[f])[0]
            for f in frames
        ]
    )

    C_gt = dv.camera_centers(E_gt)

    E_pred = arrays["extrinsic"].astype(np.float64)
    C_pred = dv.camera_centers(E_pred)

    s, A, b = dv.umeyama(
        C_gt,
        C_pred,
    )

    C_fit = (s * (A @ C_gt.T)).T + b

    residual = np.linalg.norm(
        C_fit - C_pred,
        axis=1,
    )

    radius = float(
        np.median(
            np.linalg.norm(
                C_pred - np.median(C_pred, axis=0),
                axis=1,
            )
        )
    )

    radius = max(
        radius,
        1e-12,
    )

    # ---------------------------------------------------------------
    # Transform held-out GT cameras into variant world frame.
    # ---------------------------------------------------------------

    E_gt_held = []
    K_held = []

    for f in heldout:
        Egt, K0 = dv.co3d_to_opencv_camera(
            recs[f]
        )

        E_gt_held.append(Egt)

        K_held.append(
            dv.original_to_518_affine(recs[f]) @ K0
        )

    E_held = np.stack(
        [
            dv.transform_camera_to_variant(
                e,
                s,
                A,
                b,
            )
            for e in E_gt_held
        ]
    )

    heldout_source = dv.build_heldout_source(
        scene_root,
        variant,
        train_source,
        E_held,
        np.stack(K_held),
        [
            f"frame{f:06d}.png"
            for f in heldout
        ],
        common_heldout,
    )

    alignment = {
        "scale": float(s),
        "camera_center_residual_median_scene_radius": float(
            np.median(residual) / radius
        ),
    }

    return (
        train_source,
        heldout_source,
        alignment,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def gain_bias(render, gt, mask):
    """Per-channel gain + bias exposure correction."""

    out = render.copy()

    for c in range(3):
        x = render[..., c][mask]
        y = gt[..., c][mask]

        if len(x) < 10:
            continue

        if float(np.var(x)) < 1e-12:
            continue

        design = np.stack(
            [
                x,
                np.ones_like(x),
            ],
            axis=1,
        )

        sol, *_ = np.linalg.lstsq(
            design,
            y,
            rcond=None,
        )

        out[..., c] = np.clip(
            render[..., c] * sol[0] + sol[1],
            0.0,
            1.0,
        )

    return out


def paired_stats(deltas):
    """Paired t-test summary."""

    d = np.asarray(
        [
            x
            for x in deltas
            if np.isfinite(x)
        ],
        dtype=np.float64,
    )

    n = len(d)

    if n < 2:
        return {
            "n": n,
        }

    mean = float(d.mean())
    sd = float(d.std(ddof=1))

    se = sd / np.sqrt(n)

    if se > 0:
        t = mean / se
    else:
        t = float("nan")

    try:
        from scipy import stats

        p = float(
            2 * stats.t.sf(
                abs(t),
                df=n - 1,
            )
        )

        tcrit = float(
            stats.t.ppf(
                0.975,
                df=n - 1,
            )
        )

    except Exception:
        p = float("nan")
        tcrit = 1.96

    return {
        "n": n,
        "mean_delta": mean,
        "sd": sd,
        "t": float(t),
        "p": p,
        "ci95": [
            mean - tcrit * se,
            mean + tcrit * se,
        ],
        "mde_80pct_power": float(
            2.802 * sd / np.sqrt(n)
        ),
        "scenes_positive": int(
            (d > 0).sum()
        ),
    }


def score_scene(
    cat: str,
    seq: str,
    arms_present: dict,
):
    """PSNR / SSIM / LPIPS for both masks, raw + exposure corrected."""

    from PIL import Image
    import lpips as lpips_lib
    import torch

    global _LPIPS

    try:
        _LPIPS
    except NameError:
        _LPIPS = (
            lpips_lib
            .LPIPS(net="alex")
            .cpu()
            .eval()
        )

    man = common.scene_manifest(
        cat,
        seq,
    )

    recs = dv.load_annotations(
        cat,
        seq,
    )

    held = man["heldout_frames"]

    root = common.scene_root(
        cat,
        seq,
    )

    gt_paths = [
        root
        / "common"
        / "heldout_images"
        / f"frame{f:06d}.png"
        for f in held
    ]

    def read_image(path):
        return (
            np.asarray(
                Image.open(path).convert("RGB"),
                dtype=np.float32,
            )
            / 255.0
        )

    def lpips_score(a, b, mask):
        ys, xs = dv.bbox_from_mask(
            mask
        )

        ta = (
            torch.from_numpy(a[ys, xs])
            .permute(2, 0, 1)[None]
            .float()
            * 2
            - 1
        )

        tb = (
            torch.from_numpy(b[ys, xs])
            .permute(2, 0, 1)[None]
            .float()
            * 2
            - 1
        )

        with torch.no_grad():
            return float(
                _LPIPS(ta, tb).item()
            )

    out = {}

    for arm, render_dir in arms_present.items():
        paths = sorted(
            Path(render_dir).glob("*.png")
        )

        if len(paths) != len(gt_paths):
            print(
                f"    {arm}: render count mismatch "
                f"{len(paths)} vs {len(gt_paths)}"
            )
            continue

        arm_scores = {}

        for region in (
            "foreground",
            "content",
        ):
            if region == "foreground":
                masks = [
                    dv.foreground_mask_from_record(
                        recs[f]
                    )
                    for f in held
                ]
            else:
                masks = [
                    dv.content_mask_from_record(
                        recs[f]
                    )
                    for f in held
                ]

            psnr_values = []
            ssim_values = []
            lpips_values = []
            exposure_psnr_values = []

            for gt_path, render_path, mask in zip(
                gt_paths,
                paths,
                masks,
            ):
                if not mask.any():
                    continue

                gt = read_image(gt_path)
                render = read_image(render_path)

                psnr_values.append(
                    dv.psnr(
                        render,
                        gt,
                        mask,
                    )
                )

                ssim_values.append(
                    dv.ssim_value(
                        render,
                        gt,
                        mask,
                    )
                )

                lpips_values.append(
                    lpips_score(
                        render,
                        gt,
                        mask,
                    )
                )

                corrected = gain_bias(
                    render,
                    gt,
                    mask,
                )

                exposure_psnr_values.append(
                    dv.psnr(
                        corrected,
                        gt,
                        mask,
                    )
                )

            arm_scores[region] = {
                "psnr": float(
                    np.mean(psnr_values)
                ),
                "ssim": float(
                    np.mean(ssim_values)
                ),
                "lpips": float(
                    np.mean(lpips_values)
                ),
                "psnr_exposure_corrected": float(
                    np.mean(exposure_psnr_values)
                ),
            }

        out[arm] = arm_scores

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        default="w3a3",
        help="Quantized arm to train/evaluate, e.g. w3a3 or w2a4",
    )

    parser.add_argument(
        "--scenes",
        choices=[
            "subset",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=7000,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--gpu-poll-seconds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--gpu-stable-checks",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build quantized-arm sources only; do not train.",
    )

    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Score existing renders only.",
    )

    args = parser.parse_args()

    variant = args.variant

    ARMS = (
        "full",
        "w4a4",
        variant,
    )

    if args.scenes == "subset":
        scenes = common.EXPENSIVE_SCENES
    else:
        scenes = frozen_scenes()

    print(
        f"variant={variant}  "
        f"scenes={len(scenes)} ({args.scenes})  "
        f"iterations={args.iterations}"
    )

    rows = []
    start_time = time.time()

    for index, name in enumerate(
        scenes,
        1,
    ):
        cat, seq = common.split_scene(
            name
        )

        root = common.scene_root(
            cat,
            seq,
        )

        print(
            f"\n[{index}/{len(scenes)}] {name}",
            flush=True,
        )

        n_held = len(exact_heldout_frames(cat, seq))

        iteration = args.iterations

        # -----------------------------------------------------------
        # Build/train selected quantized arm
        # -----------------------------------------------------------

        if not args.evaluate_only:
            train_src, held_src, align = build_variant_sources(
                cat,
                seq,
                variant,
                args.stride,
            )

            print(
                f"  {variant} sources ready; "
                f"Sim(3) residual "
                f"{align['camera_center_residual_median_scene_radius']:.4f} "
                f"scene radii"
            )

            if not args.prepare_only:
                model_dir = (
                    root
                    / "models"
                    / variant
                )

                dv.train_3dgs(
                    train_src,
                    model_dir,
                    iteration,
                    args.gpu_poll_seconds,
                    args.gpu_stable_checks,
                )

        if args.prepare_only:
            continue

        # -----------------------------------------------------------
        # Render all available arms at same iteration
        # -----------------------------------------------------------

        arms_present = {}

        for arm in ARMS:
            model_dir = (
                root
                / "models"
                / arm
            )

            checkpoint = (
                model_dir
                / "point_cloud"
                / f"iteration_{iteration}"
                / "point_cloud.ply"
            )

            if not checkpoint.is_file():
                print(
                    f"    {arm}: no iteration_{iteration} checkpoint, "
                    f"skipping arm"
                )
                continue

            held_src = (
                root
                / "sources"
                / arm
                / "heldout"
            )

            if not held_src.is_dir():
                print(
                    f"    {arm}: no heldout source at {held_src}, "
                    f"skipping arm"
                )
                continue

            out_dir = (
                root
                / "heldout"
                / f"{arm}_it{iteration}"
            )

            render_dir = dv.render_heldout(
                held_src,
                model_dir,
                out_dir,
                iteration,
                n_held,
                args.gpu_poll_seconds,
                args.gpu_stable_checks,
            )

            arms_present[arm] = render_dir

        # -----------------------------------------------------------
        # Score scene
        # -----------------------------------------------------------

        scores = score_scene(
            cat,
            seq,
            arms_present,
        )

        row = {
            "scene": name,
        }

        for arm, regions in scores.items():
            for region, metrics in regions.items():
                for metric, value in metrics.items():
                    row[
                        f"{arm}_{region}_{metric}"
                    ] = value

        rows.append(row)

        for arm in ARMS:
            if arm not in scores:
                continue

            fg = scores[arm]["foreground"]

            print(
                f"    {arm:6s} "
                f"fg PSNR={fg['psnr']:6.3f}  "
                f"(exp-corr {fg['psnr_exposure_corrected']:6.3f})  "
                f"SSIM={fg['ssim']:.4f}  "
                f"LPIPS={fg['lpips']:.4f}",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # Prepare-only exit
    # -----------------------------------------------------------------------

    if args.prepare_only:
        print(
            f"\nprepare-only: {variant} sources built; "
            f"no GPU training performed."
        )
        return

    # -----------------------------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------------------------

    summary = {
        "variant": variant,
        "iterations": args.iterations,
        "n_scenes": len(rows),
        "minutes": (
            time.time() - start_time
        )
        / 60.0,
        "arms": {},
        "pairs": {},
    }

    metric_names = (
        "psnr",
        "ssim",
        "lpips",
        "psnr_exposure_corrected",
    )

    for arm in ARMS:
        for region in (
            "foreground",
            "content",
        ):
            available = {}

            for metric in metric_names:
                key = (
                    f"{arm}_{region}_{metric}"
                )

                values = [
                    row[key]
                    for row in rows
                    if key in row
                ]

                if values:
                    available[metric] = float(
                        np.mean(values)
                    )

            if available:
                summary["arms"].setdefault(
                    arm,
                    {},
                )[region] = available

    # -----------------------------------------------------------------------
    # Paired comparisons
    # -----------------------------------------------------------------------

    pairs = (
        ("full", "w4a4"),
        ("full", variant),
        ("w4a4", variant),
    )

    for x, y in pairs:
        for region in (
            "foreground",
            "content",
        ):
            for metric in (
                "psnr",
                "psnr_exposure_corrected",
            ):
                key_x = (
                    f"{x}_{region}_{metric}"
                )

                key_y = (
                    f"{y}_{region}_{metric}"
                )

                deltas = [
                    row[key_x] - row[key_y]
                    for row in rows
                    if key_x in row
                    and key_y in row
                ]

                if len(deltas) >= 2:
                    name = (
                        f"{x}_minus_{y}"
                        f"__{region}"
                        f"__{metric}"
                    )

                    summary["pairs"][name] = paired_stats(
                        deltas
                    )

    summary["per_scene"] = rows

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUT
        / f"three_arm_{variant}_{args.scenes}_it{args.iterations}.json"
    )

    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    # -----------------------------------------------------------------------
    # Print final summary
    # -----------------------------------------------------------------------

    print()
    print("=" * 86)

    print(
        f"{'arm':8s}"
        f"{'fg PSNR':>10s}"
        f"{'fg exp-corr':>14s}"
        f"{'fg SSIM':>11s}"
        f"{'fg LPIPS':>11s}"
        f"{'content PSNR':>15s}"
    )

    for arm in ARMS:
        if arm not in summary["arms"]:
            continue

        fg = summary["arms"][arm].get(
            "foreground"
        )

        content = summary["arms"][arm].get(
            "content"
        )

        if not fg or not content:
            continue

        print(
            f"{arm:8s}"
            f"{fg['psnr']:10.4f}"
            f"{fg['psnr_exposure_corrected']:14.4f}"
            f"{fg['ssim']:11.4f}"
            f"{fg['lpips']:11.4f}"
            f"{content['psnr']:15.4f}"
        )

    print()
    print(
        "Paired tests "
        "(foreground, raw PSNR):"
    )

    for key, value in summary["pairs"].items():
        if "foreground__psnr" not in key:
            continue

        if value.get("n", 0) < 2:
            continue

        print(
            f"  {key:42s} "
            f"mean={value['mean_delta']:+.4f} "
            f"p={value['p']:.4g} "
            f"CI=["
            f"{value['ci95'][0]:+.4f},"
            f"{value['ci95'][1]:+.4f}"
            f"] "
            f"{value['scenes_positive']}/{value['n']}"
        )

    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()