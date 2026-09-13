#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]

CODE = REPO / "code"
DIAG = CODE / "downstream_3dgs" / "diagnostics"

sys.path.insert(0, str(CODE))
sys.path.insert(0, str(DIAG))

import common  # noqa: E402

dv = common.dv


ARCHIVE = Path("/var/tmp/poli22wo/renders7k_8scenes")
DOWNSTREAM = Path("/var/tmp/poli22wo/quantsplat/downstream_validation_v1")
OUT = Path("/var/tmp/poli22wo/quantsplat/oracle_sweep/results")


SCENES = [
    "apple/110_13051_23361",
    "ball/123_14363_28981",
    "bowl/70_5792_13401",
    "broccoli/412_56288_108844",
    "hydrant/167_18184_34441",
    "remote/350_36761_68623",
    "teddybear/187_20215_38541",
    "toaster/372_41229_82130",
]


_LPIPS = None


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB"),
        dtype=np.float32,
    ) / 255.0


def bbox(mask):
    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        raise RuntimeError("empty mask")

    return (
        slice(int(ys.min()), int(ys.max()) + 1),
        slice(int(xs.min()), int(xs.max()) + 1),
    )


def psnr(a, b, mask):
    mse = float(((a - b) ** 2)[mask].mean())

    if mse <= 1e-15:
        return float("inf")

    return -10.0 * math.log10(mse)


def ssim(a, b, mask):
    from skimage.metrics import structural_similarity

    ys, xs = bbox(mask)

    return float(
        structural_similarity(
            a[ys, xs],
            b[ys, xs],
            channel_axis=2,
            data_range=1.0,
        )
    )


def lpips_value(a, b, mask):
    global _LPIPS

    import torch

    if _LPIPS is None:
        import lpips

        _LPIPS = lpips.LPIPS(
            net="alex"
        ).cpu().eval()

    ys, xs = bbox(mask)

    ta = (
        torch.from_numpy(
            np.ascontiguousarray(a[ys, xs])
        )
        .permute(2, 0, 1)[None]
        .float()
        * 2
        - 1
    )

    tb = (
        torch.from_numpy(
            np.ascontiguousarray(b[ys, xs])
        )
        .permute(2, 0, 1)[None]
        .float()
        * 2
        - 1
    )

    with torch.no_grad():
        return float(
            _LPIPS(ta, tb).item()
        )


def frame_id(path: Path) -> int:
    return int(
        path.stem.replace("frame", "")
    )


def exact_gt(scene: str):
    cat, seq = scene.split("/", 1)

    gt_dir = ARCHIVE / cat / seq / "gt"

    gt_paths = sorted(
        gt_dir.glob("frame*.png")
    )

    if len(gt_paths) != 9:
        raise RuntimeError(
            f"{scene}: expected 9 GT images, "
            f"found {len(gt_paths)}"
        )

    frames = [
        frame_id(p)
        for p in gt_paths
    ]

    return frames, gt_paths


def arm_render_paths(
    scene: str,
    arm: str,
):
    cat, seq = scene.split("/", 1)

    if arm == "w3a3":
        render_dir = (
            DOWNSTREAM
            / cat
            / seq
            / "heldout"
            / "w3a3_it7000"
            / "renders"
        )

    else:
        render_dir = (
            ARCHIVE
            / cat
            / seq
            / arm
        )

    paths = sorted(
        render_dir.glob("*.png")
    )

    if len(paths) != 9:
        raise RuntimeError(
            f"{scene}/{arm}: expected 9 renders, "
            f"found {len(paths)} in {render_dir}"
        )

    return paths


def masks(
    scene: str,
    frames: list[int],
    region: str,
):
    cat, seq = scene.split("/", 1)

    recs = dv.load_annotations(
        cat,
        seq,
    )

    if region == "foreground":
        return [
            dv.foreground_mask_from_record(
                recs[f]
            )
            for f in frames
        ]

    if region == "content":
        return [
            dv.content_mask_from_record(
                recs[f]
            )
            for f in frames
        ]

    raise ValueError(region)


def score(
    scene: str,
    arm: str,
    region: str,
):
    frames, gt_paths = exact_gt(
        scene
    )

    render_paths = arm_render_paths(
        scene,
        arm,
    )

    ms = masks(
        scene,
        frames,
        region,
    )

    ps = []
    ss = []
    lp = []

    for gt_path, render_path, mask in zip(
        gt_paths,
        render_paths,
        ms,
    ):
        gt = read_rgb(
            gt_path
        )

        render = read_rgb(
            render_path
        )

        if gt.shape != render.shape:
            raise RuntimeError(
                f"{scene}/{arm}: "
                f"shape mismatch "
                f"{gt.shape} vs {render.shape}"
            )

        ps.append(
            psnr(
                render,
                gt,
                mask,
            )
        )

        ss.append(
            ssim(
                render,
                gt,
                mask,
            )
        )

        lp.append(
            lpips_value(
                render,
                gt,
                mask,
            )
        )

    return {
        "psnr": float(
            np.mean(ps)
        ),
        "ssim": float(
            np.mean(ss)
        ),
        "lpips": float(
            np.mean(lp)
        ),
    }


def paired_stats(deltas):
    from scipy import stats

    d = np.asarray(
        deltas,
        dtype=np.float64,
    )

    n = len(d)

    mean = float(
        d.mean()
    )

    if n < 2:
        return {
            "n": n,
            "mean_delta": mean,
        }

    sd = float(
        d.std(ddof=1)
    )

    se = sd / math.sqrt(n)

    result = stats.ttest_1samp(
        d,
        0.0,
    )

    crit = float(
        stats.t.ppf(
            0.975,
            n - 1,
        )
    )

    return {
        "n": n,
        "mean_delta": mean,
        "sd": sd,
        "p": float(
            result.pvalue
        ),
        "ci95": [
            float(
                mean - crit * se
            ),
            float(
                mean + crit * se
            ),
        ],
        "scenes_positive": int(
            (d > 0).sum()
        ),
    }


def discover_archive_arms():
    common_arms = None

    for scene in SCENES:
        cat, seq = scene.split(
            "/",
            1,
        )

        root = (
            ARCHIVE
            / cat
            / seq
        )

        here = {
            p.name
            for p in root.iterdir()
            if p.is_dir()
            and p.name != "gt"
            and len(
                list(
                    p.glob("*.png")
                )
            ) == 9
        }

        if common_arms is None:
            common_arms = here
        else:
            common_arms = (
                common_arms & here
            )

    return sorted(
        common_arms or []
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help=(
            "Archived arms to compare. "
            "W3A3 is added automatically."
        ),
    )

    parser.add_argument(
        "--ref",
        default="full",
    )

    args = parser.parse_args()

    if args.arms is None:
        archived_arms = (
            discover_archive_arms()
        )
    else:
        archived_arms = args.arms

    arms = list(
        dict.fromkeys(
            archived_arms
            + ["w3a3"]
        )
    )

    if args.ref not in arms:
        raise RuntimeError(
            f"Reference arm "
            f"{args.ref!r} "
            f"not in arms: {arms}"
        )

    print(
        f"Scenes: {len(SCENES)}"
    )

    print(
        f"Arms ({len(arms)}): "
        + ", ".join(arms)
    )

    rows = []

    for i, scene in enumerate(
        SCENES,
        1,
    ):
        print(
            f"\n[{i}/8] {scene}"
        )

        row = {
            "scene": scene,
        }

        for arm in arms:
            fg = score(
                scene,
                arm,
                "foreground",
            )

            ct = score(
                scene,
                arm,
                "content",
            )

            for metric, value in fg.items():
                row[
                    f"{arm}_foreground_{metric}"
                ] = value

            for metric, value in ct.items():
                row[
                    f"{arm}_content_{metric}"
                ] = value

            print(
                f"  {arm:20s} "
                f"fg PSNR={fg['psnr']:7.3f} "
                f"SSIM={fg['ssim']:.4f} "
                f"LPIPS={fg['lpips']:.4f}"
            )

        rows.append(
            row
        )

    summary = {
        "n_scenes": len(rows),
        "n_views_per_scene": 9,
        "arms": {},
        "pairs": {},
        "per_scene": rows,
    }

    for arm in arms:
        summary["arms"][arm] = {}

        for region in (
            "foreground",
            "content",
        ):
            summary["arms"][arm][region] = {}

            for metric in (
                "psnr",
                "ssim",
                "lpips",
            ):
                key = (
                    f"{arm}_"
                    f"{region}_"
                    f"{metric}"
                )

                values = [
                    row[key]
                    for row in rows
                ]

                summary["arms"][arm][region][metric] = float(
                    np.mean(values)
                )

    for arm in arms:
        if arm == args.ref:
            continue

        for region in (
            "foreground",
            "content",
        ):
            for metric in (
                "psnr",
                "ssim",
                "lpips",
            ):
                ref_key = (
                    f"{args.ref}_"
                    f"{region}_"
                    f"{metric}"
                )

                arm_key = (
                    f"{arm}_"
                    f"{region}_"
                    f"{metric}"
                )

                deltas = [
                    row[ref_key]
                    - row[arm_key]
                    for row in rows
                ]

                summary["pairs"][
                    f"{args.ref}_minus_{arm}"
                    f"__{region}"
                    f"__{metric}"
                ] = paired_stats(
                    deltas
                )

    print()
    print("=" * 92)
    print(
        "PRIMARY: FOREGROUND MASK, RAW METRICS"
    )

    print(
        f"{'arm':24s}"
        f"{'PSNR':>10s}"
        f"{'SSIM':>10s}"
        f"{'LPIPS':>10s}"
    )

    for arm in arms:
        values = (
            summary["arms"][arm]
            ["foreground"]
        )

        print(
            f"{arm:24s}"
            f"{values['psnr']:10.4f}"
            f"{values['ssim']:10.4f}"
            f"{values['lpips']:10.4f}"
        )

    print()
    print(
        f"PAIRED TESTS: "
        f"{args.ref} minus comparison arm "
        f"(n=8 scenes)"
    )

    print(
        "PSNR/SSIM: positive => reference better"
    )

    print(
        "LPIPS: negative => reference better"
    )

    print()

    for arm in arms:
        if arm == args.ref:
            continue

        print(
            f"{args.ref} vs {arm}"
        )

        for metric in (
            "psnr",
            "ssim",
            "lpips",
        ):
            key = (
                f"{args.ref}_minus_{arm}"
                f"__foreground"
                f"__{metric}"
            )

            stats = (
                summary["pairs"][key]
            )

            print(
                f"  {metric:6s}: "
                f"delta="
                f"{stats['mean_delta']:+.4f}, "
                f"p="
                f"{stats['p']:.4g}, "
                f"95% CI=["
                f"{stats['ci95'][0]:+.4f}, "
                f"{stats['ci95'][1]:+.4f}], "
                f"positive scenes="
                f"{stats['scenes_positive']}/"
                f"{stats['n']}"
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = (
        OUT
        / "w3a3_vs_existing_8scenes_9views.json"
    )

    out.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        f"Saved: {out}"
    )


if __name__ == "__main__":
    main()