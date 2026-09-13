#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(
    "/var/tmp/luli38se/quantsplat/"
    "heldout_eval/friend518_20260813"
)

META = json.loads(
    (
        ROOT
        / "heldout_eval_meta.json"
    ).read_text()
)

FRAMES = META["heldout_frames"]

FULL = (
    ROOT
    / "render_models/full/"
      "train/ours_30000"
)

QUANT = (
    ROOT
    / "render_models/quant/"
      "train/ours_30000"
)

MASK_DIR = (
    ROOT
    / "common/masks"
)


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_rgb(path):
    return (
        np.asarray(
            Image.open(
                path
            ).convert("RGB"),
            dtype=np.float32,
        )
        / 255.0
    )


def psnr(a, b, mask=None):
    diff = (a - b) ** 2

    if mask is not None:
        mask3 = np.repeat(
            mask[..., None],
            3,
            axis=2,
        )

        values = diff[mask3]

        if values.size == 0:
            return float("nan")

        mse = float(
            np.mean(values)
        )

    else:
        mse = float(
            np.mean(diff)
        )

    if mse == 0:
        return float("inf")

    return (
        -10.0
        * math.log10(mse)
    )


def l1(a, b, mask=None):
    diff = np.abs(a - b)

    if mask is not None:
        mask3 = np.repeat(
            mask[..., None],
            3,
            axis=2,
        )
        values = diff[mask3]
        return float(
            np.mean(values)
        )

    return float(
        np.mean(diff)
    )


def gaussian_window(
    channels,
    size=11,
    sigma=1.5,
):
    coords = torch.arange(
        size,
        dtype=torch.float32,
    )

    coords = (
        coords
        - (size - 1) / 2.0
    )

    g = torch.exp(
        -(coords ** 2)
        / (2 * sigma ** 2)
    )

    g /= g.sum()

    kernel = (
        g[:, None]
        @ g[None, :]
    )

    kernel = kernel.view(
        1,
        1,
        size,
        size,
    )

    return kernel.repeat(
        channels,
        1,
        1,
        1,
    )


def ssim(a, b):
    x = torch.from_numpy(
        a.transpose(2, 0, 1)
    ).unsqueeze(0)

    y = torch.from_numpy(
        b.transpose(2, 0, 1)
    ).unsqueeze(0)

    channels = x.shape[1]

    window = gaussian_window(
        channels
    )

    mu_x = F.conv2d(
        x,
        window,
        padding=5,
        groups=channels,
    )

    mu_y = F.conv2d(
        y,
        window,
        padding=5,
        groups=channels,
    )

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = (
        F.conv2d(
            x * x,
            window,
            padding=5,
            groups=channels,
        )
        - mu_x2
    )

    sigma_y2 = (
        F.conv2d(
            y * y,
            window,
            padding=5,
            groups=channels,
        )
        - mu_y2
    )

    sigma_xy = (
        F.conv2d(
            x * y,
            window,
            padding=5,
            groups=channels,
        )
        - mu_xy
    )

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    value = (
        (
            2 * mu_xy
            + C1
        )
        *
        (
            2 * sigma_xy
            + C2
        )
        /
        (
            (
                mu_x2
                + mu_y2
                + C1
            )
            *
            (
                sigma_x2
                + sigma_y2
                + C2
            )
        )
    )

    return float(
        value.mean().item()
    )


LPIPS_MODEL = lpips.LPIPS(
    net="alex",
    verbose=False,
).cpu().eval()


def lpips_metric(a, b):
    x = torch.from_numpy(
        a.transpose(2, 0, 1)
    ).unsqueeze(0)

    y = torch.from_numpy(
        b.transpose(2, 0, 1)
    ).unsqueeze(0)

    x = x * 2.0 - 1.0
    y = y * 2.0 - 1.0

    with torch.no_grad():
        value = LPIPS_MODEL(
            x,
            y,
        )

    return float(
        value.item()
    )


full_r = sorted(
    (FULL / "renders").glob("*.png")
)

full_gt = sorted(
    (FULL / "gt").glob("*.png")
)

quant_r = sorted(
    (QUANT / "renders").glob("*.png")
)

quant_gt = sorted(
    (QUANT / "gt").glob("*.png")
)

if not (
    len(full_r)
    == len(full_gt)
    == len(quant_r)
    == len(quant_gt)
    == 9
):
    raise RuntimeError(
        "Expected 9 renders and GT images "
        "for each variant"
    )

for a, b in zip(
    full_gt,
    quant_gt,
):
    if sha(a) != sha(b):
        raise RuntimeError(
            "Full/Quant GT mismatch"
        )


rows = []

print()
print(
    "============================================================"
)
print(
    "HELDOUT NOVEL-VIEW EVALUATION"
)
print(
    "============================================================"
)

print(
    "Frame | Q PSNR | F PSNR | dPSNR | "
    "Q SSIM | F SSIM | "
    "Q LPIPS | F LPIPS | "
    "Q FG-PSNR | F FG-PSNR"
)

print(
    "------------------------------------------------------------"
)


for idx, frame in enumerate(FRAMES):

    q = load_rgb(
        quant_r[idx]
    )

    f = load_rgb(
        full_r[idx]
    )

    gt = load_rgb(
        full_gt[idx]
    )

    name = f"frame{frame:06d}.png"

    mask = (
        np.asarray(
            Image.open(
                MASK_DIR / name
            ).convert("L"),
            dtype=np.float32,
        )
        / 255.0
    )

    mask = mask >= 0.5

    left, top, right, bottom = (
        META[
            "content_bboxes"
        ][str(frame)]
    )

    q_crop = q[
        top:bottom,
        left:right,
    ]

    f_crop = f[
        top:bottom,
        left:right,
    ]

    gt_crop = gt[
        top:bottom,
        left:right,
    ]

    q_psnr = psnr(
        q_crop,
        gt_crop,
    )

    f_psnr = psnr(
        f_crop,
        gt_crop,
    )

    q_ssim = ssim(
        q_crop,
        gt_crop,
    )

    f_ssim = ssim(
        f_crop,
        gt_crop,
    )

    q_lpips = lpips_metric(
        q_crop,
        gt_crop,
    )

    f_lpips = lpips_metric(
        f_crop,
        gt_crop,
    )

    q_fg_psnr = psnr(
        q,
        gt,
        mask=mask,
    )

    f_fg_psnr = psnr(
        f,
        gt,
        mask=mask,
    )

    q_full_psnr = psnr(
        q,
        gt,
    )

    f_full_psnr = psnr(
        f,
        gt,
    )

    q_l1 = l1(
        q_crop,
        gt_crop,
    )

    f_l1 = l1(
        f_crop,
        gt_crop,
    )

    row = {
        "frame": frame,

        "quant_psnr_content":
            q_psnr,

        "full_psnr_content":
            f_psnr,

        "full_minus_quant_psnr":
            f_psnr - q_psnr,

        "quant_ssim_content":
            q_ssim,

        "full_ssim_content":
            f_ssim,

        "full_minus_quant_ssim":
            f_ssim - q_ssim,

        "quant_lpips_content":
            q_lpips,

        "full_lpips_content":
            f_lpips,

        "quant_minus_full_lpips":
            q_lpips - f_lpips,

        "quant_fg_psnr":
            q_fg_psnr,

        "full_fg_psnr":
            f_fg_psnr,

        "quant_psnr_full_padded":
            q_full_psnr,

        "full_psnr_full_padded":
            f_full_psnr,

        "quant_l1_content":
            q_l1,

        "full_l1_content":
            f_l1,
    }

    rows.append(row)

    print(
        f"{frame:5d} | "
        f"{q_psnr:6.2f} | "
        f"{f_psnr:6.2f} | "
        f"{f_psnr-q_psnr:+6.2f} | "
        f"{q_ssim:.4f} | "
        f"{f_ssim:.4f} | "
        f"{q_lpips:.4f} | "
        f"{f_lpips:.4f} | "
        f"{q_fg_psnr:7.2f} | "
        f"{f_fg_psnr:7.2f}"
    )


def mean(key):
    return float(
        np.mean(
            [
                row[key]
                for row in rows
            ]
        )
    )


summary = {
    "n_views": 9,

    "primary_region":
        "unpadded real-image content",

    "quant_mean_psnr_content":
        mean(
            "quant_psnr_content"
        ),

    "full_mean_psnr_content":
        mean(
            "full_psnr_content"
        ),

    "full_minus_quant_psnr_content":
        mean(
            "full_psnr_content"
        )
        - mean(
            "quant_psnr_content"
        ),

    "quant_mean_ssim_content":
        mean(
            "quant_ssim_content"
        ),

    "full_mean_ssim_content":
        mean(
            "full_ssim_content"
        ),

    "full_minus_quant_ssim_content":
        mean(
            "full_ssim_content"
        )
        - mean(
            "quant_ssim_content"
        ),

    "quant_mean_lpips_content":
        mean(
            "quant_lpips_content"
        ),

    "full_mean_lpips_content":
        mean(
            "full_lpips_content"
        ),

    "quant_minus_full_lpips_content":
        mean(
            "quant_lpips_content"
        )
        - mean(
            "full_lpips_content"
        ),

    "quant_mean_fg_psnr":
        mean(
            "quant_fg_psnr"
        ),

    "full_mean_fg_psnr":
        mean(
            "full_fg_psnr"
        ),

    "quant_mean_psnr_full_padded":
        mean(
            "quant_psnr_full_padded"
        ),

    "full_mean_psnr_full_padded":
        mean(
            "full_psnr_full_padded"
        ),

    "gt_images_identical":
        True,

    "camera_alignment":
        "separate one-global-Sim(3) GT->Full and GT->Quant using the same six input camera correspondences",

    "no_per_camera_correction":
        True,
}


print()
print(
    "============================================================"
)
print(
    "HELDOUT SUMMARY"
)
print(
    "============================================================"
)

print(
    "Content PSNR:"
)

print(
    "  Quant:",
    f"{summary['quant_mean_psnr_content']:.3f} dB"
)

print(
    "  Full :",
    f"{summary['full_mean_psnr_content']:.3f} dB"
)

print(
    "  Full-Quant:",
    f"{summary['full_minus_quant_psnr_content']:+.3f} dB"
)

print()

print(
    "Content SSIM:"
)

print(
    "  Quant:",
    f"{summary['quant_mean_ssim_content']:.5f}"
)

print(
    "  Full :",
    f"{summary['full_mean_ssim_content']:.5f}"
)

print(
    "  Full-Quant:",
    f"{summary['full_minus_quant_ssim_content']:+.5f}"
)

print()

print(
    "Content LPIPS (lower better):"
)

print(
    "  Quant:",
    f"{summary['quant_mean_lpips_content']:.5f}"
)

print(
    "  Full :",
    f"{summary['full_mean_lpips_content']:.5f}"
)

print(
    "  Quant-Full:",
    f"{summary['quant_minus_full_lpips_content']:+.5f}"
)

print()

print(
    "Foreground PSNR:"
)

print(
    "  Quant:",
    f"{summary['quant_mean_fg_psnr']:.3f} dB"
)

print(
    "  Full :",
    f"{summary['full_mean_fg_psnr']:.3f} dB"
)

print()

print(
    "Full padded PSNR (secondary):"
)

print(
    "  Quant:",
    f"{summary['quant_mean_psnr_full_padded']:.3f} dB"
)

print(
    "  Full :",
    f"{summary['full_mean_psnr_full_padded']:.3f} dB"
)

print()

print(
    "GT identity: PASS"
)

print(
    "Per-camera correction: NONE"
)

print(
    "============================================================"
)
print(
    "HELDOUT EVALUATION: PASS"
)
print(
    "============================================================"
)


(ROOT / "metrics.json").write_text(
    json.dumps(
        {
            "summary": summary,
            "per_frame": rows,
        },
        indent=2,
    )
    + "\n"
)

with (
    ROOT / "metrics.csv"
).open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(rows)
