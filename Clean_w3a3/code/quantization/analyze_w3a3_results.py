#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT = Path(
    "/var/tmp/poli22wo/quantsplat/oracle_sweep/results/"
    "w3a3_vs_existing_8scenes_9views.json"
)

OUT = Path(
    "/var/tmp/poli22wo/quantsplat/oracle_sweep/results"
)

OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------

data = json.loads(RESULT.read_text())
rows = data["per_scene"]


records = []

for row in rows:
    scene = row["scene"]

    full_psnr = row["full_foreground_psnr"]
    w4_psnr = row["w4a4_foreground_psnr"]
    w3_psnr = row["w3a3_foreground_psnr"]

    full_ssim = row["full_foreground_ssim"]
    w4_ssim = row["w4a4_foreground_ssim"]
    w3_ssim = row["w3a3_foreground_ssim"]

    full_lpips = row["full_foreground_lpips"]
    w4_lpips = row["w4a4_foreground_lpips"]
    w3_lpips = row["w3a3_foreground_lpips"]

    records.append(
        {
            "scene": scene,

            "full_psnr": full_psnr,
            "w4a4_psnr": w4_psnr,
            "w3a3_psnr": w3_psnr,

            "full_minus_w4a4_psnr": full_psnr - w4_psnr,
            "full_minus_w3a3_psnr": full_psnr - w3_psnr,

            "full_minus_w4a4_ssim": full_ssim - w4_ssim,
            "full_minus_w3a3_ssim": full_ssim - w3_ssim,

            # Negative means Full is better because lower LPIPS is better.
            "full_minus_w4a4_lpips": full_lpips - w4_lpips,
            "full_minus_w3a3_lpips": full_lpips - w3_lpips,
        }
    )


df = pd.DataFrame(records)


# Short scene names for plots/tables
df["short_scene"] = df["scene"].str.split("/").str[0]


# ---------------------------------------------------------------------
# Print compact per-scene table
# ---------------------------------------------------------------------

display = df[
    [
        "short_scene",
        "full_psnr",
        "w4a4_psnr",
        "w3a3_psnr",
        "full_minus_w4a4_psnr",
        "full_minus_w3a3_psnr",
    ]
].copy()

display.columns = [
    "scene",
    "Full",
    "W4A4",
    "W3A3",
    "Full-W4A4",
    "Full-W3A3",
]


print()
print("=" * 82)
print("PER-SCENE FOREGROUND PSNR")
print("=" * 82)

print(
    display.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

print()
print("=" * 82)
print("SUMMARY")
print("=" * 82)

print(
    f"Mean Full-W4A4 PSNR delta: "
    f"{df['full_minus_w4a4_psnr'].mean():+.4f} dB"
)

print(
    f"Mean Full-W3A3 PSNR delta: "
    f"{df['full_minus_w3a3_psnr'].mean():+.4f} dB"
)

print(
    f"Median Full-W3A3 PSNR delta: "
    f"{df['full_minus_w3a3_psnr'].median():+.4f} dB"
)

print(
    f"Full beats W3A3 on PSNR: "
    f"{(df['full_minus_w3a3_psnr'] > 0).sum()}/"
    f"{len(df)} scenes"
)

print(
    f"Full beats W3A3 on SSIM: "
    f"{(df['full_minus_w3a3_ssim'] > 0).sum()}/"
    f"{len(df)} scenes"
)

print(
    f"Full beats W3A3 on LPIPS: "
    f"{(df['full_minus_w3a3_lpips'] < 0).sum()}/"
    f"{len(df)} scenes"
)


# ---------------------------------------------------------------------
# Identify strongest / weakest W3A3 effects
# ---------------------------------------------------------------------

ranked = df.sort_values(
    "full_minus_w3a3_psnr",
    ascending=False,
)

print()
print("Largest W3A3 degradation:")

for _, row in ranked.iterrows():
    print(
        f"  {row['short_scene']:12s} "
        f"{row['full_minus_w3a3_psnr']:+.3f} dB"
    )


# ---------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------

csv_path = OUT / "w3a3_per_scene_deltas.csv"

df.to_csv(
    csv_path,
    index=False,
)

print()
print(f"Saved CSV: {csv_path}")


# ---------------------------------------------------------------------
# Plot: Full-W4A4 vs Full-W3A3 PSNR degradation
# ---------------------------------------------------------------------

x = np.arange(len(df))
width = 0.36

fig, ax = plt.subplots(
    figsize=(11, 5.5)
)

ax.bar(
    x - width / 2,
    df["full_minus_w4a4_psnr"],
    width,
    label="Full - W4A4",
)

ax.bar(
    x + width / 2,
    df["full_minus_w3a3_psnr"],
    width,
    label="Full - W3A3",
)

ax.axhline(
    0,
    linewidth=1,
)

ax.set_xticks(x)
ax.set_xticklabels(
    df["short_scene"],
    rotation=35,
    ha="right",
)

ax.set_ylabel(
    "PSNR difference (dB)"
)

ax.set_title(
    "Downstream 3DGS degradation by quantization level"
)

ax.legend()

fig.tight_layout()

plot_path = OUT / "w3a3_vs_w4a4_psnr_deltas.png"

fig.savefig(
    plot_path,
    dpi=200,
)

print(
    f"Saved plot: {plot_path}"
)


# ---------------------------------------------------------------------
# Plot 2: absolute PSNR for Full / W4A4 / W3A3
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(11, 5.5)
)

width = 0.25

ax.bar(
    x - width,
    df["full_psnr"],
    width,
    label="Full",
)

ax.bar(
    x,
    df["w4a4_psnr"],
    width,
    label="W4A4",
)

ax.bar(
    x + width,
    df["w3a3_psnr"],
    width,
    label="W3A3",
)

ax.set_xticks(x)

ax.set_xticklabels(
    df["short_scene"],
    rotation=35,
    ha="right",
)

ax.set_ylabel(
    "Foreground PSNR (dB)"
)

ax.set_title(
    "3DGS rendering quality: Full vs W4A4 vs W3A3"
)

ax.legend()

fig.tight_layout()

absolute_plot = OUT / "full_w4a4_w3a3_psnr.png"

fig.savefig(
    absolute_plot,
    dpi=200,
)

print(
    f"Saved plot: {absolute_plot}"
)