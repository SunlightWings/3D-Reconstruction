#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PATHS
# ============================================================

REPO = Path("/home/utn/luli38se/cv/3D-Reconstruction")

REGION_CSV = (
    REPO
    / "results/region_characterization/dataset_region_summary.csv"
)

GEOMETRY_SCENE_CSV = (
    REPO
    / "results/per_scene.csv"
)

DOWNSTREAM_JSON = Path(
    "/home/utn/luli38se/cv/"
    "downstream_full_dataset_v1_state/"
    "aggregate_metrics.json"
)

OUT = REPO / "results/teaching_team_plots"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.titlesize": 16,
        "legend.fontsize": 10,
        "savefig.dpi": 300,
    }
)

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
FULL_COLOR = COLORS[0]
QUANT_COLOR = COLORS[1]
POSITIVE_COLOR = COLORS[0]
NEGATIVE_COLOR = COLORS[3] if len(COLORS) > 3 else COLORS[1]


# ============================================================
# HELPERS
# ============================================================

def load_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def to_float(value):
    if value is None:
        return np.nan

    value = str(value).strip()
    value = value.replace("%", "")
    value = value.replace("x", "")

    if value == "":
        return np.nan

    return float(value)


def find_column(
    columns,
    exact=(),
    required_tokens=(),
):
    """
    Robustly locate a column even if the analyzer used
    slightly different naming.
    """

    lower = {
        c.lower(): c
        for c in columns
    }

    for candidate in exact:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    for column in columns:
        name = column.lower()

        if all(
            token.lower() in name
            for token in required_tokens
        ):
            return column

    raise RuntimeError(
        "Could not locate column.\n"
        f"Required tokens: {required_tokens}\n"
        f"Available columns: {columns}"
    )


def scene_name(row):
    category = (
        row.get("category")
        or row.get("category_name")
        or ""
    )

    sequence = (
        row.get("sequence")
        or row.get("sequence_name")
        or ""
    )

    return f"{category}/{sequence}"


def save(fig, filename):
    path = OUT / filename

    fig.savefig(
        path,
        bbox_inches="tight",
        dpi=300,
    )

    plt.close(fig)

    print("Saved:", path)


# ============================================================
# PLOT 1
#
# WHERE DOES FULL-QUANT DISAGREEMENT CONCENTRATE?
# ============================================================

def plot_region_disagreement():

    rows = load_csv(REGION_CSV)

    if not rows:
        raise RuntimeError(
            "Region summary CSV is empty"
        )

    columns = list(rows[0].keys())

    cue_col = find_column(
        columns,
        exact=(
            "label",
            "cue_label",
            "region",
            "cue",
        ),
    )

    disagreement_col = find_column(
        columns,
        exact=(
            "median_disagreement_enrichment",
            "median_high_disagreement_enrichment",
            "disagreement_enrichment",
        ),
        required_tokens=(
            "disagreement",
            "enrichment",
        ),
    )

    pretty_names = {
        "object_boundary":
            "Object boundary",

        "thin_structure":
            "Thin / protrusion",

        "thin/protrusion heuristic":
            "Thin / protrusion",

        "high_depth_gradient":
            "High depth gradient",

        "high gt depth gradient":
            "High depth gradient",

        "high_texture":
            "High texture",

        "rgb_edge":
            "RGB edge",

        "corner":
            "Harris corner",

        "harris corner":
            "Harris corner",

        "low_texture":
            "Low texture",

        "smooth_flat":
            "Smooth / flat interior",

        "smooth/flat interior":
            "Smooth / flat interior",
    }

    data = []

    for row in rows:

        raw_label = row[cue_col].strip()

        label = pretty_names.get(
            raw_label.lower(),
            pretty_names.get(
                raw_label.lower().replace(" ", "_"),
                raw_label.replace("_", " ").title(),
            ),
        )

        value = to_float(
            row[disagreement_col]
        )

        if np.isfinite(value):
            data.append(
                (label, value)
            )

    if not data:
        raise RuntimeError(
            "No disagreement enrichment values found"
        )

    # Largest at top.
    data.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    labels = [
        x[0]
        for x in data
    ][::-1]

    values = np.array(
        [
            x[1]
            for x in data
        ][::-1]
    )

    colors = [
        POSITIVE_COLOR
        if x >= 1.0
        else NEGATIVE_COLOR
        for x in values
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6.5)
    )

    bars = ax.barh(
        labels,
        values,
        color=colors,
    )

    ax.axvline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="1× expected from coverage",
    )

    ax.set_xlabel(
        "Median high-disagreement enrichment (×)"
    )

    ax.set_title(
        "Where Full–Quant VGGT disagreement concentrates"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        ax.text(
            value + 0.03,
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:.2f}×",
            va="center",
            fontsize=10,
        )

    ax.legend(
        loc="lower right"
    )

    fig.text(
        0.5,
        0.01,
        (
            "Values >1× mean the region is over-represented "
            "among the top-disagreement pixels. "
            "Region cues overlap."
        ),
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=[0, 0.04, 1, 1]
    )

    save(
        fig,
        "01_region_disagreement_enrichment.png",
    )


# ============================================================
# PLOT 2
#
# FINAL FULL vs QUANT 3DGS RESULTS
# ============================================================

def plot_final_3dgs_metrics():

    d = load_json(
        DOWNSTREAM_JSON
    )

    metrics = [
        (
            "PSNR ↑",
            "mean_full_psnr",
            "mean_quant_psnr",
            "dB",
        ),
        (
            "SSIM ↑",
            "mean_full_ssim",
            "mean_quant_ssim",
            "",
        ),
        (
            "LPIPS ↓",
            "mean_full_lpips",
            "mean_quant_lpips",
            "",
        ),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4.5),
    )

    for ax, (
        title,
        full_key,
        quant_key,
        unit,
    ) in zip(axes, metrics):

        full = float(d[full_key])
        quant = float(d[quant_key])

        bars = ax.bar(
            ["Full VGGT", "W4A4"],
            [full, quant],
            color=[
                FULL_COLOR,
                QUANT_COLOR,
            ],
            width=0.65,
        )

        ax.set_title(title)
        ax.grid(
            axis="y",
            alpha=0.2,
        )

        # Honest zero baseline for bar charts.
        ax.set_ylim(
            bottom=0
        )

        for bar, value in zip(
            bars,
            [full, quant],
        ):

            if title.startswith("PSNR"):
                text = (
                    f"{value:.3f}"
                    f" {unit}"
                )
            else:
                text = f"{value:.5f}"

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
                text,
                ha="center",
                va="bottom",
                fontsize=10,
            )

        if title.startswith("PSNR"):

            delta = full - quant

            delta_text = (
                f"Full − Quant = "
                f"{delta:+.204f}"
            )

            # Correct formatting below.
            delta_text = (
                f"Full − Quant = "
                f"{delta:+.3f} dB"
            )

        elif title.startswith("SSIM"):

            delta = full - quant

            delta_text = (
                f"Full − Quant = "
                f"{delta:+.5f}"
            )

        else:

            # Lower LPIPS is better.
            delta = quant - full

            delta_text = (
                f"Quant − Full = "
                f"{delta:+.5f}"
            )

        ax.text(
            0.5,
            0.94,
            delta_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )

    fig.suptitle(
        "Final 3DGS reconstruction quality across 40 scenes",
        y=1.02,
    )

    fig.text(
        0.5,
        -0.02,
        (
            "30k-iteration GraphDECO 3DGS. "
            "Higher is better for PSNR/SSIM; "
            "lower is better for LPIPS."
        ),
        ha="center",
        fontsize=9,
    )

    fig.tight_layout()

    save(
        fig,
        "02_final_3dgs_full_vs_quant.png",
    )


# ============================================================
# PLOT 3
#
# SORTED PER-SCENE 3DGS PSNR DIFFERENCE
# ============================================================

def plot_scene_psnr_difference():

    d = load_json(
        DOWNSTREAM_JSON
    )

    scenes = d["scenes"]

    values = []

    for x in scenes:

        name = (
            f"{x['category']}/"
            f"{x['sequence']}"
        )

        delta = float(
            x["full_minus_quant_psnr"]
        )

        values.append(
            (name, delta)
        )

    values.sort(
        key=lambda x: x[1]
    )

    labels = [
        x[0]
        for x in values
    ]

    delta = np.array(
        [
            x[1]
            for x in values
        ]
    )

    colors = [
        POSITIVE_COLOR
        if x > 0
        else NEGATIVE_COLOR
        for x in delta
    ]

    fig, ax = plt.subplots(
        figsize=(11, 13)
    )

    y = np.arange(
        len(labels)
    )

    ax.barh(
        y,
        delta,
        color=colors,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        labels,
        fontsize=8,
    )

    ax.axvline(
        0,
        color="black",
        linewidth=1.2,
    )

    mean_delta = float(
        d["mean_full_minus_quant_psnr"]
    )

    ax.axvline(
        mean_delta,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label=(
            f"Mean = "
            f"{mean_delta:+.3f} dB"
        ),
    )

    full_better = int(
        d["scenes_full_better_psnr"]
    )

    ax.set_xlabel(
        "Full − Quant held-out PSNR (dB)"
    )

    ax.set_title(
        "Per-scene downstream effect of quantized VGGT initialization\n"
        f"Full has higher PSNR in {full_better}/40 scenes"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    ax.legend(
        loc="lower right"
    )

    fig.text(
        0.5,
        0.01,
        (
            "Positive = Full VGGT initialization performs better. "
            "Negative = W4A4 initialization performs better."
        ),
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=[0, 0.03, 1, 1]
    )

    save(
        fig,
        "03_per_scene_psnr_difference.png",
    )


# ============================================================
# PLOT 4
#
# PER-SCENE DISAGREEMENT USEFULNESS
# Spearman rho(d, excess error)
# ============================================================

def plot_scene_disagreement_usefulness():

    rows = load_csv(
        GEOMETRY_SCENE_CSV
    )

    if not rows:
        raise RuntimeError(
            "Per-scene geometry CSV is empty"
        )

    columns = list(
        rows[0].keys()
    )

    rho_col = find_column(
        columns,
        exact=(
            "rho_excess",
            "rho_d_excess",
            "spearman_d_excess",
            "spearman_disagreement_excess",
        ),
        required_tokens=(
            "rho",
            "excess",
        ),
    )

    values = []

    for row in rows:

        name = scene_name(row)

        rho = to_float(
            row[rho_col]
        )

        if (
            name != "/"
            and np.isfinite(rho)
        ):
            values.append(
                (name, rho)
            )

    values.sort(
        key=lambda x: x[1]
    )

    names = [
        x[0]
        for x in values
    ]

    rho = np.array(
        [
            x[1]
            for x in values
        ]
    )

    n = len(rho)
    x = np.arange(1, n + 1)

    median_rho = float(
        np.median(rho)
    )

    positive = int(
        np.sum(rho > 0)
    )

    fig, ax = plt.subplots(
        figsize=(11, 6.5)
    )

    # Light stems make this a ranked lollipop/dot plot.
    for xi, value in zip(x, rho):
        ax.plot(
            [xi, xi],
            [0, value],
            linewidth=1,
            alpha=0.35,
        )

    ax.scatter(
        x,
        rho,
        s=55,
        zorder=3,
    )

    ax.axhline(
        0,
        color="black",
        linewidth=1.2,
    )

    ax.axhline(
        median_rho,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label=f"Median ρ = {median_rho:.3f}",
    )

    ax.set_xlim(
        0,
        n + 1,
    )

    ax.set_xlabel(
        "Scenes ranked by disagreement–error relationship"
    )

    ax.set_ylabel(
        "Spearman ρ(disagreement, excess geometric error)"
    )

    ax.set_title(
        "Full–Quant disagreement is meaningful across almost all scenes\n"
        f"{positive}/{n} scenes show a positive relationship"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    ax.legend(
        loc="upper left"
    )

    # Label only a few extreme scenes to keep the figure clean.
    label_indices = sorted(
        set(
            list(range(min(2, n)))
            + list(range(max(0, n - 3), n))
        )
    )

    for i in label_indices:

        short_name = names[i]

        ax.annotate(
            short_name,
            (x[i], rho[i]),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8,
            rotation=25,
            ha="left",
        )

    fig.text(
        0.5,
        0.01,
        (
            "Each point is one CO3D scene. "
            "Positive ρ means larger Full–Quant disagreement tends to coincide "
            "with larger quantization-induced excess GT error."
        ),
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=[0, 0.04, 1, 1]
    )

    save(
        fig,
        "04_per_scene_disagreement_ranked.png",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Generating teaching-team figures..."
    )

    print()
    print("[1/4] Region disagreement")
    plot_region_disagreement()

    print()
    print("[2/4] Final 3DGS metrics")
    plot_final_3dgs_metrics()

    print()
    print("[3/4] Per-scene PSNR")
    plot_scene_psnr_difference()

    print()
    print("[4/4] Per-scene disagreement")
    plot_scene_disagreement_usefulness()

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print("Output directory:")
    print(OUT)

    print()
    for p in sorted(
        OUT.glob("*.png")
    ):
        print(
            f"{p.name}: "
            f"{p.stat().st_size / 1024:.1f} KiB"
        )


if __name__ == "__main__":
    main()
