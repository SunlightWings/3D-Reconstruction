"""Gate D -- is the effect large enough to detect?

D1 paired statistics and power (recomputed independently from the committed
aggregate table), D2 disagreement-vs-downstream-damage correlation rerun with
the Gate B object-masked metric, D3 harsher quantization (best-effort; see
docstring on test_d3 for why this one is very likely to come back BLOCKED).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

dv = common.dv


def test_d1_paired_statistics() -> Dict[str, Any]:
    agg = common.read_json(common.RESULTS_DIR / "aggregate_metrics.json")
    rows = agg["scenes"]
    full = np.array([r["full_psnr"] for r in rows])
    quant = np.array([r["quant_psnr"] for r in rows])
    delta = full - quant
    n = len(delta)
    mean, sd = float(delta.mean()), float(delta.std(ddof=1))
    se = sd / np.sqrt(n)
    t_stat, p_value = stats.ttest_rel(full, quant)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    ci = (mean - tcrit * se, mean + tcrit * se)
    n_favor_full = int(np.sum(delta > 0))

    # Minimum detectable effect at 80% power, alpha=0.05, two-sided, given the observed sd.
    from scipy.stats import norm

    z_alpha, z_beta = norm.ppf(0.975), norm.ppf(0.80)
    mde_n40 = (z_alpha + z_beta) * sd / np.sqrt(n)
    n_needed_for_mde_0p1 = ((z_alpha + z_beta) * sd / 0.1) ** 2

    return {
        "description": "Paired Full-vs-W4A4 PSNR statistics across all 40 scenes, recomputed independently from results/downstream_3dgs/aggregate_metrics.json.",
        "n_scenes": n,
        "mean_delta_db": mean,
        "sd_delta_db": sd,
        "paired_t": float(t_stat),
        "p_value": float(p_value),
        "ci95": [float(ci[0]), float(ci[1])],
        "scenes_favor_full": n_favor_full,
        "scenes_favor_quant": n - n_favor_full,
        "min_detectable_effect_at_n40_80pct_power_db": float(mde_n40),
        "n_scenes_needed_to_detect_0.1db_at_80pct_power": float(n_needed_for_mde_0p1),
        "read": "Effect is statistically real (CI excludes zero) but tiny relative to its own noise; any future null claim must be an effect-size bound, not a null result.",
    }


def _spearman(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rho, _p = stats.spearmanr(x[ok], y[ok])
    return float(rho)


def test_d2_disagreement_correlation_rerun() -> Dict[str, Any]:
    per_scene_csv = common.load_per_scene_csv()
    agg = common.read_json(common.RESULTS_DIR / "aggregate_metrics.json")
    content_loss = {(r["category"], r["sequence"]): r["full_minus_quant_psnr"] for r in agg["scenes"]}

    state = common.load_state()
    b1 = state["tests"].get("B1", {}).get("data")
    if not b1:
        raise RuntimeError("D2 requires B1 (object-mask metrics) to have completed first")
    masked_loss = {}
    for row in b1["per_scene"]:
        category, sequence = common.split_scene(row["scene"])
        masked_loss[(category, sequence)] = row["full_minus_quant"]

    disagreement_cols = ["rho_d_excess", "excess_error_median", "top10_precision_excess", "auc_top10_excess"]
    keys = [k for k in content_loss if k in per_scene_csv and k in masked_loss]

    result_by_metric = {}
    for col in disagreement_cols:
        disagree_vals = [per_scene_csv[k][col] for k in keys]
        result_by_metric[col] = {
            "rho_content_mask_baseline": _spearman(disagree_vals, [content_loss[k] for k in keys]),
            "rho_object_mask_rerun": _spearman(disagree_vals, [masked_loss[k] for k in keys]),
        }

    return {
        "description": "Spearman correlation between validated per-scene disagreement metrics (results/per_scene.csv) and per-scene downstream PSNR loss, before (content-mask) and after (Gate B1 object-mask) correcting the metric.",
        "n_scenes": len(keys),
        "by_metric": result_by_metric,
        "read": "If the object-masked correlations remain near zero, the geometry-to-rendering damage link is not supported by this harness even after fixing the masking bug in Gate B.",
    }


def test_d3_harsher_quantization() -> Dict[str, Any]:
    """W3A3 / W2A4 would require running QuantVGGT's own smooth+rotation quantization
    calibration (learned weight/activation clipping) at a new bit-width. Only a W4A4
    calibrated checkpoint (a44_quant_model_tracker_fixed_e20.pt_sym) exists anywhere on
    this workstation; there is no W3A3/W2A4 calibration artifact, and reusing the W4A4
    calibration parameters at a different bit-width would not represent that bit-width
    (learned clipping ranges are bit-width specific) -- it would silently produce a
    meaningless result rather than a real harsher-quantization control.

    This is therefore reported as BLOCKED rather than faked. Producing a real W3A3
    control requires re-running QuantVGGT/evaluation/quarot's calibration pipeline
    (see QuantVGGT/README.md) to generate new qs_frame_parameters_total.pth /
    qs_global_parameters_total.pth artifacts at wbit=3, abit=3, which is a separate,
    larger undertaking outside what this diagnostic pass can respons-ibly automate
    unattended.
    """
    calib_root = Path("/var/tmp/luli38se/quantsplat/models/QuantVGGT")
    existing = sorted(p.name for p in calib_root.glob("*_quant_*")) if calib_root.is_dir() else []
    return {
        "status": "BLOCKED",
        "description": "Harsher quantization (W3A3 / W2A4) damage-threshold sweep.",
        "reason": (
            "Only a W4A4-calibrated QuantVGGT checkpoint exists in this environment "
            f"({existing}). W3A3/W2A4 need their own learned-clipping calibration run "
            "(QuantVGGT's quarot smooth+rotation pipeline), which was not attempted here "
            "because it is a separate model-calibration workload, not a downstream-3DGS "
            "diagnostic, and reusing the W4A4 calibration at a different bit-width would "
            "produce a number that looks like a result but measures nothing real."
        ),
        "next_step": "Run QuantVGGT's calibration/quantization pipeline at wbit=3,abit=3 (and 2,4) to produce new qs_*_parameters_total.pth artifacts, then rerun run_full_dataset_disagreement.py + this downstream pipeline against them.",
    }


TESTS = [
    ("D1", test_d1_paired_statistics),
    ("D2", test_d2_disagreement_correlation_rerun),
    ("D3", test_d3_harsher_quantization),
]
