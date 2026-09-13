#!/usr/bin/env python3
"""Renders results/downstream_3dgs/diagnostics/state.json into
results/downstream_3dgs/DIAGNOSTIC_RESULTS.md. Safe to call repeatedly (and is
called automatically after every single test finishes) -- always reflects
whatever subset of the 15 tests has completed so far.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def fmt(x: Any, spec: str = ".3f") -> str:
    if x is None:
        return "n/a"
    try:
        if isinstance(x, float) and (x != x):  # NaN
            return "n/a"
        return format(x, spec)
    except (TypeError, ValueError):
        return str(x)


def status_badge(status: str) -> str:
    return {"done": "DONE", "failed": "FAILED", "pending": "PENDING"}.get(status, status.upper())


def section(test_id: str, title: str, tests: Dict[str, Any], body_fn) -> str:
    entry = tests.get(test_id, {"status": "pending"})
    status = entry.get("status", "pending")
    out = [f"### {test_id}: {title}  [{status_badge(status)}]", ""]
    if status == "pending":
        out.append("_Not run yet._")
    elif status == "failed":
        err = (entry.get("error") or "").strip().splitlines()
        out.append("**Failed.** Last error line:")
        out.append("```")
        out.append(err[-1] if err else "(no error message captured)")
        out.append("```")
    else:
        try:
            out.append(body_fn(entry.get("data") or {}))
        except Exception as e:  # noqa: BLE001
            out.append(f"_(report renderer error while formatting this section: {e})_")
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- Gate A

def body_a2(d: Dict[str, Any]) -> str:
    return (
        f"- Scenes evaluated: {d.get('n_scenes')}\n"
        f"- Overall noise floor (mean of per-scene median pair-PSNR): **{fmt(d.get('overall_mean_of_scene_medians'), '.2f')} dB**\n"
        f"- Overall noise floor (median of per-scene medians): {fmt(d.get('overall_median_of_scene_medians'), '.2f')} dB\n"
        "\nIf the main 40-scene run's absolute PSNR (~9.9 dB, see aggregate_metrics.json) sits at or below this "
        "floor, the renders carry essentially no scene information."
    )


def body_a1(d: Dict[str, Any]) -> str:
    below = d.get("scenes_below_15db") or []
    lines = [
        f"- Scenes evaluated: {d.get('n_scenes')}",
        f"- Mean self-fit PSNR, Full: **{fmt(d.get('mean_full_self_fit_psnr'), '.2f')} dB**",
        f"- Mean self-fit PSNR, W4A4: **{fmt(d.get('mean_w4a4_self_fit_psnr'), '.2f')} dB**",
        f"- Scenes with either variant below 15 dB (adapter-bug threshold): {len(below)}"
        + (f" -- {', '.join(below[:10])}{' ...' if len(below) > 10 else ''}" if below else ""),
    ]
    return "\n".join(lines)


def body_a5(d: Dict[str, Any]) -> str:
    import numpy as _np
    legacy = [r["variants"]["full"].get("median_loo_error_legacy_frame_inconsistent")
              for r in (d.get("per_scene") or [])]
    legacy = [x for x in legacy if x is not None and _np.isfinite(x)]
    legacy_txt = (f"\n- For traceability, the previously published frame-inconsistent figure was "
                  f"{fmt(float(_np.median(legacy)), '.4f')} (Full). It measured the residual in the PREDICTED world "
                  f"frame while dividing by the GT-frame scene radius; the two differ by the Sim(3) scale "
                  f"(~0.07), so it was ~13x too small and reported PASS for a condition that fails. "
                  f"See ongoing_logs.md entries #017 and #018." if legacy else "")
    return (
        f"- Scenes evaluated: {d.get('n_scenes')}\n"
        f"- Median LOO camera-center error (Full), as a fraction of scene radius: **{fmt(d.get('median_of_scene_medians_full'), '.4f')}**\n"
        f"- Median LOO camera-center error (W4A4): **{fmt(d.get('median_of_scene_medians_w4a4'), '.4f')}**\n"
        f"- Worst scene (Full): {d.get('worst_scene_full')}\n"
        f"- Worst scene (W4A4): {d.get('worst_scene_w4a4')}"
        + legacy_txt +
        "\n\nTarget: well under 0.02 (2% of scene radius). The residual and the normalising radius are now "
        "measured in the same frame.\n\n"
        "Note: entry #023 measured the in-sample residual of the BEST similarity fit at 0.0485 scene radii, already "
        "2.4x this target, so the GT-to-predicted map is not a similarity and no better Sim(3) estimator reaches it."
    )


def body_a6(d: Dict[str, Any]) -> str:
    lines = [
        f"- Scenes tested (GPU-training subset): {', '.join(d.get('scenes_tested') or [])}",
        f"- Mean self-fit PSNR, foreground mask: **{fmt(d.get('mean_self_fit_psnr_foreground'), '.2f')} dB**",
        f"- Mean self-fit PSNR, content mask: {fmt(d.get('mean_self_fit_psnr_content'), '.2f')} dB",
        f"- Minimum across scenes (foreground): {fmt(d.get('min_self_fit_psnr_foreground'), '.2f')} dB",
        f"- Scenes below target: {len(d.get('scenes_below_target') or [])}",
        "",
        "| scene | content | foreground |",
        "|---|---:|---:|",
    ]
    for r in d.get("per_scene") or []:
        lines.append(f"| {r['scene']} | {fmt(r.get('content_psnr'), '.2f')} | {fmt(r.get('foreground_psnr'), '.2f')} |")
    lines.append(
        f"\nTarget: >= {fmt(d.get('target_db'), '.0f')} dB. Renders the oracle models at their OWN training views, so a "
        "failure would indict the renderer/adapter or the GT camera path."
    )
    lines.append(
        "\nNote the asymmetry: the model was TRAINED on these cameras, so it fits them whether or not they are "
        "geometrically correct. A6 failing would indict the camera path; A6 passing does not exonerate it."
    )
    return "\n".join(lines)


def body_a4(d: Dict[str, Any]) -> str:
    figs = d.get("per_scene") or []
    lines = [f"- Overlay figures generated for: {', '.join(d.get('figures_generated_for') or [])}"]
    for row in figs:
        v = row.get("variants", {})
        full_frac = v.get("full", {}).get("fraction_reprojected_points_on_silhouette")
        quant_frac = v.get("w4a4", {}).get("fraction_reprojected_points_on_silhouette")
        lines.append(
            f"  - `{row['scene']}` (frame {row.get('heldout_frame')}): "
            f"full {fmt(full_frac, '.1%')} on-silhouette, w4a4 {fmt(quant_frac, '.1%')} on-silhouette "
            f"-- ![overlay]({Path(row['figure']).as_posix()})"
        )
    resid = d.get("all_40_scenes_camera_residual_median_scene_radii", {})
    lines.append(
        f"- All-40-scene camera-center residual (median/scene radius): "
        f"mean={fmt(resid.get('mean'), '.4f')}, median={fmt(resid.get('median'), '.4f')}, max={fmt(resid.get('max'), '.4f')}"
    )
    return "\n".join(lines)


def body_a3(d: Dict[str, Any]) -> str:
    lines = [
        f"- Scenes tested (GPU-training subset): {', '.join(d.get('scenes_tested') or [])}",
        f"- Mean oracle PSNR: **{fmt(d.get('mean_oracle_psnr'), '.2f')} dB**",
        f"- Min oracle PSNR across tested scenes: {fmt(d.get('min_oracle_psnr'), '.2f')} dB",
        "",
        "| scene | oracle PSNR | full PSNR (ref) | w4a4 PSNR (ref) |",
        "|---|---:|---:|---:|",
    ]
    for r in d.get("per_scene") or []:
        lines.append(
            f"| {r['scene']} | {fmt(r.get('mean_psnr'), '.2f')} | {fmt(r.get('full_psnr_for_reference'), '.2f')} | {fmt(r.get('w4a4_psnr_for_reference'), '.2f')} |"
        )
    tgt = d.get("target_db")
    deriv = (d.get("target_derivation") or {}).get("derivation", "")
    lines.append(
        f"\nPrimary mask: **{d.get('primary_mask', 'foreground')}**. Content-mask mean (secondary): "
        f"{fmt(d.get('mean_oracle_psnr_content_secondary'), '.2f')} dB."
    )
    lines.append(
        f"\nTarget: **>= {fmt(tgt, '.3f')} dB**, derived rather than hand-picked. {deriv}"
    )
    lines.append(
        "\nThe previous 20 dB content-mask target was unreachable by construction: with a pixel-perfect object "
        "and the best possible flat background the content mask caps at 16.81 dB, because the object is a mean "
        "15.3% of that region and CO3D supplies GT depth for ~0.1% of background pixels (ongoing_logs.md #002, "
        "confirmed to 0.14 dB by #009)."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- Gate B

def body_b_generic(d: Dict[str, Any]) -> str:
    lines = [
        f"- Scenes evaluated: {d.get('n_scenes')}",
        f"- Mean Full PSNR ({d.get('mask_kind')} mask): **{fmt(d.get('mean_full_psnr'), '.2f')} dB**",
        f"- Mean W4A4 PSNR: **{fmt(d.get('mean_w4a4_psnr'), '.2f')} dB**",
        f"- Mean Full-minus-Quant: {fmt(d.get('mean_full_minus_quant_psnr'), '+.3f')} dB",
    ]
    if "baseline_content_mask_mean_full_psnr" in d:
        lines.append(
            f"- Baseline (rectangular content-mask) Full PSNR for comparison: {fmt(d.get('baseline_content_mask_mean_full_psnr'), '.2f')} dB"
        )
    return "\n".join(lines)


def body_b3(d: Dict[str, Any]) -> str:
    return (
        f"- Scenes evaluated: {d.get('n_scenes')}\n"
        f"- A2 noise-floor reference: {fmt(d.get('a2_noise_floor_reference_db'), '.2f')} dB\n"
        f"- Mean object (foreground) PSNR, Full: **{fmt(d.get('mean_object_full_psnr'), '.2f')} dB**\n"
        f"- Mean background PSNR, Full: **{fmt(d.get('mean_background_full_psnr'), '.2f')} dB**\n"
        "\nIf background PSNR sits near the noise floor while object PSNR sits well above it, the aggregate metric "
        "in the main run was diluted by an unreconstructable background."
    )


# --------------------------------------------------------------------------- Gate C

def body_c1(d: Dict[str, Any]) -> str:
    lines = [
        f"- Scenes tested: {', '.join(d.get('scenes_tested') or [])}",
        f"- Mean PSNR, random init (GT-frame cameras): **{fmt(d.get('mean_random_psnr'), '.2f')} dB**",
        f"- Mean PSNR, Full (GT-frame cameras): **{fmt(d.get('mean_full_psnr_gt_frame'), '.2f')} dB**",
        f"- Mean PSNR, W4A4 (GT-frame cameras): **{fmt(d.get('mean_w4a4_psnr_gt_frame'), '.2f')} dB**",
        f"- Scenes where random is within 0.5 dB of full (init doesn't matter): {d.get('n_scenes_where_random_almost_equals_full_lt_0.5db')}",
        "",
        "| scene | random | w4a4 | full |",
        "|---|---:|---:|---:|",
    ]
    for r in d.get("per_scene") or []:
        a = r.get("arms", {})
        lines.append(f"| {r['scene']} | {fmt(a.get('random'), '.2f')} | {fmt(a.get('w4a4'), '.2f')} | {fmt(a.get('full'), '.2f')} |")
    lines.append("\nTarget ordering: random < quant <= full, with a visible margin. If random ~= full, initialization is inert in this harness.")
    return "\n".join(lines)


def body_c2(d: Dict[str, Any]) -> str:
    return (
        f"- Scenes tested: {', '.join(d.get('scenes_tested') or [])}\n"
        f"- Mean Full-minus-Quant gap with densification OFF: **{fmt(d.get('mean_gap_densify_off'), '+.3f')} dB**\n"
        f"- Mean Full-minus-Quant gap with densification ON (baseline, same scenes): {fmt(d.get('mean_gap_baseline_with_densify'), '+.3f')} dB\n"
        "\nExpectation: the gap grows once adaptive density control can no longer repair the initial geometry."
    )


def body_c3(d: Dict[str, Any]) -> str:
    by_iter = d.get("mean_gap_by_iteration") or {}
    lines = [
        f"- Scenes tested: {', '.join(d.get('scenes_tested') or [])}",
        f"- Gap shrinks with more iterations: **{d.get('gap_shrinks_with_iterations')}**",
        "",
        "| iteration | mean Full-minus-Quant gap (dB) |",
        "|---:|---:|",
    ]
    for it, gap in sorted(by_iter.items(), key=lambda kv: int(kv[0])):
        lines.append(f"| {it} | {fmt(gap, '+.3f')} |")
    lines.append("\nA gap that shrinks toward 30k iterations converts the finding from a quality claim into a convergence-speed claim (a legitimate, weaker but still publishable result). A flat line from iteration 500 onward would instead point back to a Gate A failure.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- Gate D

def body_d1(d: Dict[str, Any]) -> str:
    ci = d.get("ci95") or [None, None]
    return (
        f"- n = {d.get('n_scenes')}, mean delta = {fmt(d.get('mean_delta_db'), '.3f')} dB, sd = {fmt(d.get('sd_delta_db'), '.3f')} dB\n"
        f"- Paired t = {fmt(d.get('paired_t'), '.2f')}, p = {fmt(d.get('p_value'), '.4f')}\n"
        f"- 95% CI: [{fmt(ci[0], '.3f')}, {fmt(ci[1], '.3f')}] dB\n"
        f"- Scenes favoring Full: {d.get('scenes_favor_full')} / {d.get('n_scenes')}\n"
        f"- Minimum detectable effect at n=40, 80% power: {fmt(d.get('min_detectable_effect_at_n40_80pct_power_db'), '.3f')} dB\n"
        f"- Scenes needed to reliably detect a 0.1 dB effect at 80% power: ~{fmt(d.get('n_scenes_needed_to_detect_0.1db_at_80pct_power'), '.0f')}\n"
        f"\n{d.get('read', '')}"
    )


def body_d2(d: Dict[str, Any]) -> str:
    lines = [f"- Scenes compared: {d.get('n_scenes')}", "", "| disagreement metric | rho (content-mask baseline) | rho (object-mask rerun) |", "|---|---:|---:|"]
    for metric, vals in (d.get("by_metric") or {}).items():
        lines.append(f"| {metric} | {fmt(vals.get('rho_content_mask_baseline'), '.3f')} | {fmt(vals.get('rho_object_mask_rerun'), '.3f')} |")
    lines.append(f"\n{d.get('read', '')}")
    return "\n".join(lines)


def body_d3(d: Dict[str, Any]) -> str:
    return (
        f"**Status: {d.get('status')}**\n\n"
        f"{d.get('reason', '')}\n\n"
        f"**Next step:** {d.get('next_step', '')}"
    )


# --------------------------------------------------------------------------- Gate E

def body_e1(d: Dict[str, Any]) -> str:
    lines = [
        f"- Scenes evaluated: {d.get('n_scenes')}",
        f"- Mean F-score @ tau, Full: **{fmt(d.get('mean_fscore_full'), '.3f')}**",
        f"- Mean F-score @ tau, W4A4: **{fmt(d.get('mean_fscore_w4a4'), '.3f')}**",
        f"- Mean gap (Full - Quant): {fmt(d.get('mean_gap_fscore'), '+.4f')}",
        "\nThis compares the trained Gaussian centres directly against the dense CO3D ground-truth point cloud -- "
        "it survives photometric self-healing that can hide geometric error from PSNR.",
    ]
    return "\n".join(lines)


def body_e3(d: Dict[str, Any]) -> str:
    return (
        f"- Scenes evaluated: {d.get('n_scenes')}\n"
        f"- Mean opacity-weighted floater mass, Full: **{fmt(d.get('mean_floater_mass_full'), '.4f')}**\n"
        f"- Mean opacity-weighted floater mass, W4A4: **{fmt(d.get('mean_floater_mass_w4a4'), '.4f')}**\n"
        f"- Mean Gaussian count, Full: {fmt(d.get('mean_n_gaussians_full'), ',.0f')}\n"
        f"- Mean Gaussian count, W4A4: {fmt(d.get('mean_n_gaussians_w4a4'), ',.0f')}"
    )


def body_e2(d: Dict[str, Any]) -> str:
    return (
        f"- Scenes evaluated: {d.get('n_scenes')}\n"
        f"- Mean depth error (fraction of scene radius), Full: **{fmt(d.get('mean_depth_error_full'), '.4f')}**\n"
        f"- Mean depth error, W4A4: **{fmt(d.get('mean_depth_error_w4a4'), '.4f')}**\n"
        f"- Mean boundary/interior depth-error enrichment, Full: {fmt(d.get('mean_boundary_enrichment_full'), '.2f')}x\n"
        f"- Mean boundary/interior depth-error enrichment, W4A4: {fmt(d.get('mean_boundary_enrichment_w4a4'), '.2f')}x"
    )


GATES = [
    ("Gate A", "Is the harness measuring anything?", [
        ("A2", "Noise-floor calibration", body_a2),
        ("A1", "Training-view PSNR (self-fit test)", body_a1),
        ("A5", "Leave-one-out Sim(3) extrapolation error", body_a5),
        ("A6", "Oracle self-fit at its own training views", body_a6),
        ("A4", "Held-out reprojection overlay", body_a4),
        ("A3", "Oracle control: GT geometry end-to-end", body_a3),
    ]),
    ("Gate B", "Is the metric looking at the right pixels?", [
        ("B1", "Object-mask metrics", body_b_generic),
        ("B2", "Boundary-band metrics", body_b_generic),
        ("B3", "Object / background split", body_b3),
    ]),
    ("Gate C", "Does the geometry prior matter here at all?", [
        ("C1", "Random-initialisation control", body_c1),
        ("C2", "Densification disabled", body_c2),
        ("C3", "Iteration sweep", body_c3),
    ]),
    ("Gate D", "Is the effect large enough to detect?", [
        ("D1", "Paired statistics and power", body_d1),
        ("D2", "Disagreement -> downstream-damage correlation, rerun", body_d2),
        ("D3", "Harsher quantization", body_d3),
    ]),
    ("Gate E", "Geometry evidence, not photometry", [
        ("E1", "Chamfer distance of final Gaussians", body_e1),
        ("E2", "Rendered depth error on held-out views", body_e2),
        ("E3", "Floater and opacity audit", body_e3),
    ]),
]


def render() -> None:
    state = common.load_state()
    tests = state.get("tests", {})
    n_done = sum(1 for t in tests.values() if t.get("status") == "done")
    n_failed = sum(1 for t in tests.values() if t.get("status") == "failed")

    out = []
    out.append("# Downstream-3DGS Diagnostic Gates: Results")
    out.append("")
    out.append(
        f"_Auto-generated by `code/downstream_3dgs/diagnostics/run_all_diagnostics.py`. "
        f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}. "
        f"{n_done}/15 tests done, {n_failed} failed, {15 - n_done - n_failed} pending._"
    )
    out.append("")
    out.append(
        "Implements the 15 diagnostic tests from the Full-VGGT-vs-W4A4-QuantVGGT downstream-3DGS "
        "diagnostic checklist. A2/A1/A5/B1-B3/D1/D2/E1/E3 run against all 40 scenes and reuse "
        "already-computed renders/models (no new training). A3/C1/C2/C3 require brand-new 3DGS "
        f"training and run on an 8-scene subset ({', '.join(common.EXPENSIVE_SCENES)}) to keep "
        "wall-clock bounded. D3 needs a new QuantVGGT calibration artifact that does not exist in "
        "this environment and is reported as blocked rather than faked."
    )
    out.append("")

    for gate_name, gate_desc, entries in GATES:
        out.append(f"## {gate_name}: {gate_desc}")
        out.append("")
        for test_id, title, body_fn in entries:
            out.append(section(test_id, title, tests, body_fn))

    out.append("## Notes")
    out.append("")
    out.append(
        "- All PSNR/SSIM numbers reuse `run_downstream_validation.py`'s own `psnr`/`ssim_value` "
        "implementations, so they are directly comparable to `results/downstream_3dgs/aggregate_metrics.json`.\n"
        "- The 8-scene expensive subset spans 8 categories and both signs of the baseline "
        "Full-minus-Quant PSNR delta.\n"
        "- Figures referenced above are written under `results/downstream_3dgs/diagnostics/figures/`.\n"
        "- Raw state: `results/downstream_3dgs/diagnostics/state.json`."
    )
    out.append("")

    common.REPORT_PATH.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    render()
    print(f"Wrote {common.REPORT_PATH}")
