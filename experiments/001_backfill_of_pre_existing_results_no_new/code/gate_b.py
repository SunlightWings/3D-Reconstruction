"""Gate B -- is the metric looking at the right pixels?

B1 object-mask metrics, B2 boundary-band metrics, B3 object/background split.
All three are CPU-only recomputations over renders that already exist on disk
(no GPU, no retraining) across all 40 scenes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.ndimage import binary_erosion

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

dv = common.dv
dis = common.dis


def _masks_for_scene(category: str, sequence: str, heldout_frames):
    """RGB-PSNR masks: gated on the CO3D foreground silhouette and the padded content
    rectangle only. Unlike characterize_disagreement_regions.py's cue_masks(), these must
    NOT also require GT-depth validity -- PSNR is a 2D RGB comparison and does not need a
    3D point at that pixel; gating on depth validity here would (and did) wrongly zero out
    almost the entire background region, since CO3D only supplies confident depth over the
    foreground object.
    """
    records = dv.load_annotations(category, sequence)
    content_masks = {}
    fg_masks = {}
    band_masks = {}
    bg_masks = {}
    for f in heldout_frames:
        content = dv.content_mask_from_record(records[f])
        gt = dis.decode_gt_frame(records[f], dv.CO3D_ROOT, common.pad_transform)
        fg = (gt["fg"] >= 0.5) & content
        eroded2 = binary_erosion(fg, iterations=2)
        band = fg & ~eroded2  # same boundary-ring definition as characterize_disagreement_regions.py
        content_masks[f] = content
        fg_masks[f] = fg
        band_masks[f] = band
        bg_masks[f] = content & ~fg
    return content_masks, fg_masks, band_masks, bg_masks


def _psnr_ssim_for_masks(category: str, sequence: str, heldout_frames, masks: Dict[int, np.ndarray]) -> Dict[str, Any]:
    root = common.scene_root(category, sequence)
    gt_dir = root / "common" / "heldout_images"
    out = {}
    for variant in ("full", "w4a4"):
        renders_dir = root / "heldout" / variant / "renders"
        render_paths = sorted(renders_dir.glob("*.png"))
        gt_paths = [gt_dir / f"frame{f:06d}.png" for f in heldout_frames]
        if len(render_paths) != len(heldout_frames):
            raise RuntimeError(f"{category}/{sequence}/{variant}: expected {len(heldout_frames)} renders")
        mask_list = [masks[f] for f in heldout_frames]
        result = common.evaluate_renders(gt_paths, render_paths, mask_list)
        out[variant] = result["mean_psnr"]
    out["full_minus_quant"] = out["full"] - out["w4a4"]
    return out


def _run_masked_metric(mask_kind: str) -> Dict[str, Any]:
    per_scene = []
    for name in common.all_scenes():
        category, sequence = common.split_scene(name)
        manifest = common.scene_manifest(category, sequence)
        heldout_frames = manifest["heldout_frames"]
        content_masks, fg_masks, band_masks, _bg_masks = _masks_for_scene(category, sequence, heldout_frames)
        masks = {"content": content_masks, "foreground": fg_masks, "boundary_band": band_masks}[mask_kind]
        row = {"scene": name}
        row.update(_psnr_ssim_for_masks(category, sequence, heldout_frames, masks))
        per_scene.append(row)
        common.log(f"B/{mask_kind} {name}: full={row['full']:.2f} dB, w4a4={row['w4a4']:.2f} dB, delta={row['full_minus_quant']:+.3f} dB")
    finite = lambda key: [r[key] for r in per_scene if np.isfinite(r[key])]  # noqa: E731
    return {
        "mask_kind": mask_kind,
        "n_scenes": len(per_scene),
        "mean_full_psnr": float(np.mean(finite("full"))),
        "mean_w4a4_psnr": float(np.mean(finite("w4a4"))),
        "mean_full_minus_quant_psnr": float(np.mean(finite("full_minus_quant"))),
        "per_scene": per_scene,
    }


def test_b1_object_mask_metrics() -> Dict[str, Any]:
    result = _run_masked_metric("foreground")
    result["description"] = "PSNR restricted to the true CO3D foreground/object mask (not the padded content rectangle)."
    result["expect"] = "absolute PSNR rises substantially versus the rectangular content-mask baseline"
    baseline = common.read_json(common.RESULTS_DIR / "aggregate_metrics.json")
    result["baseline_content_mask_mean_full_psnr"] = baseline["mean_full_psnr"]
    result["baseline_content_mask_mean_quant_psnr"] = baseline["mean_quant_psnr"]
    return result


def test_b2_boundary_band_metrics() -> Dict[str, Any]:
    result = _run_masked_metric("boundary_band")
    result["description"] = "PSNR restricted to the object-boundary band (foreground minus a 2px-eroded interior), matching characterize_disagreement_regions.py's band definition."
    result["expect"] = "if quantization damage reaches rendering at all, it shows here first and largest"
    return result


def test_b3_object_background_split() -> Dict[str, Any]:
    per_scene = []
    for name in common.all_scenes():
        category, sequence = common.split_scene(name)
        manifest = common.scene_manifest(category, sequence)
        heldout_frames = manifest["heldout_frames"]
        _content_masks, fg_masks, _band, bg_masks = _masks_for_scene(category, sequence, heldout_frames)
        obj = _psnr_ssim_for_masks(category, sequence, heldout_frames, fg_masks)
        bg = _psnr_ssim_for_masks(category, sequence, heldout_frames, bg_masks)
        row = {
            "scene": name,
            "object_full_psnr": obj["full"], "object_w4a4_psnr": obj["w4a4"],
            "background_full_psnr": bg["full"], "background_w4a4_psnr": bg["w4a4"],
        }
        per_scene.append(row)
        common.log(f"B3 {name}: object full={obj['full']:.2f} dB, background full={bg['full']:.2f} dB")

    noise_floor = common.load_state()["tests"].get("A2", {}).get("data", {}).get("overall_median_of_scene_medians")
    return {
        "description": "Splits PSNR into object (foreground) vs background (padded content minus foreground) to quantify how much of the aggregate PSNR is diluted by unreconstructable background.",
        "expect": "background PSNR near the A2 noise floor; object PSNR clearly above it",
        "a2_noise_floor_reference_db": noise_floor,
        "mean_object_full_psnr": float(np.mean([r["object_full_psnr"] for r in per_scene])),
        "mean_background_full_psnr": float(np.mean([r["background_full_psnr"] for r in per_scene])),
        "per_scene": per_scene,
    }


TESTS = [
    ("B1", test_b1_object_mask_metrics),
    ("B2", test_b2_boundary_band_metrics),
    ("B3", test_b3_object_background_split),
]
