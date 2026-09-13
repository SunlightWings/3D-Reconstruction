#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "run_downstream_validation.py"
HEAVY_ROOT = Path("/var/tmp/luli38se/quantsplat/downstream_validation_v1")
STATE_ROOT = Path("/home/utn/luli38se/cv/downstream_full_dataset_v1_state")
COMPLETED_ROOT = STATE_ROOT / "completed"
ITER_EARLY = 7000
ITER_FINAL = 30000
HELDOUT_VIEWS = 9
STRIDE = 4


def load_base():
    if not BASE_SCRIPT.is_file():
        raise FileNotFoundError(BASE_SCRIPT)
    spec = importlib.util.spec_from_file_location("downstream_base_7k", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.OUT_ROOT = HEAVY_ROOT
    return mod


base = load_base()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def discover_scenes():
    pairs = set()
    for p in base.PRED_ROOT.rglob("full_meta.json"):
        try:
            m = read_json(p)
        except Exception:
            continue
        if m.get("category") and m.get("sequence"):
            pairs.add((str(m["category"]), str(m["sequence"])))
    scenes = sorted(pairs)
    if len(scenes) != 40:
        raise RuntimeError(f"Expected 40 scenes, found {len(scenes)}")
    return scenes


def checkpoint(category: str, sequence: str, variant: str, iteration: int) -> Path:
    return (
        HEAVY_ROOT / category / sequence / "models" / variant / "point_cloud"
        / f"iteration_{iteration}" / "point_cloud.ply"
    )


def final_metrics_path(category: str, sequence: str) -> Path:
    p = COMPLETED_ROOT / category / sequence / "metrics.json"
    if p.is_file():
        return p
    q = HEAVY_ROOT / category / sequence / "metrics.json"
    if q.is_file():
        return q
    raise FileNotFoundError(f"Missing 30k metrics for {category}/{sequence}")


def srow(category: str, sequence: str, early: dict, final: dict) -> dict:
    e = early["summary"]
    f = final["summary"]
    return {
        "category": category,
        "sequence": sequence,
        "full_psnr_7k": float(e["full_mean_psnr_content"]),
        "quant_psnr_7k": float(e["quant_mean_psnr_content"]),
        "full_minus_quant_psnr_7k": float(e["full_minus_quant_psnr_content"]),
        "full_ssim_7k": float(e["full_mean_ssim_content"]),
        "quant_ssim_7k": float(e["quant_mean_ssim_content"]),
        "full_minus_quant_ssim_7k": float(e["full_minus_quant_ssim_content"]),
        "full_lpips_7k": float(e["full_mean_lpips_content"]),
        "quant_lpips_7k": float(e["quant_mean_lpips_content"]),
        "quant_minus_full_lpips_7k": float(e["quant_minus_full_lpips_content"]),
        "full_psnr_30k": float(f["full_mean_psnr_content"]),
        "quant_psnr_30k": float(f["quant_mean_psnr_content"]),
        "full_minus_quant_psnr_30k": float(f["full_minus_quant_psnr_content"]),
        "full_ssim_30k": float(f["full_mean_ssim_content"]),
        "quant_ssim_30k": float(f["quant_mean_ssim_content"]),
        "full_minus_quant_ssim_30k": float(f["full_minus_quant_ssim_content"]),
        "full_lpips_30k": float(f["full_mean_lpips_content"]),
        "quant_lpips_30k": float(f["quant_mean_lpips_content"]),
        "quant_minus_full_lpips_30k": float(f["quant_minus_full_lpips_content"]),
    }


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def median(rows, key):
    return float(np.median([r[key] for r in rows]))


def aggregate(rows):
    # Improvements from 7k -> 30k. Positive means quality improved.
    full_psnr_gain = [r["full_psnr_30k"] - r["full_psnr_7k"] for r in rows]
    quant_psnr_gain = [r["quant_psnr_30k"] - r["quant_psnr_7k"] for r in rows]
    full_ssim_gain = [r["full_ssim_30k"] - r["full_ssim_7k"] for r in rows]
    quant_ssim_gain = [r["quant_ssim_30k"] - r["quant_ssim_7k"] for r in rows]
    full_lpips_gain = [r["full_lpips_7k"] - r["full_lpips_30k"] for r in rows]
    quant_lpips_gain = [r["quant_lpips_7k"] - r["quant_lpips_30k"] for r in rows]

    out = {
        "scene_count": len(rows),
        "iteration_early": ITER_EARLY,
        "iteration_final": ITER_FINAL,
        "at_7k": {
            "mean_full_psnr": mean(rows, "full_psnr_7k"),
            "mean_quant_psnr": mean(rows, "quant_psnr_7k"),
            "mean_full_minus_quant_psnr": mean(rows, "full_minus_quant_psnr_7k"),
            "median_full_minus_quant_psnr": median(rows, "full_minus_quant_psnr_7k"),
            "scenes_full_better_psnr": int(sum(r["full_minus_quant_psnr_7k"] > 0 for r in rows)),
            "mean_full_ssim": mean(rows, "full_ssim_7k"),
            "mean_quant_ssim": mean(rows, "quant_ssim_7k"),
            "mean_full_minus_quant_ssim": mean(rows, "full_minus_quant_ssim_7k"),
            "median_full_minus_quant_ssim": median(rows, "full_minus_quant_ssim_7k"),
            "scenes_full_better_ssim": int(sum(r["full_minus_quant_ssim_7k"] > 0 for r in rows)),
            "mean_full_lpips": mean(rows, "full_lpips_7k"),
            "mean_quant_lpips": mean(rows, "quant_lpips_7k"),
            "mean_quant_minus_full_lpips": mean(rows, "quant_minus_full_lpips_7k"),
            "median_quant_minus_full_lpips": median(rows, "quant_minus_full_lpips_7k"),
            "scenes_full_better_lpips": int(sum(r["quant_minus_full_lpips_7k"] > 0 for r in rows)),
        },
        "at_30k": {
            "mean_full_psnr": mean(rows, "full_psnr_30k"),
            "mean_quant_psnr": mean(rows, "quant_psnr_30k"),
            "mean_full_minus_quant_psnr": mean(rows, "full_minus_quant_psnr_30k"),
            "median_full_minus_quant_psnr": median(rows, "full_minus_quant_psnr_30k"),
            "scenes_full_better_psnr": int(sum(r["full_minus_quant_psnr_30k"] > 0 for r in rows)),
            "mean_full_ssim": mean(rows, "full_ssim_30k"),
            "mean_quant_ssim": mean(rows, "quant_ssim_30k"),
            "mean_full_minus_quant_ssim": mean(rows, "full_minus_quant_ssim_30k"),
            "median_full_minus_quant_ssim": median(rows, "full_minus_quant_ssim_30k"),
            "scenes_full_better_ssim": int(sum(r["full_minus_quant_ssim_30k"] > 0 for r in rows)),
            "mean_full_lpips": mean(rows, "full_lpips_30k"),
            "mean_quant_lpips": mean(rows, "quant_lpips_30k"),
            "mean_quant_minus_full_lpips": mean(rows, "quant_minus_full_lpips_30k"),
            "median_quant_minus_full_lpips": median(rows, "quant_minus_full_lpips_30k"),
            "scenes_full_better_lpips": int(sum(r["quant_minus_full_lpips_30k"] > 0 for r in rows)),
        },
        "additional_optimization_7k_to_30k": {
            "mean_full_psnr_gain": float(np.mean(full_psnr_gain)),
            "mean_quant_psnr_gain": float(np.mean(quant_psnr_gain)),
            "quant_minus_full_psnr_gain": float(np.mean(quant_psnr_gain) - np.mean(full_psnr_gain)),
            "scenes_quant_gained_more_psnr": int(sum(q > f for q, f in zip(quant_psnr_gain, full_psnr_gain))),
            "mean_full_ssim_gain": float(np.mean(full_ssim_gain)),
            "mean_quant_ssim_gain": float(np.mean(quant_ssim_gain)),
            "quant_minus_full_ssim_gain": float(np.mean(quant_ssim_gain) - np.mean(full_ssim_gain)),
            "scenes_quant_gained_more_ssim": int(sum(q > f for q, f in zip(quant_ssim_gain, full_ssim_gain))),
            "mean_full_lpips_improvement": float(np.mean(full_lpips_gain)),
            "mean_quant_lpips_improvement": float(np.mean(quant_lpips_gain)),
            "quant_minus_full_lpips_improvement": float(np.mean(quant_lpips_gain) - np.mean(full_lpips_gain)),
            "scenes_quant_improved_more_lpips": int(sum(q > f for q, f in zip(quant_lpips_gain, full_lpips_gain))),
        },
        "gap_change_7k_to_30k": {
            "mean_psnr_full_advantage_change_30k_minus_7k": mean(rows, "full_minus_quant_psnr_30k") - mean(rows, "full_minus_quant_psnr_7k"),
            "mean_ssim_full_advantage_change_30k_minus_7k": mean(rows, "full_minus_quant_ssim_30k") - mean(rows, "full_minus_quant_ssim_7k"),
            "mean_lpips_full_advantage_change_30k_minus_7k": mean(rows, "quant_minus_full_lpips_30k") - mean(rows, "quant_minus_full_lpips_7k"),
        },
        "scenes": rows,
    }
    return out


def main():
    base.preflight()
    scenes = discover_scenes()

    missing = []
    for category, sequence in scenes:
        for variant in ("full", "w4a4"):
            for it in (ITER_EARLY, ITER_FINAL):
                p = checkpoint(category, sequence, variant, it)
                if not p.is_file():
                    missing.append(str(p))
    if missing:
        print("Missing required checkpoints:")
        for p in missing:
            print("  ", p)
        raise SystemExit(f"ABORT: {len(missing)} required checkpoint(s) missing")

    print("CHECKPOINT COVERAGE PASS: 40/40 scenes have Full+W4A4 at 7k and 30k", flush=True)
    rows = []
    for idx, (category, sequence) in enumerate(scenes, 1):
        name = f"{category}/{sequence}"
        print("\n" + "=" * 72, flush=True)
        print(f"[{idx:02d}/40] {name} @ 7000", flush=True)
        print("=" * 72, flush=True)

        p = base.prepare_scene(category, sequence, STRIDE, HELDOUT_VIEWS)
        scene_root = Path(p["scene_root"])
        conv_root = scene_root / "convergence" / f"iteration_{ITER_EARLY}"
        full_model = scene_root / "models" / "full"
        quant_model = scene_root / "models" / "w4a4"

        full_renders = base.render_heldout(
            p["full_heldout_source"], full_model, conv_root / "heldout" / "full",
            ITER_EARLY, HELDOUT_VIEWS, 10, 1,
        )
        quant_renders = base.render_heldout(
            p["quant_heldout_source"], quant_model, conv_root / "heldout" / "w4a4",
            ITER_EARLY, HELDOUT_VIEWS, 10, 1,
        )
        early = base.evaluate_scene(
            conv_root,
            p["heldout_frames"],
            p["content_masks"],
            full_renders,
            quant_renders,
            p["heldout_image_dir"],
        )
        final = read_json(final_metrics_path(category, sequence))
        row = srow(category, sequence, early, final)
        rows.append(row)
        print(
            f"7k:  F-Q PSNR={row['full_minus_quant_psnr_7k']:+.3f} dB  "
            f"F-Q SSIM={row['full_minus_quant_ssim_7k']:+.5f}  "
            f"Q-F LPIPS={row['quant_minus_full_lpips_7k']:+.5f}",
            flush=True,
        )

    agg = aggregate(rows)
    out_json = STATE_ROOT / "convergence_7k_vs_30k.json"
    out_csv = STATE_ROOT / "convergence_7k_vs_30k.csv"
    write_json(out_json, agg)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    a7 = agg["at_7k"]
    a30 = agg["at_30k"]
    gain = agg["additional_optimization_7k_to_30k"]
    print("\n" + "=" * 72)
    print("7K vs 30K CONVERGENCE RESULT")
    print("=" * 72)
    print(f"7k  mean Full-Quant PSNR: {a7['mean_full_minus_quant_psnr']:+.4f} dB")
    print(f"30k mean Full-Quant PSNR: {a30['mean_full_minus_quant_psnr']:+.4f} dB")
    print(f"7k  Full better PSNR:     {a7['scenes_full_better_psnr']}/40")
    print(f"30k Full better PSNR:     {a30['scenes_full_better_psnr']}/40")
    print()
    print(f"Full PSNR gain 7k->30k:   {gain['mean_full_psnr_gain']:+.4f} dB")
    print(f"Quant PSNR gain 7k->30k:  {gain['mean_quant_psnr_gain']:+.4f} dB")
    print(f"Extra Quant PSNR gain:    {gain['quant_minus_full_psnr_gain']:+.4f} dB")
    print(f"Quant gained more PSNR:   {gain['scenes_quant_gained_more_psnr']}/40 scenes")
    print()
    print(f"7k  mean F-Q SSIM:        {a7['mean_full_minus_quant_ssim']:+.5f}")
    print(f"30k mean F-Q SSIM:        {a30['mean_full_minus_quant_ssim']:+.5f}")
    print(f"Quant gained more SSIM:   {gain['scenes_quant_gained_more_ssim']}/40 scenes")
    print()
    print(f"7k  mean Q-F LPIPS:       {a7['mean_quant_minus_full_lpips']:+.5f}")
    print(f"30k mean Q-F LPIPS:       {a30['mean_quant_minus_full_lpips']:+.5f}")
    print(f"Quant improved more LPIPS:{gain['scenes_quant_improved_more_lpips']}/40 scenes")
    print()
    print("JSON:", out_json)
    print("CSV: ", out_csv)


if __name__ == "__main__":
    main()
