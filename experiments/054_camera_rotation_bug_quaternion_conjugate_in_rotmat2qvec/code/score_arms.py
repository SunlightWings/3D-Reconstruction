"""Score any set of held-out render arms with run_w2a4_downstream.score_scene (metrics unchanged), paired over scenes.

Used for arms that did not come out of run_w2a4_downstream itself (e.g. gsplat_posefix.py). An arm is scored on a
scene only if its renders directory holds exactly one PNG per held-out frame. Every requested comparison reports the
paired mean, 95 % CI, t-test p, Wilcoxon p and a sign count, oriented so that POSITIVE = first arm is BETTER on every
metric (PSNR/SSIM higher, LPIPS lower).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path("/home/utn/luli38se/cv/3D-Reconstruction")
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "diagnostics"))
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "oracle_sweep"))
import common  # noqa: E402

RES = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="first:second, e.g. gs_w3a3_pose:gs_w3a3")
    ap.add_argument("--scenes", choices=["subset", "all"], default="subset")
    ap.add_argument("--iterations", type=int, default=7000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("rwd", REPO / "code" / "quantization" / "run_w2a4_downstream.py")
    rwd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rwd)

    names = common.EXPENSIVE_SCENES if a.scenes == "subset" else common.all_scenes()
    rows = []
    for n in names:
        cat, seq = n.split("/", 1)
        root = common.scene_root(cat, seq)
        n_held = len(common.scene_manifest(cat, seq)["heldout_frames"])
        present = {}
        for arm in a.arms:
            d = root / "heldout" / f"{arm}_it{a.iterations}" / "renders"
            if d.is_dir() and len(list(d.glob("*.png"))) == n_held:
                present[arm] = d
        if not present:
            continue
        s = rwd.score_scene(cat, seq, present)
        row = {"scene": n}
        for arm, regs in s.items():
            for reg in ("foreground", "content"):
                for m in ("psnr", "ssim", "lpips"):
                    row[f"{arm}_{reg}_{m}"] = regs[reg][m]
        rows.append(row)
        print(f"  {n[:30]:30s} " + "  ".join(f"{arm}={row.get(f'{arm}_foreground_psnr', float('nan')):.3f}"
                                            for arm in a.arms), flush=True)

    print("\n" + "=" * 100)
    print(f"FOREGROUND means over scenes where each arm is complete")
    summary = {"arms": {}, "pairs": {}, "per_scene": rows}
    for arm in a.arms:
        v = [r for r in rows if f"{arm}_foreground_psnr" in r]
        if v:
            summary["arms"][arm] = {"n": len(v), **{m: float(np.mean([r[f"{arm}_foreground_{m}"] for r in v]))
                                                    for m in ("psnr", "ssim", "lpips")}}
            print(f"  {arm:28s} n={len(v):2d}  PSNR {summary['arms'][arm]['psnr']:.4f}  "
                  f"SSIM {summary['arms'][arm]['ssim']:.4f}  LPIPS {summary['arms'][arm]['lpips']:.4f}")
    print("\npaired, POSITIVE = first arm BETTER")
    for pr in a.pairs:
        x, y = pr.split(":")
        for m, sgn in (("psnr", 1), ("ssim", 1), ("lpips", -1)):
            d = np.array([sgn * (r[f"{x}_foreground_{m}"] - r[f"{y}_foreground_{m}"]) for r in rows
                          if f"{x}_foreground_{m}" in r and f"{y}_foreground_{m}" in r])
            if len(d) < 3:
                continue
            t = stats.ttest_1samp(d, 0)
            se = d.std(ddof=1) / np.sqrt(len(d))
            c = stats.t.ppf(0.975, len(d) - 1)
            w = stats.wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
            summary["pairs"][f"{x}__vs__{y}__{m}"] = {"n": len(d), "mean": float(d.mean()), "t_p": float(t.pvalue),
                                                      "wilcoxon_p": float(w), "ci95": [float(d.mean() - c * se), float(d.mean() + c * se)],
                                                      "first_better": int((d > 0).sum())}
            print(f"  {x:22s} vs {y:22s} {m:5s} mean {d.mean():+.4f}  t p {t.pvalue:.4f}  Wilcoxon p {w:.4f}  "
                  f"CI [{d.mean() - c * se:+.4f},{d.mean() + c * se:+.4f}]  better {(d > 0).sum()}/{len(d)}")
    RES.mkdir(parents=True, exist_ok=True)
    p = RES / f"score_{a.out}.json"
    p.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
