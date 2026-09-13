"""Re-score the 40-scene full/w4a4 held-out renders under BOTH masks.

The headline Full-minus-Quant result (+0.204 dB, p=0.0092) was computed on the content
mask, which is ~85% background on this dataset. A single-scene spot check during the metric
rewrite showed the per-scene delta changing SIGN between masks on apple (+0.130 dB content,
-0.405 dB foreground). This checks whether that is isolated or systematic, and recomputes
D1's paired statistics on the foreground mask.

No training, no rendering: reads renders already on disk.
"""
import json
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP.parent / "diagnostics"))
import common  # noqa: E402

dv = common.dv
OUT = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")


def paired_stats(deltas):
    """D1's paired test plus the minimum detectable effect at 80% power."""
    d = np.asarray([x for x in deltas if np.isfinite(x)], dtype=np.float64)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    t = mean / se if se > 0 else float("nan")
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
    except Exception:
        p = float("nan")
        tcrit = 1.96
    return {
        "n": n, "mean_delta": mean, "sd": sd, "t": float(t), "p": p,
        "ci95": [mean - tcrit * se, mean + tcrit * se],
        "mde_80pct_power": float(2.802 * sd / np.sqrt(n)),  # (z_.975 + z_.80) = 2.802
        "scenes_favouring_full": int((d > 0).sum()),
    }


def main():
    rows = []
    for name in common.all_scenes():
        cat, seq = common.split_scene(name)
        man = common.scene_manifest(cat, seq)
        recs = dv.load_annotations(cat, seq)
        held = man["heldout_frames"]
        root = common.scene_root(cat, seq)
        gt = [root / "common" / "heldout_images" / f"frame{f:06d}.png" for f in held]
        fr = sorted((root / "heldout" / "full" / "renders").glob("*.png"))
        qr = sorted((root / "heldout" / "w4a4" / "renders").glob("*.png"))
        if not (len(fr) == len(qr) == len(gt)):
            print(f"  SKIP {name}: render count mismatch")
            continue
        row = {"scene": name}
        for masktype in ("content", "foreground"):
            masks = [(dv.foreground_mask_from_record(recs[f]) if masktype == "foreground"
                      else dv.content_mask_from_record(recs[f])) for f in held]
            rf = common.evaluate_renders(gt, fr, masks, with_ssim=True)
            rq = common.evaluate_renders(gt, qr, masks, with_ssim=True)
            row[f"{masktype}_full_psnr"] = rf["mean_psnr"]
            row[f"{masktype}_quant_psnr"] = rq["mean_psnr"]
            row[f"{masktype}_delta"] = rf["mean_psnr"] - rq["mean_psnr"]
            row[f"{masktype}_full_ssim"] = rf.get("mean_ssim")
            row[f"{masktype}_quant_ssim"] = rq.get("mean_ssim")
            row[f"{masktype}_delta_ssim"] = (rf.get("mean_ssim") or 0) - (rq.get("mean_ssim") or 0)
        row["sign_agrees"] = bool(np.sign(row["content_delta"]) == np.sign(row["foreground_delta"]))
        rows.append(row)
        print(f"  {name:34s} content {row['content_delta']:+.3f}  foreground {row['foreground_delta']:+.3f}  "
              f"{'agree' if row['sign_agrees'] else 'FLIP'}", flush=True)

    agree = sum(r["sign_agrees"] for r in rows)
    out = {
        "description": "40-scene full/w4a4 held-out renders re-scored under both masks.",
        "n_scenes": len(rows),
        "content": {
            "mean_full_psnr": float(np.mean([r["content_full_psnr"] for r in rows])),
            "mean_quant_psnr": float(np.mean([r["content_quant_psnr"] for r in rows])),
            "mean_delta": float(np.mean([r["content_delta"] for r in rows])),
            "mean_delta_ssim": float(np.mean([r["content_delta_ssim"] for r in rows])),
            "d1": paired_stats([r["content_delta"] for r in rows]),
        },
        "foreground": {
            "mean_full_psnr": float(np.mean([r["foreground_full_psnr"] for r in rows])),
            "mean_quant_psnr": float(np.mean([r["foreground_quant_psnr"] for r in rows])),
            "mean_delta": float(np.mean([r["foreground_delta"] for r in rows])),
            "mean_delta_ssim": float(np.mean([r["foreground_delta_ssim"] for r in rows])),
            "d1": paired_stats([r["foreground_delta"] for r in rows]),
        },
        "sign_agreement_rate": agree / len(rows),
        "scenes_sign_agree": agree,
        "scenes_sign_flip": len(rows) - agree,
        "flipped_scenes": [r["scene"] for r in rows if not r["sign_agrees"]],
        "per_scene": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "main40_both_masks.json").write_text(json.dumps(out, indent=2))

    print("\n=== 40-scene means ===")
    for m in ("content", "foreground"):
        d = out[m]
        print(f"{m:11s} full={d['mean_full_psnr']:7.4f}  w4a4={d['mean_quant_psnr']:7.4f}  "
              f"delta={d['mean_delta']:+.4f} dB")
        s = d["d1"]
        print(f"            D1: n={s['n']} mean={s['mean_delta']:+.4f} sd={s['sd']:.4f} t={s['t']:.3f} "
              f"p={s['p']:.4g} CI95=[{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}] "
              f"MDE={s['mde_80pct_power']:.4f} favouring_full={s['scenes_favouring_full']}/{s['n']}")
    print(f"\nsign agreement: {agree}/{len(rows)} = {agree/len(rows)*100:.1f}%  "
          f"({len(rows)-agree} scenes flip)")
    print(f"saved {OUT/'main40_both_masks.json'}")


if __name__ == "__main__":
    main()
