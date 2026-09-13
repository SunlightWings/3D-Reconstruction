"""Re-score the existing C1 renders (random / w4a4 / full) plus the oracle under
BOTH the content mask and the foreground mask.

C1's published verdict was computed on the content mask, which is ~85% background
(see ongoing_logs.md entry #002). Since the init prior only ever touches geometry
the model can actually place, the ordering test belongs on the foreground.

No training and no rendering: this reads renders that already exist on disk.
"""
import json
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP.parent / "diagnostics"))
import common  # noqa: E402
from reeval import fgmask  # noqa: E402

dv = common.dv
OUT = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")
ARMS = ["c1_random", "c1_w4a4", "c1_full", "oracle"]


def score_arm(arm: str, masktype: str):
    rows = []
    for name in common.EXPENSIVE_SCENES:
        cat, seq = common.split_scene(name)
        rdir = common.DIAG_HEAVY_ROOT / cat / seq / "heldout" / arm / "renders"
        if not rdir.is_dir():
            return None, f"missing renders: {rdir}"
        man = common.scene_manifest(cat, seq)
        recs = dv.load_annotations(cat, seq)
        held = man["heldout_frames"]
        gt = [common.scene_root(cat, seq) / "common" / "heldout_images" / f"frame{f:06d}.png" for f in held]
        masks = [(fgmask(recs[f]) if masktype == "foreground" else dv.content_mask_from_record(recs[f]))
                 for f in held]
        rp = sorted(rdir.glob("*.png"))
        if len(rp) != len(gt):
            return None, f"{name} {arm}: {len(rp)} renders vs {len(gt)} gt"
        r = common.evaluate_renders(gt, rp, masks, with_ssim=True)
        rows.append({"scene": name, "psnr": r["mean_psnr"], "ssim": r.get("mean_ssim")})
    return {
        "mean_psnr": float(np.mean([x["psnr"] for x in rows])),
        "mean_ssim": float(np.mean([x["ssim"] for x in rows if x["ssim"] is not None])),
        "min_psnr": float(np.min([x["psnr"] for x in rows])),
        "per_scene": rows,
    }, None


def main():
    out = {}
    for masktype in ("content", "foreground"):
        out[masktype] = {}
        for arm in ARMS:
            s, err = score_arm(arm, masktype)
            if err:
                print(f"  SKIP {arm} [{masktype}]: {err}")
                continue
            out[masktype][arm] = s
            print(f"  {arm:12s} [{masktype:10s}] mean PSNR={s['mean_psnr']:7.3f}  SSIM={s['mean_ssim']:.4f}")

    # 4-way ordering and the per-scene "init is inert" count, both masks
    for masktype in ("content", "foreground"):
        d = out[masktype]
        if not {"c1_random", "c1_full"} <= set(d):
            continue
        order = sorted(d, key=lambda a: -d[a]["mean_psnr"])
        print(f"\n[{masktype}] ordering (best first): " + " > ".join(
            f"{a.replace('c1_','')} {d[a]['mean_psnr']:.3f}" for a in order))
        rnd = {r["scene"]: r["psnr"] for r in d["c1_random"]["per_scene"]}
        ful = {r["scene"]: r["psnr"] for r in d["c1_full"]["per_scene"]}
        inert = sum(1 for s in rnd if abs(ful[s] - rnd[s]) < 0.5)
        out[masktype]["_summary"] = {
            "ordering": order,
            "full_minus_random": d["c1_full"]["mean_psnr"] - d["c1_random"]["mean_psnr"],
            "scenes_random_within_0.5dB_of_full": inert,
            "n_scenes": len(rnd),
        }
        print(f"[{masktype}] full - random = {out[masktype]['_summary']['full_minus_random']:+.3f} dB; "
              f"random within 0.5 dB of full on {inert}/{len(rnd)} scenes")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c1_both_masks.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT/'c1_both_masks.json'}")


if __name__ == "__main__":
    main()
