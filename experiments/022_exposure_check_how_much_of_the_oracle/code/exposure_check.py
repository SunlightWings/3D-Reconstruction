"""How much of the oracle's 14.83 dB is photometric rather than geometric?

Two probes, both foreground-masked, neither needing training:

(a) NEAREST-POSE GT BASELINE. For each held-out GT frame, find the training GT frame whose
    camera centre is nearest, and score that training image directly against the held-out
    one. This is what a method scores by ignoring geometry entirely and copying the closest
    real photograph. If the oracle is not clearly above it, the oracle is not demonstrating
    reconstruction so much as photometric similarity between nearby views.

(b) GAIN+BIAS FIT. Fit one global affine correction per image (a scalar gain and bias per
    colour channel, least squares over the masked pixels) between the oracle's render and
    the GT, then re-score. Whatever PSNR this recovers was never a geometry error at all --
    it was exposure/white-balance mismatch. CO3D frames are captured with auto-exposure, so
    this is a real effect and not a contrivance.

The difference between (b) and the raw score is the photometric share; what remains is the
part a better geometry could still win.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP.parent / "diagnostics"))
import common  # noqa: E402
from reeval import fgmask  # noqa: E402

dv = common.dv
OUT = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")


def read(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def gain_bias_fit(render, gt, mask):
    """Per-channel least-squares gain+bias mapping render -> gt over masked pixels."""
    out = render.copy()
    for c in range(3):
        x = render[..., c][mask]
        y = gt[..., c][mask]
        if len(x) < 10 or float(np.var(x)) < 1e-12:
            continue
        A = np.stack([x, np.ones_like(x)], axis=1)
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        out[..., c] = np.clip(render[..., c] * sol[0] + sol[1], 0.0, 1.0)
    return out


def main():
    rows = []
    for name in common.EXPENSIVE_SCENES:
        cat, seq = common.split_scene(name)
        man = common.scene_manifest(cat, seq)
        recs = dv.load_annotations(cat, seq)
        held, train = man["heldout_frames"], man["input_frames"]
        root = common.scene_root(cat, seq)
        gt_dir = root / "common" / "heldout_images"
        train_dir = root / "common" / "train_images"
        renders = sorted((common.DIAG_HEAVY_ROOT / cat / seq / "heldout" / "oracle" / "renders").glob("*.png"))

        C_train = np.stack([dv.camera_center(dv.co3d_to_opencv_camera(recs[f])[0]) for f in train])
        raw, fitted, nearest = [], [], []
        for i, f in enumerate(held):
            mask = fgmask(recs[f])
            if not mask.any():
                continue
            gt = read(gt_dir / f"frame{f:06d}.png")
            rd = read(renders[i])
            raw.append(dv.psnr(rd, gt, mask))
            fitted.append(dv.psnr(gain_bias_fit(rd, gt, mask), gt, mask))
            # (a) nearest training view by camera centre
            c = dv.camera_center(dv.co3d_to_opencv_camera(recs[f])[0])
            k = int(np.argmin(np.linalg.norm(C_train - c, axis=1)))
            nearest.append(dv.psnr(read(train_dir / f"image_{k+1}.png"), gt, mask))

        rows.append({"scene": name,
                     "oracle_raw": float(np.mean(raw)),
                     "oracle_gain_bias": float(np.mean(fitted)),
                     "nearest_train_gt": float(np.mean(nearest))})
        r = rows[-1]
        print(f"  {name:33s} oracle={r['oracle_raw']:6.3f}  +gain/bias={r['oracle_gain_bias']:6.3f} "
              f"({r['oracle_gain_bias']-r['oracle_raw']:+.3f})  nearest-GT={r['nearest_train_gt']:6.3f}")

    raw = float(np.mean([r["oracle_raw"] for r in rows]))
    fit = float(np.mean([r["oracle_gain_bias"] for r in rows]))
    near = float(np.mean([r["nearest_train_gt"] for r in rows]))
    out = {
        "description": "Photometric vs geometric decomposition of the oracle's held-out score (foreground mask).",
        "mean_oracle_raw": raw,
        "mean_oracle_after_gain_bias": fit,
        "photometric_headroom_db": fit - raw,
        "mean_nearest_pose_train_gt": near,
        "oracle_minus_nearest_gt_db": raw - near,
        "per_scene": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exposure_check.json").write_text(json.dumps(out, indent=2))
    print(f"\noracle raw                       : {raw:.3f} dB")
    print(f"oracle after per-image gain+bias : {fit:.3f} dB  (photometric headroom {fit-raw:+.3f} dB)")
    print(f"nearest-pose training GT frame   : {near:.3f} dB  (oracle beats it by {raw-near:+.3f} dB)")
    print(f"saved {OUT/'exposure_check.json'}")


if __name__ == "__main__":
    main()
