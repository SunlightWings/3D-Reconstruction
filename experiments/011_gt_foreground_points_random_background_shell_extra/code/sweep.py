"""Oracle-only 3DGS config sweep. Metric identical to A3: content-mask PSNR/SSIM
on the 9 held-out views, 8-scene subset."""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/utn/luli38se/cv/3D-Reconstruction/code/downstream_3dgs/diagnostics")
import common
dv = common.dv
GS_ROOT = Path("/home/utn/luli38se/cv/gaussian-splatting")
GS_PY = dv.GS_PY
SWEEP = Path("/var/tmp/luli38se/quantsplat/oracle_sweep")
# A real 8-scene train+render sweep never finishes this fast; see the guard in run_config().
MIN_REAL_RUN_MINUTES = 3.0

def sh(cmd, cwd):
    r = subprocess.run(cmd, cwd=str(cwd), env=dv.gs_env(), capture_output=True, text=True)
    if r.returncode != 0:
        print("FAILED:", " ".join(map(str, cmd))); print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit(1)
    return r

def run_config(tag, extra_train, iters_eval, source_key="oracle", train_iters=None, allow_fast=False):
    train_iters = train_iters or max(iters_eval)
    per_iter = {i: [] for i in iters_eval}
    t0 = time.time()
    for name in common.EXPENSIVE_SCENES:
        cat, seq = common.split_scene(name)
        heavy = common.DIAG_HEAVY_ROOT / cat / seq
        src = heavy / source_key / "train"
        held_src = heavy / source_key / "heldout"
        if source_key != "oracle":
            src = SWEEP / source_key / cat / seq / "train"
            held_src = SWEEP / source_key / cat / seq / "heldout"
        model = SWEEP / tag / cat / seq
        need = [i for i in iters_eval if not (model/"point_cloud"/f"iteration_{i}"/"point_cloud.ply").is_file()]
        if need:
            cmd = [str(GS_PY), "train.py", "-s", str(src), "-m", str(model),
                   "--iterations", str(train_iters), "--data_device", "cpu", "--resolution", "1",
                   "--eval" if False else "--quiet", "--disable_viewer",
                   "--save_iterations", *[str(i) for i in iters_eval],
                   "--test_iterations", "-1"] + extra_train
            sh(cmd, GS_ROOT)
        man = common.scene_manifest(cat, seq)
        held = man["heldout_frames"]
        recs = dv.load_annotations(cat, seq)
        masks = [dv.content_mask_from_record(recs[f]) for f in held]
        gt_dir = common.scene_root(cat, seq) / "common" / "heldout_images"
        gt_paths = [gt_dir / f"frame{f:06d}.png" for f in held]
        for it in iters_eval:
            rc = model / "train" / f"ours_{it}"
            if rc.exists(): shutil.rmtree(rc)
            # -d "" : render.py inherits `depths` from the training cfg_args and would
            # then demand depth_params.json in the heldout source. Depth is a training-only
            # signal, so switch it off explicitly at render time.
            sh([str(GS_PY), "render.py", "-m", str(model), "-s", str(held_src),
                "--iteration", str(it), "--skip_test", "--quiet", "-d", ""], GS_ROOT)
            rp = sorted((rc/"renders").glob("*.png"))
            assert len(rp) == len(gt_paths), f"{name} it={it}: {len(rp)} renders vs {len(gt_paths)} gt"
            res = common.evaluate_renders(gt_paths, rp, masks, with_ssim=True)
            per_iter[it].append({"scene": name, "psnr": res["mean_psnr"], "ssim": res.get("mean_ssim")})
            print(f"  [{tag}] {name} it={it}: PSNR={res['mean_psnr']:.2f} SSIM={res.get('mean_ssim')}", flush=True)
    minutes = (time.time()-t0)/60.
    # Guard against a silently-resumed run. A sweep tag's directory is both its source and
    # its GraphDECO model directory, so a prep script that copies another tag's directory
    # also copies that tag's trained checkpoints; the `need` check above then finds them,
    # skips training entirely, and re-renders the OLD model under the NEW tag. That failure
    # is invisible in the numbers -- it just reproduces the parent config exactly -- but it
    # is obvious in the clock: a real 8-scene run takes minutes, a pure re-render takes ~1.
    if minutes < MIN_REAL_RUN_MINUTES and not allow_fast:
        raise RuntimeError(
            f"[{tag}] finished in {minutes:.2f} min, under the {MIN_REAL_RUN_MINUTES} min floor for a real "
            f"{len(common.EXPENSIVE_SCENES)}-scene training run. This almost always means training was "
            f"skipped because trained checkpoints already existed under {SWEEP/tag} -- e.g. a prep script "
            f"copied another tag's directory along with its point_cloud/. Delete the model artifacts "
            f"(point_cloud/, cfg_args, cameras.json, exposure.json, input.ply, */ours_*) and re-run, or pass "
            f"allow_fast=True if this really is a render-only re-scoring."
        )
    out = {"tag": tag, "extra_train": extra_train, "source": source_key,
           "minutes": minutes, "by_iter": {}}
    for it in iters_eval:
        rows = per_iter[it]
        out["by_iter"][it] = {"mean_psnr": float(np.mean([r["psnr"] for r in rows])),
                              "mean_ssim": float(np.mean([r["ssim"] for r in rows if r["ssim"] is not None])),
                              "min_psnr": float(np.min([r["psnr"] for r in rows])), "per_scene": rows}
        print(f"[{tag}] it={it}: MEAN PSNR={out['by_iter'][it]['mean_psnr']:.3f} dB  SSIM={out['by_iter'][it]['mean_ssim']:.4f}", flush=True)
    (SWEEP/"results").mkdir(parents=True, exist_ok=True)
    (SWEEP/"results"/f"{tag}.json").write_text(json.dumps(out, indent=2))
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--iters", type=int, nargs="+", default=[7000])
    p.add_argument("--source", default="oracle")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    p.add_argument("--allow-fast", action="store_true",
                   help="skip the minimum-runtime guard (render-only re-scoring)")
    a = p.parse_args()
    run_config(a.tag, a.extra, a.iters, a.source, allow_fast=a.allow_fast)
