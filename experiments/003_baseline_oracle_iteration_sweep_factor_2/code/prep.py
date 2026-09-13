"""Build oracle-style 3DGS sources with a custom input-view list and/or
foreground-masked training images (background replaced by a constant)."""
import os, sys, shutil
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, "/home/utn/luli38se/cv/3D-Reconstruction/code/downstream_3dgs/diagnostics")
import common, gate_a
dv, dis = common.dv, common.dis
SWEEP = Path("/var/tmp/luli38se/quantsplat/oracle_sweep")

MODEL_ARTIFACTS = ("point_cloud", "cameras.json", "cfg_args", "exposure.json", "input.ply")


def strip_model_artifacts(root: Path) -> None:
    """Remove GraphDECO model output from a copied sweep directory.

    A sweep tag's directory is BOTH its data source and its GraphDECO model directory, so
    copying one to seed a new config also copies its trained checkpoints. sweep.py's
    resume-skip would then find them, skip training, and re-render the old model under the
    new tag -- producing numbers identical to the parent config with no error raised.
    Every prep script that copies a directory must call this afterwards.
    """
    import shutil as _sh
    for junk in MODEL_ARTIFACTS:
        p = root / junk
        if p.is_dir():
            _sh.rmtree(p)
        elif p.exists():
            p.unlink()
    for renders in list(root.glob("train/ours_*")) + list(root.glob("heldout/ours_*")):
        _sh.rmtree(renders)


def pick_frames(recs, held, n):
    ok = sorted(f for f in recs if f not in set(held))
    idx = np.linspace(0, len(ok)-1, n).round().astype(int)
    return [ok[i] for i in dict.fromkeys(idx)]

def fg518(rec):
    h, w = [int(v) for v in rec["image"]["size"]]
    nw, nh, left, top, _ = dv.preprocess_geometry(w, h)
    tr = {"resized_width": nw, "resized_height": nh, "padding": {"left": left, "top": top}}
    with Image.open(dv.CO3D_ROOT / rec["mask"]["path"]) as im:
        raw = np.asarray(im.convert("L"))
    d = np.ones_like(raw, dtype=np.float32)
    _a, _b, m = dis.resize_depth_and_masks(d, d > 0, raw, tr)
    return m >= 0.5

def build(source_key, n_views=None, mask_bg=None, stride=4):
    for name in common.EXPENSIVE_SCENES:
        cat, seq = common.split_scene(name)
        man = common.scene_manifest(cat, seq); recs = dv.load_annotations(cat, seq)
        held = man["heldout_frames"]
        frames = man["input_frames"] if n_views is None else pick_frames(recs, held, n_views)
        root = SWEEP / source_key / cat / seq
        ok = root / "PREPARED.ok"
        if ok.is_file(): print(f"  skip {source_key} {name}"); continue
        if root.exists(): shutil.rmtree(root)
        tr_root, he_root = root/"train", root/"heldout"
        (tr_root/"images").mkdir(parents=True, exist_ok=True); he_root.mkdir(parents=True, exist_ok=True)
        # training images (optionally background-masked to a constant)
        tensor = dv.preprocess_images([dv.resolve_image_path(recs[f]) for f in frames])
        names = [f"image_{i}.png" for i in range(1, len(frames)+1)]
        for i, f in enumerate(frames):
            arr = np.transpose(tensor[i], (1,2,0)).copy()
            if mask_bg is not None:
                m = fg518(recs[f]); arr[~m] = mask_bg
            Image.fromarray(np.clip(arr*255,0,255).astype(np.uint8)).save(tr_root/"images"/names[i])
        os.symlink(common.scene_root(cat,seq)/"common"/"heldout_images", he_root/"images", target_is_directory=True)
        pts, cols, Es, Ks = [], [], [], []
        for f in frames:
            fr = gate_a._oracle_decode_frame(recs[f])
            Es.append(fr["E"]); Ks.append(fr["K"])
            v = fr["valid"][::stride,::stride]; w3 = fr["world"][::stride,::stride,:]
            p = w3[v]
            if len(p)==0: continue
            rgb = np.transpose(dv.preprocess_images([dv.resolve_image_path(recs[f])])[0],(1,2,0))[::stride,::stride,:][v]
            pts.append(p.astype(np.float32)); cols.append(np.clip(rgb*255,0,255).astype(np.uint8))
        P, C = np.concatenate(pts), np.concatenate(cols)
        dv.write_colmap_text_model(tr_root, np.stack(Es), np.stack(Ks), names, P, C)
        Eh, Kh = [], []
        for f in held:
            E, K0 = dv.co3d_to_opencv_camera(recs[f]); Eh.append(E); Kh.append(dv.original_to_518_affine(recs[f]) @ K0)
        (he_root/"sparse"/"0").mkdir(parents=True, exist_ok=True)
        dv.write_colmap_text_model(he_root, np.stack(Eh), np.stack(Kh), [f"frame{f:06d}.png" for f in held], P[:10], C[:10])
        shutil.copy2(tr_root/"sparse"/"0"/"points3D.txt", he_root/"sparse"/"0"/"points3D.txt")
        ok.write_text(f"PASS views={len(frames)} points={len(P)} frames={frames}\n")
        print(f"  built {source_key} {name}: {len(frames)} views, {len(P)} pts")

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "views12": build("views12", n_views=12)
    elif which == "maskedbg": build("maskedbg", mask_bg=0.0)
    elif which == "maskedbg12": build("maskedbg12", n_views=12, mask_bg=0.0)
