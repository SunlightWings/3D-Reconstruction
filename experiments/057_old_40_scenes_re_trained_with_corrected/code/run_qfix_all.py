"""Corrected-rotation 3DGS for a list of scenes, same configuration as every earlier run.

GraphDECO, 7000 iterations, defaults as invoked by dv.train_3dgs (--data_device cpu --resolution 1); sources are the
stride-4 VGGT point grid with 6 input views; 9 held-out views rendered with dv.render_heldout.

Source modes
  qfix    scenes whose sources were written BEFORE the rotmat2qvec fix: sources/<arm>_qfix are quaternion-corrected
          copies made by build_qfix_sources.py (run first, skips copies that already exist).
  direct  scenes whose sources were written AFTER the fix: sources/<arm> are already correct and are linked as
          sources/<arm>_qfix.
Either way the model is models/<arm>_qfix and the renders heldout/<arm>_qfix_it<iters>. Anything already complete is
skipped. After each scene/arm an outputs index (npz, meta json, ply, scene manifest, renders) is saved.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/utn/luli38se/cv/3D-Reconstruction")
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "diagnostics"))
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "oracle_sweep"))
import common  # noqa: E402

dv = common.dv
RES = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-file", type=Path, default=None, help="one category/sequence per line")
    ap.add_argument("--old40", action="store_true", help="the 40 scenes of the frozen manifest")
    ap.add_argument("--arms", nargs="+", default=["full", "w4a4"])
    ap.add_argument("--source-mode", choices=["qfix", "direct"], required=True)
    ap.add_argument("--iterations", type=int, default=7000)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    scenes = common.all_scenes() if a.old40 else [l.strip() for l in a.scenes_file.read_text().splitlines() if l.strip()]
    it = a.iterations
    print(f"PHASE {a.tag}: {len(scenes)} scenes x {a.arms}, source mode {a.source_mode}, {it} iterations  "
          f"({time.strftime('%H:%M')})", flush=True)
    if a.source_mode == "qfix":
        subprocess.run([sys.executable, "-u", str(REPO / "code/quantization/build_qfix_sources.py"), "--arms", *a.arms]
                       + (["--all"] if a.old40 else ["--scenes", *scenes]), check=True)

    index_path = RES / f"outputs_index_{a.tag}.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {}
    total, done, run_secs = len(scenes) * len(a.arms), 0, []
    for name in scenes:
        c, s = name.split("/", 1)
        root = common.scene_root(c, s)
        g, _m, _w = dv.find_group(c, s)
        n_held = len(common.scene_manifest(c, s)["heldout_frames"])
        for arm in a.arms:
            q = f"{arm}_qfix"
            src = root / "sources" / q
            if a.source_mode == "direct" and not src.exists():
                if not (root / "sources" / arm / "train" / "PREPARED.ok").is_file():
                    raise SystemExit(f"{name}: sources/{arm} not prepared")
                os.symlink(root / "sources" / arm, src, target_is_directory=True)
            model = root / "models" / q
            ply = model / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
            rdir = root / "heldout" / f"{q}_it{it}" / "renders"
            complete = ply.is_file() and rdir.is_dir() and len(list(rdir.glob("*.png"))) == n_held
            t0 = time.time()
            dv.train_3dgs(src / "train", model, it, 20, 1)
            dv.render_heldout(src / "heldout", model, root / "heldout" / f"{q}_it{it}", it, n_held, 20, 1)
            dt = time.time() - t0
            if not complete:
                run_secs.append(dt)
            done += 1
            index.setdefault(name, {})[arm] = {"npz": str(g / f"{arm}.npz"), "meta_json": str(g / f"{arm}_meta.json"),
                                               "ply": str(ply), "scene_manifest": str(root / "scene_manifest.json"),
                                               "renders": str(rdir), "iterations": it, "cameras": "rotation-corrected"}
            index_path.write_text(json.dumps(index, indent=2))
            avg = sum(run_secs) / len(run_secs) if run_secs else 0
            print(f"[{done}/{total}] {name} {arm}: {'skipped (already done)' if complete else f'trained+rendered in {dt:.0f}s'}"
                  f"  | {time.strftime('%H:%M')}  ETA this phase ~{avg * (total - done) / 60:.0f} min", flush=True)
    print(f"PHASE {a.tag} DONE ({time.strftime('%H:%M')})  index: {index_path}", flush=True)


if __name__ == "__main__":
    main()
