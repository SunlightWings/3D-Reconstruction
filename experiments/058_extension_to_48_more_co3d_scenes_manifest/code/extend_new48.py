"""Extend the dataset from the 40 frozen scenes to all 88 CO3D sequences on disk, with the SAME frame selection, grouping,
VGGT inference and scene preparation as the original run.

The original frame-selection code was never committed -- only its output, frozen_dataset_manifest.json. It is rebuilt:
  usable frame     image, depth, depth-mask and foreground-mask files exist and valid_gt_pixels >= --min-valid
  valid_gt_pixels  count(depth_mask & depth > 0 & finite & foreground > 0) -- reproduces the stored counts exactly
  groups           sorted usable frames; np.array_split into six temporal strata; group j = j-th element of each
                   stratum (reproduces all 40 stored scenes' groups exactly from their stored usable frames)
Downstream uses g0000 (first complete six-view group), as for all 40 original scenes. Context fillers for incomplete
final groups are NOT reproduced; those groups are never used downstream and are marked as such.

Subcommands
  validate   measure how well the usable-frame rule reproduces the 40 stored scenes (usable sets and g0000)
  manifest   write the manifest for the 48 new scenes (CO3D test split: no GT point cloud, no quality score)
  infer      full then W4A4 VGGT on g0000 of each new scene, exactly as run_full_dataset_disagreement.py
  prepare    dv.prepare_scene(stride 4, 9 held-out) for each new scene (sources written with the fixed rotmat2qvec)
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path("/home/utn/luli38se/cv/3D-Reconstruction")
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "diagnostics"))
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "oracle_sweep"))
import common  # noqa: E402

dv = common.dv
CO3D = Path("/var/tmp/luli38se/co3d_single_all")
FROZEN = Path("/var/tmp/luli38se/quantsplat/disagreement_dataset_v1/frozen_dataset_manifest.json")
PRED = Path("/var/tmp/luli38se/quantsplat/disagreement_dataset_v1/run/predictions")
NEWDIR = Path("/var/tmp/luli38se/quantsplat/dataset_v2_new48")
NEWMAN = NEWDIR / "manifest_new48.json"
RES = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results")
_ANN = {}


def annotations(cat):
    if cat not in _ANN:
        _ANN[cat] = json.load(gzip.open(CO3D / cat / "frame_annotations.jgz"))
    return _ANN[cat]


def frame_row(a):
    img, dep, dmk, fgm = (CO3D / a["image"]["path"], CO3D / a["depth"]["path"], CO3D / a["depth"]["mask_path"],
                          CO3D / a["mask"]["path"])
    row = {"frame_number": int(a["frame_number"]), "image_path": str(img), "depth_path": str(dep),
           "depth_mask_path": str(dmk), "foreground_mask_path": str(fgm),
           "files_ok": all(p.is_file() for p in (img, dep, dmk, fgm)), "valid_gt_pixels": 0, "fg_frac": 0.0,
           "frame_type": a.get("meta", {}).get("frame_type"), "image_size": a["image"].get("size"),
           "focal": a.get("viewpoint", {}).get("focal_length"), "pp": a.get("viewpoint", {}).get("principal_point"),
           "depth_scale": a["depth"].get("scale_adjustment"), "timestamp": a.get("frame_timestamp")}
    if row["files_ok"]:
        raw = np.asarray(Image.open(dep), dtype=np.uint16)
        d = np.frombuffer(raw.tobytes(), dtype=np.float16).astype(np.float64).reshape(raw.shape) * a["depth"]["scale_adjustment"]
        dm = np.asarray(Image.open(dmk)) > 0
        fg = np.asarray(Image.open(fgm).convert("L"))
        # exact stored definition: reproduces frozen_dataset_manifest valid_gt_pixels on 160/160 checked frames
        row["valid_gt_pixels"] = int((dm & (d > 0) & np.isfinite(d) & (fg > 0)).sum())
        row["fg_frac"] = float((fg > 127).mean())
        row["fg_pixels"] = int((fg > 0).sum())
    return row


def scene_rows(cat, seq):
    ann = sorted((a for a in annotations(cat) if a["sequence_name"] == seq), key=lambda a: a["frame_number"])
    with ThreadPoolExecutor(12) as ex:
        return list(ex.map(frame_row, ann))


def usable(rows, min_valid):
    return [r for r in rows if r["files_ok"] and r["valid_gt_pixels"] >= min_valid]


def groups_from(frames):
    strata = np.array_split(np.array(sorted(frames)), 6)
    out = []
    for j in range(max(len(x) for x in strata)):
        prim = [int(x[j]) for x in strata if j < len(x)]
        out.append({"group_id": f"g{j:04d}", "frame_numbers": prim, "primary_frame_numbers": prim,
                    "context_only_frame_numbers": [], "complete": len(prim) == 6})
    return out


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def new_scenes():
    old = set(common.all_scenes())
    return sorted(f"{p.parent.parent.name}/{p.parent.name}" for p in CO3D.glob("*/*/images")
                  if f"{p.parent.parent.name}/{p.parent.name}" not in old)


def cmd_validate(a):
    frozen = json.loads(FROZEN.read_text())
    resid = []   # collected only at the chosen --fg-min (was reset every sweep pass, so it always printed empty)
    rows_by = {}
    for s in frozen["scenes"]:
        rows_by[(s["category"], s["sequence"])] = scene_rows(s["category"], s["sequence"])
    for fg_min in a.sweep:
        exact = g0 = 0
        sym = []
        for s in frozen["scenes"]:
            key = (s["category"], s["sequence"])
            U = {r["frame_number"] for r in usable(rows_by[key], fg_min)}
            S = {u["frame_number"] for u in s["usable_frames"]}
            exact += U == S
            sym.append(len(U ^ S))
            g0 += groups_from(U)[0]["frame_numbers"] == s["groups"][0]["frame_numbers"]
            if fg_min == a.min_valid:
                for r in rows_by[key]:
                    if (r["frame_number"] in S) != (r["frame_number"] in U):
                        resid.append((f"{key[0]}/{key[1]}", r["frame_number"], "stored-only" if r["frame_number"] in S else "rule-only",
                                      r["valid_gt_pixels"], round(r["fg_frac"], 4), r["frame_type"], r["image_size"],
                                      r["focal"], r["pp"], r["depth_scale"], r["timestamp"]))
        print(f"min_valid {fg_min}: usable sets exact {exact}/40, total differing frames {sum(sym)} "
              f"(of {sum(len(s['usable_frames']) for s in frozen['scenes'])}), g0000 (used downstream) reproduced {g0}/40", flush=True)
    print(f"\nresidual differences at min_valid {a.min_valid}: {len(resid)}")
    for r in resid[:40]:
        print("  ", r)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "new48_rule_validation.json").write_text(json.dumps({"sweep": a.sweep, "chosen_min_valid": a.min_valid,
                                                                 "residuals": resid}, indent=2))


def cmd_manifest(a):
    NEWDIR.mkdir(parents=True, exist_ok=True)
    scenes = []
    for n in new_scenes():
        cat, seq = n.split("/", 1)
        rows = scene_rows(cat, seq)
        if any(r["valid_gt_pixels"] > 0 for r in rows):
            U, rule = usable(rows, a.min_valid), "gt_depth"
        else:
            # CO3D test split withholds GT depth: the depth PNGs and depth masks are empty for EVERY frame, so the
            # depth condition would reject the whole sequence. Nothing downstream uses GT depth; keep the rest of the
            # rule (all files exist) and require a non-empty foreground mask instead.
            U, rule = [r for r in rows if r["files_ok"] and r.get("fg_pixels", 0) > 0], "no_gt_depth_fg_mask"
        keep = ("frame_number", "image_path", "depth_path", "depth_mask_path", "foreground_mask_path", "valid_gt_pixels")
        if len(U) < 6:
            raise SystemExit(f"{n}: only {len(U)} usable frames under rule {rule}")
        g = groups_from([r["frame_number"] for r in U])
        scenes.append({"category": cat, "sequence": seq, "point_cloud_path": None, "point_cloud_quality_score": None,
                       "viewpoint_quality_score": None, "split_frame_types": sorted({r["frame_type"] for r in rows}),
                       "n_annotated_frames": len(rows), "frame_rule_used": rule,
                       "usable_frames": [{k: r[k] for k in keep} for r in U], "groups": g})
        print(f"  {n}: annotated {len(rows)} usable {len(U)} rule {rule} g0000 {g[0]['frame_numbers']}", flush=True)
    man = {"version": "new48-1", "source_root": str(CO3D),
           "frame_rule": f"gt_depth: files exist and valid_gt_pixels = count(depth_mask & depth>0 & finite & fg>0) >= {a.min_valid}; "
                         "no_gt_depth_fg_mask (sequences whose GT depth is withheld): files exist and foreground mask non-empty",
           "grouping": "sorted usable frames; np.array_split into six temporal strata; group j takes j-th element of each "
                       "stratum; incomplete final groups kept primary-only (context fillers not reproduced; unused downstream)",
           "note": "the 48 CO3D sequences on disk not in frozen_dataset_manifest.json; CO3D test split (no GT point cloud)",
           "scenes": scenes}
    NEWMAN.write_text(json.dumps(man, indent=2) + "\n")
    (NEWDIR / "manifest_new48.sha256").write_text(sha(NEWMAN) + "\n")
    (NEWDIR / "new48.txt").write_text("\n".join(f"{s['category']}/{s['sequence']}" for s in scenes) + "\n")
    print(f"wrote {NEWMAN} ({len(scenes)} scenes), sha {sha(NEWMAN)[:16]}")


def cmd_infer(a):
    import torch
    import run_full_dataset_disagreement as rfd   # same preprocess / semantic hash / idle / valid / save as the original run
    from vggt_inference_core import load_variant, infer
    man = json.loads(NEWMAN.read_text())
    msha = sha(NEWMAN)
    for variant in ("full", "w4a4"):
        for _ in range(120):
            if rfd.idle():
                break
            time.sleep(30)
        else:
            raise SystemExit("GPU never became idle; refusing inference")
        model = load_variant(variant, "cuda:0")
        for i, s in enumerate(man["scenes"], 1):
            g = s["groups"][0]
            if not g["complete"]:
                raise SystemExit(f"{s['category']}/{s['sequence']}: g0000 incomplete")
            d = PRED / s["category"] / s["sequence"] / g["group_id"]
            npz, met, frames = d / f"{variant}.npz", d / f"{variant}_meta.json", g["frame_numbers"]
            if rfd.valid(npz, met, variant, frames):
                print(f"[infer {variant} {i}/48] {s['category']}/{s['sequence']} skipped (exists)", flush=True)
                continue
            lookup = {r["frame_number"]: r["image_path"] for r in s["usable_frames"]}
            x = rfd.preprocess([lookup[f] for f in frames])
            ih = rfd.semantic(x)
            if variant == "w4a4" and json.loads((d / "full_meta.json").read_text())["input_semantic_sha256"] != ih:
                raise SystemExit("Full/W4A4 input semantic mismatch")
            arr, rt = infer(model, x, frames)
            rfd.save(d, variant, arr, {"category": s["category"], "sequence": s["sequence"], "group_id": g["group_id"],
                                       "frame_numbers": frames, "primary_frame_numbers": g["primary_frame_numbers"],
                                       "context_only_frame_numbers": [], "variant": variant, "input_semantic_sha256": ih,
                                       "manifest_new48_sha256": msha, "runtime": rt})
            print(f"[infer {variant} {i}/48] {s['category']}/{s['sequence']} {rt['seconds']:.1f}s", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


def cmd_prepare(a):
    man = json.loads(NEWMAN.read_text())
    for i, s in enumerate(man["scenes"], 1):
        t0 = time.time()
        dv.prepare_scene(s["category"], s["sequence"], 4, 9)
        print(f"[prepare {i}/48] {s['category']}/{s['sequence']} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "manifest", "infer", "prepare"])
    ap.add_argument("--min-valid", type=int, default=1)
    ap.add_argument("--sweep", type=int, nargs="+", default=[1, 1000, 5000, 20000])
    a = ap.parse_args()
    {"validate": cmd_validate, "manifest": cmd_manifest, "infer": cmd_infer, "prepare": cmd_prepare}[a.cmd](a)


if __name__ == "__main__":
    main()
