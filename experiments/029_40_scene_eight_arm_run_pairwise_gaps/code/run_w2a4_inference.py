"""Generate W2A4 VGGT predictions for the scenes the downstream pipeline uses.

Writes `w2a4.npz` + `w2a4_meta.json` alongside the existing `full.npz` / `w4a4.npz`,
under the identical contract enforced by run_full_dataset_disagreement.py: same frame
list, same preprocessing, same array shapes, a SHA over the saved file, and a check that
the semantic hash of the input images matches the `full` arm's. If the inputs differ at
all, the run aborts rather than producing an incomparable prediction.

Only the group that `run_downstream_validation.find_group()` selects per scene is
inferred -- the frozen manifest has 1330 groups, but the downstream comparison uses one
six-primary-view group per scene, so 8 (subset) or 40 (all) inferences suffice.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/utn/luli38se/cv/3D-Reconstruction")
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "code" / "quantization"))
sys.path.insert(0, str(REPO / "code" / "downstream_3dgs" / "diagnostics"))

import common  # noqa: E402  (diagnostics helper: gives us dv + the scene list)
from quant_loader import load_quantized  # noqa: E402

dv = common.dv
FROZEN = Path("/var/tmp/luli38se/quantsplat/disagreement_dataset_v1/frozen_dataset_manifest.json")
EXPECT_MANIFEST_SHA = "1ce2f3f8d43cc17f61d84545b82f132a3b64d49d5acf70ff0fe0ac6aa9968e06"


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def semantic(x) -> str:
    return hashlib.sha256(x.numpy().astype("<f4", copy=False).tobytes(order="C")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="w2a4")
    ap.add_argument("--scenes", choices=["subset", "all"], default="subset")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    if sha_file(FROZEN) != EXPECT_MANIFEST_SHA:
        raise SystemExit("frozen manifest SHA mismatch -- refusing to run")

    from prepare_co3d_scene import import_local_preprocessor
    from vggt_inference_core import infer

    # Must reproduce run_full_dataset_disagreement.py's wrapper EXACTLY: the raw
    # load_and_preprocess_images defaults to crop mode, and calling it unwrapped yields
    # different pixels -- which the input-semantic-SHA guard below correctly rejects.
    _P, _impl, _sig = import_local_preprocessor(Path("/home/utn/luli38se/cv/QuantVGGT"))

    def preprocess(paths):
        kw = {"mode": "pad"}
        if "target_size" in _sig.parameters:
            kw["target_size"] = 518
        x = _P([str(q) for q in paths], **kw).detach().cpu().float().contiguous()
        if tuple(x.shape) != (6, 3, 518, 518) or not torch.isfinite(x).all():
            raise SystemExit("bad frozen preprocessing output")
        return x

    scenes = common.EXPENSIVE_SCENES if a.scenes == "subset" else common.all_scenes()
    print(f"variant={a.variant}  scenes={len(scenes)} ({a.scenes})")

    print(f"loading {a.variant} ...")
    t0 = time.time()
    model = load_quantized(a.variant, a.device)
    model.to(a.device)
    print(f"  loaded in {time.time()-t0:.1f}s: {model._quant_loader_info['provenance']}")

    manifest = json.loads(FROZEN.read_text())
    by_scene = {f"{s['category']}/{s['sequence']}": s for s in manifest["scenes"]}

    done, timings = 0, []
    for name in scenes:
        cat, seq = common.split_scene(name)
        # the SAME group the downstream pipeline uses
        group_dir, full_meta, _w = dv.find_group(cat, seq)
        frames = list(map(int, full_meta["frame_numbers"]))
        out_npz = group_dir / f"{a.variant}.npz"
        out_meta = group_dir / f"{a.variant}_meta.json"
        if out_npz.is_file() and out_meta.is_file():
            print(f"  skip {name} (exists)")
            done += 1
            continue

        lookup = {r["frame_number"]: r["image_path"] for r in by_scene[name]["usable_frames"]}
        x = preprocess([lookup[f] for f in frames])
        ih = semantic(x)
        if full_meta.get("input_semantic_sha256") != ih:
            raise SystemExit(f"{name}: input semantic SHA differs from the full arm -- inputs are not identical")

        arr, rt = infer(model, x, frames, device=a.device)
        tmp = group_dir / f".{a.variant}.tmp.npz"
        np.savez(tmp, **arr)
        os.replace(tmp, out_npz)
        meta = {
            "category": cat, "sequence": seq, "group_id": full_meta.get("group_id"),
            "frame_numbers": frames,
            "primary_frame_numbers": full_meta.get("primary_frame_numbers"),
            "context_only_frame_numbers": full_meta.get("context_only_frame_numbers"),
            "variant": a.variant,
            "input_semantic_sha256": ih,
            "frozen_manifest_sha256": EXPECT_MANIFEST_SHA,
            "prediction_sha256": sha_file(out_npz),
            "runtime": rt,
            "quant_provenance": model._quant_loader_info["provenance"],
            "wbit": model._quant_loader_info["wbit"], "abit": model._quant_loader_info["abit"],
        }
        out_meta.write_text(json.dumps(meta, indent=2) + "\n")
        timings.append(rt["seconds"])
        done += 1
        print(f"  {name}: {rt['seconds']:.2f}s, peak {rt['peak_allocated_mib']:.0f} MiB -> {out_npz.name}",
              flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    if timings:
        print(f"\n{done}/{len(scenes)} scenes; mean inference {np.mean(timings):.2f}s")
    else:
        print(f"\n{done}/{len(scenes)} scenes (all pre-existing)")


if __name__ == "__main__":
    main()
