"""Emit an ongoing_logs.md entry straight from a sweep result JSON.

Numbers are read from the measured JSON, never retyped, so the log cannot drift
from what was actually measured.
"""
import json
import sys
from pathlib import Path

SWEEP = Path("/var/tmp/luli38se/quantsplat/oracle_sweep")
COMMIT = "e411cb8786172a49571be68fc3ba3f7d399ebd96"
TARGET = 20.0

META = {
    "densify_off": ("#005", "Densification disabled (factor 1)",
                    "`oracle`", "default except densification off",
                    "`--densify_until_iter 0` (no adaptive density control at all)",
                    "6 input, 9 held-out", "`content`", "GS default black; unmasked full frames", "off"),
    "densify_3000": ("#006", "Densification stopped at 3000 (factor 1)",
                     "`oracle`", "default", "`--densify_until_iter 3000` (default is 15000)",
                     "6 input, 9 held-out", "`content`", "GS default black; unmasked full frames", "off"),
    "no_opacity_reset": ("#007", "Opacity reset disabled (factor 3)",
                         "`oracle`", "default", "default densification; `--opacity_reset_interval 100000` (> iterations, so never fires)",
                         "6 input, 9 held-out", "`content`", "GS default black; unmasked full frames", "off"),
    "views12": ("#008", "12 input views instead of 6 (factor 4)",
                "`oracle` rebuilt on 12 views", "default", "default",
                "12 input (evenly spaced, held-out frames excluded), 9 held-out",
                "`content`", "GS default black; unmasked full frames", "off"),
    "maskedbg": ("#009", "Foreground-masked training images, constant background (factor 5)",
                 "`oracle`", "default", "default",
                 "6 input, 9 held-out", "`content` (also re-scored under `foreground`)",
                 "training images have background replaced by constant black; held-out GT unchanged", "off"),
    "depthreg": ("#010", "Depth regularisation against GT depth (factor 6)",
                 "`oracle` + GT inverse-depth maps", "default", "default",
                 "6 input, 9 held-out", "`content`", "GS default black; unmasked full frames",
                 "ON: `-d depths --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01`; GT depth covers 2.6-6.2% of pixels, foreground only"),
    "views24": ("#012", "24 input views (follow-up to factor 4)",
                "`oracle` rebuilt on 24 views", "default", "default",
                "24 input (evenly spaced, held-out frames excluded), 9 held-out",
                "`content`", "GS default black; unmasked full frames", "off"),
    "bgshell": ("#011", "GT foreground points + random background shell (extra factor, not a pure oracle)",
                "`oracle` GT foreground points **plus 100,000 random background points** per scene (radii 1.2-8.0 x scene radius, grey). NOT a pure GT-geometry oracle: the background points are a random prior, not measurements.",
                "default", "default", "6 input, 9 held-out", "`content`",
                "GS default black; unmasked full frames", "off"),
}

SCENES = ["apple/110_13051_23361", "ball/123_14363_28981", "bowl/70_5792_13401",
          "broccoli/412_56288_108844", "hydrant/167_18184_34441", "remote/350_36761_68623",
          "teddybear/187_20215_38541", "toaster/372_41229_82130"]


def emit(tag, interpretation):
    d = json.loads((SWEEP / "results" / f"{tag}.json").read_text())
    num, title, init, iters, densify, views, mask, bg, depth = META[tag]
    its = sorted(int(k) for k in d["by_iter"])
    by = {int(k): v for k, v in d["by_iter"].items()}
    best_it = max(its, key=lambda i: by[i]["mean_psnr"])
    best = by[best_it]["mean_psnr"]

    extra = " ".join(d["extra_train"]) if d["extra_train"] else "(none)"
    out = []
    out.append(f"\n---\n\n### {num} — {title}\n")
    out.append(f"- **Date/time:** 2026-09-07 CEST (this run took {d['minutes']:.1f} min wall for 8 scenes)")
    out.append(f"- **Git commit:** `{COMMIT}` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)")
    out.append("- **Command executed:**\n  ```\n  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag "
               f"{tag} --iters 500 1000 3000 7000 --source {d['source']}"
               + (f" --extra {extra}" if d["extra_train"] else "") + "\n  ```")
    out.append(f"  Per scene this runs GraphDECO `train.py` with `--iterations {max(its)} --data_device cpu --resolution 1 "
               f"--quiet --disable_viewer --save_iterations {' '.join(str(i) for i in its)} --test_iterations -1"
               + (f" {extra}" if d["extra_train"] else "") + "`, then `render.py --skip_test` per checkpoint.\n")
    out.append("**Config:**\n")
    out.append("| Field | Value |\n|---|---|")
    out.append(f"| Init source | {init} |")
    out.append(f"| Iterations | {' / '.join(str(i) for i in its)} (checkpointed in one training run per scene) |")
    out.append(f"| Densify | {densify} |")
    out.append(f"| Input views | {views} |")
    out.append(f"| Mask type | {mask} |")
    out.append(f"| Background handling | {bg} |")
    out.append(f"| Depth regularisation | {depth} |\n")
    out.append("**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, "
               "bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, "
               "teddybear/187_20215_38541, toaster/372_41229_82130).\n")
    out.append("**Results (content-mask PSNR, per scene and mean):**\n")
    out.append("| scene | " + " | ".join(str(i) for i in its) + " |")
    out.append("|---|" + "---:|" * len(its))
    per = {i: {r["scene"]: r for r in by[i]["per_scene"]} for i in its}
    for s in SCENES:
        row = [f"{per[i][s]['psnr']:.2f}" if s in per[i] else "n/a" for i in its]
        out.append(f"| {s} | " + " | ".join(row) + " |")
    out.append("| **mean PSNR** | " + " | ".join(f"**{by[i]['mean_psnr']:.3f}**" for i in its) + " |")
    out.append("| **mean SSIM** | " + " | ".join(f"{by[i]['mean_ssim']:.4f}" for i in its) + " |")
    out.append("| min PSNR | " + " | ".join(f"{by[i]['min_psnr']:.2f}" for i in its) + " |")
    out.append("\nLPIPS not computed (the A3-comparable metric path computes PSNR/SSIM only).\n")
    verdict = "PASS" if best >= TARGET else "FAIL"
    out.append(f"**PASS/FAIL:** Target: **oracle >= {TARGET:.0f} dB mean content-mask PSNR on the 8-scene subset.** "
               f"Best measured for this config: **{best:.3f} dB at {best_it} iterations**. **{verdict}"
               + (f"**, short by {TARGET - best:.2f} dB." if verdict == "FAIL" else ".**"))
    out.append(f"\n**Interpretation:** {interpretation}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    print(emit(sys.argv[1], sys.argv[2]))
