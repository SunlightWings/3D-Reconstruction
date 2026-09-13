"""Gate A -- is the harness measuring anything at all?

A1 train-view PSNR, A2 noise-floor calibration, A3 oracle GT-geometry control,
A4 held-out reprojection overlay, A5 leave-one-out Sim(3) extrapolation error.
"""

from __future__ import annotations

import itertools
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

dv = common.dv
dis = common.dis


# ---------------------------------------------------------------------------
# A2: noise-floor calibration -- PSNR between distinct heldout GT frames of the
# same scene. No GPU, no retrain: purely reuses images already on disk.
# ---------------------------------------------------------------------------

def test_a2_noise_floor() -> Dict[str, Any]:
    scenes = common.all_scenes()
    per_scene = []
    for name in scenes:
        category, sequence = common.split_scene(name)
        manifest = common.scene_manifest(category, sequence)
        records = dv.load_annotations(category, sequence)
        heldout = manifest["heldout_frames"]
        gt_dir = common.scene_root(category, sequence) / "common" / "heldout_images"
        masks = {f: dv.content_mask_from_record(records[f]) for f in heldout}

        from PIL import Image

        images = {}
        for f in heldout:
            p = gt_dir / f"frame{f:06d}.png"
            images[f] = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0

        pair_psnrs = []
        for a, b in itertools.combinations(heldout, 2):
            mask = masks[a] & masks[b]
            if not mask.any():
                continue
            pair_psnrs.append(dv.psnr(images[a], images[b], mask))
        finite = [p for p in pair_psnrs if np.isfinite(p)]
        per_scene.append(
            {
                "scene": name,
                "n_pairs": len(pair_psnrs),
                "mean_pair_psnr": float(np.mean(finite)) if finite else float("nan"),
                "median_pair_psnr": float(np.median(finite)) if finite else float("nan"),
            }
        )
        common.log(f"A2 {name}: noise-floor median PSNR = {per_scene[-1]['median_pair_psnr']:.2f} dB")

    all_medians = [r["median_pair_psnr"] for r in per_scene if np.isfinite(r["median_pair_psnr"])]
    return {
        "description": "PSNR between distinct heldout GT frames of the same scene (an uninformative-render floor).",
        "n_scenes": len(per_scene),
        "overall_mean_of_scene_medians": float(np.mean(all_medians)) if all_medians else float("nan"),
        "overall_median_of_scene_medians": float(np.median(all_medians)) if all_medians else float("nan"),
        "per_scene": per_scene,
    }


# ---------------------------------------------------------------------------
# A1: training-view (self-fit) PSNR. Reuses already-trained full/w4a4 models;
# only issues a render pass (no new training).
# ---------------------------------------------------------------------------

def _render_train_split(source: Path, model: Path, iterations: int, n_expected: int, private_out: Path) -> Tuple[Path, Path]:
    """Renders `source`'s cameras through `model` and returns (renders_dir, gt_dir).

    GraphDECO's render.py always writes to model/train/ours_<iter>/{renders,gt} regardless
    of which source path was given -- the SAME scratch directory the main pipeline's own
    render_heldout() already populated with the 9 held-out renders for this exact model.
    Checking that shared path for "already rendered" would silently reuse those stale
    held-out images instead of a fresh render of `source`'s own cameras, so the scratch
    dir is always cleared first and the result is copied out to a private, source-specific
    directory that resumability is checked against instead.
    """
    private_renders, private_gt = private_out / "renders", private_out / "gt"
    if private_renders.is_dir() and len(list(private_renders.glob("*.png"))) == n_expected:
        return private_renders, private_gt
    dv.wait_for_gpu(60, 2)
    render_cache = model / "train" / f"ours_{iterations}"
    if render_cache.exists():
        shutil.rmtree(render_cache)
    common.run(
        [str(dv.GS_PY), "render.py", "-m", str(model), "-s", str(source),
         "--iteration", str(iterations), "--skip_test"],
        cwd=dv.GS_ROOT, env=dv.gs_env(),
    )
    fresh_renders, fresh_gt = render_cache / "renders", render_cache / "gt"
    if len(list(fresh_renders.glob("*.png"))) != n_expected:
        raise RuntimeError(f"Expected {n_expected} train-split renders, found {len(list(fresh_renders.glob('*.png')))} in {fresh_renders}")
    for d, dest in ((fresh_renders, private_renders), (fresh_gt, private_gt)):
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(d, dest)
    return private_renders, private_gt


def test_a1_train_view_psnr() -> Dict[str, Any]:
    scenes = common.all_scenes()
    per_scene = []
    for name in scenes:
        category, sequence = common.split_scene(name)
        manifest = common.scene_manifest(category, sequence)
        records = dv.load_annotations(category, sequence)
        frames = manifest["input_frames"]
        masks = [dv.content_mask_from_record(records[f]) for f in frames]
        root = common.scene_root(category, sequence)
        row = {"scene": name}
        for variant in ("full", "w4a4"):
            source = Path(manifest[f"{variant}_train_source"])
            model = root / "models" / variant
            private_out = common.DIAG_HEAVY_ROOT / category / sequence / "train_split_render" / variant
            renders_dir, gt_dir = _render_train_split(source, model, common.TRAIN_ITERATIONS, len(frames), private_out)
            renders = sorted(renders_dir.glob("*.png"))
            gts = sorted(gt_dir.glob("*.png"))
            if len(renders) != len(frames) or len(gts) != len(frames):
                raise RuntimeError(f"{name}/{variant}: expected {len(frames)} train renders, got {len(renders)}")
            result = common.evaluate_renders(gts, renders, masks, with_ssim=False)
            row[f"{variant}_mean_psnr"] = result["mean_psnr"]
        per_scene.append(row)
        common.log(
            f"A1 {name}: full self-fit PSNR={row['full_mean_psnr']:.2f} dB, "
            f"w4a4 self-fit PSNR={row['w4a4_mean_psnr']:.2f} dB"
        )

    full_vals = [r["full_mean_psnr"] for r in per_scene]
    quant_vals = [r["w4a4_mean_psnr"] for r in per_scene]
    below_15 = [r["scene"] for r in per_scene if r["full_mean_psnr"] < 15 or r["w4a4_mean_psnr"] < 15]
    return {
        "description": "PSNR on the model's own TRAINING views (self-fit / overfit capability check).",
        "expect": f">= 30 dB at {common.TRAIN_ITERATIONS} iterations; < 15 dB indicates an adapter bug",
        "n_scenes": len(per_scene),
        "mean_full_self_fit_psnr": float(np.mean(full_vals)),
        "mean_w4a4_self_fit_psnr": float(np.mean(quant_vals)),
        "scenes_below_15db": below_15,
        "per_scene": per_scene,
    }


# ---------------------------------------------------------------------------
# A3: oracle control -- replace VGGT geometry entirely with GT-unprojected
# points and GT cameras (no Sim(3) needed: everything already lives in the
# GT frame). Upper bound on what this harness can produce.
# ---------------------------------------------------------------------------

def _oracle_decode_frame(record: Dict[str, Any]) -> Dict[str, Any]:
    """GT depth decode aligned to dv's own pad geometry (not dis's independent one),
    so the unprojected points land in exactly the same 518x518 pixel grid as the
    images dv.preprocess_images() already produced for this scene."""
    h, w = [int(v) for v in record["image"]["size"]]
    new_w, new_h, left, top, _mask = dv.preprocess_geometry(w, h)
    transform = {"resized_width": new_w, "resized_height": new_h, "padding": {"left": left, "top": top}}
    E, K0 = dv.co3d_to_opencv_camera(record)
    K = dv.original_to_518_affine(record) @ K0

    from PIL import Image

    depth_meta = record["depth"]
    depth_path = dv.CO3D_ROOT / depth_meta["path"]
    dm_path = dv.CO3D_ROOT / depth_meta["mask_path"]
    fg_path = dv.CO3D_ROOT / record["mask"]["path"]
    with Image.open(depth_path) as img:
        bits = np.array(img, dtype=np.uint16)
        depth_raw = np.frombuffer(bits.tobytes(order="C"), dtype=np.float16).astype(np.float32).reshape(bits.shape)
    with Image.open(dm_path) as img:
        dm_raw = np.asarray(img)
    with Image.open(fg_path) as img:
        fg_raw = np.asarray(img.convert("L"))
    scale_adjustment = float(depth_meta.get("scale_adjustment", 1.0))
    depth_actual = depth_raw * scale_adjustment
    depth_518, valid_518, fg_518 = dis.resize_depth_and_masks(depth_actual, dm_raw != 0, fg_raw, transform)
    gt_valid = valid_518 & (fg_518 >= 0.5)
    world = dis.unproject_to_world(depth_518, gt_valid, K, E)
    return {"E": E, "K": K, "valid": gt_valid, "world": world}


def _build_oracle_variant(category: str, sequence: str, stride: int) -> Dict[str, Any]:
    manifest = common.scene_manifest(category, sequence)
    records = dv.load_annotations(category, sequence)
    frames = manifest["input_frames"]
    heldout_frames = manifest["heldout_frames"]
    root = common.DIAG_HEAVY_ROOT / category / sequence / "oracle"
    train_root = root / "train"
    heldout_root = root / "heldout"
    ok = root / "PREPARED.ok"

    scene_common = common.scene_root(category, sequence) / "common"
    train_names = [f"image_{i}.png" for i in range(1, len(frames) + 1)]
    heldout_names = [f"frame{f:06d}.png" for f in heldout_frames]

    if not ok.is_file():
        if root.exists():
            shutil.rmtree(root)
        train_root.mkdir(parents=True, exist_ok=True)
        heldout_root.mkdir(parents=True, exist_ok=True)
        import os

        os.symlink(scene_common / "train_images", train_root / "images", target_is_directory=True)
        os.symlink(scene_common / "heldout_images", heldout_root / "images", target_is_directory=True)

        pts_all, cols_all = [], []
        E_train = []
        K_train = []
        for f in frames:
            frame = _oracle_decode_frame(records[f])
            E_train.append(frame["E"])
            K_train.append(frame["K"])
            v = frame["valid"][::stride, ::stride]
            w = frame["world"][::stride, ::stride, :]
            pts = w[v]
            if len(pts) == 0:
                continue
            tensor = dv.preprocess_images([dv.resolve_image_path(records[f])])[0]
            rgb = np.transpose(tensor, (1, 2, 0))[::stride, ::stride, :][v]
            cols = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
            pts_all.append(pts.astype(np.float32))
            cols_all.append(cols)
        points = np.concatenate(pts_all, axis=0)
        colors = np.concatenate(cols_all, axis=0)
        if len(points) < 1000:
            raise RuntimeError(f"{category}/{sequence}: too few valid GT oracle points ({len(points)})")

        dv.write_colmap_text_model(train_root, np.stack(E_train), np.stack(K_train), train_names, points, colors)

        E_held = []
        K_held = []
        for f in heldout_frames:
            Egt, Korig = dv.co3d_to_opencv_camera(records[f])
            E_held.append(Egt)
            K_held.append(dv.original_to_518_affine(records[f]) @ Korig)
        (heldout_root / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        dv.write_colmap_text_model(heldout_root, np.stack(E_held), np.stack(K_held), heldout_names,
                                    points[:10], colors[:10])
        # points3D.txt is only required to exist for GraphDECO's COLMAP reader; rendering
        # from a loaded checkpoint never reads it.
        import shutil as _sh

        _sh.copy2(train_root / "sparse" / "0" / "points3D.txt", heldout_root / "sparse" / "0" / "points3D.txt")
        ok.write_text(f"PASS point_count={len(points)}\n", encoding="utf-8")

    return {
        "train_source": train_root,
        "heldout_source": heldout_root,
        "model": common.DIAG_HEAVY_ROOT / category / sequence / "models" / "oracle",
        "heldout_frames": heldout_frames,
        "train_frames": frames,
    }


A3_CEILING_FILE = Path("/var/tmp/luli38se/quantsplat/oracle_sweep/results/metric_ceiling.json")


def _a3_target() -> Dict[str, Any]:
    """Read A3's pass threshold from the measured ceiling/noise-floor file.

    The threshold is DERIVED, not hand-picked, and it is scored on the FOREGROUND mask.

    Why the gate changed (see ongoing_logs.md entries #002, #004, #009, #014):
      * A3 originally demanded >= 20 dB on the CONTENT mask. That mask is the whole
        non-letterbox rectangle, of which the object is a mean 15.3%, and CO3D supplies
        GT depth for ~0.1% of background pixels. With a pixel-perfect object and the best
        possible flat background the content-mask score is 16.81 dB, so the old gate was
        unreachable by construction and its FAIL carried no information about geometry.
        That ceiling was confirmed independently to within 0.14 dB by the `maskedbg`
        sweep config (predicted 6.26 dB, measured 6.13 dB).
      * The foreground mask has no such binding ceiling, so a gate there is passable.
      * The threshold carries over the ORIGINAL gate's own margin above the A2 noise
        floor rather than inventing a number; `derive_a3_target()` in
        code/downstream_3dgs/oracle_sweep/ceiling.py holds the arithmetic.

    Regenerate the file with: python3 code/downstream_3dgs/oracle_sweep/ceiling.py
    """
    if not A3_CEILING_FILE.is_file():
        raise RuntimeError(
            f"A3 target file missing: {A3_CEILING_FILE}. "
            "Run: python3 code/downstream_3dgs/oracle_sweep/ceiling.py"
        )
    return common.read_json(A3_CEILING_FILE)["a3_target"]


def test_a3_oracle_control() -> Dict[str, Any]:
    per_scene = []
    for name in common.EXPENSIVE_SCENES:
        category, sequence = common.split_scene(name)
        common.log(f"A3 {name}: building oracle GT-geometry source")
        built = _build_oracle_variant(category, sequence, stride=4)
        dv.train_3dgs(built["train_source"], built["model"], common.TRAIN_ITERATIONS, 60, 2)
        render_out = common.DIAG_HEAVY_ROOT / category / sequence / "heldout" / "oracle"
        renders_dir = dv.render_heldout(built["heldout_source"], built["model"], render_out, common.TRAIN_ITERATIONS, len(built["heldout_frames"]), 60, 2)

        records = dv.load_annotations(category, sequence)
        gt_dir = common.scene_root(category, sequence) / "common" / "heldout_images"
        gt_paths = [gt_dir / f"frame{f:06d}.png" for f in built["heldout_frames"]]
        render_paths = sorted(renders_dir.glob("*.png"))

        row = {"scene": name}
        for masktype in ("foreground", "content"):
            masks = [(dv.foreground_mask_from_record(records[f]) if masktype == "foreground"
                      else dv.content_mask_from_record(records[f])) for f in built["heldout_frames"]]
            result = common.evaluate_renders(gt_paths, render_paths, masks)
            row[f"{masktype}_psnr"] = result["mean_psnr"]
            row[f"{masktype}_ssim"] = result.get("mean_ssim")
        # primary metric is the foreground mask
        row["mean_psnr"] = row["foreground_psnr"]
        row["mean_ssim"] = row["foreground_ssim"]
        agg = common.load_aggregate_row(category, sequence)
        row["full_psnr_for_reference"] = agg["full_psnr"]
        row["w4a4_psnr_for_reference"] = agg["quant_psnr"]
        per_scene.append(row)
        common.log(f"A3 {name}: oracle foreground PSNR={row['foreground_psnr']:.2f} dB "
                   f"(content={row['content_psnr']:.2f}; full={agg['full_psnr']:.2f}, w4a4={agg['quant_psnr']:.2f})")

    target = _a3_target()
    fg_vals = [r["foreground_psnr"] for r in per_scene]
    content_vals = [r["content_psnr"] for r in per_scene]
    below = [r["scene"] for r in per_scene if r["foreground_psnr"] < target["target_db"]]
    return {
        "description": "Upper-bound control: GT depth/points and GT cameras end to end, no VGGT involved. "
                       "Scored on the FOREGROUND mask (primary); content mask reported as secondary.",
        "expect": f">= {target['target_db']:.3f} dB on the foreground mask. {target['derivation']} "
                  "If the oracle misses this, the harness cannot measure reconstruction quality.",
        "target_db": target["target_db"],
        "target_derivation": target,
        "primary_mask": "foreground",
        "scenes_tested": common.EXPENSIVE_SCENES,
        "mean_oracle_psnr": float(np.mean(fg_vals)),
        "min_oracle_psnr": float(np.min(fg_vals)),
        "mean_oracle_psnr_content_secondary": float(np.mean(content_vals)),
        "scenes_below_target": below,
        "passed": len(below) == 0 and float(np.mean(fg_vals)) >= target["target_db"],
        "per_scene": per_scene,
    }


# ---------------------------------------------------------------------------
# A4: held-out reprojection overlay (qualitative + on-silhouette fraction).
# ---------------------------------------------------------------------------

def test_a4_reprojection_overlay() -> Dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenes = common.EXPENSIVE_SCENES[:5]
    per_scene = []
    for name in scenes:
        category, sequence = common.split_scene(name)
        manifest = common.scene_manifest(category, sequence)
        group_dir, full_meta, wmeta = dv.find_group(category, sequence)
        records = dv.load_annotations(category, sequence)
        frame = manifest["heldout_frames"][0]

        row = {"scene": name, "heldout_frame": frame, "variants": {}}
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, variant, npz_name in zip(axes, ("full", "w4a4"), ("full.npz", "w4a4.npz")):
            arrays = dv.load_npz(group_dir / npz_name)
            pts, _colors = dv.sampled_points_and_colors(
                arrays["world_points_from_depth"],
                dv.preprocess_images([dv.resolve_image_path(records[f]) for f in manifest["input_frames"]]),
                stride=8,
            )
            align = manifest["camera_alignment"][variant]
            s, A, b = align["scale"], np.array(align["rotation"]), np.array(align["translation"])
            Egt, K0 = dv.co3d_to_opencv_camera(records[frame])
            K = dv.original_to_518_affine(records[frame]) @ K0
            E = dv.transform_camera_to_variant(Egt, s, A, b)
            R, t = E[:, :3], E[:, 3]
            cam = (R @ pts.T + t[:, None]).T
            z = cam[:, 2]
            front = z > 1e-6
            uv = (K @ cam[front].T).T
            uv = uv[:, :2] / uv[:, 2:3]

            from PIL import Image

            gt_path = common.scene_root(category, sequence) / "common" / "heldout_images" / f"frame{frame:06d}.png"
            gt_img = np.asarray(Image.open(gt_path).convert("RGB"))
            fg = dv.content_mask_from_record(records[frame])
            try:
                gt_decoded = dis.decode_gt_frame(records[frame], dv.CO3D_ROOT, common.pad_transform)
                fg_silhouette = gt_decoded["fg"] >= 0.5
            except Exception:
                fg_silhouette = fg
            from scipy.ndimage import binary_dilation

            dilated = binary_dilation(fg_silhouette, iterations=6)
            in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < 518) & (uv[:, 1] >= 0) & (uv[:, 1] < 518)
            ub = uv[in_bounds].astype(int)
            on_silhouette = dilated[np.clip(ub[:, 1], 0, 517), np.clip(ub[:, 0], 0, 517)]
            frac_on = float(on_silhouette.mean()) if len(on_silhouette) else float("nan")

            ax.imshow(gt_img)
            ax.scatter(uv[in_bounds, 0], uv[in_bounds, 1], s=1.5, alpha=0.35, c="red")
            ax.set_title(f"{variant}: {frac_on*100:.1f}% on silhouette")
            ax.axis("off")
            row["variants"][variant] = {
                "camera_center_residual_scene_radii": align["camera_center_residual_median_scene_radius"],
                "fraction_reprojected_points_on_silhouette": frac_on,
            }
        fig_path = common.FIGURES_DIR / f"A4_reprojection_{category}_{sequence}.png"
        fig.suptitle(f"{name} heldout frame {frame}")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=110)
        plt.close(fig)
        row["figure"] = str(fig_path.relative_to(common.REPO))
        per_scene.append(row)
        common.log(f"A4 {name}: figure saved to {fig_path.name}")

    # Cheap, all-40-scene numeric signal: the camera-center residual already computed by
    # the main pipeline and stored in every scene_manifest.json.
    residuals = []
    for name in common.all_scenes():
        category, sequence = common.split_scene(name)
        manifest = common.scene_manifest(category, sequence)
        for variant in ("full", "w4a4"):
            residuals.append(manifest["camera_alignment"][variant]["camera_center_residual_median_scene_radius"])

    return {
        "description": "Visual check: does the Sim(3)-mapped point cloud land on the held-out image's object silhouette?",
        "figures_generated_for": scenes,
        "per_scene": per_scene,
        "all_40_scenes_camera_residual_median_scene_radii": {
            "mean": float(np.mean(residuals)),
            "median": float(np.median(residuals)),
            "max": float(np.max(residuals)),
        },
    }


# ---------------------------------------------------------------------------
# A5: leave-one-out Sim(3) extrapolation error, all 40 scenes, both variants.
# ---------------------------------------------------------------------------

def test_a5_loo_sim3() -> Dict[str, Any]:
    per_scene_csv = common.load_per_scene_csv()
    per_scene = []
    for name in common.all_scenes():
        category, sequence = common.split_scene(name)
        manifest = common.scene_manifest(category, sequence)
        frames = manifest["input_frames"]
        records = dv.load_annotations(category, sequence)
        C_gt = dv.camera_centers(np.stack([dv.co3d_to_opencv_camera(records[f])[0] for f in frames]))
        group_dir, _fm, _wm = dv.find_group(category, sequence)
        radius = per_scene_csv[(category, sequence)]["scene_radius"]

        row = {"scene": name, "variants": {}}
        for variant, npz_name in (("full", "full.npz"), ("w4a4", "w4a4.npz")):
            arrays = dv.load_npz(group_dir / npz_name)
            C_pred = dv.camera_centers(arrays["extrinsic"].astype(np.float64))
            errs = []
            errs_legacy = []
            for k in range(len(frames)):
                idx = [i for i in range(len(frames)) if i != k]
                try:
                    s, A, b = dv.umeyama(C_gt[idx], C_pred[idx])
                except Exception:
                    continue
                # FRAME-CONSISTENT: carry the held-out PREDICTED centre back into the GT
                # frame and compare against the GT centre there, so the residual and the
                # radius that normalises it live in the same frame.
                #
                # The previous form measured `||s*A@C_gt[k] + b - C_pred[k]||` -- a residual in
                # the PREDICTED frame -- and divided it by `scene_radius`, which is the CO3D GT
                # point cloud's radius. The two frames differ by the Sim(3) scale (~0.072 on the
                # 8-scene subset), so that figure was ~13x too small and reported PASS for a gate
                # that actually fails. See ongoing_logs.md entry #017.
                back_pred_k = (C_pred[k] - b) @ A / s
                errs.append(float(np.linalg.norm(back_pred_k - C_gt[k]) / radius))
                errs_legacy.append(float(np.linalg.norm(s * (A @ C_gt[k]) + b - C_pred[k]) / radius))
            row["variants"][variant] = {
                "median_loo_error_scene_radii": float(np.median(errs)) if errs else float("nan"),
                "max_loo_error_scene_radii": float(np.max(errs)) if errs else float("nan"),
                "median_loo_error_legacy_frame_inconsistent": (
                    float(np.median(errs_legacy)) if errs_legacy else float("nan")),
                "per_fold": errs,
            }
        per_scene.append(row)

    all_full = [r["variants"]["full"]["median_loo_error_scene_radii"] for r in per_scene]
    all_quant = [r["variants"]["w4a4"]["median_loo_error_scene_radii"] for r in per_scene]
    return {
        "description": "Leave-one-camera-out Sim(3) extrapolation error: fit on 5 of 6 input cameras, predict the 6th. "
                       "The held-out predicted camera is mapped back into the GT frame so that the residual and the "
                       "normalising radius share a frame (see ongoing_logs.md #017); "
                       "median_loo_error_legacy_frame_inconsistent reproduces the old, ~13x-too-small figure.",
        "expect": "median well under 0.02 (2% of scene radius)",
        "n_scenes": len(per_scene),
        "median_of_scene_medians_full": float(np.median(all_full)),
        "median_of_scene_medians_w4a4": float(np.median(all_quant)),
        "worst_scene_full": per_scene[int(np.argmax(all_full))]["scene"],
        "worst_scene_w4a4": per_scene[int(np.argmax(all_quant))]["scene"],
        "per_scene": per_scene,
    }


# ---------------------------------------------------------------------------
# A6: oracle self-fit -- render the oracle models' OWN training views.
#
# A1 establishes that the full/w4a4 models can fit their training views (~50 dB).
# A6 asks the same question of the oracle arm, which is the only arm built from
# GT cameras and GT-unprojected points. If the oracle cannot reproduce the very
# images it was trained on, the fault is in the GT camera path -- the cameras
# written into the oracle COLMAP model, or the intrinsics/pad affine used to
# build them -- and not in the reconstruction or in VGGT.
#
# Reuses the existing oracle models; no training.
# ---------------------------------------------------------------------------

A6_SELF_FIT_TARGET_DB = 30.0


def test_a6_oracle_selffit() -> Dict[str, Any]:
    per_scene = []
    for name in common.EXPENSIVE_SCENES:
        category, sequence = common.split_scene(name)
        manifest = common.scene_manifest(category, sequence)
        records = dv.load_annotations(category, sequence)
        frames = manifest["input_frames"]
        built = _build_oracle_variant(category, sequence, stride=4)
        model = built["model"]
        if not (model / "point_cloud" / f"iteration_{common.TRAIN_ITERATIONS}" / "point_cloud.ply").is_file():
            raise RuntimeError(f"A6 {name}: oracle model missing; run A3 first ({model})")
        private_out = common.DIAG_HEAVY_ROOT / category / sequence / "train_split_render" / "oracle"
        renders_dir, gt_dir = _render_train_split(built["train_source"], model, common.TRAIN_ITERATIONS, len(frames), private_out)
        renders = sorted(renders_dir.glob("*.png"))
        gts = sorted(gt_dir.glob("*.png"))

        row = {"scene": name}
        for masktype in ("content", "foreground"):
            masks = [(dv.foreground_mask_from_record(records[f]) if masktype == "foreground"
                      else dv.content_mask_from_record(records[f])) for f in frames]
            result = common.evaluate_renders(gts, renders, masks, with_ssim=False)
            row[f"{masktype}_psnr"] = result["mean_psnr"]
        per_scene.append(row)
        common.log(f"A6 {name}: oracle self-fit content={row['content_psnr']:.2f} dB, "
                   f"foreground={row['foreground_psnr']:.2f} dB")

    content_vals = [r["content_psnr"] for r in per_scene]
    fg_vals = [r["foreground_psnr"] for r in per_scene]
    below = [r["scene"] for r in per_scene if r["foreground_psnr"] < A6_SELF_FIT_TARGET_DB]
    return {
        "description": "Oracle self-fit: the oracle models rendered at their OWN training views, both masks.",
        "expect": f">= {A6_SELF_FIT_TARGET_DB:.0f} dB. Below that indicts the GT camera path "
                  "(oracle COLMAP cameras / pad affine on intrinsics), not the reconstruction.",
        "target_db": A6_SELF_FIT_TARGET_DB,
        "scenes_tested": common.EXPENSIVE_SCENES,
        "mean_self_fit_psnr_content": float(np.mean(content_vals)),
        "mean_self_fit_psnr_foreground": float(np.mean(fg_vals)),
        "min_self_fit_psnr_foreground": float(np.min(fg_vals)),
        "scenes_below_target": below,
        "passed": len(below) == 0,
        "per_scene": per_scene,
    }


TESTS = [
    ("A2", test_a2_noise_floor),
    ("A1", test_a1_train_view_psnr),
    ("A5", test_a5_loo_sim3),
    ("A4", test_a4_reprojection_overlay),
    ("A3", test_a3_oracle_control),
    ("A6", test_a6_oracle_selffit),
]
