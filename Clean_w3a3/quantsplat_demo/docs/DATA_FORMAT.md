# Data layout

The packer creates this folder automatically:

```text
data/
  catalog.json
  metrics.json
  geometry.json        # optional
  runtime.json         # optional
  images/
    apple/110_13051_23361/
      gt/frame000002.png ...
      full/00000.png ...
      w4a4/00000.png ...
      w3a3/00000.png ...
      w3a3_conf/00000.png ...   # future
      inputs/image_1.png ...   # optional
```

## catalog.json

Contains `schema_version: 1`, `iterations`, `input_views`, a `models` array, and a `scenes` array. Each scene has:

- `id`: `category/sequence`.
- `label`: display label.
- `frames`: exact held-out frame IDs in ascending order.
- `gt`: relative GT image paths in that same order.
- `renders`: map from arm ID to equally ordered render paths.
- `pairing`: verification notes for each arm.
- `inputs`: optional six training image paths.
- `input_frames`: optional original input frame IDs from metadata.

An absent `renders[arm]` means images are pending. There is no fallback to another arm. Paths are relative to `data/` and may not escape it.

## metrics.json

Uses your existing `w3a3_vs_existing_8scenes_9views.json` format:

```json
{
  "n_scenes": 8,
  "n_views_per_scene": 9,
  "arms": {
    "full": {"foreground": {"psnr": 9.9708, "ssim": 0.2403, "lpips": 0.6553}}
  },
  "per_scene": [
    {"scene": "apple/110_13051_23361", "full_foreground_psnr": 12.670}
  ],
  "pairs": {}
}
```

The small JSON above illustrates the schema using previously reported values; it is not a complete evaluation file. Use your original full JSON. The supplied starter contains all eight reported scenes and is explicitly marked as a rounded console transcription.

Regions: `foreground` and `content`. Optional metrics append `_exposure_corrected` to `psnr`, `ssim`, `lpips`. Pair keys are `full_minus_<arm>__<region>__<metric>`; entries contain `n`, `mean_delta`, `p`, and `ci95`.

For a new confidence method, use `w3a3_conf` consistently in `arms`, `per_scene`, and `pairs`. Do not fill unavailable values with zero. Absent fields are shown as pending. The app displays supplied aggregate and paired-test values and never derives p-values from rounded per-scene summaries.

## geometry.json

Uses the existing geometry comparison structure: `{ "w3a3": { "mean": { "n_scenes": 8, "cam_rot_deg": ..., "cam_centre_err": ..., "cam_spread_ratio": ..., "depth_rel_err": ..., "point_err": ... }, "per_scene": [...] } }`.

## runtime.json

Contains `records`, each with `scene`, `variant`, `seconds`, `peak_allocated_mib`, and a relative source filename. The packer reads these from the selected group's `<arm>_meta.json`. These values are not treated as a controlled end-to-end benchmark.

## Important pairing rule

The reference archive has frame-numbered GT and sequentially indexed renders. The newer supplied scoring README explicitly corrects an older README's claim that filenames match directly. Pair by sorted order ONLY under that documented mapping. The local W3A3 packer additionally verifies its held-out COLMAP camera-image names, and, where available, GT pixel equality. It does not take the first nine PNGs from a 196-view run.
