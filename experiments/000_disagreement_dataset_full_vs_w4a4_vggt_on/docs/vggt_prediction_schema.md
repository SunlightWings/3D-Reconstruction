# VGGT prediction schema for the controlled Quant→3DGS experiment

This schema describes `predictions.npz` produced by
`scripts/model_tests/quant_3dgs/run_vggt_variant.py`. It is derived from the
local QuantVGGT/VGGT source, especially `vggt/models/vggt.py`,
`vggt/utils/pose_enc.py`, `vggt/utils/geometry.py`, and
`vggt/heads/head_act.py`. It is not yet a 3DGS or COLMAP adapter.

## Frozen input contract

- Scene/category: `602_93503_186945` / `toilet`
- Split: `input_12`
- Ordered frames: `1, 22, 36, 56, 77, 91, 112, 126, 147, 167, 181, 202`
- Input shape: `[12, 3, 518, 518]`
- Preprocessing: local VGGT `load_and_preprocess_images(..., mode="pad",
  target_size=518)`, performed once in Phase 1
- Input SHA-256:
  `41fdcacfa3d403f6aa76dda70d22d845c44ce295eb61fb195bf77a98d744fb87`

The runner has no JPEG preprocessing path. Both variants consume the identical
serialized tensor.

## Arrays

All predicted floating-point arrays are saved as C-contiguous NumPy `float32`.
`frame_numbers` is `int64`. Only the leading model batch dimension (which must
equal one) is removed; singleton semantic/channel dimensions are retained.

| Array | Shape | Meaning |
|---|---:|---|
| `pose_enc` | `[12, 9]` | Raw absolute camera encoding: translation (3), quaternion (4), horizontal and vertical field of view (2). |
| `depth` | `[12, 518, 518, 1]` | Depth-head output in the predicted camera coordinate system. The local head uses exponential activation, hence valid predictions are positive. |
| `depth_conf` | `[12, 518, 518]` | Depth-head confidence. The local head applies `1 + exp(raw)`; it is a model score, not a calibrated probability. |
| `intrinsic` | `[12, 3, 3]` | Pixel-space camera intrinsics decoded from `pose_enc` for an exact image size of 518×518. |
| `extrinsic` | `[12, 3, 4]` | OpenCV world-to-camera transform `[R | t]`. |
| `world_points` | `[12, 518, 518, 3]` | Direct point-head prediction in VGGT's predicted world/reference coordinate frame. |
| `world_points_conf` | `[12, 518, 518]` | Point-head confidence using `1 + exp(raw)`; not a calibrated probability. |
| `world_points_from_depth` | `[12, 518, 518, 3]` | Depth backprojection using the saved `depth`, `intrinsic`, and `extrinsic`, via the local VGGT geometry utility. |
| `frame_numbers` | `[12]` | Manifest frame numbers in frozen input order. |

The original model/output dtypes before conversion are recorded per array in
`inference_meta.json`.

## Camera and pixel convention

`pose_encoding_to_extri_intri` decodes `pose_enc` using VGGT's
`absT_quaR_FoV` convention. The extrinsic maps a world point to camera space:

`p_camera = R * p_world + t`

Camera axes follow OpenCV: +x right, +y down, +z forward. Intrinsics are
pixel-space matrices with positive `fx`, `fy`; the local decoder fixes the
principal point at `(width/2, height/2)`, which is `(259, 259)` for this frozen
518×518 input.

The repository unprojection code creates zero-based integer pixel grids
`u = 0..517`, `v = 0..517`, computes camera points as
`((u-cx)z/fx, (v-cy)z/fy, z)`, then inverts `[R | t]` to obtain
`world_points_from_depth`. This array therefore shares the predicted world
frame with the decoded cameras. It is distinct from the direct
`world_points` head and both are retained deliberately.

The Phase 1 padding and scale metadata required to map these 518×518 pixel
coordinates back to original CO3D RGB coordinates remains in
`preprocess_meta.json`; no such mapping is applied during Phase 3.

## Deterministic hashes

`prediction_content_sha256` is independent of ZIP container metadata. Arrays
are visited in the exact order shown in the table. For each array, the hash is
updated with UTF-8 array name, one NUL byte, compact JSON shape, one NUL byte,
NumPy dtype name, one NUL byte, and C-contiguous raw bytes.

`prediction_hash.sha256` separately contains the ordinary SHA-256 of the
`predictions.npz` file in standard `sha256sum` format.

