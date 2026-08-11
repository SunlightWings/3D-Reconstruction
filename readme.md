# Shared VGGT to 3DGS Adapter

This branch contains the shared adapter for the controlled comparison:

- Full VGGT + 3DGS
- QuantVGGT + 3DGS

Both pipelines must use the **same input tensor, same adapter, and same 3DGS configuration**.

# Shared adapter file name -> vggt_to_3dgs.py

## Canonical Input

Use exactly these 6 frames in this order:

```text
[1, 43, 91, 112, 174, 202]
```

The shared preprocessed input is:

```text
inputs/input_tensor.pt
```

Expected tensor:

```text
shape: [6, 3, 518, 518]
dtype: float32
range: [0, 1]
```

Tensor-content SHA256:

```text
157f8b61cae1fb20322c70a840bad89269562127ec1db0564a622a4dd1b0aeec
```

Both Full VGGT and QuantVGGT should use this same tensor directly.

## Raw VGGT Output

The adapter expects a `.pt` file containing at least:

```text
intrinsic
extrinsic
world_points_from_depth
frame_numbers
```

Expected shapes:

```text
intrinsic:               [1, 6, 3, 3]
extrinsic:               [1, 6, 3, 4]
world_points_from_depth: [6, 518, 518, 3]
frame_numbers:           [6]
```

For Full VGGT:

```text
results/full_vggt_raw.pt
```

QuantVGGT should export the same structure, for example:

```text
results/quant_vggt_raw.pt
```

If QuantVGGT produces a different raw structure, change the QuantVGGT export code to match this format. Do not create a separate Quant-specific adapter.

## Run Adapter for Full VGGT

```bash
python vggt_to_3dgs.py \
  --raw results/full_vggt_raw.pt \
  --input_tensor inputs/input_tensor.pt \
  --output results/full_vggt_adapter
```

## Run Adapter for QuantVGGT

Use the exact same script:

```bash
python vggt_to_3dgs.py \
  --raw results/quant_vggt_raw.pt \
  --input_tensor inputs/input_tensor.pt \
  --output results/quant_vggt_adapter
```

Only the raw input file and output directory should change.

## What the Adapter Does

The adapter:

- uses `world_points_from_depth` for the initial 3D geometry
- uses the VGGT-predicted intrinsics and extrinsics
- uses the canonical 518x518 input images
- exports one camera per view
- uses the COLMAP `PINHOLE` camera model
- performs deterministic point sampling with stride 4
- removes only non-finite 3D points
- creates a COLMAP-compatible scene for 3DGS

The adapter does NOT apply:

- confidence filtering
- visibility filtering
- reprojection-error filtering
- tracking
- fine tracking
- bundle adjustment
- random point sampling

This must remain identical for Full VGGT and QuantVGGT.

## Adapter Output

The output looks like:

```text
results/full_vggt_adapter/
├── images/
│   ├── image_1.png
│   ├── image_2.png
│   ├── image_3.png
│   ├── image_4.png
│   ├── image_5.png
│   └── image_6.png
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── initial_points.ply
```

The QuantVGGT adapter output should have the same structure.

For the current experiment, the adapter produces approximately:

```text
6 registered images
6 cameras
101400 initial 3D points
```

## Validate Adapter Output

For Full VGGT:

```bash
python - <<'PY'
import pycolmap

rec = pycolmap.Reconstruction(
    "results/full_vggt_adapter/sparse/0"
)

print(rec.summary())

for image_id, image in rec.images.items():
    print(image_id, image.name)

print("3D points:", len(rec.points3D))
PY
```

For QuantVGGT, replace the path with:

```text
results/quant_vggt_adapter/sparse/0
```

Expected image names:

```text
image_1.png
image_2.png
image_3.png
image_4.png
image_5.png
image_6.png
```

## Experimental Pipeline

```text
                 same input_tensor.pt
                         |
                +--------+--------+
                |                 |
            Full VGGT         QuantVGGT
                |                 |
        full_vggt_raw.pt   quant_vggt_raw.pt
                |                 |
                +--------+--------+
                         |
                 SAME ADAPTER
                 vggt_to_3dgs.py
                         |
                    SAME 3DGS
                         |
                  compare results
```

The only intended difference is **Full VGGT vs QuantVGGT**. Everything after raw VGGT inference should remain identical.