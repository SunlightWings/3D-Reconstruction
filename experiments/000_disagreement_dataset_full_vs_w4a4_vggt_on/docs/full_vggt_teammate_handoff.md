# Full VGGT teammate handoff

Run Full VGGT only against the frozen Phase 1 tensor. This handoff intentionally
does not authorize 3DGS, COLMAP conversion, bundle adjustment, camera refinement,
or use of held-out frames.

## Frozen experiment

- Scene: `602_93503_186945` (`toilet`)
- Split: `input_12`
- Frames, in required order: `1, 22, 36, 56, 77, 91, 112, 126, 147, 167, 181, 202`
- Tensor:
  `/home/utn/luli38se/cv/3D-Reconstruction/outputs/model_tests/quant_3dgs/prepared/602_93503_186945/input_12/input_tensor.pt`
- Required tensor shape: `[12, 3, 518, 518]`
- Required tensor SHA-256:
  `41fdcacfa3d403f6aa76dda70d22d845c44ce295eb61fb195bf77a98d744fb87`
- Full checkpoint:
  `/home/utn/luli38se/cv/QuantVGGT/VGGT-1B/model_tracker_fixed_e20.pt`

The runner checks the tensor hash before model construction. Do not load or
preprocess the original JPEGs. Do not change input order, resolution,
preprocessing, autocast, camera decoding, point unprojection, or output
conversion. Do not substitute held-out images. Do not enable bundle adjustment
or any camera optimization.

## Exact commands

```bash
cd /home/utn/luli38se/cv/3D-Reconstruction
/home/utn/luli38se/cv/.venv/bin/python scripts/model_tests/quant_3dgs/run_vggt_variant.py --variant full
```

The expected output directory is:

`outputs/model_tests/quant_3dgs/predictions/602_93503_186945/input_12/full/`

Validate it immediately:

```bash
cd /home/utn/luli38se/cv/3D-Reconstruction
/home/utn/luli38se/cv/.venv/bin/python scripts/model_tests/quant_3dgs/validate_vggt_predictions.py --prediction-dir outputs/model_tests/quant_3dgs/predictions/602_93503_186945/input_12/full
```

Optionally create the same visual sanity check used for W4A4:

```bash
cd /home/utn/luli38se/cv/3D-Reconstruction
/home/utn/luli38se/cv/.venv/bin/python scripts/model_tests/quant_3dgs/inspect_vggt_outputs.py --prediction-dir outputs/model_tests/quant_3dgs/predictions/602_93503_186945/input_12/full --output outputs/model_tests/quant_3dgs/diagnostics/602_93503_186945/input_12/full_overview.png
```

Do not use `--force` unless replacement of an already completed Full run has
been explicitly approved.
