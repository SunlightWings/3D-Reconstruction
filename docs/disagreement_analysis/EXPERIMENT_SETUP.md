# Experiment setup

## Frozen inputs

The selection was fixed before inference: `toytruck/190_20494_39385` frames `[1,41,75,123,162,202]`; `bench/415_57112_110099` and `toaster/372_41229_82130` frames `[1,41,81,122,162,202]`. Each Full and W4A4 run consumed the identical saved CPU tensor for its scene: `[6,3,518,518]`, `float32`, range `[0,1]`.

Preprocessing was local VGGT pad/no-crop preprocessing: Pillow bicubic RGB resize to an approximately 518-pixel content dimension, white padding to 518×518, no ImageNet normalization. No JPEG was independently preprocessed per model.

## Models and conversion

Full VGGT and the locally audited W4A4 resumed quantization state share the same base checkpoint. Both use the same common VGGT output conversion, `torch.inference_mode`, bfloat16 autocast, and depth-to-point unprojection. There was no tracking, bundle adjustment, confidence filtering, COLMAP refinement, GT pose optimization, or per-frame scale fitting.

## Ground truth

CO3D depth PNG `uint16` values were decoded as float16 **bit patterns**, then converted to float32 and multiplied by `scale_adjustment`; they are not ordinary integer depth values. Corrected GT depth/camera unprojection had median nearest-neighbour error to the official sequence point cloud of 0.3524%, 0.1751%, and 0.2989% of robust radius for toytruck, bench, and toaster respectively.

## Alignment and metrics

For each model and scene, one proper Sim(3) was fitted only from the six corresponding GT and predicted camera centres:

`P_model = s (A P_GT) + b`.

Predicted points were inverse-mapped to GT coordinates before per-pixel comparisons. Primary GT analysis used deterministic 5,000 foreground-valid samples per input frame. The localization analysis reports all valid foreground pixels and uses boundary/interior and depth-gradient regions. Exact paths, hashes, and analysis variants are in [PROVENANCE.md](PROVENANCE.md).
