# QuantSplat experimental results

Frozen: 2026-09-14

## 1. Learned camera-confidence predictor

Question:
Can W3A3 output statistics predict frames whose camera geometry is unreliable?

Untouched category-disjoint evaluation:
- Analytic consistency: AUROC 0.3714, AP 0.3875
- Learned confidence: AUROC 0.7302, AP 0.7056

Interpretation:
The learned model is substantially more informative about W3A3 camera risk
than the hand-designed analytic score.

The confidence network predicts error/risk. It is not merely a focal
calibrator and it does not directly predict camera correction vectors.

## 2. Camera residual corrector

The residual corrector is a separate learned component.

It predicts corrections for:
- camera rotation,
- camera center,
- focal parameters.

Its output is applied to W3A3 camera geometry before downstream 3DGS.

## 3. Custom held-out cup: method ablation

Compared:
- Full VGGT
- W4A4
- W3A3 raw
- W3A3 + residual corrector
- W3A3 + learned-confidence gate
- W3A3 + corrector + confidence

A real held-out image was excluded from 3DGS optimization and rendered as
a novel test view.

## 4. View-count study

Input-view counts:
N = 3, 6, 9, 12

The same real cup target was held out from 3DGS.

PSNR table:

| Method | N=3 | N=6 | N=9 | N=12 |
|---|---:|---:|---:|---:|
| Full VGGT | 13.772 | 16.631 | 12.770 | 15.885 |
| W4A4 | 13.854 | 15.441 | 12.317 | 15.074 |
| W3A3 raw | 13.948 | 15.374 | 13.122 | 15.233 |
| W3A3 + corrector | 13.958 | 15.380 | 12.759 | 15.176 |
| W3A3 + confidence | 11.880 | 15.681 | 14.485 | 15.173 |
| W3A3 + corrector + confidence | 12.331 | 15.511 | 14.223 | 15.407 |

The complete PSNR/SSIM/LPIPS/MAE measurements are supplied as JSON/CSV.

## 5. Bottle replication

Fixed held-out bottle view:

| Method | PSNR | MAE |
|---|---:|---:|
| W3A3 raw | 13.978 | 0.16113 |
| W3A3 + corrector | 13.409 | 0.17639 |
| W3A3 + confidence | 12.521 | 0.19818 |
| W3A3 + corrector + confidence | 12.233 | 0.20804 |

This replication is important: confidence gating and correction are not
universally beneficial.

## 6. Multi-view stability

Three fixed cup views were excluded simultaneously from 3DGS optimization.

| Method | PSNR mean | SSIM mean | LPIPS mean |
|---|---:|---:|---:|
| Full VGGT | 21.288 | 0.7206 | 0.3198 |
| W4A4 | 19.230 | 0.6021 | 0.3868 |
| W3A3 raw | 18.045 | 0.5116 | 0.5141 |
| W3A3 + corrector | 18.013 | 0.5164 | 0.4858 |
| W3A3 + confidence | 17.624 | 0.4982 | 0.5018 |
| W3A3 + corrector + confidence | 18.161 | 0.5225 | 0.4874 |

Among W3A3 variants, the combined method gives the highest mean PSNR and
SSIM, while correction alone gives the lowest LPIPS.

## 7. Scientific conclusion

The evidence does NOT support the claim that confidence gating or residual
correction always improves W3A3.

Supported conclusion:

Learned reliability estimation identifies risky aggressively-quantized
geometry, and residual camera correction can recover part of the resulting
downstream quality loss. The benefit of hard confidence-based view removal
depends strongly on scene and view availability.
