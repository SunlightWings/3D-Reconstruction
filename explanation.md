# Full-vs-W4A4 Geometric Disagreement Validation

## 1. What this experiment investigates

QuantVGGT reduces VGGT precision using W4A4 simulated quantization. The central
question of this experiment is whether quantization changes the predicted 3D
geometry in a consistent and measurable way, and whether those changes identify
locations where quantization makes the geometry worse.

The experiment compares:

- Full-precision VGGT
- W4A4 QuantVGGT
- CO3D ground-truth geometry

on exactly the same images.

---

## 2. What "geometric disagreement" means

For an image pixel u:

- P_F(u) = Full VGGT predicted 3D point
- P_Q(u) = W4A4 QuantVGGT predicted 3D point
- P_GT(u) = ground-truth 3D point

After aligning Full and W4A4 into the same GT coordinate system, geometric
disagreement is:

    d(u) = ||P_Q(u) - P_F(u)|| / R

where R is the robust GT scene radius.

This is a full 3D geometric displacement, not only a depth difference.

Large d(u) means Full and W4A4 predict substantially different 3D geometry at
that pixel.

---

## 3. Quantization-induced excess geometric error

Disagreement alone does not tell us which model is correct, so ground truth was
used to validate the signal.

Full GT error:

    e_F(u) = ||P_F(u) - P_GT(u)|| / R

W4A4 GT error:

    e_Q(u) = ||P_Q(u) - P_GT(u)|| / R

Signed change caused by quantization:

    delta_e(u) = e_Q(u) - e_F(u)

If delta_e > 0, W4A4 is worse than Full at that pixel.

Positive quantization-induced excess error is:

    e_excess(u) = max(e_Q(u) - e_F(u), 0)

Therefore, e_excess measures the additional geometric error associated with
quantization relative to Full VGGT.

---

## 4. Dataset and experiment

The frozen experiment contains:

- 40 CO3D sequences
- 7,889 primary evaluation frames
- 1,330 six-view VGGT groups
- 91 context-only frame occurrences
- 0 duplicated primary frames
- 0 missing primary frames

Frozen dataset manifest SHA256:

    1ce2f3f8d43cc17f61d84545b82f132a3b64d49d5acf70ff0fe0ac6aa9968e06

Each Full and W4A4 group used the same six images and identical deterministic
518x518 pad/no-crop preprocessing.

The semantic input hash was recorded for both variants and required to match.

For each six-view group:

1. Run Full VGGT.
2. Run W4A4 QuantVGGT.
3. Fit one proper Sim(3) from GT camera centers to Full cameras.
4. Fit one separate proper Sim(3) from GT camera centers to W4A4 cameras.
5. Transform predictions into the common GT coordinate system.
6. Score only primary frames.
7. Compare Full, W4A4, and GT geometry.

There is no per-frame alignment, no bundle adjustment, and no fitting against GT
depth.

CO3D depth is decoded using the validated convention:

    uint16 PNG bits
    -> reinterpret as float16
    -> float32
    -> multiply by scale_adjustment

---

## 5. Main disagreement result

Across the 40 evaluated scenes:

| Metric | Result |
|---|---:|
| Scenes | 40 |
| Primary frames | 7,889 |
| Six-view groups | 1,330 |
| Positive rho(disagreement, excess error) | 39 / 40 |
| rho > 0.4 | 24 / 40 |
| Median rho(disagreement, excess error) | 0.443 |
| Excess-error AUC > 0.5 | 40 / 40 |
| Excess-error AUC > 0.75 | 38 / 40 |
| Median excess-error AUC | 0.906 |
| Median top-10% precision | 44.3% |
| Random top-10% precision | ~10% |
| Shuffled disagreement median AUC | 0.500 |

The important result is that Full-vs-W4A4 disagreement consistently ranks
pixels that suffer large quantization-induced excess geometric error.

The disagreement should therefore be interpreted as a quantization-risk /
uncertainty signal, not as an exact calibrated estimate of total GT error.

---

## 6. Where does the disagreement occur?

A second CPU-only characterization classified pixels using overlapping
visual/geometric cues.

Enrichment is defined as:

    P(region | top-10% excess error) / P(region)

A value above 1 means that region occurs more often among the highest
quantization-damaged pixels than expected from its normal frequency.

| Region | Excess-error enrichment | Disagreement enrichment |
|---|---:|---:|
| Object boundary | 2.02x | 2.65x |
| High GT depth gradient | 1.39x | 1.35x |
| High texture | 1.20x | 1.25x |
| RGB edge | 1.20x | 1.22x |
| Harris corner | 1.11x | 1.12x |
| Low texture | 0.87x | 0.86x |
| Smooth/flat interior | 0.72x | 0.59x |

Object boundaries are the strongest consistent spatial pattern:

- they represent about 4.6% of valid pixels,
- but about 9.4% of the top-10% excess-error pixels,
- giving 2.02x median enrichment,
- with enrichment above 1x in 97.5% of scenes.

High depth gradients, high texture, and RGB edges also show consistent but
weaker enrichment.

Smooth/flat regions and low-texture regions are comparatively less vulnerable.

Thin/protrusion regions showed strong disagreement enrichment, but the
excess-error confidence interval crossed 1, so this is treated as suggestive
rather than a primary conclusion.

These region cues overlap. Their percentages must not be added together, and
this analysis is associative rather than causal.

---

## 7. Final conclusion

W4A4 quantization causes measurable changes in VGGT's predicted 3D geometry.

Across the evaluated CO3D subset, large Full-vs-W4A4 geometric disagreement
consistently identifies locations where quantization introduces additional
geometric error.

The signal is particularly enriched near object boundaries and, more modestly,
at depth transitions, textured regions, and RGB edges.

This supports using Full-vs-W4A4 disagreement as an offline supervision signal
for a lightweight uncertainty predictor.

The predictor will receive only QuantVGGT features at deployment time. Full
VGGT is required only during offline target generation.

---

## 8. Pipeline files

`code/run_full_dataset_disagreement.py`

Runs Full VGGT and W4A4 over the frozen 40-scene dataset and stores resumable
prediction outputs.

`code/analyze_full_dataset_disagreement.py`

CPU-only validation and GT analysis. Computes disagreement, absolute GT error,
quantization-induced excess error, scene-level correlations, AUC, top-decile
precision, shuffled controls, and dataset summaries.

`code/characterize_disagreement_regions.py`

CPU-only spatial characterization. Measures where disagreement and excess error
are enriched: boundaries, RGB edges, corners, texture regions, thin-structure
heuristics, depth gradients, and smooth interiors.

The remaining files in `code/` are shared inference, preprocessing, prediction
contract, and W4A4 loading helpers.

---

## 9. How to run

The current experiment expects the CO3D subset, VGGT checkpoint, QuantVGGT
W4A4 artifacts, and experiment output directories to exist at the workstation
paths configured in the pipeline.

Python environment:

    /home/utn/luli38se/cv/.venv

### Stage 1: Full + W4A4 inference

    /home/utn/luli38se/cv/.venv/bin/python -u \
      code/run_full_dataset_disagreement.py

The runner is resumable and skips already-valid predictions.

### Stage 2: dataset-wide GT disagreement analysis

    /home/utn/luli38se/cv/.venv/bin/python -u \
      code/analyze_full_dataset_disagreement.py

### Stage 3: region characterization

    /home/utn/luli38se/cv/.venv/bin/python -u \
      code/characterize_disagreement_regions.py \
      --workers 4

The final compact results are stored in `results/`.

Large raw datasets, model checkpoints, cached tensors, and the 2,660 dense
prediction NPZ files are intentionally not committed to Git.
