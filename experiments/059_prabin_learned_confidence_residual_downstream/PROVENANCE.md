# Experiment 059 — learned W3A3 confidence and camera residual correction

Owner: Prabin Sharma Poudel
Frozen: 2026-09-14

This experiment bundle contains the learned downstream studies performed on
QuantVGGT W3A3.

Included evidence covers:

- category-disjoint learned camera-confidence prediction;
- analytic-vs-learned confidence comparison;
- learned camera residual correction;
- correction / confidence / combined downstream ablations;
- real custom cup held-out novel-view reconstruction;
- fixed held-out view-count study at N={3,6,9,12};
- independent bottle replication;
- PSNR, SSIM, LPIPS and MAE evaluation;
- three-fixed-held-out-view visual-stability evaluation;
- inference/memory measurements;
- frozen confidence and residual-corrector models;
- deployment/inference code and machine-readable CSV/JSON results.

Important scientific conclusion:

The learned reliability predictor successfully predicts W3A3 camera risk,
and residual correction can recover part of quantization-induced degradation.
Hard confidence-based frame rejection is not universally beneficial: its
downstream effect depends on scene and available view count.
