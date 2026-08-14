# Disagreement analysis freeze

This directory freezes the evidence used to justify Full–W4A4 geometric disagreement as an offline QuantSplat supervision signal. Across three frozen CO3D GT scenes, disagreement is a **scene-dependent** predictor of absolute W4A4 error, but a substantially stronger indicator of W4A4-induced excess geometric risk.

- [Scientific report](DISAGREEMENT_ANALYSIS.md)
- [Experiment setup](EXPERIMENT_SETUP.md)
- [Provenance](PROVENANCE.md)
- [Compact machine-readable results](../../results/disagreement_analysis/)

The decision is **MIXED** for absolute-error supervision and **SUPPORTED / PROCEED** for quantization-induced uncertainty supervision. This is not a claim of universal generalization from three scenes.
