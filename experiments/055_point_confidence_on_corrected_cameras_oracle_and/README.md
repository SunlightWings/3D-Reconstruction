# 055 — Point confidence on corrected cameras: oracle and learned predictor, W3A3, 8 scenes

_No `ongoing_logs.md` entry exists for this work yet. Every number below is read directly from the files in `results/`._

**Question.** Given the W3A3 VGGT prior, can a per-point confidence signal make 3DGS reconstruct better? Tested on the corrected cameras (054), W3A3 geometry, 8 scenes, 7000 iterations, foreground mask.

**Signals.** `oracle` = true per-point error of W3A3 against full precision (upper bound, not deployable). `predictor` = a logistic model on W3A3-only features (`code/predictor_confidence.py`, weights in the result json). **Uses.** `prune` = drop the least-confident 50 % of init points; `weight` = set each point's initial opacity from confidence (GraphDECO patched to read `GS_INIT_OPACITY_NPY`, see `code/gaussian_model.py` and `code/gaussian_model.init_opacity.diff`; default behaviour unchanged). `rand_prune` = random 50 % (point-count control).

**Means** (from `results/score_confidence_w3a3_qfix.json`):

| arm | n | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| full_qfix | 8 | 14.574 | 0.3195 | 0.3464 |
| w3a3_qfix | 8 | 12.983 | 0.2485 | 0.4390 |
| w3a3_qfix_oracle_prune | 8 | 13.142 | 0.2551 | 0.4407 |
| w3a3_qfix_oracle_weight | 8 | 13.019 | 0.2498 | 0.4376 |
| w3a3_qfix_rand_prune | 8 | 13.019 | 0.2480 | 0.4393 |
| w3a3_qfix_predictor_prune | 8 | 12.533 | 0.2477 | 0.4478 |
| w3a3_qfix_predictor_weight | 8 | 12.931 | 0.2491 | 0.4379 |

**Paired differences** (first − second, over scenes; for LPIPS the sign is as stored in the file):

| comparison | metric | mean diff | t-test p | Wilcoxon p | scenes first better |
|---|---|---:|---:|---:|---:|
| full_qfix vs w3a3_qfix | psnr | 1.5904 | 0.0118 | 0.0156 | 7/8 |
| full_qfix vs w3a3_qfix | ssim | 0.0710 | 0.0070 | 0.0156 | 7/8 |
| full_qfix vs w3a3_qfix | lpips | 0.0926 | 0.0038 | 0.0078 | 8/8 |
| w3a3_qfix_oracle_prune vs w3a3_qfix | psnr | 0.1589 | 0.3045 | 0.4609 | 4/8 |
| w3a3_qfix_oracle_prune vs w3a3_qfix | ssim | 0.0065 | 0.0598 | 0.0547 | 7/8 |
| w3a3_qfix_oracle_prune vs w3a3_qfix | lpips | -0.0017 | 0.8717 | 0.8438 | 3/8 |
| w3a3_qfix_oracle_weight vs w3a3_qfix | psnr | 0.0354 | 0.5836 | 0.8438 | 5/8 |
| w3a3_qfix_oracle_weight vs w3a3_qfix | ssim | 0.0013 | 0.3752 | 0.1953 | 7/8 |
| w3a3_qfix_oracle_weight vs w3a3_qfix | lpips | 0.0014 | 0.4175 | 0.4609 | 5/8 |
| w3a3_qfix_rand_prune vs w3a3_qfix | psnr | 0.0358 | 0.6188 | 0.9453 | 4/8 |
| w3a3_qfix_rand_prune vs w3a3_qfix | ssim | -0.0006 | 0.6845 | 0.6406 | 4/8 |
| w3a3_qfix_rand_prune vs w3a3_qfix | lpips | -0.0003 | 0.8593 | 0.7422 | 3/8 |
| w3a3_qfix_oracle_prune vs w3a3_qfix_rand_prune | psnr | 0.1231 | 0.4638 | 0.8438 | 4/8 |
| w3a3_qfix_oracle_prune vs w3a3_qfix_rand_prune | ssim | 0.0071 | 0.0276 | 0.0547 | 7/8 |
| w3a3_qfix_oracle_prune vs w3a3_qfix_rand_prune | lpips | -0.0013 | 0.9015 | 0.9453 | 5/8 |
| w3a3_qfix_predictor_prune vs w3a3_qfix | psnr | -0.4505 | 0.1078 | 0.1094 | 1/8 |
| w3a3_qfix_predictor_prune vs w3a3_qfix | ssim | -0.0008 | 0.7463 | 1.0000 | 4/8 |
| w3a3_qfix_predictor_prune vs w3a3_qfix | lpips | -0.0088 | 0.5003 | 0.5469 | 3/8 |
| w3a3_qfix_predictor_weight vs w3a3_qfix | psnr | -0.0522 | 0.5279 | 0.4609 | 2/8 |
| w3a3_qfix_predictor_weight vs w3a3_qfix | ssim | 0.0006 | 0.7436 | 0.7422 | 5/8 |
| w3a3_qfix_predictor_weight vs w3a3_qfix | lpips | 0.0011 | 0.6281 | 0.5469 | 4/8 |
| w3a3_qfix_predictor_prune vs w3a3_qfix_rand_prune | psnr | -0.4863 | 0.1217 | 0.1094 | 2/8 |
| w3a3_qfix_predictor_prune vs w3a3_qfix_rand_prune | ssim | -0.0002 | 0.9182 | 0.6406 | 3/8 |
| w3a3_qfix_predictor_prune vs w3a3_qfix_rand_prune | lpips | -0.0084 | 0.5505 | 0.8438 | 4/8 |

**Predictor ranking quality** (from `results/predictor_confidence_w3a3.json`, mean over scenes):

- `auroc_top_decile`: 0.843
- `spearman_conf_vs_neg_err`: 0.607

**Reading.** Compare each confidence arm against `w3a3_qfix` and against `w3a3_qfix_rand_prune` in the table above; the p-values are the evidence on whether point-level confidence changes the reconstruction.

---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/confidence_3dgs.py`
- `code/gaussian_model.init_opacity.diff`
- `code/gaussian_model.py`
- `code/predictor_confidence.py`
- `code/run_confidence_chain.sh`
- `code/score_arms.py`

**results/**

- `results/conf_predictor/apple_110_13051_23361_w3a3.npy`
- `results/conf_predictor/ball_123_14363_28981_w3a3.npy`
- `results/conf_predictor/bowl_70_5792_13401_w3a3.npy`
- `results/conf_predictor/broccoli_412_56288_108844_w3a3.npy`
- `results/conf_predictor/hydrant_167_18184_34441_w3a3.npy`
- `results/conf_predictor/remote_350_36761_68623_w3a3.npy`
- `results/conf_predictor/teddybear_187_20215_38541_w3a3.npy`
- `results/conf_predictor/toaster_372_41229_82130_w3a3.npy`
- `results/confidence_3dgs_w3a3_oracle_it7000.json`
- `results/confidence_3dgs_w3a3_predictor_it7000.json`
- `results/predictor_confidence_w3a3.json`
- `results/score_confidence_w3a3_qfix.json`

**logs/**

- `logs/confidence_chain.log`
- `logs/confidence_oracle_qfix.log.gz`
- `logs/confidence_predictor_qfix.log.gz`
- `logs/confidence_score.log`
- `logs/predictor_confidence_w3a3.log`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.
