# Handoff: VGGT → 3DGS sparse-view pipeline, and how to quantize to any bit-width

Everything needed to reproduce the Full-VGGT → 3DGS baseline, and to calibrate and evaluate
a new quantization setting (W3A3, W1A1, W2A4, …) end to end.

**Read section 6 before drawing any conclusion from a PSNR number.** Two published results in
this project reversed once the metric was corrected, and the corrections are not optional.

---

## 1. What the pipeline does

Six unposed RGB images of an object go in. VGGT predicts cameras, depth and a point map.
Those initialize a 3D Gaussian Splatting scene, which is trained on the six images and then
rendered from nine held-out viewpoints and compared against the real photographs.

The question is whether *quantized* VGGT geometry produces measurably worse renders than
full-precision VGGT geometry.

```
6 RGB images ─► VGGT (full | w4a4 | w2a4 | …) ─► cameras + depth + points
                                                        │
                              Sim(3) align to GT frame ─┤
                                                        ▼
                                        3DGS train (7k iters)
                                                        │
                                       render 9 held-out views
                                                        │
                                     PSNR / SSIM / LPIPS vs GT
```

---

## 2. Environment

| Component | Location / version |
|---|---|
| VGGT + QuantVGGT | `~/cv/QuantVGGT` (from https://github.com/wlfeng0509/QuantVGGT) |
| VGGT-1B weights | `~/cv/QuantVGGT/VGGT-1B/model_tracker_fixed_e20.pt` (4.7 GB) |
| 3DGS | `~/cv/gaussian-splatting` (GraphDECO official) |
| 3DGS python | `/opt/saltstack/salt/bin/python3.10` (has the CUDA rasterizer built) |
| Main python | `~/cv/.venv/bin/python` (3.12) |
| CO3D-v2 | `/var/tmp/luli38se/co3d_single_all` (8.5 GB) |
| Calibration set | `calib_data.pt` (1.3 GB, 42 samples) — **download, see 2.1** |
| Frozen scene manifest | `data/frozen_dataset_manifest.json` — **included in this zip**, see 2.2 |
| GPU used | RTX 4000 Ada, 20 GB |

Two separate interpreters are load-bearing: 3DGS runs under `GS_PY` because the rasterizer
extension is compiled against it, everything else under the venv. `code/downstream_3dgs/run_downstream_validation.py`
holds both paths (`GS_PY`, `CV_PY`) and the `gs_env()` helper that sets `PYTHONPATH` for the
3DGS subprocess. Don't try to merge them.

Python deps beyond the two repos' requirements: `lpips`, `scikit-image`, `scipy`, `opencv-python`,
`pycolmap` (only for the LLFF arm).

### 2.1 Files you must download (too large to ship)

| File | Size | Where from | Put it at |
|---|---|---|---|
| `model_tracker_fixed_e20.pt` | 4.7 GB | https://huggingface.co/wlfeng/QuantVGGT | `~/cv/QuantVGGT/VGGT-1B/` |
| `calib_data.pt` | 1.3 GB | same HF repo (the filtered CO3D calibration set) | anywhere; pass with `--calib-path` |
| W4A4 qs parameters | 3.8 GB | same HF repo | `~/cv/QuantVGGT/evaluation/outputs/w4a4/a44_model_tracker_fixed_e20.pt_sym/` |
| CO3D-v2 | ~8.5 GB | https://github.com/facebookresearch/co3d | set `CO3D_ROOT` in `run_downstream_validation.py` |

`calib_data.pt` is the authors' k-means-filtered CO3D calibration set (42 multi-view samples).
You can rebuild it instead with `QuantVGGT/evaluation/make_calibation.py`, but then your
calibration is not comparable to anything calibrated with theirs — keep one or the other
throughout.

`calibrate_w2a4.py` defaults to `/var/tmp/luli38se/quantsplat/models/QuantVGGT/calib_data.pt`;
override with `--calib-path`. `quant_loader.py` and `run_downstream_validation.py` have the
other paths hard-coded near the top of each file — edit them for your machine.

### 2.2 The frozen scene manifest (shipped: `data/frozen_dataset_manifest.json`)

This 4.7 MB file **is** the experiment definition: which 40 CO3D scenes, which frames are usable
per scene, and which six-view groups exist. Every result in this project is tied to it.

Its SHA-256 is hard-coded as `EXPECTED_MANIFEST_SHA256` in `run_full_dataset_disagreement.py`,
`run_downstream_validation.py` and `code/quantization/run_w2a4_inference.py`, and those scripts
**abort** if it does not match:

```
1ce2f3f8d43cc17f61d84545b82f132a3b64d49d5acf70ff0fe0ac6aa9968e06
```

Copy it to `/var/tmp/<user>/quantsplat/disagreement_dataset_v1/frozen_dataset_manifest.json`
(or edit the `MAN` / `FROZEN` path constants). Do not regenerate it — a regenerated manifest
picks different frames, changes the hash, and makes your numbers incomparable to the shipped
results.

---

## 3. Run the full-precision baseline (already done — results included)

The 40-scene Full and W4A4 arms are already trained; models and renders live under
`/var/tmp/luli38se/quantsplat/downstream_validation_v1/<category>/<sequence>/`. Included
results:

| File | What it is |
|---|---|
| `results/downstream_3dgs/aggregate_metrics.json` | 40-scene Full vs W4A4 summary (content mask) |
| `results/downstream_3dgs/aggregate_metrics.csv` | per-scene version of the above |
| `results/downstream_3dgs/convergence_7k_vs_30k.json` | why 7k iterations is enough |
| `results/downstream_3dgs/DIAGNOSTIC_RESULTS.md` | the 16 diagnostic gates and their verdicts |
| `sweep_results/main40_both_masks.json` | **the corrected 40-scene result — use this one** |
| `sweep_results/*.json` | every configuration experiment (see `ongoing_logs.md`) |

To regenerate from scratch (many GPU-hours):

```bash
# 1. VGGT inference for full + w4a4 over the frozen scene manifest
python3 code/run_full_dataset_disagreement.py

# 2. 3DGS train + render + evaluate, 40 scenes
python3 code/downstream_3dgs/run_full_dataset_downstream.py --iterations 7000
```

Both are resumable — re-running skips completed scenes.

---

## 4. Quantize to a new bit-width (W3A3, W1A1, …)

### 4.1 Why you cannot skip calibration

Quantization here is not a post-hoc rounding of weights. QuantVGGT learns per-layer clipping
thresholds (`lwc` weight clipping, `lac` activation clipping) *for a specific bit-width*. At 4
bits a weight takes one of 16 values; at 2 bits, one of 4; at 1 bit, one of 2. Thresholds
fitted for one are wrong for another.

**Reusing the W4A4 parameters at another bit-width produces a model that looks quantized but
whose clipping was fitted for a different quantizer.** Any number from it is meaningless. This
is why gate D3 in `DIAGNOSTIC_RESULTS.md` is marked BLOCKED rather than filled with a plausible
figure.

Note the W4A4 parameters in `QuantVGGT/evaluation/outputs/w4a4/` were **downloaded from the
authors' HuggingFace release**, not calibrated locally. Anything you calibrate yourself is not
strictly comparable to them — see section 7.

### 4.2 Run the calibration

```bash
python3 code/quantization/calibrate_w2a4.py --wbit 3 --abit 3 --exp-name a33
python3 code/quantization/calibrate_w2a4.py --wbit 1 --abit 1 --exp-name a11
```

Despite the filename it takes any bit-width. Useful flags:

| Flag | Meaning |
|---|---|
| `--wbit` / `--abit` | weight / activation bits |
| `--exp-name` | short tag; `a33` for W3A3, by the authors' `a44` convention |
| `--epochs` | default 15 per block |
| `--nsamples` | limit calibration samples (default: all 42) |
| `--dry-run` | verify setup and exit without calibrating |

**Cost:** 48 transformer blocks (24 frame + 24 global) × 15 epochs × ~17 s ≈ **3.5 hours** on a
20 GB GPU, ~8 GB peak. It writes an 80 MB file per block as it goes, then merges into
`qs_frame_parameters_total.pth` + `qs_global_parameters_total.pth` (~1.9 GB each) under
`QuantVGGT/evaluation/outputs/w<W>a<A>/<exp>_model_tracker_fixed_e20.pt_sym/`.

Watch `app.log` in that directory. The `mse` per block is the quantized block's output error
against the full-precision block. It should fall over the 15 epochs (W2A4 block 0 went
0.00267 → 0.00078). **If mse plateaus immediately or grows, that bit-width is not converging —
stop rather than spend the remaining hours.**

### 4.3 Register the new variant

Add an entry to `VARIANTS` in `code/quantization/quant_loader.py`:

```python
"w3a3": {
    "wbit": 3, "abit": 3, "exp_name": "a33",
    "qs_dir": QUANTVGGT_ROOT / "evaluation/outputs/w3a3/a33_model_tracker_fixed_e20.pt_sym",
    "provenance": "calibrated locally on <date>",
},
```

Check it loads: `python3 code/quantization/quant_loader.py w3a3`

### 4.4 Predictions, then 3DGS

```bash
python3 code/quantization/run_w2a4_inference.py  --variant w3a3 --scenes all
python3 code/quantization/run_w2a4_downstream.py --scenes all --iterations 7000
```

The inference step **aborts** if the preprocessed input images do not hash identically to the
`full` arm's. That check is deliberate — an unnoticed input difference would invalidate the
comparison while still producing plausible numbers. It also records inference seconds and peak
GPU memory per scene, which is the efficiency half of the claim.

To compare arms other than `full`/`w4a4`/`w2a4`, edit `ARMS` at the top of
`run_w2a4_downstream.py`.

---

## 5. Iterations: 7000, not 30000

Every arm is rendered at 7k. Measured justification:

| source | 7k → 30k gain |
|---|---|
| `convergence_7k_vs_30k.json`, 40 scenes | Full **+0.095 dB**, Quant **+0.065 dB**, SSIM **−0.016** |
| oracle sweep, 8 scenes (`ongoing_logs.md` #003) | 14.023 → 14.061 dB (**+0.038**) |

4.3× the compute for under 0.1 dB, and SSIM gets worse. **All arms must be rendered at the same
iteration** — otherwise bit-width is confounded with training length. The runner enforces this
by rendering every arm at `--iterations`, never mixing checkpoints.

---

## 6. Metrics — read this before trusting a number

### 6.1 Use the foreground mask

The "content mask" is the whole non-letterbox image. On CO3D the object is a mean **15.3%** of
it, and CO3D supplies ground-truth depth for **~0.1%** of background pixels — so the background
is unreconstructable by any arm, and a content-mask PSNR is mostly measuring it.

This is not a theoretical worry. Re-scoring the 40-scene result on the object:

| | content mask | foreground mask |
|---|---:|---:|
| Full − W4A4 | +0.2038 dB | +0.1115 dB |
| p | 0.0092 | **0.174** |
| 95% CI | [+0.054, +0.354] | **[−0.051, +0.274]** (includes zero) |
| scenes favouring Full | 31/40 | **20/40** (chance) |

**15 of 40 scenes reverse sign between the two masks.** The published "Full is better" result
does not survive. Both masks are reported by the runner; foreground is primary.

### 6.2 Correct for exposure

CO3D is captured with auto-exposure, and the evaluation applies no correction. Fitting a single
per-image per-channel gain+bias before scoring recovers **+3.5 to +5.8 dB** — roughly 30× the
effect being measured. On the 8-scene subset the correction *flips the sign* of the Full−W4A4
delta (+0.187 raw → −0.103 corrected; both non-significant).

The runner reports `psnr` and `psnr_exposure_corrected` side by side. If they disagree, exposure
is doing the talking.

### 6.3 Report an effect-size bound, not a null

With n=40 the minimum detectable effect at 80% power is ~0.23 dB. Anything smaller cannot be
resolved. Say "the advantage is +0.11 dB, 95% CI [−0.05, +0.27]", never "there is no difference".

### 6.4 Known harness limits

| Issue | Size | Where |
|---|---|---|
| Predicted↔GT camera alignment error | 0.081 (full) / 0.138 (w4a4) scene radii vs a 0.02 target | `ongoing_logs.md` #018 |
| — costs | **2.2 dB**, 38% of the oracle-to-VGGT gap | #021 |
| — irreducible: GT→predicted map is not a similarity | best in-sample Sim(3) residual 0.0485 | #023 |
| Oracle (GT geometry + GT cameras) reaches only | 14.83 dB foreground | #004 |
| — and is beaten by copying the nearest training photo | 15.41 dB | #022 |

The last row is the uncomfortable one: an arm with *perfect* geometry and *perfect* cameras
scores worse than doing no reconstruction at all. Any claim about geometry quality has to
survive that.

---

## 7. If you calibrate a new bit-width, also re-calibrate W4A4

The shipped W4A4 came from the authors; anything you calibrate locally uses 42 calibration
samples on this machine. If your W3A3 looks worse than W4A4, you cannot tell whether 3-bit is
worse or your calibration is worse.

The control is one extra run — `--wbit 4 --abit 4 --exp-name a44_local` — and comparing
locally-calibrated W4A4 against locally-calibrated W3A3. It also measures your calibration
quality against the authors' as a by-product.

---

## 8. File map

```
code/
  run_full_dataset_disagreement.py       full + w4a4 VGGT inference over the frozen manifest
  vggt_inference_core.py                 model load + inference + NPZ contract
  w4a4_memory_optimized_loader.py        original W4A4 loader (superseded by quant_loader.py)
  analyze_full_dataset_disagreement.py   geometric disagreement analysis; owns resize_depth_and_masks
  quantization/
    calibrate_w2a4.py                    ← calibrate ANY bit-width
    quant_loader.py                      ← load any calibrated bit-width
    run_w2a4_inference.py                ← predictions for a quantized arm
    run_w2a4_downstream.py               ← 3DGS + 3-arm comparison + statistics
  downstream_3dgs/
    run_downstream_validation.py         core pipeline: sources, train, render, metrics
    run_full_dataset_downstream.py       40-scene driver
    diagnostics/                         16 diagnostic gates (gate_a..gate_e)
    oracle_sweep/                        config sweep, ceilings, alignment probes
    llff/                                LLFF arm (BLOCKED — needs the benchmark data)
results/downstream_3dgs/                 metrics + DIAGNOSTIC_RESULTS.md
sweep_results/                           all experiment JSONs
ongoing_logs.md                          append-only experiment log — 25 entries
CLAUDE.md                                logging rules
```

`ongoing_logs.md` is the real documentation. Its DECISION STATE section at the top says which
gates pass, what conclusion the evidence licenses, and what to run next.

---

## 9. Fastest path to a result

```bash
# 1. calibrate (3.5 h)
python3 code/quantization/calibrate_w2a4.py --wbit 3 --abit 3 --exp-name a33
# 2. register w3a3 in quant_loader.py VARIANTS
# 3. predictions (minutes)
python3 code/quantization/run_w2a4_inference.py --variant w3a3 --scenes subset
# 4. 3DGS + metrics on 8 scenes first (~1 h) — confirms the geometry is sane
python3 code/quantization/run_w2a4_downstream.py --scenes subset --iterations 7000
# 5. only if that looks sane, all 40
python3 code/quantization/run_w2a4_downstream.py --scenes all --iterations 7000
```

Read the `foreground / psnr_exposure_corrected` column and the paired test for
`full_minus_w3a3__foreground__psnr_exposure_corrected`. If its CI includes zero, there is no
measurable degradation at that bit-width — which is a real finding, and at 4 bits it is the
current answer.
