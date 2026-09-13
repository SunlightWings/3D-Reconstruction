# Paper evidence: quantized geometry priors for sparse-view 3DGS

Consolidated technical record for writing up this work. Every number here was measured, and
each is traced to the artifact that produced it and the `ongoing_logs.md` entry that logged
it. Nothing in this file is an estimate.

**Status marker:** ✅ measured · ⏳ running · ⛔ blocked · ❌ refuted

---

## 1. The claim the project started with, and what happened to it

**Original research question.** *Can a lightweight confidence predictor reduce the novel-view
rendering degradation caused by QuantVGGT geometry errors, while preserving inference-time
efficiency?* (proposal §IV)

The question presupposes three things. Each was tested.

| Presupposition | Verdict | Evidence |
|---|---|---|
| Quantization (W4A4) causes measurable rendering degradation | ❌ **refuted** | §2.1 |
| Quantization reduces inference cost | ❌ **refuted as implemented** | §2.2 |
| Geometric disagreement predicts downstream damage | ❌ **refuted** | §2.3 |

None of this makes the topic dead — §5 sets out what the evidence does support — but the
paper cannot claim the original three.

---

## 2. Primary negative results

### 2.1 ✅ No measurable W4A4 rendering degradation on the object

40 CO3D scenes, 6 input views, 9 held-out, paired per scene. `ongoing_logs.md` #020,
artifact `sweep_results/main40_both_masks.json`.

| | content mask | foreground mask |
|---|---:|---:|
| mean Full − W4A4 | +0.2038 dB | +0.1115 dB |
| t | 2.743 | 1.385 |
| p | **0.0092** | **0.174** |
| 95% CI | [+0.0535, +0.3542] | **[−0.0514, +0.2744]** |
| scenes favouring Full | 31/40 | **20/40** |
| MDE at 80% power | 0.2082 dB | 0.2256 dB |

**Sign agreement between masks: 25/40 = 62.5%. Fifteen scenes reverse sign.**

The content-mask column reproduces the previously published result exactly, so the only
change is the evaluation region. Reported as an effect-size bound, per this project's own
rule (gate D1): *the Full-over-W4A4 advantage on the object is +0.1115 dB, 95% CI
[−0.05, +0.27] dB; effects below 0.226 dB are unresolvable at n = 40.* Detecting 0.1 dB at
80% power would need ~173 scenes.

### 2.2 ✅ Quantization as implemented is not cheaper

Measured over all 1330 six-view groups per variant, from the prediction metadata
(`predictions/index.csv`).

| variant | mean inference | mean peak GPU memory |
|---|---:|---:|
| full | 2.19 s | 9,893 MiB |
| w4a4 | 2.05 s | **15,384 MiB** |

6% faster, **55% more memory**. This is *simulated* quantization: weights are quantized and
de-quantized around full-precision matmuls, so the model holds original weights and
quantization parameters simultaneously and runs no low-bit kernels. Any efficiency claim
must state which of the two it refers to. A deployed low-bit engine was not measured.

### 2.3 ✅ Disagreement does not predict downstream damage

Spearman ρ between four geometric-disagreement metrics and downstream damage, object mask,
40 scenes: 0.127, −0.012, 0.229, 0.161 (`DIAGNOSTIC_RESULTS.md`, gate D2). This matters
because the proposal's supervision signal (§VI.A.3) is exactly that disagreement.

---

## 3. The measurement corrections — a contribution in their own right

Four defects were found in the evaluation chain. Two reversed a published verdict. This
section is arguably the most transferable part of the work: each is a trap any
VGGT→3DGS evaluation can fall into.

### 3.1 ✅ The metric measured the wrong region

The "content mask" is the whole non-letterbox image. On CO3D the object occupies a mean
**15.3%** of it, and CO3D provides ground-truth depth for **~0.1%** of background pixels —
so ~85% of the evaluated area is unreconstructable by any arm.

Consequences, both measured:
- The A3 oracle gate (target ≥20 dB) is **unreachable by construction**: with a
  pixel-perfect object and the best possible flat background the ceiling is **16.81 dB**,
  and only 2/8 scenes could individually reach 20 (#002, `metric_ceiling.json`).
- **Independent confirmation:** the `maskedbg` sweep arm renders a black background by
  construction; the ceiling model predicted 6.26 dB, measured **6.13 dB** (#009).
- Fixing the region **flipped gate C1 from FAIL to PASS** (#014): full − random went
  +0.481 → **+1.457 dB**, and "initialization is inert" went 4/8 → **1/8** scenes.

### 3.2 ✅ A units bug that turned a failing gate into a passing one

`test_a5_loo_sim3` computed its leave-one-out residual in the **predicted** world frame but
divided by `scene_radius`, the CO3D **ground-truth** point-cloud radius. The two frames
differ by the Sim(3) scale, measured at **0.0716**.

| | full | w4a4 | vs 0.02 target |
|---|---:|---:|---|
| as published (frame-inconsistent) | 0.0066 | 0.0106 | PASS |
| frame-consistent, 40 scenes | **0.0813** | **0.1380** | **FAIL by 4–7×** |

The legacy path reproduces 0.0066 exactly, confirming only frame handling changed. Robust to
the choice of radius: 0.1385 against the point-cloud radius, 0.0473 against camera spread —
both fail (#017, #018).

### 3.3 ✅ Auto-exposure dominates the effect being measured

CO3D is captured with auto-exposure and the evaluation applied no correction. A single
per-image per-channel gain+bias fit recovers **+3.542 dB** on the oracle arm (14.828 →
18.370), which is ~30× the effect under study (#022).

On the 8-scene subset the correction **flips the sign** of the Full−W4A4 delta: +0.187 raw
→ **−0.103** corrected (both non-significant, n=8).

### 3.4 ✅ 30k iterations buys nothing

| source | 7k → 30k |
|---|---|
| `convergence_7k_vs_30k.json`, 40 scenes | Full **+0.095 dB**, Quant **+0.065 dB**, SSIM **−0.016** |
| oracle sweep, 8 scenes (#003) | 14.023 → 14.061 dB (**+0.038**) |

4.3× the compute for <0.1 dB, and SSIM degrades. All pipelines now default to 7k
(`common.TRAIN_ITERATIONS`). **All arms in a comparison must render at the same iteration**,
or bit-width is confounded with training length.

---

## 4. The harness itself is a bottleneck — quantified

Established by an oracle control: ground-truth geometry and ground-truth cameras, no VGGT.

| finding | measurement | entry |
|---|---|---|
| Oracle held-out score | **14.828 dB** foreground (target 19.430, derived) | #004 |
| Oracle self-fit at its own training views | **47.86 dB** (target 30) — renderer/adapter are sound | #015 |
| Oracle is beaten by copying the nearest training photo | 15.414 dB vs 14.828 — loses by **0.586 dB**, wins on 3/8 scenes | #022 |
| Alignment cost, injected into the oracle | **2.215 dB = 38.2%** of the oracle-minus-full gap | #021 |
| Geometry + other | 3.585 dB = 61.8% | #021 |
| GT camera path is exact | **100%** self on-silhouette, 96.4% cross-view | #016 |
| GT→predicted map is **not a similarity** | best in-sample Sim(3) residual **0.0485** = 2.4× the target | #023 |
| Rotation-constrained Sim(3) | 0.0813 → 0.0772 (only 5–10% better) | #023 |

**Nine 3DGS configurations were swept** and none reached the target (#003, #005–#013):
best was 12 views + depth regularisation at **15.589 dB**, against the plain oracle's 14.828
and a 19.430 target. Ruled out: early stopping, densification schedule, opacity reset,
background initialisation (100k seeded points buy +0.11 dB), and view count (6→12 gains
+1.12 dB, 12→24 gains **−0.14 dB**).

**A6 caveat, stated so the gate is not over-read:** self-fit is circular with respect to
camera correctness — the model was *trained* on those cameras, so it fits them whether or not
they are right. A6 failing would indict the camera path; A6 passing does not exonerate it.

---

## 5. What the evidence does support — the viable paper

### 5.1 ✅ A measurement instrument that works where rendering does not

Rendering cannot separate full from W4A4 (p = 0.174). Geometry error against full-precision
VGGT separates arms cleanly, at ~2 s/scene versus ~1 h/arm for 3DGS
(`ablation_compare.py`, #026, artifact `w4a4_ablation_subset.json`):

| arm | cam rot (°) | cam centre | spread ratio | depth rel | point err |
|---|---:|---:|---:|---:|---:|
| w4a4 (full pipeline) | 1.4230 | 0.02435 | 1.005 | 0.01826 | 0.05142 |
| w4a4_rtn (all components off) | 26.2667 | 0.13944 | 0.856 | 0.08095 | 0.68261 |
| w2a4 | 113.9351 | 0.68659 | **0.010** | 0.13184 | 87.35657 |

Metrics are computed after a Sim(3) fit on the six input camera centres, because every
variant predicts in its own arbitrary world frame. `cam_spread_ratio` is the collapse
detector.

### 5.2 ✅ Quantization degradation is not uniform — it hits the pose head first

W2A4, locally calibrated, 209.6 min, converged normally per block (#025):

| | full | w2a4 |
|---|---:|---:|
| camera-centre spread | 0.44–0.94 | **0.004–0.011** (~1% of full) |
| max pairwise camera rotation | 73.7° | **1.0–2.6°** |
| leave-one-out camera error | 0.079 | **5.30** scene radii |
| median depth | 0.77–1.03 | 0.85 |
| world-point spread | — | ~60% of full |

At 2-bit the model places all six cameras at essentially one pose — it has stopped estimating
pose — while depth degrades gracefully. **The failure is specific to the pose head.**

Calibration error by depth is the quantitative signature: W4A4 rises ~50× across the network
(8.6e-6 → 5.0e-4); W2A4 rises ~440× (7.8e-4 → 0.339).

### 5.3 ✅ QuantVGGT's machinery does substantial work at 4 bits

Removing all four components (smooth, rotation, learned weight clipping, learned activation
clipping) costs **18.5× on camera rotation**, 5.7× on camera centre, 4.4× on depth, 13.3× on
world points (§5.1). Per-component attribution ⏳ pending.

### 5.4 The reframing the evidence points to

Not *"confidence weighting recovers quantization loss"* — at 4 bits there is no measurable
loss, and at 2 bits the model is collapsed rather than degraded. Instead:

> Quantization damage in feed-forward geometry models is **structured**: it concentrates in
> the pose head and appears abruptly between 4 and 2 bits. Rendering metrics on this data are
> too blunt to see it; direct geometry metrics see it cheaply. The ablation ladder supplies a
> controlled range of damage between "intact" and "collapsed" in which a per-pixel confidence
> predictor can be tested where its premise actually holds.

---

## 6. Reproducibility

| artifact | location |
|---|---|
| Experiment log, 27 entries | `ongoing_logs.md` (append-only; DECISION STATE at top) |
| Diagnostic gates + verdicts | `results/downstream_3dgs/DIAGNOSTIC_RESULTS.md` |
| All experiment JSONs | `/var/tmp/luli38se/quantsplat/oracle_sweep/results/` (27 files) |
| VGGT predictions, 2668 files, 64 GB | `predictions/` + `index.csv` + its own `README.md` |
| Scene definition | `frozen_dataset_manifest.json`, SHA `1ce2f3f8d43cc17f61d84545b82f132a3b64d49d5acf70ff0fe0ac6aa9968e06` |
| Setup guide for a new machine | `HANDOFF.md` |

Scripts abort if the manifest hash does not match, and the inference stage aborts if a
variant's preprocessed inputs do not hash identically to the `full` arm's — that guard fired
during this work and caught a real preprocessing mismatch (crop vs pad mode).

**Two implementation bugs found and fixed, both recorded where they occurred:** `render.py`
inheriting the training `depths` config and demanding `depth_params.json` in the held-out
source (#010); and a prep script copying a sweep directory along with its trained
checkpoints, so the resume-skip silently re-rendered the parent config instead of training —
caught by a 1.44 min runtime and bit-identical output, now guarded by a minimum-runtime check
(#013).

---

## 7. Threats to validity

1. **Single dataset.** Everything is CO3D-v2, 40 object-centric scenes, 6 input views. LLFF
   was attempted and is ⛔ **blocked on data access** (pycolmap installed and verified; the
   8-scene benchmark unobtainable from this network; of two demo scenes only one
   reconstructs) — #024.
2. **Calibration provenance is mixed.** The shipped W4A4 came from the authors' HuggingFace
   release; W2A4 and the ablation arms are calibrated locally on 42 samples. `a44_local`
   exists specifically so ablations compare against a locally calibrated full pipeline.
3. **Simulated quantization only** (§2.2). No low-bit kernels were run.
4. **Underpowered for small effects.** MDE 0.226 dB at n=40.
5. **8-scene subset** for all GPU-training experiments except #018, #020, #023.
6. **Irreducible alignment error.** The GT→predicted map is not a similarity (§3.2, §4), so
   ~2.2 dB of every downstream comparison is alignment, not geometry.
7. **The oracle loses to a nearest-photo baseline** (§4). Any downstream quality claim on
   this harness must survive that.

---

## 8. Open and pending

| item | status |
|---|---|
| Per-component ablation attribution (4 arms) | ⏳ calibrating, ETA ~23:00 2026-09-09 |
| Downstream 3DGS for all ablation arms | ⏳ queued behind calibration |
| W3A3 — the untested middle between 4-bit (no effect) and 2-bit (collapse) | not started |
| Exposure-corrected 40-scene D1 | not started; #022 makes it the highest-value next run |
| LLFF cross-dataset validation | ⛔ blocked on benchmark data |
| D3 harsher quantization | partially unblocked — W2A4 artifact now exists; W3A3 still ⛔ |
