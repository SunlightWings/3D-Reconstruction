# Ongoing Experiment Log

Append-only. Every experiment run gets an entry, in the format defined in [CLAUDE.md](CLAUDE.md). Past entries are never edited or overwritten; a superseded entry is corrected by a later entry that says so. `DECISION STATE` is the single exception: it is rewritten in place after each new entry.

---

## DECISION STATE

_Last rewritten: 2026-09-13 07:18 CEST, after entry #053._

### Gates that currently pass / fail

| Gate | Status | Evidence |
|---|---|---|
| A1 self-fit (full/w4a4) | **PASS** | 50.92 / 50.77 dB (#001). |
| A2 noise floor | **REFERENCE** | 12.94 dB content, 40 scenes (#001). |
| A3 oracle control | **FAIL** | Oracle 14.828 dB foreground; loses to copying the nearest training photo (15.414 dB) (#022). |
| A4 reprojection overlay | **FAIL** | 8.70 % on-silhouette (#023). |
| A5 LOO Sim(3) | **FAIL** | 0.0813 full / 0.1380 w4a4 vs 0.02 target (#018). |
| A6 oracle self-fit | **PASS** | 47.86 dB (#015). |
| C1 random-init control | **PASS** | Foreground margin +1.457 dB (#014). |
| **D1 full vs W4A4, n = 40** | **FAIL on PSNR/SSIM; marginal on LPIPS** | Raw PSNR +0.1359 dB, p 0.099; raw LPIPS −0.0108, p 0.00797, Bonferroni ≈ 0.048 (#029). |
| **D3 harsher quantization (W3A3)** | **PASS — the first arm that degrades rendering** | full − w3a3 +0.7828 dB, p 0.00629, 7/8; w4a4 − w3a3 +0.4544 dB, p 0.0263 (#051). n = 8, selected subset. |
| Ablation attribution | **PASS** | Rotation is the only strongly load-bearing component: no_rot 2.05× camera error, no_lac 1.28×, others ~1× (#027, #028). |
| Rendering tracks geometry | **PARTIAL — weaker than first reported** | Six arms, exact permutation p: LPIPS ρ +0.943 p 0.0167, SSIM −0.886 p 0.0333, PSNR −0.829 p 0.0583; none < 0.05 without the bowl scene (#032, correcting #029). |
| W4A4 depth vs GT | **PASS (real, tiny)** | Object absrel 0.00542 → 0.00645, worse in 1214/1330 groups (#033). |
| W4A4 point-error source | **MEASURED** | ≈ thirds: depth shape 0.32, per-view scale drift ≈ 0.31, cameras ≈ 0.36 (#035). |
| Error predictable from own output | **PASS all pixels / FAIL on object / FAIL for cameras** | AUROC 0.888 scene-held-out, 0.887 category-held-out; object 0.589–0.668; camera self-consistency ρ −0.032, p 0.248 (#036). |
| Exposure correction mechanism | **PASS** | Recovers +0.4534 / +0.5246 dB extra for no_rot / rtn (p ≤ 0.0004) (#037). Raw metrics are cited everywhere as a result. |
| Power for the W4A4 PSNR null | **UNDERPOWERED** | ~112 scenes needed at +0.136 dB (have 40) (#038). |
| Cameras vs points (swap, n = 40) | **PASS — cameras** | Bad cameras +0.9721 dB p 5.1e-06; bad points +0.0780 p 0.61; cameras = 85 % of joint loss (#039). |
| **Cameras vs points on W3A3 (swap, n = 8)** | **PASS — cameras; robust on LPIPS only** | Cameras LPIPS +0.0330, 8/8, Wilcoxon p 0.0078; PSNR +0.7001, t p 0.032 but Wilcoxon p 0.109 (5/8). Points null: +0.1355 dB p 0.571. Cameras = 91 % of joint PSNR loss (#053). |
| Pose vs focal (camsplit, n = 40) | **PASS — both, pose larger** | Pose only +0.6943 dB / LPIPS +0.0612; focal only +0.4876 / +0.0206; pose alone = 92 % of the joint LPIPS loss (#043). |
| Geometry accuracy vs render quality | **DECOUPLED** | Bad-point model: worse final 3D (p ≤ 1.5e-09) but renders like full; bad-camera model: 3D like full, renders ~1 dB worse (#040). |
| Damage tolerance (synthetic sweep, n = 12) | **MEASURED** | Rotation: no effect ≤ 2°, LPIPS from 4°, PSNR from 8°. Points: no PSNR effect up to 1.0 scene radius (#042). |
| **W3A3 geometry decomposition** | **MEASURED — camera-borne** | Pose 2.69° mean (straddles the 2–4° threshold, 8/8 worse than w4a4, p 0.0078); focal 0.1015 (74 % of rtn's damaging level); points 0.1007 scene radii, 10× inside the harmless regime (#050). |
| Pose confidence from order-ensembling | **PASS** | Spread vs w4a4 pose error ρ +0.880, p 7e-14; costs 4× inference (#045). |
| Evaluate on farther views to gain sensitivity | **FAIL — sign reversed** | Damage shrinks with view distance (swapK ρ −0.425, p 7e-08) (#044). |
| Ensemble size scaling | **SATURATES** | 2/4/8/16 orderings: rtn 6.07° → 5.42/5.13/4.68/4.37°; never crosses 4° (#049). |
| Ensemble correction downstream | **FAIL** | Pose-corrected +0.1070 dB of a 0.9721 dB gap, p 0.38; with focal, +0.0417, p 0.67 (#048). |
| Ensembling corrects intrinsics | **FAIL** | Focal error 0.17445 → 0.17700, p 0.485: a systematic bias, not order-noise (#048). |
| Order-ensembling as pose corrector | **PASS (geometry only)** | w4a4_rtn 6.07° → 4.87° (p 5.3e-07); full-precision control only +6.4 % (#047). |
| Classical SfM as pose corrector | **FAIL / BLOCKED** | COLMAP reconstructs nothing on 28/40 scenes; usable on 7, worse in 6 of those (#046). |
| **Confidence ceiling (point pruning), arm 1** | **FAIL** | Perfect true-error pruning of `rtn` recovers −12 % of a 0.93 dB gap (#030). |
| **Confidence ceiling (point pruning), arm 2** | **FAIL** | Perfect pruning of `w3a3` recovers −1.4 % of a 0.78 dB gap; −0.0904 dB vs its matched random control, p 0.635 (#052). |
| LLFF cross-dataset arm | **BLOCKED (data only)** | #024. |

### Conclusion currently licensed by the evidence

**Quantization does damage novel-view rendering — at 3 bits, not at 4.** W4A4 does not measurably degrade
PSNR or SSIM at n = 40 and only marginally LPIPS (#029). W3A3 does: +0.7828 dB, p 0.00629, 7/8 scenes, and it
separates from W4A4 itself (+0.4544 dB, p 0.0263) (#051). This is the result the project lacked until
2026-09-12, and it is the premise the proposal needs.

**The damage is camera-borne, and this is now predictive rather than retrospective.** #050 decomposed W3A3's
geometry *before* any rendering was measured — points 10× inside the harmless regime, pose straddling the
2–4° tolerance, focal at 74 % of a known-damaging level — and predicted camera-borne damage; #051 confirmed
it. Across four arms the account is consistent: `w4a4` (1.42°, inside tolerance → no damage), `w3a3` (2.69°,
straddling → damage), `no_rot` and `rtn` (far outside → large damage). The mechanism is #040: 3DGS's
appearance optimisation absorbs misplaced points, while camera poses are held fixed during training and are
never corrected. **#053 measured this directly on W3A3:** swapping in its points costs nothing (+0.1355 dB, p 0.571),
swapping in its cameras costs +0.7001 dB and worsens LPIPS on 8/8 scenes (Wilcoxon p 0.0078) — robust on LPIPS, not yet on
PSNR (Wilcoxon p 0.109).

**The proposal's intervention is closed by measurement, on two independent arms.** A *perfect* point-level
confidence signal — true error against full precision, unavailable to any deployed system — recovers −12 %
of `rtn`'s gap (#030) and −1.4 % of `w3a3`'s (#052), in both cases indistinguishable from dropping an equal
number of random points. Because the oracle is the ceiling, **no learned predictor at any accuracy can help
via point pruning**, including the AUROC 0.888 predictor of #036. The detector half of the proposal stands;
the intervention half does not.

**Camera-level correction also fails downstream, for a stated reason.** The order-spread detector predicts
pose error at ρ 0.880 (#045) and order-averaging genuinely reduces it, 6.07° → 4.87° (#047) — but the
correction recovers +0.1070 dB of a 0.9721 dB gap (p 0.38, #048), because averaging saturates at 4.37° even
at 16 orderings (#049), still inside the damaged regime, and focal error is a systematic bias every ordering
shares (p 0.485, #048). COLMAP fails outright on six wide-baseline views (#046).

**Held-out rendering on this benchmark is decoupled from 3D accuracy and governed by camera correctness**
(#039, #040) — the central methodological finding for the report.

**Retracted by later entries:** "exposure-corrected metrics are the column to trust" (#027/#028 → #029);
the LPIPS clip hypothesis (#028 → #029); "rare catastrophic W4A4 failures" as a quantization effect (one
symmetric bowl scene, #031); the dose-response's significance on all three metrics (#029 → #032); "W4A4
depth error lives in the background" (#034 → #035); "3DGS moves bad points to the right place"
(conversation → #040); the times stated in #034/#035 (corrected in #036); **the teammate-reported W3A3 gap
of +1.1239 dB (44 % above the like-for-like value rebuilt locally, #051) — must not be cited.**

### Single experiment that would most change the picture

**W3A3 inference on all 40 scenes** (their `.npz` only, ~1 GB; no 3DGS needed on their side). Every W3A3
claim in #050–#052 rests on n = 8, and those 8 were *selected in #002 to span both signs of the
full-minus-quant delta* — a biased subset for effect-size estimation. At n = 40 the minimum detectable
effect is 0.23 dB (#038) and the selection objection disappears. Nothing else available would strengthen the
report as much for as little compute.

Second, now done (#053): the W3A3 swap probe confirmed cameras over points. Its PSNR camera effect fails the
rank test at n = 8 (Wilcoxon p 0.109), which is one more reason the 40-scene predictions above are the
priority.

### Open blockers

- **W3A3 QS checkpoints were never transferred** (#050). Only `.npz` predictions and trained PLYs arrived,
  so W3A3 cannot be re-inferred locally. This blocks extending the order-ensembling detector (#045) and
  corrector (#047) to W3A3 — the case #049 named as the one where that machinery could plausibly pay off.
- **W3A3 preprocessing is unverified** (#050). `input_semantic_sha256` in the local `w3a3_meta.json` files is
  copied from our own `full_meta.json`, not independently confirmed; their meta files were not sent.
  `w3a3_compare_contract.json` (40 scenes, sha256 of every train and held-out GT PNG) was produced for them
  to check against and has not been returned. Their code has also drifted — their results JSON predates #037.
- **W3A3 n = 8 and the subset is selected, not random** — see "Single experiment" above.
- **Soft confidence *weighting* (as opposed to hard pruning) is untested.** It is the only remaining reading
  of the proposal title and would need the GraphDECO trainer modified to accept per-point weights. Given
  #052 shows no separation from random at all, it is not expected to change the conclusion; not built.
- **A3 / harness ceiling** — the oracle still loses to a nearest-photo baseline; every null inherits this.
- **LLFF** — data access only (#024).
- **Nothing is committed.** `code/quantization/` and 17 other paths are untracked or modified at `f0c8e32`.

---

## Entries

### #001 — Backfill of pre-existing results (no new run)

- **Date/time:** 2026-09-07 (backfill; underlying runs completed 2026-08-23 13:48:28 +02:00 for the 40-scene main run, 2026-09-07 18:07:07 for the diagnostics)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; commit dated 2026-08-30, i.e. this is the tree state at backfill time, not necessarily the tree the runs executed on)
- **Command executed:** **UNRECORDED.** The runs predate this log and no command line was persisted; `results/downstream_3dgs/diagnostics/state.json` stores results only. The entry points were `code/downstream_3dgs/run_full_dataset_downstream.py` and `code/downstream_3dgs/diagnostics/run_all_diagnostics.py`, but their exact invocations and flags are not recoverable and are **not** reconstructed here.

**Config (as documented, not as observed at run time):**

| Field | Value |
|---|---|
| Init source | `full` and `w4a4` (main run); `oracle` and `random` additionally in A3/C1 |
| Iterations | 30000 default; C3 swept 500 / 1000 / 3000 / 7000 / 15000 / 30000 |
| Densify | On by default; C2 re-trained with `--densify_until_iter 0` |
| Input views | `--stride 4`, `--heldout-views 9` (defaults) |
| Mask type | `content` (rectangular) for the main aggregate; `foreground` (B1), `boundary_band` (B2), object/background split (B3) |
| Background handling | **UNRECORDED** — `gs_render_depth.py` selects white vs black via `dataset.white_background`; the value used for these runs was not persisted |
| Depth regularisation | **UNRECORDED** — no depth-regularisation flag is exposed by the runners; E2 renders depth for evaluation only |

**Scenes used:** 40 CO3D scenes for the main run and A1/A2/A4/A5/B1-B3/D1/D2/E1-E3. The 8-scene GPU subset for A3/C1/C2/C3: apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130.

**Results (40-scene main run, from `aggregate_metrics.json`):**

| Metric | Full | W4A4 | Full - W4A4 |
|---|---:|---:|---:|
| Mean PSNR (dB) | 9.9156 | 9.7117 | +0.2038 |
| Mean SSIM | 0.31485 | 0.31198 | +0.00287 |
| Mean LPIPS | 0.66245 | 0.67396 | -0.01151 (Full better) |

Median Full-minus-Quant PSNR +0.2322 dB; scenes favouring Full: 31/40 PSNR, 20/40 SSIM, 28/40 LPIPS. Status COMPLETE, 40/40 scenes.

Geometry metrics (40 scenes): Chamfer F-score @ tau 0.039 (Full) / 0.034 (W4A4). Floater mass 0.9768 / 0.9802. Mean Gaussian count 393,108 / 390,908.

**Per-scene PSNR/SSIM/LPIPS:** not reproduced here — they exist in `results/downstream_3dgs/aggregate_metrics.csv` and `results/per_scene.csv`. Per-scene oracle/random PSNR for the 8-scene subset is tabulated in `DIAGNOSTIC_RESULTS.md` (A3, C1). Copying them into this entry was avoided rather than risk transcription error; future entries must include them inline as the rule requires.

**PASS/FAIL:**

- Target: **A3 oracle >= 20 dB.** Measured **14.06 dB** mean (min 12.04 dB). **FAIL.**
- Target: **absolute PSNR above the A2 noise floor of 12.94 dB.** Measured **9.9156 dB** (Full). **FAIL.**
- Target: **A1 self-fit >= 15 dB.** Measured **50.92 dB** (Full) / 50.77 dB (W4A4), 0/40 below threshold. **PASS.**
- Target: **A5 LOO camera error well under 0.02 scene radii.** Measured **0.0066** (Full) / **0.0106** (W4A4). **PASS.**
- Target: **A4 camera-centre residual under the same 0.02.** Measured mean **0.0253**, max **0.1096**. **FAIL.**
- Target: **C1 ordering random < quant <= full with a visible margin.** Measured 13.48 / 13.87 / 13.96 dB, 4/8 scenes with random within 0.5 dB of full. **FAIL.**
- Target: **D2 non-zero disagreement-to-damage correlation.** Measured |rho| <= 0.229. **FAIL.**
- **D3: BLOCKED** — no W3A3/W2A4 calibration artifact exists; see open blockers.

**Interpretation:** Rules out the metric-masking and background-dilution explanations (Gate B) and rules in a harness-level failure — GT geometry itself cannot clear the noise floor (A3), so the +0.204 dB Full-vs-W4A4 delta cannot yet be read as a quality difference.

<!--
ENTRY TEMPLATE — copy for each new run.

### #NNN — <short title>

- **Date/time:** YYYY-MM-DD HH:MM TZ
- **Git commit:** <full hash> (branch <name>; dirty? yes/no)
- **Command executed:**
  ```
  <exact command line, copy-pasteable>
  ```

**Config:**

| Field | Value |
|---|---|
| Init source | full / w4a4 / oracle / random |
| Iterations | |
| Densify | |
| Input views | stride N, held-out M |
| Mask type | content / foreground / boundary_band / background |
| Background handling | |
| Depth regularisation | on / off |

**Scenes used:** <list, or named set + count>

**Results:** mean and per-scene PSNR / SSIM / LPIPS; plus Chamfer F-score, floater mass, Gaussian count where relevant.

| scene | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| | | | |
| **mean** | | | |

**PASS/FAIL:** Target: <target written out>. Measured: <value>. PASS/FAIL.

**Interpretation:** <one line: what this rules in or rules out>

Then rewrite DECISION STATE above.
-->

---

### #002 — Evaluation-region vs available-geometry audit (analysis, no training)

- **Date/time:** 2026-09-07 20:52 CEST
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 code/downstream_3dgs/oracle_sweep/ceiling.py
  ```
  (plus an inline `python3 -c` depth-validity probe using the same `common`/`dis` helpers; the probe is reproduced by the `depth_valid_in_BG` computation in that script's sibling analysis)

**Config:** No training. Pure measurement over the existing prepared scene assets.

| Field | Value |
|---|---|
| Init source | n/a (no 3DGS run) |
| Iterations | n/a |
| Densify | n/a |
| Input views | 6 (`input_frames` from the frozen manifest), 9 held-out |
| Mask type | `content` (the A3 metric region) vs `foreground`, compared |
| Background handling | n/a |
| Depth regularisation | n/a |

**Scenes used:** all 8 of the GPU subset for the ceiling computation; the depth-validity probe was run on the first 4 (apple, ball, bowl, broccoli).

**Results:**

Foreground fraction of the evaluated (content-mask) region, and the PSNR ceiling if the foreground were reconstructed *pixel-perfectly* while the background is filled with the best possible constant colour:

| scene | fg fraction of content mask | ceiling, bg = optimal constant | ceiling, bg = black |
|---|---:|---:|---:|
| apple/110_13051_23361 | 0.186 | 17.69 | 6.88 |
| ball/123_14363_28981 | 0.163 | 15.36 | 5.91 |
| bowl/70_5792_13401 | 0.232 | 17.34 | 7.50 |
| broccoli/412_56288_108844 | 0.080 | 12.44 | 4.84 |
| hydrant/167_18184_34441 | 0.131 | 13.20 | 5.82 |
| remote/350_36761_68623 | 0.106 | 20.74 | 7.22 |
| teddybear/187_20215_38541 | 0.192 | 22.84 | 5.61 |
| toaster/372_41229_82130 | 0.136 | 14.87 | 6.27 |
| **mean** | **0.153** | **16.81** | **6.26** |

GT depth validity, as a fraction of pixels (input frame 1 of each scene):

| scene | depth valid within content mask | depth valid within background |
|---|---:|---:|
| apple/110_13051_23361 | 0.039 | 0.001 |
| ball/123_14363_28981 | 0.107 | 0.001 |
| bowl/70_5792_13401 | 0.063 | 0.001 |
| broccoli/412_56288_108844 | 0.033 | 0.000 |

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Measured ceiling with a *pixel-perfect* foreground and an optimally chosen flat background: **16.81 dB mean**, with only 2/8 scenes (remote 20.74, teddybear 22.84) individually able to clear 20 dB. **FAIL — the target is unreachable under this metric by any 3DGS configuration that does not reconstruct the background.**

**Interpretation:** Rules out every configuration change as a route to 20 dB under the A3 content-mask metric, because 84.7% of the evaluated pixels are background for which CO3D supplies essentially no ground-truth depth (0.1% valid), so the oracle cannot be initialised there at all.

---

### #003 — Baseline oracle, iteration sweep (factor 2)

- **Date/time:** 2026-09-07 21:00-21:09 CEST (8.9 min wall for the 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (dirty, as above)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag base_itersweep --iters 500 1000 3000 7000
  ```
  which per scene invokes, from `/home/utn/luli38se/cv/gaussian-splatting`:
  ```
  /opt/saltstack/salt/bin/python3.10 train.py -s <oracle>/train -m <model> --iterations 7000 \
      --data_device cpu --resolution 1 --quiet --disable_viewer \
      --save_iterations 500 1000 3000 7000 --test_iterations -1
  /opt/saltstack/salt/bin/python3.10 render.py -m <model> -s <oracle>/heldout --iteration <it> --skip_test --quiet
  ```

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` (GT cameras + GT-unprojected foreground points; reused the existing `PREPARED.ok` sources) |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in a single training run per scene) |
| Densify | default (`densify_from_iter` 500, `densify_until_iter` 15000, interval 100, grad threshold 0.0002) |
| Input views | 6 input, 9 held-out |
| Mask type | `content` (identical to A3) |
| Background handling | GS default black; training images are unmasked full frames |
| Depth regularisation | off (`depth_l1_weight` unused, no `-d` depths folder) |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR / SSIM):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 15.75 | 15.36 | 14.41 | 15.80 |
| ball/123_14363_28981 | 10.73 | 10.44 | 13.32 | 13.40 |
| bowl/70_5792_13401 | 12.05 | 13.08 | 14.79 | 15.26 |
| broccoli/412_56288_108844 | 9.29 | 10.87 | 11.67 | 12.46 |
| hydrant/167_18184_34441 | 11.28 | 10.46 | 10.93 | 11.75 |
| remote/350_36761_68623 | 10.97 | 13.81 | 13.62 | 14.60 |
| teddybear/187_20215_38541 | 16.25 | 14.29 | 15.83 | 16.48 |
| toaster/372_41229_82130 | 10.71 | 11.07 | 11.04 | 12.41 |
| **mean PSNR** | **12.129** | **12.423** | **13.202** | **14.023** |
| **mean SSIM** | 0.4502 | 0.4308 | 0.3759 | 0.3768 |
| min PSNR | 9.29 | 10.44 | 10.93 | 11.75 |

Reference point at 30000 iterations, same config, from A3: **14.061 dB**. LPIPS not computed for this sweep (the A3 metric path used for comparability computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR.** Best measured: **14.023 dB at 7000 iterations** (14.061 dB at 30000). **FAIL**, by ~6 dB.

**Interpretation:** Rules out early stopping as the fix — mean PSNR rises monotonically with iterations (12.13 -> 14.02 -> 14.06 at 30k) rather than peaking early and decaying, so the sparse-view failure here is not the kind of overfitting that fewer iterations repairs; the falling SSIM (0.450 -> 0.377) alongside rising PSNR does indicate densification is trading structural fidelity for photometric mean.

---

### #004 — Harness validation: A3 reproduction and foreground-mask re-scoring (analysis, no training)

- **Date/time:** 2026-09-07 20:59 CEST
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (dirty, as above)
- **Command executed:** inline `python3 -c` re-scoring the existing A3 oracle renders under both masks, using `common.evaluate_renders` and the `fgmask` helper now preserved in `code/downstream_3dgs/oracle_sweep/reeval.py`. Equivalent re-runnable form:
  ```
  python3 code/downstream_3dgs/oracle_sweep/reeval.py <tag> foreground 500 1000 3000 7000
  ```
  (the inline form read `DIAG_HEAVY_ROOT/<scene>/heldout/oracle/renders` rather than a sweep tag)

**Config:** no training; re-scored the already-existing A3 oracle renders at 30000 iterations. All other config fields as in the A3 row of #001.

**Scenes used:** the 8-scene GPU subset.

**Results:**

| scene | content-mask PSNR | content SSIM | foreground-mask PSNR | foreground SSIM |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 15.45 | 0.3738 | 17.33 | 0.4698 |
| ball/123_14363_28981 | 14.05 | 0.3265 | 14.91 | 0.3304 |
| bowl/70_5792_13401 | 14.76 | 0.3800 | 17.45 | 0.2773 |
| broccoli/412_56288_108844 | 12.04 | 0.4224 | 15.09 | 0.2731 |
| hydrant/167_18184_34441 | 12.83 | 0.3016 | 12.90 | 0.3027 |
| remote/350_36761_68623 | 14.47 | 0.3251 | 11.79 | 0.2638 |
| teddybear/187_20215_38541 | 15.83 | 0.3593 | 15.68 | 0.3554 |
| toaster/372_41229_82130 | 13.06 | 0.2977 | 13.47 | 0.2977 |
| **mean** | **14.061** | **0.3483** | **14.828** | **0.3213** |

**PASS/FAIL:**
- Target: **reproduce A3's published 14.06 dB mean** to confirm this sweep's metric is comparable. Measured **14.061 dB**, per-scene identical to the DIAGNOSTIC_RESULTS.md A3 table. **PASS.**
- Target: **oracle >= 20 dB** under a foreground-only mask (the region where GT geometry actually exists). Measured **14.828 dB**. **FAIL.**

**Interpretation:** Rules out "the metric is merely diluted by unreconstructable background" as a complete explanation — restricting the metric to the object, where the oracle actually has GT points, buys only +0.77 dB and still misses 20 dB by 5.2 dB, so the object itself is genuinely badly reconstructed; and it rules in this sweep harness as metric-identical to A3.

---

### #005 — Densification disabled (factor 1)

- **Date/time:** 2026-09-07 CEST (this run took 6.1 min wall for 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag densify_off --iters 500 1000 3000 7000 --source oracle --extra --densify_until_iter 0
  ```
  Per scene this runs GraphDECO `train.py` with `--iterations 7000 --data_device cpu --resolution 1 --quiet --disable_viewer --save_iterations 500 1000 3000 7000 --test_iterations -1 --densify_until_iter 0`, then `render.py --skip_test` per checkpoint.

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in one training run per scene) |
| Densify | `--densify_until_iter 0` (no adaptive density control at all) |
| Input views | 6 input, 9 held-out |
| Mask type | `content` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR, per scene and mean):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 15.75 | 15.27 | 14.58 | 14.13 |
| ball/123_14363_28981 | 10.78 | 11.35 | 12.77 | 13.15 |
| bowl/70_5792_13401 | 12.10 | 11.44 | 12.24 | 13.47 |
| broccoli/412_56288_108844 | 9.28 | 9.25 | 10.63 | 11.73 |
| hydrant/167_18184_34441 | 11.28 | 11.95 | 11.84 | 11.83 |
| remote/350_36761_68623 | 10.97 | 13.45 | 13.44 | 13.54 |
| teddybear/187_20215_38541 | 16.24 | 16.12 | 16.17 | 16.39 |
| toaster/372_41229_82130 | 10.73 | 10.14 | 11.83 | 12.39 |
| **mean PSNR** | **12.140** | **12.371** | **12.938** | **13.329** |
| **mean SSIM** | 0.4502 | 0.4463 | 0.4391 | 0.4297 |
| min PSNR | 9.28 | 9.25 | 10.63 | 11.73 |

LPIPS not computed (the A3-comparable metric path computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Best measured for this config: **13.329 dB at 7000 iterations**. **FAIL**, short by 6.67 dB.

**Interpretation:** Rules out adaptive density control as the cause of the gap: with densification fully off the mean is 13.329 dB at 7000 vs the baseline 14.023 dB, i.e. removing densification makes the photometric score worse, not better, while raising SSIM (0.4297 vs 0.3768) -- the floaters densification adds are buying PSNR and costing structure.


---

### #006 — Densification stopped at 3000 (factor 1)

- **Date/time:** 2026-09-07 CEST (this run took 9.1 min wall for 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag densify_3000 --iters 500 1000 3000 7000 --source oracle --extra --densify_until_iter 3000
  ```
  Per scene this runs GraphDECO `train.py` with `--iterations 7000 --data_device cpu --resolution 1 --quiet --disable_viewer --save_iterations 500 1000 3000 7000 --test_iterations -1 --densify_until_iter 3000`, then `render.py --skip_test` per checkpoint.

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in one training run per scene) |
| Densify | `--densify_until_iter 3000` (default is 15000) |
| Input views | 6 input, 9 held-out |
| Mask type | `content` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR, per scene and mean):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 15.75 | 14.94 | 14.58 | 14.91 |
| ball/123_14363_28981 | 10.75 | 11.70 | 13.70 | 13.94 |
| bowl/70_5792_13401 | 12.10 | 12.79 | 14.51 | 14.70 |
| broccoli/412_56288_108844 | 9.30 | 10.25 | 10.92 | 11.05 |
| hydrant/167_18184_34441 | 11.28 | 11.43 | 11.97 | 13.09 |
| remote/350_36761_68623 | 10.97 | 12.19 | 12.98 | 12.91 |
| teddybear/187_20215_38541 | 16.25 | 14.00 | 15.82 | 15.81 |
| toaster/372_41229_82130 | 10.71 | 8.56 | 11.14 | 11.00 |
| **mean PSNR** | **12.139** | **11.982** | **13.202** | **13.427** |
| **mean SSIM** | 0.4503 | 0.4231 | 0.3605 | 0.3507 |
| min PSNR | 9.30 | 8.56 | 10.92 | 11.00 |

LPIPS not computed (the A3-comparable metric path computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Best measured for this config: **13.427 dB at 7000 iterations**. **FAIL**, short by 6.57 dB.

**Interpretation:** Rules out an intermediate densification cutoff as the fix: 13.427 dB at 7000 vs the baseline 14.023 dB with the default 15000 cutoff. Its 3000-iteration value (13.202 dB) is bit-identical to the baseline's, as it must be since the two configs only diverge after iteration 3000 -- a useful determinism check on the harness.


---

### #007 — Opacity reset disabled (factor 3)

- **Date/time:** 2026-09-07 CEST (this run took 10.0 min wall for 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag no_opacity_reset --iters 500 1000 3000 7000 --source oracle --extra --opacity_reset_interval 100000
  ```
  Per scene this runs GraphDECO `train.py` with `--iterations 7000 --data_device cpu --resolution 1 --quiet --disable_viewer --save_iterations 500 1000 3000 7000 --test_iterations -1 --opacity_reset_interval 100000`, then `render.py --skip_test` per checkpoint.

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in one training run per scene) |
| Densify | default densification; `--opacity_reset_interval 100000` (> iterations, so never fires) |
| Input views | 6 input, 9 held-out |
| Mask type | `content` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR, per scene and mean):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 15.75 | 15.19 | 14.37 | 14.60 |
| ball/123_14363_28981 | 10.76 | 11.45 | 13.21 | 13.26 |
| bowl/70_5792_13401 | 12.05 | 11.39 | 14.42 | 14.49 |
| broccoli/412_56288_108844 | 9.29 | 9.08 | 10.65 | 10.66 |
| hydrant/167_18184_34441 | 11.28 | 11.27 | 11.90 | 13.31 |
| remote/350_36761_68623 | 10.97 | 11.77 | 14.09 | 14.10 |
| teddybear/187_20215_38541 | 16.24 | 14.44 | 15.31 | 15.25 |
| toaster/372_41229_82130 | 10.71 | 11.70 | 12.93 | 12.95 |
| **mean PSNR** | **12.131** | **12.036** | **13.361** | **13.577** |
| **mean SSIM** | 0.4502 | 0.4288 | 0.3685 | 0.3588 |
| min PSNR | 9.29 | 9.08 | 10.65 | 10.66 |

LPIPS not computed (the A3-comparable metric path computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Best measured for this config: **13.577 dB at 7000 iterations**. **FAIL**, short by 6.42 dB.

**Interpretation:** Rules out opacity reset as the cause: disabling it gives 13.577 dB at 7000 vs the baseline 14.023 dB, so the periodic opacity cull is not what is destroying the reconstruction -- removing it is mildly harmful, not helpful.


---

### #008 — 12 input views instead of 6 (factor 4)

- **Date/time:** 2026-09-07 CEST (this run took 12.2 min wall for 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag views12 --iters 500 1000 3000 7000 --source views12
  ```
  Per scene this runs GraphDECO `train.py` with `--iterations 7000 --data_device cpu --resolution 1 --quiet --disable_viewer --save_iterations 500 1000 3000 7000 --test_iterations -1`, then `render.py --skip_test` per checkpoint.

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` rebuilt on 12 views |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in one training run per scene) |
| Densify | default |
| Input views | 12 input (evenly spaced, held-out frames excluded), 9 held-out |
| Mask type | `content` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR, per scene and mean):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 16.22 | 15.80 | 15.41 | 15.94 |
| ball/123_14363_28981 | 11.18 | 11.28 | 13.91 | 14.45 |
| bowl/70_5792_13401 | 12.21 | 13.29 | 14.31 | 15.93 |
| broccoli/412_56288_108844 | 9.42 | 11.25 | 12.38 | 13.35 |
| hydrant/167_18184_34441 | 10.13 | 11.72 | 12.38 | 13.29 |
| remote/350_36761_68623 | 8.62 | 11.41 | 15.24 | 15.44 |
| teddybear/187_20215_38541 | 15.70 | 16.71 | 16.96 | 18.52 |
| toaster/372_41229_82130 | 11.45 | 10.54 | 13.73 | 14.21 |
| **mean PSNR** | **11.867** | **12.748** | **14.289** | **15.141** |
| **mean SSIM** | 0.4478 | 0.4394 | 0.4270 | 0.4221 |
| min PSNR | 8.62 | 10.54 | 12.38 | 13.29 |

LPIPS not computed (the A3-comparable metric path computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Best measured for this config: **15.141 dB at 7000 iterations**. **FAIL**, short by 4.86 dB.

**Interpretation:** Rules IN input-view count as the one factor that actually helps: 15.141 dB at 7000 vs the baseline 14.023 dB (+1.118 dB), with SSIM also up (0.4221 vs 0.3768) rather than traded away. This is consistent with the entry #002 finding that the metric is dominated by background the 6-view oracle cannot initialise -- extra views add real background coverage. Still 4.86 dB short of target.


---

### #009 — Foreground-masked training images, constant background (factor 5)

- **Date/time:** 2026-09-07 CEST (this run took 4.8 min wall for 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag maskedbg --iters 500 1000 3000 7000 --source maskedbg
  ```
  Per scene this runs GraphDECO `train.py` with `--iterations 7000 --data_device cpu --resolution 1 --quiet --disable_viewer --save_iterations 500 1000 3000 7000 --test_iterations -1`, then `render.py --skip_test` per checkpoint.

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in one training run per scene) |
| Densify | default |
| Input views | 6 input, 9 held-out |
| Mask type | `content` (also re-scored under `foreground`) |
| Background handling | training images have background replaced by constant black; held-out GT unchanged |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR, per scene and mean):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 6.84 | 6.83 | 6.82 | 6.84 |
| ball/123_14363_28981 | 5.76 | 5.77 | 5.74 | 5.75 |
| bowl/70_5792_13401 | 7.33 | 7.33 | 7.37 | 7.37 |
| broccoli/412_56288_108844 | 4.85 | 4.84 | 4.84 | 4.84 |
| hydrant/167_18184_34441 | 5.79 | 5.77 | 5.75 | 5.74 |
| remote/350_36761_68623 | 6.90 | 6.85 | 6.80 | 6.79 |
| teddybear/187_20215_38541 | 5.56 | 5.56 | 5.57 | 5.57 |
| toaster/372_41229_82130 | 6.11 | 6.10 | 6.09 | 6.11 |
| **mean PSNR** | **6.142** | **6.129** | **6.122** | **6.126** |
| **mean SSIM** | 0.0547 | 0.0509 | 0.0494 | 0.0511 |
| min PSNR | 4.85 | 4.84 | 4.84 | 4.84 |

LPIPS not computed (the A3-comparable metric path computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Best measured for this config: **6.142 dB at 500 iterations**. **FAIL**, short by 13.86 dB.

**Interpretation:** Rules out foreground-masked training: under the A3 content-mask metric it collapses to 6.126 dB, and even re-scored on the foreground alone (the only region it models) it reaches 14.342 dB, still below the plain oracle's 14.828 dB foreground score from entry #004 -- so removing background supervision does not improve the object either.


**Secondary metric for this config (foreground-only mask, since the model deliberately does not represent the background):**

| iterations | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| mean foreground PSNR | 13.544 | 13.737 | 14.115 | **14.342** |
| mean foreground SSIM | 0.2442 | 0.2292 | 0.2238 | 0.2315 |
| min foreground PSNR | 8.33 | 8.26 | 8.14 | 8.61 |

Command: `python3 code/downstream_3dgs/oracle_sweep/reeval.py maskedbg foreground 500 1000 3000 7000`.
Reference: plain oracle at 30000 scores 14.828 dB under this same foreground mask (entry #004). Target under this
secondary metric is still 20 dB: **FAIL**, short by 5.66 dB.

The content-mask collapse to 6.126 dB is itself a confirmation of entry #002's model, which predicted a 6.26 dB
ceiling for a perfectly-reconstructed foreground on a black background.

---

### #011 — GT foreground points + random background shell (extra factor, not a pure oracle)

- **Date/time:** 2026-09-07 CEST (this run took 10.5 min wall for 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag bgshell --iters 500 1000 3000 7000 --source bgshell
  ```
  Per scene this runs GraphDECO `train.py` with `--iterations 7000 --data_device cpu --resolution 1 --quiet --disable_viewer --save_iterations 500 1000 3000 7000 --test_iterations -1`, then `render.py --skip_test` per checkpoint.

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` GT foreground points **plus 100,000 random background points** per scene (radii 1.2-8.0 x scene radius, grey). NOT a pure GT-geometry oracle: the background points are a random prior, not measurements. |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in one training run per scene) |
| Densify | default |
| Input views | 6 input, 9 held-out |
| Mask type | `content` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR, per scene and mean):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 15.74 | 15.35 | 15.42 | 15.83 |
| ball/123_14363_28981 | 10.74 | 11.22 | 13.19 | 13.75 |
| bowl/70_5792_13401 | 12.05 | 12.58 | 12.93 | 14.66 |
| broccoli/412_56288_108844 | 9.25 | 10.39 | 12.03 | 12.31 |
| hydrant/167_18184_34441 | 11.28 | 11.82 | 11.54 | 12.96 |
| remote/350_36761_68623 | 10.97 | 13.07 | 14.21 | 14.18 |
| teddybear/187_20215_38541 | 16.24 | 16.08 | 14.87 | 16.23 |
| toaster/372_41229_82130 | 10.73 | 10.24 | 12.72 | 13.17 |
| **mean PSNR** | **12.126** | **12.593** | **13.365** | **14.136** |
| **mean SSIM** | 0.4501 | 0.4264 | 0.3746 | 0.3783 |
| min PSNR | 9.25 | 10.24 | 11.54 | 12.31 |

LPIPS not computed (the A3-comparable metric path computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Best measured for this config: **14.136 dB at 7000 iterations**. **FAIL**, short by 5.86 dB.

**Interpretation:** Rules out missing background INITIALISATION as the bottleneck: adding 100,000 random background points per scene gives 14.136 dB at 7000 vs the baseline 14.023 dB, a +0.113 dB change that is negligible against the 5.9 dB gap to target. Combined with entry #002 (no GT background depth exists) this shifts the diagnosis: the background is not merely un-initialised, it is not reconstructable from 6 views at all, so seeding geometry there does not help.


---

### #012 — 24 input views (follow-up to factor 4)

- **Date/time:** 2026-09-07 CEST (this run took 13.0 min wall for 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag views24 --iters 500 1000 3000 7000 --source views24
  ```
  Per scene this runs GraphDECO `train.py` with `--iterations 7000 --data_device cpu --resolution 1 --quiet --disable_viewer --save_iterations 500 1000 3000 7000 --test_iterations -1`, then `render.py --skip_test` per checkpoint.

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` rebuilt on 24 views |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in one training run per scene) |
| Densify | default |
| Input views | 24 input (evenly spaced, held-out frames excluded), 9 held-out |
| Mask type | `content` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR, per scene and mean):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 16.29 | 16.04 | 16.79 | 16.98 |
| ball/123_14363_28981 | 12.58 | 12.19 | 13.73 | 13.90 |
| bowl/70_5792_13401 | 11.98 | 13.89 | 15.63 | 15.81 |
| broccoli/412_56288_108844 | 9.04 | 9.94 | 12.70 | 12.94 |
| hydrant/167_18184_34441 | 10.79 | 11.72 | 12.78 | 12.30 |
| remote/350_36761_68623 | 10.51 | 13.80 | 16.09 | 15.55 |
| teddybear/187_20215_38541 | 16.59 | 16.02 | 17.97 | 18.55 |
| toaster/372_41229_82130 | 11.34 | 10.81 | 13.57 | 13.98 |
| **mean PSNR** | **12.392** | **13.051** | **14.909** | **15.001** |
| **mean SSIM** | 0.4535 | 0.4532 | 0.4412 | 0.4229 |
| min PSNR | 9.04 | 9.94 | 12.70 | 12.30 |

LPIPS not computed (the A3-comparable metric path computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Best measured for this config: **15.001 dB at 7000 iterations**. **FAIL**, short by 5.00 dB.

**Interpretation:** Rules out 'just add more views' as a route to target: doubling 12 -> 24 views yields 15.001 dB at 7000, which is 0.140 dB BELOW views12's 15.141 dB, so the view-count gain has saturated. The per-doubling gain at 3000 iterations falls from +1.087 dB (6->12) to +0.620 dB (12->24), and at 7000 it is negative. The 6-view result was therefore not primarily view-starved.


---

### #010 — Depth regularisation against GT depth (factor 6)

- **Date/time:** 2026-09-07 CEST (this run took 8.7 min wall for 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag depthreg --iters 500 1000 3000 7000 --source depthreg --extra -d depths --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01
  ```
  Per scene this runs GraphDECO `train.py` with `--iterations 7000 --data_device cpu --resolution 1 --quiet --disable_viewer --save_iterations 500 1000 3000 7000 --test_iterations -1 -d depths --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01`, then `render.py --skip_test` per checkpoint.

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` + GT inverse-depth maps |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in one training run per scene) |
| Densify | default |
| Input views | 6 input, 9 held-out |
| Mask type | `content` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | ON: `-d depths --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01`; GT depth covers 2.6-6.2% of pixels, foreground only |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR, per scene and mean):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 16.00 | 15.29 | 14.98 | 14.90 |
| ball/123_14363_28981 | 10.47 | 10.95 | 14.91 | 14.82 |
| bowl/70_5792_13401 | 11.96 | 12.40 | 14.80 | 15.35 |
| broccoli/412_56288_108844 | 9.41 | 10.67 | 12.24 | 13.03 |
| hydrant/167_18184_34441 | 10.71 | 11.69 | 12.50 | 13.78 |
| remote/350_36761_68623 | 11.12 | 13.05 | 13.89 | 14.69 |
| teddybear/187_20215_38541 | 17.77 | 15.28 | 17.36 | 17.76 |
| toaster/372_41229_82130 | 10.93 | 10.37 | 12.88 | 13.95 |
| **mean PSNR** | **12.297** | **12.461** | **14.195** | **14.786** |
| **mean SSIM** | 0.4489 | 0.4207 | 0.3499 | 0.3561 |
| min PSNR | 9.41 | 10.37 | 12.24 | 13.03 |

LPIPS not computed (the A3-comparable metric path computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Best measured for this config: **14.786 dB at 7000 iterations**. **FAIL**, short by 5.21 dB.

**Interpretation:** Rules IN depth regularisation as a real but insufficient help: 14.786 dB at 7000 vs the baseline 14.023 dB (+0.763 dB), achieved even though GT depth covers only 2.6-6.2% of pixels and lies almost entirely on the object. Second-best single factor after view count, still 5.21 dB short. NOTE: the first attempt at this config failed (rc=1) because render.py inherits the training cfg_args 'depths' setting and then demands depth_params.json in the heldout source; fixed by passing -d "" at render time (sweep.py), and re-run.


---

### #013 — COMBINED: 12 input views + GT depth regularisation (best config found)

- **Date/time:** 2026-09-07 CEST (this run took 12.0 min wall for 8 scenes)
- **Git commit:** `e411cb8786172a49571be68fc3ba3f7d399ebd96` (branch `disagreement-dataset-validation`; working tree dirty — adds `CLAUDE.md`, `ongoing_logs.md`, `code/downstream_3dgs/oracle_sweep/`)
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sweep.py --tag views12_depth --iters 500 1000 3000 7000 --source views12_depth --extra -d depths --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01
  ```
  Per scene this runs GraphDECO `train.py` with `--iterations 7000 --data_device cpu --resolution 1 --quiet --disable_viewer --save_iterations 500 1000 3000 7000 --test_iterations -1 -d depths --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01`, then `render.py --skip_test` per checkpoint.

**Config:**

| Field | Value |
|---|---|
| Init source | `oracle` rebuilt on 12 views, plus GT inverse-depth maps for those 12 frames |
| Iterations | 500 / 1000 / 3000 / 7000 (checkpointed in one training run per scene) |
| Densify | default |
| Input views | 12 input (evenly spaced, held-out excluded), 9 held-out |
| Mask type | `content` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | ON: `-d depths --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01` |

**Scenes used:** the 8-scene GPU subset (apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401, broccoli/412_56288_108844, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_38541, toaster/372_41229_82130).

**Results (content-mask PSNR, per scene and mean):**

| scene | 500 | 1000 | 3000 | 7000 |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 16.05 | 15.00 | 15.53 | 15.92 |
| ball/123_14363_28981 | 10.76 | 11.40 | 14.50 | 14.65 |
| bowl/70_5792_13401 | 11.91 | 13.28 | 14.37 | 15.16 |
| broccoli/412_56288_108844 | 9.48 | 10.35 | 12.86 | 13.77 |
| hydrant/167_18184_34441 | 10.49 | 11.86 | 13.24 | 14.23 |
| remote/350_36761_68623 | 9.12 | 12.72 | 15.65 | 16.38 |
| teddybear/187_20215_38541 | 15.93 | 15.51 | 18.95 | 19.96 |
| toaster/372_41229_82130 | 11.63 | 9.90 | 13.71 | 14.66 |
| **mean PSNR** | **11.920** | **12.502** | **14.851** | **15.589** |
| **mean SSIM** | 0.4472 | 0.4377 | 0.4126 | 0.4109 |
| min PSNR | 9.12 | 9.90 | 12.86 | 13.77 |

LPIPS not computed (the A3-comparable metric path computes PSNR/SSIM only).

**PASS/FAIL:** Target: **oracle >= 20 dB mean content-mask PSNR on the 8-scene subset.** Best measured for this config: **15.589 dB at 7000 iterations**. **FAIL**, short by 4.41 dB.

**Interpretation:** Rules out the whole search space explored here: stacking the only two factors that helped gives 15.589 dB at 7000, still 4.411 dB short of target. The two gains are sub-additive (+1.118 for views alone, +0.763 for depth alone, +1.566 combined against an additive prediction of +1.881), which is what a shared ceiling looks like rather than two independent deficiencies. NOTE: the first attempt at this config was INVALID and is not reported as a result -- `prep_views12_depth.py` copied the `views12` directory, which doubles as its own GraphDECO model directory, so the trained checkpoints came along and sweep.py's resume-skip re-rendered them instead of training (1.44 min runtime, output bit-identical to views12). The prep script now strips model artifacts after copying and the config was retrained from a clean source (12.02 min).


---

## Artifacts for entries #002-#013

Every number in entries #002-#013 is reproducible from these files (all under
`/var/tmp/luli38se/quantsplat/oracle_sweep/results/`):

| file | contents |
|---|---|
| `metric_ceiling.json` | entry #002: per-scene foreground fraction and the PSNR ceiling under a pixel-perfect foreground |
| `a3_oracle_30000_both_masks.json` | entry #004: the A3 oracle at 30000 re-scored under both content and foreground masks |
| `base_itersweep.json` | entry #003 |
| `densify_off.json`, `densify_3000.json`, `no_opacity_reset.json` | entries #005, #006, #007 |
| `views12.json`, `views24.json` | entries #008, #012 |
| `maskedbg.json` + `maskedbg_it*_foreground.json` | entry #009, both metrics |
| `depthreg.json` | entry #010 |
| `bgshell.json` | entry #011 |
| `views12_depth.json` | entry #013 |

Code is versioned in the repo under `code/downstream_3dgs/oracle_sweep/`:
`sweep.py` (train/render/evaluate driver, metric identical to A3), `prep.py` (view-count and masked-background
sources), `prep_depth.py`, `prep_bgshell.py`, `prep_views12_depth.py`, `reeval.py` (re-score existing renders under a
different mask), `ceiling.py`, `emit_entry.py` (generates these log entries directly from the result JSON so numbers
are never retyped), and `run_configs*.sh`.

**Two harness bugs were found and fixed during this work, both recorded in the entries that hit them:**
1. `render.py` inherits the training `cfg_args` `depths` setting and then requires `depth_params.json` in the held-out
   source, killing any depth-regularised config at render time. Fixed by passing `-d ""` at render time (#010).
2. A prep script that copies a sweep directory also copies its trained checkpoints, because a sweep tag's directory is
   both its source and its GraphDECO model directory; `sweep.py`'s resume-skip then silently re-renders the old model
   instead of training. Caught because the run finished in 1.44 min and returned output bit-identical to its parent
   config. Fixed by stripping model artifacts after the copy (#013).

---

### #014 — C1 random-init control, re-scored on the foreground mask (analysis, no training)

- **Date/time:** 2026-09-07 23:40 CEST
- **Git commit:** `f8b1f7d` (branch `disagreement-dataset-validation`, clean tree at time of run)
- **Command executed:**
  ```
  python3 code/downstream_3dgs/oracle_sweep/reeval_c1.py
  ```

**Config:** No training and no rendering. Re-scores the C1 renders that already exist on disk
(`DIAG_HEAVY_ROOT/<scene>/heldout/{c1_random,c1_w4a4,c1_full,oracle}/renders`). All training config is
whatever C1 and A3 originally used (30000 iterations, default densification, 6 input views, GT-frame cameras);
the only thing varied here is the evaluation mask.

| Field | Value |
|---|---|
| Init source | all four arms compared: `random`, `w4a4`, `full`, `oracle` |
| Iterations | 30000 (as originally trained) |
| Densify | default (as originally trained) |
| Input views | 6 input, 9 held-out |
| Mask type | **`content` vs `foreground`, compared** |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset.

**Results:**

| arm | content PSNR | content SSIM | foreground PSNR | foreground SSIM |
|---|---:|---:|---:|---:|
| random | 13.481 | 0.3563 | 13.459 | 0.3107 |
| w4a4 | 13.873 | 0.3803 | 14.814 | 0.3389 |
| full | 13.962 | 0.3764 | **14.916** | 0.3354 |
| oracle | **14.061** | 0.3483 | 14.828 | 0.3213 |

| quantity | content mask | foreground mask |
|---|---:|---:|
| ordering (best first) | oracle > full > w4a4 > random | full > oracle > w4a4 > random |
| full - random | +0.481 dB | **+1.457 dB** |
| scenes where random is within 0.5 dB of full | 4 / 8 | **1 / 8** |

**PASS/FAIL:** Target (C1's own, as written in DIAGNOSTIC_RESULTS.md): **ordering random < quant <= full, with a
visible margin**; the stated failure mode is "if random ~= full, initialization is inert in this harness."

- On the **content** mask: random 13.481 < w4a4 13.873 < full 13.962, margin +0.481 dB, init inert on 4/8 scenes. **FAIL** (this was the published verdict).
- On the **foreground** mask: random 13.459 < w4a4 14.814 <= full 14.916, margin **+1.457 dB**, init inert on **1/8** scenes. **PASS.**

**Interpretation:** Rules out "initialization is inert in this harness" — that conclusion was an artifact of scoring on a
mask that is ~85% background, where no initialization of any kind can help (cf. entries #002 and #011); on the region
where the init prior actually places geometry, the geometry prior matters and the C1 ordering holds with a 3x larger
margin.

**Flag for follow-up, not resolved here:** on the foreground mask the `oracle` arm (14.828 dB) sits *below* `full`
(14.916 dB), i.e. GT geometry is beaten by VGGT geometry. The two arms share GT-frame cameras and differ only in the
initial point cloud, and the oracle cloud is far sparser (2,566-6,356 points from GT depth valid only on the object)
than VGGT's dense prediction. That is a plausible explanation but it is **not** measured, and it should be, because a
GT-geometry control that loses to the method it is meant to bound is a gate worth distrusting.

---

### #015 — A6 oracle self-fit (new diagnostic, no training)

- **Date/time:** 2026-09-08 08:47 CEST
- **Git commit:** `f8b1f7d` + working-tree changes adding A6, the foreground mask, and the A3 gate redefinition
- **Command executed:**
  ```
  python3 -c "import sys; sys.path.insert(0,'code/downstream_3dgs/diagnostics'); import gate_a; gate_a.test_a6_oracle_selffit()"
  ```
  (registered as test `A6` in `gate_a.TESTS`, so it also runs via `run_all_diagnostics.py`)

**Config:** No training. Renders the EXISTING oracle models at their own 6 training views.

| Field | Value |
|---|---|
| Init source | `oracle` (GT cameras + GT-unprojected foreground points) |
| Iterations | 30000 (models as trained by A3) |
| Densify | default (as trained by A3) |
| Input views | 6 input; rendered at those same 6 training cameras |
| Mask type | both `foreground` (primary) and `content` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset.

**Results (self-fit PSNR at the model's own training views):**

| scene | content | foreground |
|---|---:|---:|
| apple/110_13051_23361 | 50.17 | 50.34 |
| ball/123_14363_28981 | 47.05 | 48.74 |
| bowl/70_5792_13401 | 47.81 | 46.63 |
| broccoli/412_56288_108844 | 49.60 | 46.28 |
| hydrant/167_18184_34441 | 46.95 | 48.52 |
| remote/350_36761_68623 | 48.33 | 47.73 |
| teddybear/187_20215_38541 | 46.93 | 46.07 |
| toaster/372_41229_82130 | 45.58 | 48.57 |
| **mean** | **47.801** | **47.859** |
| min (foreground) | | 46.07 |

**PASS/FAIL:** Target: **>= 30 dB self-fit.** Measured **47.859 dB** mean on the foreground
mask (47.801 dB content), minimum across scenes 46.07 dB, scenes below target: 0/8. **PASS.**

**Interpretation:** Rules out a renderer/adapter defect and a capacity limit in the oracle arm -- the oracle reproduces
its own training views at ~48 dB, so the ~14.8 dB held-out foreground score is a genuine failure to generalise to new
viewpoints, not a broken pipeline.

**Important asymmetry, stated so this gate is not over-read:** A6 is circular with respect to camera correctness. The
model was *trained* on these exact cameras, so it will fit them well whether or not they are geometrically right; a
systematic camera error would simply be baked into the Gaussians. A6 FAILING would have indicted the GT camera path,
but A6 PASSING does not exonerate it. The camera question stays open and belongs to A4/A5 (task items 5 and 6).

---

### #016 — A4 root cause: is the GT camera path self-consistent? (analysis, no training)

- **Date/time:** 2026-09-08 09:05 CEST
- **Git commit:** `1088b5a`
- **Command executed:**
  ```
  python3 code/downstream_3dgs/oracle_sweep/check_gt_projection.py
  ```

**Config:** No training, no rendering, no VGGT. Pure projection arithmetic on GT data.

| Field | Value |
|---|---|
| Init source | GT depth unprojection only (`gate_a._oracle_decode_frame`) |
| Iterations | n/a |
| Densify | n/a |
| Input views | 6 input frames per scene |
| Mask type | `foreground` (the CO3D object silhouette) |
| Background handling | n/a |
| Depth regularisation | n/a |

**Scenes used:** the 8-scene GPU subset.

**Method:** points unprojected from frame f's own GT depth using frame f's own GT camera are projected straight back
through that same camera (`self`), and into the other input frames' GT cameras (`cross-view`). The self case is a round
trip with no alignment, no prediction and no other view involved, so its answer is known in advance: it must be ~100%.

**Results (fraction of projected points landing on the object silhouette):**

| scene | self | inside image | cross-view |
|---|---:|---:|---:|
| apple/110_13051_23361 | 100.0% | 100.0% | 92.7% |
| ball/123_14363_28981 | 100.0% | 100.0% | 92.7% |
| bowl/70_5792_13401 | 100.0% | 100.0% | 97.6% |
| broccoli/412_56288_108844 | 100.0% | 100.0% | 99.3% |
| hydrant/167_18184_34441 | 100.0% | 100.0% | 95.9% |
| remote/350_36761_68623 | 100.0% | 100.0% | 99.3% |
| teddybear/187_20215_38541 | 100.0% | 100.0% | 96.2% |
| toaster/372_41229_82130 | 100.0% | 100.0% | 97.2% |
| **mean** | **100.00%** | 100.00% | **96.36%** |

**PASS/FAIL:** Target: **>= 90% self on-silhouette** (projecting GT points through their own GT camera).
Measured **100.00%**, with 100.00% of points landing inside the image. **PASS.**

**Interpretation:** Rules out the GT camera path as the cause of A4's 9-29% on-silhouette -- the pad affine applied to
the intrinsics, the CAM_FLIP sign convention, and the NDC->pixel conversion in `co3d_to_opencv_camera` are all
self-consistent, and the GT extrinsics are mutually consistent across views at 96.4% cross-view. The
remaining candidates for A4 are therefore VGGT's predicted geometry and the Sim(3) alignment between the predicted and
GT frames, which is exactly what task item 5 tests. The 3.6% cross-view shortfall is expected from genuine
occlusion (points visible in one view are hidden in another), not from a coordinate bug.

---

### #017 — Sim(3) direction comparison, and a units bug in A5 (analysis, no training)

- **Date/time:** 2026-09-08 09:20 CEST
- **Git commit:** `1088b5a`
- **Command executed:**
  ```
  python3 code/downstream_3dgs/oracle_sweep/sim3_direction.py
  ```
  plus two inline `python3 -c` probes reproducing A5's normalisation (both reproduced by
  `loo_error()` in that module and by the table below)

**Config:** No training, no rendering. Operates on the frozen prediction NPZs and GT annotations.

| Field | Value |
|---|---|
| Init source | `full` and `w4a4` predicted geometry (frozen NPZs) |
| Iterations | n/a |
| Densify | n/a |
| Input views | 6 input frames; point stride 8 for the on-silhouette test |
| Mask type | `foreground` (object silhouette) |
| Background handling | n/a |
| Depth regularisation | n/a |

**Scenes used:** the 8-scene GPU subset.

**Result 1 — Sim(3) fitting direction makes no difference to A4.**

Comparing the current `umeyama(C_gt, C_pred)`-then-invert form against the direct
`umeyama(C_pred, C_gt)`-applied-forward form:

| variant | on-silhouette (inverse) | on-silhouette (direct) |
|---|---:|---:|
| full | 9.50% | 9.51% |
| w4a4 | 9.39% | 9.40% |

Per-scene the two agree to within 0.1 percentage point on every scene.

**Result 2 — A5 normalises a predicted-frame residual by a GT-frame radius.**

`test_a5_loo_sim3` computes `||pred_k - C_pred[k]|| / radius`, where the residual is in the PREDICTED
world frame but `radius` is `scene_radius` from per_scene.csv, i.e. the 90th-percentile radius of the CO3D
**GT** point cloud. The two frames differ by the Sim(3) scale, measured here at **0.0716** on average
(the predicted world is ~14x smaller than the GT world), so the reported figure is scaled by that factor.

Recomputing the same leave-one-out quantity consistently -- mapping the predicted camera back into the GT
frame and normalising by a GT-frame radius:

| scene | A5 as coded | consistent (GT frame) | ratio |
|---|---:|---:|---:|
| apple/110_13051_23361 | 0.0107 | 0.1441 | 13.4x |
| ball/123_14363_28981 | 0.0065 | 0.0760 | 11.7x |
| bowl/70_5792_13401 | 0.0341 | 0.4549 | 13.3x |
| broccoli/412_56288_108844 | 0.0052 | 0.0826 | 15.9x |
| hydrant/167_18184_34441 | 0.0023 | 0.0449 | 19.2x |
| remote/350_36761_68623 | 0.0042 | 0.0646 | 15.4x |
| teddybear/187_20215_38541 | 0.0044 | 0.0588 | 13.3x |
| toaster/372_41229_82130 | 0.0150 | 0.1821 | 12.2x |
| **mean** | **0.0103** | **0.1385** | **~13.4x** |

The conclusion does not depend on which GT-frame radius is chosen:

| normalisation | mean LOO error | vs 0.02 target |
|---|---:|---|
| GT point-cloud radius (what per_scene.csv holds) | 0.1385 | **FAIL** |
| GT camera-spread radius | 0.0473 | **FAIL** |

**PASS/FAIL:**
- Target: **Sim(3) direction should be chosen by whichever gives better A4/A5.** On A4 the two directions are
  indistinguishable (9.50% vs 9.51%). On A5 they are not comparable as written, because the as-coded metric is
  frame-inconsistent. **Keep the existing `inverse` form** — there is no measured reason to change it, and changing
  it would churn the alignment used by every existing result for no gain.
- Target: **A5 median LOO error well under 0.02 scene radii.** As coded: 0.0103, **PASS**. Measured consistently:
  **0.1385** (or 0.0473 against camera spread), **FAIL** by 2.4x to 7x.

**Interpretation:** Rules out the Sim(3) fitting direction as an explanation for A4's low on-silhouette, and rules IN
a units bug in A5 that converts a real FAIL into a reported PASS. This resolves the A4-vs-A5 tension flagged in the
previous DECISION STATE: A4 (frame-consistent) said the camera alignment was borderline, A5 (frame-inconsistent) said
it was comfortable, and A5 was wrong. **The camera alignment between the predicted and GT frames is genuinely poor,
which is a live candidate for the held-out generalisation failure that entry #015 localised to viewpoint change.**

**Superseded:** the A5 rows in entries #001 and its DECISION STATE ("A5 PASS", 0.0066 / 0.0106) are frame-inconsistent
and should not be cited. This entry supersedes them.

---

### #018 — A5 re-run on all 40 scenes after the frame-consistency fix

- **Date/time:** 2026-09-08 09:30 CEST
- **Git commit:** `1088b5a` + working-tree fix to `test_a5_loo_sim3`
- **Command executed:**
  ```
  python3 -c "import sys; sys.path.insert(0,'code/downstream_3dgs/diagnostics'); import gate_a; gate_a.test_a5_loo_sim3()"
  ```

**Config:** No training, no rendering. Frozen prediction NPZs plus GT annotations.

| Field | Value |
|---|---|
| Init source | `full` and `w4a4` predicted cameras |
| Iterations | n/a |
| Densify | n/a |
| Input views | 6 input cameras per scene (fit 5, predict the 6th) |
| Mask type | n/a (camera-geometry test) |
| Background handling | n/a |
| Depth regularisation | n/a |

**Scenes used:** all 40 scenes (A5 is a CPU test and runs on the full set).

**Results (median over scenes of each scene's median LOO error, in scene radii):**

| measure | full | w4a4 |
|---|---:|---:|
| **frame-consistent (fixed)** | **0.0813** | **0.1380** |
| legacy, frame-inconsistent (retained in output for traceability) | 0.0066 | — |

The legacy figure reproduces the value published in DIAGNOSTIC_RESULTS.md (0.0066 for Full) **exactly**, which
confirms the fix changes only the frame handling and nothing else about the computation.

**PASS/FAIL:** Target: **median LOO camera-centre error well under 0.02 scene radii.** Measured **0.0813** (Full)
and **0.1380** (W4A4). **FAIL**, by 4x and 7x respectively. The previously published **PASS (0.0066 / 0.0106) was an
artifact of the units bug** described in entry #017.

**Interpretation:** Rules in poor predicted-to-GT camera alignment as a real defect of this harness, on the full
40-scene set rather than the 8-scene subset — the Sim(3) that carries GT held-out cameras into each prediction's world
frame does not extrapolate to a held-out camera, so the held-out views every downstream metric is computed from are
rendered from materially wrong viewpoints. This is now the leading candidate cause of the held-out generalisation
failure that entry #015 isolated, and it applies to the `full` and `w4a4` arms but NOT to the `oracle` arm, which uses
GT cameras directly with no Sim(3).

**Supersedes:** the A5 rows in entry #001 and in every DECISION STATE written before this one.

---

### #019 — LLFF downstream pipeline: BLOCKED

- **Date/time:** 2026-09-08 09:45 CEST
- **Git commit:** `b838e3a`
- **Command executed:** environment probes only —
  ```
  which colmap; ls /usr/bin/colmap /usr/local/bin/colmap /opt/*/bin/colmap
  python3 -c "import pycolmap"
  find /var/tmp/luli38se /home/utn/luli38se -iname "*llff*"
  find /var/tmp/luli38se/vggt_official -name "sparse" -o -name "cameras.txt" -o -name "*.npy"
  ```

**Status: BLOCKED.**

**Config:** n/a — nothing was run, and no unverifiable code was written.

**Scenes available:** only `llff_fern` (20 images) and `llff_flower` (25 images) at 706x529, under
`/var/tmp/luli38se/vggt_official/examples/`. These are VGGT demo inputs, not the LLFF benchmark (8 scenes).

**Results:** none. The requested design gates running `full`/`w4a4` on a COLMAP reference arm landing in published
sparse-view range, and that reference arm cannot be constructed here:

1. No COLMAP binary on PATH, in `/usr/bin`, `/usr/local/bin` or `/opt/*/bin`; `pycolmap` not installed. The only
   COLMAP-related file present is GraphDECO's `scene/colmap_loader.py`, which reads a reconstruction and cannot
   produce one.
2. The LLFF images present carry no poses: no `poses_bounds.npy`, no `sparse/`, no `cameras.txt`/`images.txt`/
   `points3D.txt`.
3. Two demo scenes is not the benchmark, so even with poses the "published sparse-view range" comparison would not
   be meaningful.

**PASS/FAIL:** Target: **the COLMAP reference arm must land in published sparse-view PSNR range before running
full/w4a4.** The arm cannot be built, so the gate cannot be evaluated. **BLOCKED — not FAIL, and explicitly not
reported as a number.**

No pose source was substituted. Seeding the reference arm from VGGT's own predicted cameras was considered and
rejected: it would make the control depend on the method under test, producing a figure that looks like a reference
but measures nothing — the same failure mode D3 refused.

**Next step:** obtain the LLFF benchmark release with `poses_bounds.npy` (8 scenes), or install COLMAP/`pycolmap` and
run sparse reconstruction per scene; then build the source builder (3/6/9 views, every 8th frame held out, whole-image
metrics) and gate on the COLMAP arm. Recorded in `code/downstream_3dgs/llff/README.md`.

**Interpretation:** Rules nothing in or out about the method — this is an environment limitation, recorded so the
absence of an LLFF result is not later mistaken for an LLFF null result.

---

### #020 — 40-scene Full-vs-W4A4 re-scored on the foreground mask; D1 recomputed

- **Date/time:** 2026-09-08 10:05 CEST
- **Git commit:** `231d1ea`
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/reeval_main40.py
  ```

**Config:** No training, no rendering. Re-scores the existing 40-scene held-out renders.

| Field | Value |
|---|---|
| Init source | `full` and `w4a4` (the main 40-scene run's own trained models) |
| Iterations | 30000 (as originally trained) |
| Densify | default (as originally trained) |
| Input views | 6 input, 9 held-out |
| Mask type | **`content` vs `foreground`, compared** |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** all 40 scenes.

**Results:**

| metric | content mask | foreground mask |
|---|---:|---:|
| mean Full PSNR | 9.9156 | 9.0279 |
| mean W4A4 PSNR | 9.7117 | 8.9164 |
| **mean Full-minus-Quant** | **+0.2038** | **+0.1115** |
| mean SSIM delta | +0.00287 | +0.00022 |

**D1 paired statistics, recomputed on both masks:**

| quantity | content mask | foreground mask |
|---|---:|---:|
| n | 40 | 40 |
| mean delta (dB) | 0.2038 | 0.1115 |
| sd | 0.4700 | 0.5093 |
| t | 2.7427 | 1.3848 |
| p | 0.0092 | 0.1740 |
| MDE at 80% power | 0.2082 | 0.2256 |
| scenes favouring Full | 31 | 20 |
| 95% CI | [+0.0535, +0.3542] | [-0.0514, +0.2744] |

The content-mask column reproduces the published D1 exactly (mean 0.204, sd 0.470, t 2.74, p 0.0092,
CI [0.054, 0.354], MDE 0.208, 31/40 favouring Full), confirming that only the mask changed.

**Per-scene sign agreement: 25/40 = 62.5%. 15 of 40 scenes change sign between masks.**

Scenes that flip: apple/110_13051_23361, broccoli/372_41112_81867, broccoli/412_56288_108844, cake/403_53094_103680, donut/391_47032_93657, hydrant/411_56064_108483, mouse/377_43416_86289, orange/374_42196_84367, orange/385_45386_90752, plant/374_42005_84358, skateboard/245_26182_52130, skateboard/366_39266_76077, toaster/372_41229_82130, toaster/416_57389_110765, toytruck/190_20494_39385.

**PASS/FAIL:**
- Target: **does the headline Full-minus-Quant result survive the mask correction?** On the content mask the effect is
  statistically real (p = 0.0092, CI excludes zero). On the foreground mask **p = 0.174, the 95% CI
  [-0.0514, +0.2744] includes zero**, and scenes favouring Full fall to 20/40 — exactly chance. **FAIL: the result does not survive.**
- Target (D1's own rule): **any null must be stated as an effect-size bound, not a null result.** On the foreground mask
  the bound is: the Full-minus-Quant advantage is +0.1115 dB with 95% CI [-0.0514, +0.2744] dB; effects
  below the 0.2256 dB minimum detectable effect at n=40 cannot be resolved by this study.

**Interpretation:** Rules out the project's central quantitative claim as currently stated — the +0.204 dB
Full-over-W4A4 advantage is an artifact of scoring a region that is ~85% background, and on the object itself the
effect is indistinguishable from zero (p = 0.174, 20/40 scenes each way). The apple spot check was NOT isolated:
15/40 scenes reverse sign between masks.

**Supersedes** the Full-vs-W4A4 delta reported in entry #001 and in `aggregate_metrics.json` as a quality claim; those
content-mask numbers remain arithmetically correct but no longer support the conclusion drawn from them.

---

### #021 — Pose vs geometry decomposition of the oracle-minus-VGGT gap

- **Date/time:** 2026-09-08 10:25 CEST
- **Git commit:** `231d1ea`
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/pose_vs_geometry.py
  ```

**Config:** Render-only, reusing the trained oracle models. Held-out camera centres displaced by a random-direction
translation of magnitude `delta * scene_radius`, orientation unchanged; seed 20260908.

| Field | Value |
|---|---|
| Init source | `oracle` (GT cameras + GT foreground points, no Sim(3) anywhere) |
| Iterations | 30000 (models as trained by A3) |
| Densify | default (as trained) |
| Input views | 6 input, 9 held-out |
| Mask type | `foreground` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset.

**Why the literally-requested arm was not built:** a Sim(3) applied to both the points and the cameras leaves every
rendered image unchanged, and the w4a4 geometry's only link to the GT frame *is* its Sim(3). "Render the w4a4 cloud
through GT cameras directly" is therefore pixel-identical to "render it through its own Sim(3)" and can separate
nothing. (The full-arm-Sim(3) swap is well-posed -- the two predicted frames agree to a median 0.0243 of scene radius,
scale ratio ~1.00, rotation < 1.5 deg on 7/8 scenes -- but it would only compare two nearly identical estimates.)
Instead the causation is run the other way: measured alignment error is injected into the one arm that has none.

**Results (foreground PSNR vs injected camera error):**

| injected error (scene radii) | provenance | mean fg PSNR | cost vs unperturbed |
|---|---|---:|---:|
| 0.0000 | oracle as it stands | 14.828 | +0.000 |
| 0.0200 | A5's pass threshold | 13.648 | +1.180 |
| 0.0813 | A5 measured, full arm (#018) | 12.613 | +2.215 |
| 0.1380 | A5 measured, w4a4 arm (#018) | 12.045 | +2.783 |

The delta=0 row reproduces the oracle's 14.828 dB from entry #004 exactly, confirming the render path is unchanged.

**Attribution:**

| quantity | value |
|---|---:|
| oracle, no alignment error | 14.828 dB |
| full arm, 40 scenes, foreground (#020) | 9.028 dB |
| total oracle-minus-full gap | 5.800 dB |
| cost of the measured full-arm alignment error | 2.215 dB |
| **share attributable to alignment** | **38.2%** |
| **share attributable to geometry and everything else** | **61.8%** (3.585 dB) |

**PASS/FAIL:** Target: **decide whether a per-pixel confidence predictor is the right instrument**, by attributing the
gap. Measured split: alignment **38.2%**, geometry-and-other **61.8%**. **Geometry is the majority share, so a
per-pixel geometry-confidence predictor addresses the larger part -- but it cannot touch the 38.2% that is pose, and
that share is not small.**

**Interpretation:** Rules out treating the gap as purely a geometry problem: roughly two fifths of it is camera
alignment, which no per-pixel geometry confidence can fix, so a confidence predictor should be paired with (or preceded
by) the alignment fix in task item 3. It also shows the exchange rate is steep and saturating -- the first 0.02 radii of
camera error costs 1.180 dB while going from 0.0813 to 0.1380 costs only a further 0.568 dB -- so even a partial
alignment improvement should pay off, and the w4a4 arm's larger misalignment explains little of the Full-vs-W4A4
difference on its own.

---

### #022 — Exposure check: how much of the oracle's 14.83 dB is photometric, not geometric?

- **Date/time:** 2026-09-08 10:35 CEST
- **Git commit:** `231d1ea`
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/exposure_check.py
  ```

**Config:** No training, no rendering. Two probes over the existing oracle held-out renders.

| Field | Value |
|---|---|
| Init source | `oracle` |
| Iterations | 30000 (as trained) |
| Densify | default (as trained) |
| Input views | 6 input, 9 held-out |
| Mask type | `foreground` |
| Background handling | GS default black; unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset.

**Probes:** (a) score each held-out GT frame against the training GT frame whose camera centre is nearest — what a
method scores by copying the closest real photograph and modelling nothing; (b) fit one per-channel gain+bias per image
by least squares between the oracle render and the GT over the masked pixels, then re-score — whatever this recovers was
exposure/white-balance mismatch, not geometry. CO3D is captured with auto-exposure, so this is a real effect.

**Results (foreground PSNR):**

| scene | oracle | oracle + gain/bias | photometric headroom | nearest-pose GT frame |
|---|---:|---:|---:|---:|
| apple/110_13051_23361 | 17.332 | 19.727 | +2.395 | 17.824 |
| ball/123_14363_28981 | 14.907 | 17.916 | +3.009 | 15.809 |
| bowl/70_5792_13401 | 17.449 | 19.187 | +1.738 | 16.981 |
| broccoli/412_56288_108844 | 15.087 | 20.341 | +5.254 | 14.674 |
| hydrant/167_18184_34441 | 12.900 | 18.331 | +5.431 | 16.590 |
| remote/350_36761_68623 | 11.795 | 14.251 | +2.456 | 12.109 |
| teddybear/187_20215_38541 | 15.680 | 20.770 | +5.090 | 15.501 |
| toaster/372_41229_82130 | 13.474 | 16.435 | +2.962 | 13.823 |
| **mean** | **14.828** | **18.370** | **+3.542** | **15.414** |

**PASS/FAIL:**
- Target: **quantify the photometric share of the oracle's score.** A single global gain+bias per image recovers
  **+3.542 dB** (14.828 -> 18.370), so that much of the oracle's deficit was never a geometry error.
- Target: **the oracle should beat the trivial nearest-photograph baseline.** It does not: nearest-pose GT scores
  **15.414 dB** against the oracle's **14.828 dB**, i.e. the oracle is **0.586 dB WORSE** than copying the
  nearest training photograph, and it wins on only 3/8 scenes. **FAIL.**

**Interpretation:** Rules out reading the oracle's 14.83 dB as a measure of reconstruction quality at all. Two
independent reasons: 3.54 dB of it is a per-image exposure offset that any global colour correction removes, and
the whole arm is beaten by a baseline that does no reconstruction whatsoever — copying the nearest training frame.
Once exposure is normalised the oracle reaches 18.370 dB, which is finally above the nearest-photo baseline and
close to the 19.430 dB A3 target from entry #002's derivation.

**This changes the diagnosis of A3.** The oracle's apparent failure is substantially photometric, and the harness does
not currently control for auto-exposure. Either the metric should fit a per-image gain+bias before scoring (standard
practice for NVS on auto-exposure capture, and what GraphDECO's own `--train_test_exp` option exists for), or the
comparison should be restricted to exposure-stable scenes. Note the existing pipeline writes an `exposure.json` per
model but the evaluation path does not apply any exposure correction.

---

### #023 — Sim(3) with rotation constraints, and the similarity-model floor

- **Date/time:** 2026-09-08 10:50 CEST
- **Git commit:** `231d1ea`
- **Command executed:**
  ```
  python3 -u code/downstream_3dgs/oracle_sweep/sim3_rotation.py --all
  ```
  plus an inline `python3 -c` probe measuring the in-sample residual of the best similarity fit.

**Config:** No training, no rendering. Frozen prediction NPZs and GT annotations.

| Field | Value |
|---|---|
| Init source | `full` and `w4a4` predicted cameras and points |
| Iterations | n/a |
| Densify | n/a |
| Input views | 6 input cameras (fit 5, predict the 6th for LOO); point stride 8 for on-silhouette |
| Mask type | `foreground` |
| Background handling | n/a |
| Depth regularisation | n/a |

**Scenes used:** all 40 scenes.

**Estimators:** `centres` = the current `dv.umeyama` on 6 camera centres; `rotation` = A from chordal rotation
averaging of the 6 per-camera estimates `A_i = R_pred_i^T R_gt_i`, with s and b then in closed form from the centres;
`joint` = SO(3) re-projection of an equal blend of the rotation-averaged and centres-only A.

**Results (median LOO error in scene radii; on-silhouette is the mean over scenes):**

| variant | estimator | median LOO | vs centres | on-silhouette |
|---|---|---:|---:|---:|
| full | centres | 0.0813 |  | 8.70% |
| full | rotation | 0.0776 | -4.7% | 8.58% |
| full | joint | 0.0772 | -5.1% | 8.62% |
| w4a4 | centres | 0.1380 |  | 8.80% |
| w4a4 | rotation | 0.1294 | -6.2% | 8.65% |
| w4a4 | joint | 0.1243 | -10.0% | 8.70% |

The `centres` rows reproduce entry #018 exactly (0.0813 full, 0.1380 w4a4), confirming only the estimator changed.

**The reason the gain is small — the similarity model itself is the floor.** Fitting a similarity on all 6 cameras and
scoring the residual on those same 6 (best case, no extrapolation at all) still leaves:

| quantity | value |
|---|---:|
| in-sample residual of the best similarity fit, median over 40 scenes | **0.0485** |
| in-sample residual, mean | 0.0666 |
| leave-one-out error, `centres` (#018) | 0.0813 |
| A5 target | 0.0200 |

**PASS/FAIL:** Target: **adding rotation constraints should materially reduce the 0.081 / 0.138 residual.** Measured:
full 0.0813 -> **0.0772** (joint), w4a4 0.1380 -> **0.1243** (joint) — improvements of 5.1% and 10.0%.
On-silhouette is unchanged to slightly worse (8.70% -> 8.62%). **FAIL — the reduction is real but not material,
and the residual remains 4-6x the target.**

**Interpretation:** Rules out a better estimator of a similarity as the fix. The in-sample floor of 0.0485 scene radii
is already 2.4x the A5 target, and it is reached with the same six cameras used for fitting — so it is not an
extrapolation or a weak-constraint problem at all: **the GT-to-predicted map is not a similarity.** VGGT's predicted
geometry is non-rigidly distorted relative to GT, and no Sim(3), however well estimated, can absorb that. Rotation
averaging recovers only the extrapolation slice (0.0328 of the 0.0813), which is why it buys ~5%.

**What this implies for the harness:** closing the remaining alignment error requires either a non-rigid alignment
(per-camera or deformable), which the pipeline currently forbids by design (`no_per_camera_correction: True` in every
metrics.json), or accepting the alignment error as a floor and reporting downstream numbers with it stated. The
`joint` estimator is a free ~5-10% improvement and is worth adopting, but it does not change any gate verdict.

---

### #024 — LLFF unblocking attempt: tooling resolved, benchmark data still BLOCKED

- **Date/time:** 2026-09-08 11:15 CEST
- **Git commit:** `231d1ea`
- **Commands executed:**
  ```
  python3 -m pip install pycolmap          # -> pycolmap 4.2.0 installed
  python3 -m pip install gdown
  python3 code/downstream_3dgs/llff/run_colmap_sfm.py llff_fern
  python3 code/downstream_3dgs/llff/run_colmap_sfm.py llff_flower
  curl -sSI -L https://people.eecs.berkeley.edu/~bmild/nerf/nerf_llff_data.zip
  python3 -c "import gdown; gdown.download('https://drive.google.com/uc?id=16VnMcF1KJYxN9QId6TClMsZRahHNMW5g', ...)"
  curl -s https://huggingface.co/api/datasets?search=llff
  ```

**Status: PARTIALLY UNBLOCKED. The LLFF downstream arm remains BLOCKED on data.**

**Resolved — SfM tooling.** `pycolmap` 4.2.0 installs from PyPI and works: `extract_features`,
`match_exhaustive` and `incremental_mapping` are all available (CPU only, `has_cuda=False`). Verified end to end on
`llff_fern`: **20/20 images registered, 2,959 points, 2 seconds**. A COLMAP reference arm is therefore constructible
in this environment, and `code/downstream_3dgs/llff/run_colmap_sfm.py` does it.

**Not resolved — the 8-scene benchmark.** Every canonical source failed:

| source | result |
|---|---|
| `people.eecs.berkeley.edu/~bmild/nerf/nerf_llff_data.zip` | HTTP 404 (redirects, then not found) |
| Google Drive `16VnMcF1KJYxN9QId6TClMsZRahHNMW5g` | 303; `gdown` refused — Drive blocks programmatic access to this file |
| Hugging Face `nielsr` / `dsrivastavv` / `pcuenq` mirrors | HTTP 401 |
| Hugging Face dataset search for "llff" | only `srikanthrangan/llffilters` and `poptree/LLFF`; the latter is `gated: manual` (401 without an accepted licence) and is actually mipnerf360 data — it contains no fern/fortress/horns/leaves/orchids/trex |

**Additional finding on the demo scenes:** of the two VGGT demo scenes, only `llff_fern` reconstructs.
`llff_flower` (25 images) fails SfM outright — every initialisation attempt is discarded with
"no initial pair", so it yields no reconstruction at all.

**PASS/FAIL:** Target: **obtain the real 8-scene LLFF benchmark with `poses_bounds.npy`, and install COLMAP or
pycolmap.** Tooling: **PASS**. Data: **BLOCKED** — 1 usable scene (fern) versus the 8 required, and it is a demo
asset rather than the benchmark, so the "published sparse-view range" gate still cannot be evaluated.

No pose source was substituted, and VGGT poses were again not used as the reference arm.

**Next step:** the benchmark needs a human with a browser (Drive's confirm page) or an accepted Hugging Face licence;
alternatively any mirror serving `nerf_llff_data.zip` over plain HTTP. Once the images and `poses_bounds.npy` are on
disk, `run_colmap_sfm.py` already covers the reconstruction half, and only the source builder (3/6/9 views, every 8th
frame held out, whole-image metrics) remains.

**Interpretation:** Rules out tooling as the LLFF blocker and localises it entirely to dataset access — worth
recording because the previous entry (#019) attributed the block partly to missing COLMAP, which is no longer true.

---

### #025 — W2A4 calibrated locally; the pose head collapses (unblocks D3 at 2 bits)

- **Date/time:** 2026-09-08 15:08-18:36 CEST (calibration 209.6 min), evaluated 18:40-19:00
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/`
- **Commands executed:**
  ```
  python3 -u code/quantization/calibrate_w2a4.py
  python3 code/quantization/quant_loader.py w2a4
  python3 -u code/quantization/run_w2a4_inference.py --scenes subset
  ```

**Config:**

| Field | Value |
|---|---|
| Init source | `w2a4` — VGGT-1B quantized to 2-bit weights / 4-bit activations |
| Quantization | QuantVGGT quarot smooth+rotation, learned weight AND activation clipping (lwc+lac), symmetric |
| Calibration | authors' filtered CO3D set, 42 samples, 15 epochs/block, batch 1, 48 blocks (24 frame + 24 global) |
| Calibration provenance | **locally calibrated** (the shipped W4A4 is the authors' HuggingFace release) |
| Iterations / densify / views / mask / background / depth reg | n/a — no 3DGS was run; this entry is VGGT-level only |

**Scenes used:** the 8-scene GPU subset, one six-view group each (the group the downstream pipeline selects).

**Results — calibration itself converged.** 48/48 blocks, 209.6 min, artifacts written:
`qs_frame_parameters_total.pth` and `qs_global_parameters_total.pth`, 1.78 GiB each. Per-block reconstruction
error fell within every block (block 0: 0.00267 -> 0.00078). But it degrades sharply with depth:
first blocks ~0.0008, last blocks ~0.34, a ~440x rise across the network. For contrast the W4A4 calibration
running under identical settings goes 0.0000086 -> 0.0005, a ~50x rise.

**Results — the model is broken. Per-scene camera statistics:**

| scene | cam spread full | cam spread w2a4 | ratio | max pair-rot w2a4 (deg) | LOO cam err full | LOO cam err w2a4 |
|---|---:|---:|---:|---:|---:|---:|
| apple/110_13051_23361 | 0.5204 | 0.0113 | 0.022 | 2.6 | 0.1441 | 2.7158 |
| ball/123_14363_28981 | 0.9419 | 0.0082 | 0.009 | 1.5 | 0.0760 | 6.0907 |
| bowl/70_5792_13401 | 0.4640 | 0.0038 | 0.008 | 1.0 | 0.4549 | 5.5326 |
| broccoli/412_56288_108844 | 0.7118 | 0.0045 | 0.006 | 1.7 | 0.0826 | 10.2075 |
| hydrant/167_18184_34441 | 0.4375 | 0.0046 | 0.010 | 2.0 | 0.0449 | 3.2489 |
| remote/350_36761_68623 | 0.6093 | 0.0070 | 0.011 | 1.2 | 0.0646 | 4.8738 |
| teddybear/187_20215_38541 | 0.5535 | 0.0032 | 0.006 | 2.6 | 0.0588 | 5.0599 |
| toaster/372_41229_82130 | 0.8439 | 0.0077 | 0.009 | 1.9 | 0.1821 | 9.5981 |
| **mean** | 0.6353 | 0.0063 | **0.010** | 1.8 | 0.0793 | 5.2963 |

Full precision places the six input cameras across a wide orbit (max pairwise rotation 73.7 deg on apple);
W2A4 places them all within a few degrees of each other. Depth degrades far more gracefully -- median depth
0.85 vs full's 0.77-1.03, world-point spread compressed to ~60% -- so the failure is specific to the POSE head,
not global.

Inference cost, measured: 2.25 s/scene mean, 15,384 MiB peak (identical to W4A4, since both are simulated
quantization over full-precision matmuls).

**PASS/FAIL:** Target: **W2A4 should produce geometry degraded enough to measure but still usable**, so that a
confidence predictor has something to repair. Measured: camera-centre spread collapses to **1.0%** of full
precision and leave-one-out camera error rises to **5.296** scene radii against full's
**0.079**. **FAIL — the arm is collapsed, not degraded.**

**Interpretation:** Rules out W2A4 as an experimental setting for the confidence-predictor idea, and rules out
"just quantize harder" as the fix for the unresolvable W4A4 result (#020): 2-bit does not produce a larger
measurable gap, it produces a model that has stopped doing pose estimation. No 3DGS was run on this arm on
purpose -- the resulting gap would measure model collapse, not a quantization effect a per-pixel confidence
could repair.

**Also unblocks D3 partially:** DIAGNOSTIC_RESULTS.md marks D3 (harsher quantization) BLOCKED for want of a
calibrated artifact. That artifact now exists for W2A4. D3 remains BLOCKED for W3A3.

---

### #026 — W4A4 ablation study: design, instrument, and the round-to-nearest arm (IN PROGRESS)

- **Date/time:** 2026-09-09, started 05:02 CEST; calibration of the remaining arms still running
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/`
- **Commands executed:**
  ```
  python3 code/quantization/calibrate_w2a4.py --wbit 4 --abit 4 --exp-name a44_rtn \
      --no-smooth --no-rot --no-lwc --no-lac
  python3 -u code/quantization/run_w2a4_inference.py --variant w4a4_rtn --scenes subset
  python3 code/quantization/ablation_compare.py --variants w4a4 w4a4_rtn w2a4 --scenes subset
  ./code/quantization/run_w4a4_ablation.sh          # 5 calibrations, ~17.5 h
  ./code/quantization/run_ablation_full.sh subset   # stage 2, queued behind it
  ```

**What is being ablated.** QuantVGGT's W4A4 pipeline has four components, each of which changes the quantized
linear layer's parameter structure (`quarot_linear.py`):

| component | flag | what it adds |
|---|---|---|
| SmoothQuant per-channel scaling | `not_smooth` | `channel_wise_scale` |
| QuaRot Hadamard rotation | `not_rot` | `rotation_matrix` |
| learned weight clipping | `lwc` | `clip_factor_w_max/min` |
| learned activation clipping | `lac` | activation-quantizer clipping |

Arms: `a44_local` (all four on -- the baseline), then one arm per component removed, plus `a44_rtn`
(all four off). Each arm needs its OWN calibration because the learned parameters are configuration-specific;
`a44_rtn` is the exception -- with smooth/lwc/lac all off QuantVGGT performs no calibration at all, so it is
round-to-nearest weights and costs seconds.

**Why `a44_local` and not the shipped `w4a4`.** The shipped W4A4 parameters were downloaded from the authors'
HuggingFace release with unknown calibration settings. Comparing an ablation against it would confound
"component removed" with "different calibration", so every ablation is compared against a locally calibrated
full pipeline instead.

**Config:**

| Field | Value |
|---|---|
| Init source | `w4a4` variants (4-bit weights, 4-bit activations throughout) |
| Calibration | authors' 42-sample CO3D set, 15 epochs/block, 48 blocks, batch 1 |
| Iterations | 7000 for the downstream stage; all arms rendered at the SAME iteration |
| Densify | GraphDECO default |
| Input views | 6 input, 9 held-out |
| Mask type | `foreground` primary, `content` secondary (both reported) |
| Background handling | GS default black, unmasked full frames |
| Depth regularisation | off |

**Scenes used:** the 8-scene GPU subset for the geometry table; the downstream stage runs the same subset first.

**Measurement design.** Two levels, deliberately:

1. *Geometry* -- each arm's VGGT output against that scene's full-precision output, after a Sim(3) fit on the six
   input camera centres (every variant predicts in its own arbitrary world frame). Metrics: mean camera-orientation
   error, camera-centre error in scene radii, camera-spread ratio (the collapse detector that caught W2A4),
   median relative depth error, median world-point error. Cost ~2 s/scene.
2. *Downstream 3DGS* -- train and render each arm, PSNR/SSIM/LPIPS under both masks, raw and exposure-corrected,
   with paired statistics per arm pair. Cost ~1 h/arm at 40 scenes.

Geometry is the sensitive instrument: at W4A4 the full-vs-quant *rendering* difference is not resolvable
(p = 0.174, #020), so an ablation scored only downstream would measure noise. But the ablation arms are far more
degraded than `w4a4`, so a downstream gap may well be resolvable there -- and the proposal's claim is a
downstream one. Geometry says which component matters; 3DGS says whether it matters to renders.

**Results so far (geometry vs full precision, 8 scenes):**

| arm | cam rot (deg) | cam centre err | cam spread ratio | depth rel err | point err |
|---|---:|---:|---:|---:|---:|
| w4a4 | 1.4230 | 0.02435 | 1.005 | 0.01826 | 0.05142 |
| w4a4_rtn | 26.2667 | 0.13944 | 0.856 | 0.08095 | 0.68261 |
| w2a4 | 113.9351 | 0.68659 | 0.010 | 0.13184 | 87.35657 |

Ratio, round-to-nearest over full pipeline: camera rotation **18.5x** worse,
camera centre **5.7x**, depth **4.4x**,
world points **13.3x**.

Downstream Sim(3) alignment residuals for the RTN arm, measured while building its 3DGS sources, range
0.0634-0.1067 scene radii across the 8 scenes, against `w4a4`'s 0.0243 (#017) -- consistent with the geometry table.

**PASS/FAIL:** Target: **the geometry instrument must separate arms that the rendering metric cannot.** The
rendering metric cannot distinguish full from w4a4 (p = 0.174, #020); the geometry metric separates
full-pipeline W4A4 from round-to-nearest W4A4 by 18x on camera rotation and from W2A4 by a further
4.3x. **PASS.** The per-component attribution is **PENDING** -- 5 calibrations still running.

**Interpretation:** Rules in QuantVGGT's smooth+rotation+clipping machinery as doing substantial work at 4 bits
(removing all of it costs 18x on camera rotation), and rules in geometry-level measurement as an instrument
that responds where the rendering metric is inert -- the first such instrument in this project. Which individual
component carries that gain is not yet measured and MUST NOT be guessed.

**Still to come in this entry's experiment (will be logged as its own entries when measured):** the four
single-component arms, and the downstream 3DGS numbers for all arms.

---

### #027 — W4A4 ablation, five arms measured: the Hadamard rotation is the load-bearing component

- **Date/time:** 2026-09-09 23:28 CEST (stage 2 completed); calibrations ran 2026-09-09 05:02 → 22:07
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/`
- **Commands executed:**
  ```
  ./code/quantization/run_w4a4_ablation.sh            # calibrations, one arm per component
  ./code/quantization/run_stage2_then_nolac.sh        # stage 2 reordered ahead of the last arm
  #   -> code/quantization/run_w2a4_inference.py --variant <arm> --scenes subset
  #   -> code/quantization/ablation_compare.py --scenes subset --tag subset --variants ...
  #   -> code/quantization/run_w2a4_downstream.py --scenes subset --iterations 7000 --arms ...
  ```
- **Wall time:** calibration 3.5 h/arm; stage 2 82.6 min for 7 arms

**Config:**

| Field | Value |
|---|---|
| Init source | locally calibrated W4A4 variants, one arm per removed component |
| Calibration | authors' 42-sample CO3D set, 15 epochs/block, 48 blocks (24 frame + 24 global), batch 1 |
| Iterations | 7000, identical for every arm |
| Densify | GraphDECO default |
| Input views | 6 input, 9 held-out |
| Mask type | `foreground` primary, `content` secondary (both reported) |
| Background handling | GS default black, unmasked full frames |
| Depth regularisation | off |
| Exposure | both raw and gain+bias-corrected PSNR reported |

**Scenes used (8-scene GPU subset):** apple/110_13051_23361, ball/123_14363_28981, bowl/70_5792_13401,
broccoli/412_56288_1088, hydrant/167_18184_34441, remote/350_36761_68623, teddybear/187_20215_385,
toaster/372_41229_82130

**Geometry error vs full-precision VGGT (n = 8):**

| arm | cam rot (deg) | cam centre err | cam spread ratio | depth rel err | point err |
|---|---:|---:|---:|---:|---:|
| w4a4 (shipped) | 1.4230 | 0.02435 | 1.005 | 0.01826 | 0.05142 |
| w4a4_local (baseline) | 1.2050 | 0.02254 | 1.000 | 0.02198 | 0.04015 |
| w4a4_no_lwc | 1.1169 | 0.01815 | 1.001 | 0.01715 | 0.04015 |
| w4a4_no_smooth | 1.1711 | 0.02102 | 0.996 | 0.01699 | 0.04851 |
| **w4a4_no_rot** | **2.4690** | **0.04117** | 0.995 | 0.02511 | **0.08084** |
| w4a4_rtn (all off) | 26.2667 | 0.13944 | 0.856 | 0.08095 | 0.68261 |
| w2a4 (reference) | 113.9351 | 0.68659 | 0.010 | 0.13184 | 87.35657 |

Against `w4a4_local`, removing the rotation costs **2.05x** camera rotation and **2.01x** point error.
Removing `lwc` or `smooth` individually costs nothing measurable (both are at or below baseline).

**Per-scene foreground PSNR (raw, 7000 it):**

| scene | full | local | no_rot | no_smooth | no_lwc | rtn |
|---|---:|---:|---:|---:|---:|---:|
| apple/110_13051_23361 | 12.670 | 12.797 | 11.655 | 12.477 | 13.631 | 9.594 |
| ball/123_14363_28981 | 11.224 | 10.726 | 10.212 | 10.427 | 10.167 | 8.528 |
| bowl/70_5792_13401 | 10.458 | 10.202 | 9.079 | 10.063 | 10.598 | 5.684 |
| broccoli/412_56288_1088 | 9.338 | 9.223 | 8.734 | 9.159 | 9.277 | 9.102 |
| hydrant/167_18184_34441 | 8.131 | 7.815 | 7.973 | 7.994 | 8.143 | 7.207 |
| remote/350_36761_68623 | 9.656 | 9.872 | 9.289 | 9.680 | 9.690 | 9.385 |
| teddybear/187_20215_385 | 9.341 | 8.411 | 8.946 | 8.672 | 8.741 | 7.973 |
| toaster/372_41229_82130 | 8.949 | 8.330 | 7.842 | 8.663 | 9.263 | 7.067 |
| **mean** | **9.9708** | **9.6721** | **9.2164** | **9.6419** | **9.9387** | **8.0677** |

**Downstream means (n = 8):**

| arm | fg PSNR | fg exp-corr | fg SSIM | fg LPIPS | content PSNR |
|---|---:|---:|---:|---:|---:|
| full | 9.9708 | 15.7667 | 0.2403 | 0.6553 | 9.8990 |
| w4a4 | 9.6424 | 15.8439 | 0.2369 | 0.6635 | 9.8931 |
| w4a4_local | 9.6721 | 15.8306 | 0.2336 | 0.6624 | 9.8191 |
| w4a4_no_lwc | 9.9387 | 15.9986 | 0.2436 | 0.6516 | 10.0610 |
| w4a4_no_rot | 9.2164 | 15.8340 | 0.2350 | 0.6710 | 9.4923 |
| w4a4_no_smooth | 9.6419 | 15.9555 | 0.2323 | 0.6598 | 9.8254 |
| w4a4_rtn | 8.0677 | 14.1246 | 0.1942 | 0.7243 | 8.4257 |

**Paired tests, foreground PSNR (the four that matter):**

| contrast | raw mean | p | CI95 | sign | exp-corr mean | p |
|---|---:|---:|---|---|---:|---:|
| full − no_rot | +0.7545 | 0.001675 | [+0.3933, +1.1156] | 8/8 | −0.0673 | 0.7725 |
| full − no_smooth | +0.3290 | 0.01246 | [+0.0959, +0.5620] | 7/8 | −0.1888 | 0.4143 |
| full − rtn | +1.9031 | 0.01054 | [+0.6028, +3.2035] | 8/8 | +1.6421 | 0.2863 |
| full − no_lwc | +0.0321 | 0.8838 | [−0.4691, +0.5334] | 3/8 | −0.2319 | 0.2988 |
| local − no_rot | +0.4557 | 0.05915 | [−0.0231, +0.9345] | 6/8 | −0.0034 | 0.9713 |
| local − rtn | +1.6044 | 0.02349 | [+0.2893, +2.9195] | 8/8 | +1.7060 | 0.2711 |

**PASS/FAIL:** Target: **attribute the W4A4 quantization gain to individual components, with the geometry
instrument separating arms at p < 0.05 and consistent sign.** Geometry separates `no_rot` from baseline by
2.05x on camera rotation and `rtn` by 21.8x, while `no_lwc` and `no_smooth` are indistinguishable from
baseline. **PASS for geometry; PASS for raw downstream on no_rot and rtn (8/8 sign, p < 0.02); FAIL for
exposure-corrected downstream on every arm (no contrast reaches p < 0.05).**

**Interpretation.** At 4 bits the QuaRot Hadamard rotation is the only component of QuantVGGT's pipeline that
is individually load-bearing -- removing it doubles camera and point error and is the single cleanest
downstream signal in this project to date (8/8 scenes, p = 0.0017); removing learned weight clipping or
SmoothQuant scaling individually costs nothing measurable, yet removing all four costs 21.8x, so the three
non-rotation components are redundant with each other rather than useless. The exposure-corrected column is
the load-bearing caveat: every raw-PSNR gap collapses to insignificance once a per-image gain+bias is fitted,
so most of what the raw metric is scoring is brightness, not geometry -- consistent with #022 and with the
harness-insensitivity finding in #020. Geometry remains the instrument that responds; the renderer does not.

**Deviation from plan, flagged:** the sixth arm `w4a4_no_lac` is **NOT MEASURED** and MUST NOT be inferred
from the other five. Its calibration was killed 1 h 07 m in (14 of 48 blocks) by the machine's nightly
shutdown. Restarted 2026-09-10 04:37 from scratch; not resumed, because upstream `resume_qs` loads finished
totals rather than continuing a run, and every other arm is a clean run.

**Operational fact worth recording:** `last -x` shows this machine shuts down at **00:35 every night** and
returns around **04:33** -- a usable window of ~20 h, not the 12:30 reset previously assumed. Any calibration
of ~3.5 h must therefore start before ~21:00.

---

### #028 — Sixth ablation arm measured; exposure correction extended to SSIM and LPIPS

- **Date/time:** 2026-09-10; `a44_nolac` calibrated 04:37–08:07, stage 2 rerun 08:07–08:31
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/`
- **Commands executed:**
  ```
  ./code/quantization/run_nolac_then_stage2.sh
  #   -> calibrate_w2a4.py --wbit 4 --abit 4 --exp-name a44_nolac --no-lac
  #   -> run_ablation_full.sh subset   (auto-detected all six arms)
  ```
- **Wall time:** calibration 3 h 30 m (rc=0); stage 2 rerun 24 min

**Why a restart and not a resume.** The first `a44_nolac` attempt was killed at 14/48 blocks by the
nightly 00:35 shutdown. It was restarted from scratch: upstream `resume_qs` loads finished totals rather
than continuing a run, and block-wise calibration feeds each block's output into the next, so a hand-rolled
resume would have had to replay layers 0–6 anyway and could diverge from a clean run. Every other arm is a
clean run; an arm not comparable to the others has no value here. Partial files retained under
`killed_run_2026-09-09T2328/`.

**Config:** identical to #027 in every field (7000 iterations, 6 input / 9 held-out views, foreground primary,
GraphDECO defaults, no depth regularisation, 8-scene subset). Only the arm set changed.

**Geometry error vs full-precision VGGT (n = 8), all six arms:**

| arm | cam rot (deg) | cam centre err | cam spread ratio | depth rel err | point err | vs `w4a4_local` |
|---|---:|---:|---:|---:|---:|---:|
| w4a4_no_lwc | 1.1169 | 0.01815 | 1.001 | 0.01715 | 0.04015 | 0.93x |
| w4a4_no_smooth | 1.1711 | 0.02102 | 0.996 | 0.01699 | 0.04851 | 0.97x |
| w4a4_local (baseline) | 1.2050 | 0.02254 | 1.000 | 0.02198 | 0.04015 | 1.00x |
| **w4a4_no_lac** | **1.5476** | 0.02645 | 0.991 | 0.02298 | 0.04957 | **1.28x** |
| **w4a4_no_rot** | **2.4690** | 0.04117 | 0.995 | 0.02511 | 0.08084 | **2.05x** |
| w4a4_rtn | 26.2667 | 0.13944 | 0.856 | 0.08095 | 0.68261 | 21.80x |

**Downstream, foreground, raw and exposure-corrected (n = 8):**

| arm | PSNR | PSNR* | SSIM | SSIM* | LPIPS | LPIPS* |
|---|---:|---:|---:|---:|---:|---:|
| full | 9.9708 | 15.7667 | 0.2403 | 0.3674 | 0.6553 | 0.6898 |
| w4a4 | 9.6424 | 15.8439 | 0.2369 | 0.3655 | 0.6635 | 0.6882 |
| w4a4_local | 9.6721 | 15.8306 | 0.2336 | 0.3662 | 0.6624 | 0.6981 |
| w4a4_no_lac | 9.8432 | 15.9031 | 0.2365 | 0.3639 | 0.6667 | 0.6927 |
| w4a4_no_lwc | 9.9387 | 15.9986 | 0.2436 | 0.3680 | 0.6516 | 0.6818 |
| w4a4_no_rot | 9.2164 | 15.8340 | 0.2350 | 0.3698 | 0.6710 | 0.7007 |
| w4a4_no_smooth | 9.6419 | 15.9555 | 0.2323 | 0.3650 | 0.6598 | 0.6773 |
| w4a4_rtn | 8.0677 | 14.1246 | 0.1942 | 0.3263 | 0.7243 | 0.7463 |

`*` = one per-channel least-squares gain+bias fitted per image over masked pixels, applied, clipped to
[0,1], then scored. One fit per image reused across all three metrics.

**Code change:** `run_w2a4_downstream.py` previously exposure-corrected PSNR only. SSIM carries a luminance
term and LPIPS is only partly exposure-robust, so two of three metrics remained confounded. All six metrics
are now computed, aggregated, paired-tested and printed raw beside corrected.

**PASS/FAIL:**
- Target: **complete the per-component attribution begun in #027.** All six arms measured. **PASS.**
- Target: **does `no_lac` overturn the #027 conclusion that rotation is the load-bearing component?**
  It does not: `no_lac` costs 1.28x camera rotation against rotation's 2.05x. **PASS, conclusion stands.**
- Target: **every metric corrected by the same transform.** **PASS.**

**Interpretation.** Learned *activation* clipping is the second-ranked single component (1.28x) while learned
*weight* clipping is inert (0.93x, at or below baseline) — so of QuantVGGT's four components, exactly two are
individually detectable at 4 bits, and the rotation is 1.6x more important than the runner-up. The
raw-versus-corrected split from #027 survives the added arm and the added metrics: every raw-PSNR contrast
that reached p < 0.05 (`no_rot` 0.0017, `no_smooth` 0.0125, `rtn` 0.0105) fails under correction, and the
first two *change sign*, i.e. corrected, those arms render marginally better than full precision. That sign
flip is not a real effect — it is noise around zero — but it is direct evidence that raw PSNR was
manufacturing the ordering rather than measuring it.

**Unexplained observation, flagged not resolved.** Exposure correction moves PSNR and SSIM *up* (SSIM
0.2403 → 0.3674 for `full`, a larger relative shift than PSNR's) but moves LPIPS *down in quality*
(0.6553 → 0.6898, higher = worse). The plausible cause is the clip to [0,1] after gain+bias destroying
highlight detail that LPIPS is sensitive to. **This is a hypothesis, not a measurement.** Exposure-corrected
LPIPS must not be used as a headline metric until an unclipped variant is tested against it.

**Next:** 40-scene run of all eight arms launched 08:43 (`run_40scene_exposure.sh`), to separate the two
distinct n=8 outcomes — a tight null (`full − no_rot`, −0.0673, CI [−0.5966, +0.4620]) from a large but
imprecise effect (`full − rtn`, +1.6421, CI [−1.7220, +5.0063]).

---

### #029 — 40-scene, eight-arm run: pairwise gaps, dose-response, and the LPIPS clip test

- **Date/time:** 2026-09-10; 3DGS 08:43–15:56 CEST; LPIPS clip test 16:40–17:10
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/`
- **Commands executed:**
  ```
  ./code/quantization/run_40scene_exposure.sh
  #   -> run_w2a4_inference.py --variant <arm> --scenes all        (7 arms)
  #   -> ablation_compare.py --scenes all --tag all40 --variants <arms> w2a4
  #   -> run_w2a4_downstream.py --scenes all --iterations 7000 --arms <8 arms>
  python -u code/quantization/lpips_clip_test.py --scenes all --iterations 7000 \
      --arms full w4a4 w4a4_local w4a4_no_rot w4a4_rtn
  ```
- **Wall time:** 7 h 13 m; 192 trainings, 112 resume-skips, 0 failures

**Config:** identical to #028 except **Scenes used: all 40 of the frozen manifest** (sha256 `1ce2f3f8…968e06`).

**Geometry vs full precision (n = 40):**

| arm | cam rot (deg) | cam centre | spread | depth rel | point err |
|---|---:|---:|---:|---:|---:|
| w4a4 | 1.2754 | 0.02063 | 1.004 | 0.02255 | 0.06144 |
| w4a4_local | 1.1014 | 0.01896 | 1.002 | 0.02450 | 0.04987 |
| w4a4_no_lwc | 1.0799 | 0.01662 | 1.005 | 0.02201 | 0.05326 |
| w4a4_no_smooth | 1.1390 | 0.02034 | 0.991 | 0.02509 | 0.05840 |
| w4a4_no_lac | 1.3160 | 0.02286 | 1.002 | 0.02933 | 0.06447 |
| w4a4_no_rot | 2.0323 | 0.03665 | 0.996 | 0.04404 | 0.10293 |
| w4a4_rtn | 14.6565 | 0.12093 | 0.889 | 0.11145 | 0.46422 |

**Downstream, foreground (n = 40):**

| arm | PSNR | PSNR* | SSIM | SSIM* | LPIPS | LPIPS* |
|---|---:|---:|---:|---:|---:|---:|
| full | 8.9642 | 15.9047 | 0.2318 | 0.3744 | 0.6604 | 0.6895 |
| w4a4 | 8.8283 | 15.9085 | 0.2305 | 0.3739 | 0.6712 | 0.6952 |
| w4a4_local | 8.8004 | 15.8643 | 0.2278 | 0.3729 | 0.6683 | 0.6993 |
| w4a4_no_rot | 8.5326 | 15.9265 | 0.2278 | 0.3744 | 0.6749 | 0.6992 |
| w4a4_no_smooth | 8.9461 | 15.9213 | 0.2309 | 0.3753 | 0.6634 | 0.6913 |
| w4a4_no_lwc | 8.9986 | 15.9774 | 0.2318 | 0.3745 | 0.6616 | 0.6893 |
| w4a4_rtn | 7.9794 | 15.4445 | 0.2109 | 0.3618 | 0.7167 | 0.7327 |
| w4a4_no_lac | 8.8642 | 15.9000 | 0.2272 | 0.3741 | 0.6688 | 0.6989 |

**Paired, full minus arm, foreground (n = 40):**

| arm | raw PSNR | p | sign | corrected PSNR | p |
|---|---:|---:|---|---:|---:|
| w4a4 | +0.1359 | 0.09876 | 23/40 | −0.0038 | 0.966 |
| w4a4_local | +0.1638 | 0.04296 | 25/40 | +0.0404 | 0.503 |
| w4a4_no_rot | +0.4316 | 0.0001689 | 30/40 | −0.0218 | 0.7989 |
| w4a4_no_smooth | +0.0181 | 0.7953 | 19/40 | −0.0165 | 0.8609 |
| w4a4_no_lwc | −0.0344 | 0.6865 | 17/40 | −0.0727 | 0.37 |
| w4a4_rtn | +0.9848 | 7.508e-06 | 32/40 | +0.4602 | 0.1838 |
| w4a4_no_lac | +0.1000 | 0.1888 | 23/40 | +0.0047 | 0.9586 |

Full minus w4a4 on every metric: PSNR +0.1359 (p 0.099), SSIM +0.0013 (p 0.575), **LPIPS −0.0108 (p 0.00797,
28/40 favouring full)**, PSNR* −0.0038 (p 0.966), SSIM* +0.0005 (p 0.793), LPIPS* −0.0057 (p 0.206).

**Dose-response, Spearman vs 40-scene camera-rotation error over the six locally calibrated arms:**
LPIPS ρ = +0.9429 (p 0.004805); SSIM ρ = −0.8857 (p 0.01885); PSNR ρ = −0.8286 (p 0.04156);
LPIPS* ρ = +0.6571 (p 0.1562); PSNR* ρ = −0.4857 (p 0.3287); SSIM* ρ = −0.4857 (p 0.3287).

**LPIPS clip test (n = 40):** clipping to [0,1] after gain+bias alters 0.06–0.17 % of masked pixels, and
unclipped LPIPS equals clipped LPIPS to four decimals on every arm (e.g. full 0.6895 vs 0.6895).

**PASS/FAIL:**
- Target: **resolve the n = 8 ambiguity of `full − rtn` exposure-corrected (+1.6421, CI [−1.7220, +5.0063]).**
  At n = 40: +0.4602, CI [−0.2277, +1.1481]. The n = 8 estimate was inflated ~3.5x. **Resolved: small and bounded.**
- Target: **does W4A4 degrade rendering vs full precision at n = 40?** PSNR and SSIM, raw and corrected: no.
  Raw LPIPS: yes, p = 0.00797; Bonferroni over six metrics ≈ 0.048. **Marginal PASS on one metric only.**
- Target (#028 hypothesis): **is the LPIPS worsening under correction caused by the clip?** **FAIL — refuted.**
- Target: **do rendering metrics track geometry error across arms?** Raw metrics: all three at p < 0.05.
  Corrected: none. **PASS for raw, FAIL for corrected.**

**Interpretation.** At the configuration QuantVGGT ships, W4A4, quantization does not measurably degrade
novel-view rendering on PSNR or SSIM and only marginally on LPIPS; measurable degradation needs damage well
beyond anything the shipped method produces (`no_rot`, `rtn`). Rendering quality does follow geometry error
monotonically across the ladder, so the pipeline is not inert — but only raw metrics show it, and exposure
correction removes the relationship, i.e. it discards genuine geometry-driven signal. **This retracts the
#027/#028 recommendation that exposure-corrected metrics are "the column to trust"** for measuring geometry
degradation. **It also retracts #028's clip hypothesis:** the affine correction itself, not the clip, is what
worsens LPIPS, so exposure-corrected LPIPS is dropped rather than repaired.

---

### #030 — Perfect-confidence oracle: pruning by TRUE error recovers none of a real gap

- **Date/time:** 2026-09-10 21:54 → 2026-09-11 00:15 CEST (self-stopped at its deadline)
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/confidence_oracle.py`
- **Command executed:**
  ```
  python -u code/quantization/confidence_oracle.py --arm w4a4_rtn --scenes 40 \
      --iterations 7000 --keep 0.5 --deadline 00:15
  ```
- **Wall time:** 2 h 21 m; stopped cleanly after 36 of 40 scenes. The last four were **not run**, not estimated.

**What it tests.** An upper bound on the proposal. Instead of a learned confidence head, the TRUE per-point error
against full-precision VGGT (Sim(3) on the six input camera centres, residual in scene radii, on the stride-4
init grid) is used as confidence, and the worst half of the initial points is removed before 3DGS.

**Conditions (per scene):** A full precision; B `w4a4_rtn`, all points; C `w4a4_rtn`, lowest-error 50 % kept;
R `w4a4_rtn`, random 50 % kept (seed 0) — the matched-point-count control. **C − R is the test**; C − B
confounds confidence with halving the Gaussian count.

**Config:**

| Field | Value |
|---|---|
| Init source | `w4a4_rtn` VGGT geometry, stride 4, 101,400 → 50,700 points |
| Confidence | true per-point error vs full precision (oracle — not available at deployment) |
| Iterations | 7000 |
| Densify | GraphDECO default |
| Input views | 6 input, 9 held-out |
| Mask type | foreground |
| Background handling | GS default black, unmasked full frames |
| Depth regularisation | off |
| Cameras | the arm's own predicted cameras, unchanged by pruning |

**Scenes used:** the first 36 of the frozen 40-scene manifest, in manifest order.

**Per-scene results (foreground):**

| scene | A PSNR | B PSNR | C PSNR | R PSNR | B LPIPS | C LPIPS | R LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| apple/110_13051_23361 | 12.670 | 9.594 | 9.728 | 9.490 | 0.7101 | 0.6992 | 0.7102 |
| apple/189_20393_38136 | 6.539 | 6.527 | 6.337 | 6.316 | 0.7506 | 0.7606 | 0.7664 |
| ball/123_14363_28981 | 11.224 | 8.528 | 8.574 | 8.538 | 0.6304 | 0.6388 | 0.6255 |
| ball/375_42693_85518 | 10.229 | 10.243 | 10.832 | 10.498 | 0.6918 | 0.6946 | 0.6977 |
| bench/415_57112_110099 | 12.687 | 10.799 | 12.031 | 11.178 | 0.6609 | 0.6300 | 0.6500 |
| bench/415_57121_110109 | 11.121 | 9.421 | 9.998 | 9.393 | 0.6981 | 0.6779 | 0.6904 |
| book/119_13962_28926 | 8.465 | 7.876 | 7.965 | 7.844 | 0.7016 | 0.6898 | 0.6953 |
| book/247_26469_51778 | 10.197 | 8.996 | 8.482 | 8.846 | 0.7786 | 0.7744 | 0.7815 |
| bowl/69_5465_12831 | 10.959 | 10.152 | 10.270 | 10.548 | 0.7098 | 0.6971 | 0.7090 |
| bowl/70_5792_13401 | 10.458 | 5.684 | 5.684 | 5.684 | 0.9102 | 0.9102 | 0.9102 |
| broccoli/372_41112_81867 | 8.211 | 7.772 | 7.589 | 7.679 | 0.7098 | 0.7078 | 0.7055 |
| broccoli/412_56288_108844 | 9.338 | 9.102 | 8.102 | 9.253 | 0.7506 | 0.7465 | 0.7492 |
| cake/374_42274_84517 | 7.306 | 5.668 | 6.022 | 5.474 | 0.7218 | 0.6829 | 0.7230 |
| cake/403_53094_103680 | 7.532 | 6.284 | 5.990 | 6.464 | 0.7698 | 0.7810 | 0.7664 |
| donut/391_47032_93657 | 8.537 | 8.321 | 7.830 | 8.133 | 0.7864 | 0.7965 | 0.7582 |
| donut/403_52964_103416 | 8.792 | 9.867 | 7.803 | 9.653 | 0.5679 | 0.6068 | 0.5999 |
| hydrant/167_18184_34441 | 8.131 | 7.207 | 7.989 | 7.295 | 0.7744 | 0.7689 | 0.7733 |
| hydrant/411_56064_108483 | 8.130 | 8.777 | 8.928 | 8.817 | 0.6498 | 0.6460 | 0.6622 |
| mouse/107_12753_23606 | 6.719 | 7.672 | 7.660 | 7.847 | 0.7223 | 0.7050 | 0.7255 |
| mouse/377_43416_86289 | 7.819 | 5.686 | 5.753 | 5.811 | 0.7491 | 0.7393 | 0.7432 |
| orange/374_42196_84367 | 9.734 | 8.158 | 7.595 | 7.792 | 0.8378 | 0.8587 | 0.8456 |
| orange/385_45386_90752 | 8.716 | 8.720 | 8.071 | 8.267 | 0.6668 | 0.6867 | 0.6843 |
| plant/247_26441_50907 | 9.124 | 7.870 | 7.758 | 7.934 | 0.6639 | 0.6486 | 0.6655 |
| plant/374_42005_84358 | 8.735 | 8.265 | 8.089 | 8.121 | 0.6382 | 0.6670 | 0.6398 |
| remote/195_20989_41543 | 9.215 | 7.901 | 8.249 | 8.068 | 0.6961 | 0.6846 | 0.6942 |
| remote/350_36761_68623 | 9.656 | 9.385 | 9.361 | 9.516 | 0.6664 | 0.6901 | 0.6627 |
| skateboard/245_26182_52130 | 9.546 | 8.047 | 8.323 | 8.521 | 0.7052 | 0.6813 | 0.6997 |
| skateboard/366_39266_76077 | 9.450 | 9.549 | 9.252 | 9.944 | 0.6813 | 0.6779 | 0.6721 |
| suitcase/410_55734_107452 | 8.462 | 6.361 | 6.537 | 6.243 | 0.7491 | 0.7244 | 0.7450 |
| suitcase/50_2928_8645 | 9.317 | 8.353 | 8.326 | 8.292 | 0.6917 | 0.7099 | 0.7005 |
| teddybear/187_20215_38541 | 9.341 | 7.973 | 7.942 | 7.648 | 0.6389 | 0.6427 | 0.6558 |
| teddybear/34_1479_4753 | 7.200 | 8.224 | 7.001 | 8.110 | 0.6926 | 0.7009 | 0.6985 |
| toaster/372_41229_82130 | 8.949 | 7.067 | 6.740 | 6.915 | 0.7133 | 0.7194 | 0.7185 |
| toaster/416_57389_110765 | 8.631 | 7.865 | 7.493 | 8.030 | 0.7076 | 0.7114 | 0.7025 |
| toytrain/240_25394_51994 | 8.422 | 8.232 | 8.292 | 8.362 | 0.9714 | 0.9812 | 0.9785 |
| toytrain/399_51323_100753 | 4.455 | 4.363 | 3.840 | 4.507 | 0.8222 | 0.8253 | 0.8352 |

**Means (n = 36):**

| condition | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| A full precision | 9.0004 | 0.2341 | 0.6683 |
| B rtn, all points | 8.0697 | 0.2128 | 0.7218 |
| C rtn, confidence-kept | 7.9566 | 0.2062 | 0.7212 |
| R rtn, random-kept | 8.0842 | 0.2137 | 0.7234 |

**Paired (n = 36):**

| contrast | PSNR | p | LPIPS | p |
|---|---:|---:|---:|---:|
| A − B (the gap to close) | +0.9307 | 4.56e-05 | −0.0536 | 6.632e-05 |
| C − B (confidence vs uniform) | −0.1131 | 0.246 | −0.0006 | 0.8223 |
| **C − R (confidence vs random)** | **−0.1276** | **0.1541** | **−0.0022** | **0.4084** |
| R − B (point-count effect) | +0.0145 | 0.6948 | +0.0015 | 0.3833 |

C beats R on PSNR in 14/36 scenes and beats B in 15/36. The single-scene smoke test (apple, C − R +0.239 dB)
was not representative.

**PASS/FAIL:** Target: **a perfect-knowledge confidence signal recovers a meaningful fraction of the A − B gap,
beyond the matched random control.** The gap is real (+0.9307 dB, p = 4.56e-05). Recovery (C − B)/(A − B) on
PSNR = −0.1131 / 0.9307 = **−12 %**, not significant; C − R is −0.1276 dB, p = 0.1541. **FAIL.**

**Interpretation.** Removing the least trustworthy half of the initial points — chosen with error values no
deployed predictor could ever see — recovers none of a real, highly significant 0.93 dB gap, and does no better
than removing a random half. This rules out init-time point pruning as the mechanism for confidence-weighted
3DGS on this arm. It does **not** rule out every use of confidence: GraphDECO densification regrows points
from 50,700 anyway, and the pruning leaves the arm's cameras untouched. The second point matters most: `rtn`
carries 14.66° of camera-rotation error, and no point-level confidence can correct a wrong camera. **Whether
`rtn`'s gap is camera error or point error is not measured** — the discriminating experiment is to render `rtn`
points with full-precision cameras, and it should be run before any conclusion about the proposal is drawn.

---

### #031 — CPU camera analysis: focal-length error, per-scene predictors, and error against ground truth

- **Date/time:** 2026-09-11 13:35–13:43 CEST (CPU only, `nice -n 15`, run beside the GPU swap probe)
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/cpu_camera_analysis.py`
- **Command executed:**
  ```
  OMP_NUM_THREADS=4 nice -n 15 /home/utn/luli38se/cv/.venv/bin/python -u \
      code/quantization/cpu_camera_analysis.py
  ```
- **Config:** no training, no rendering. Pose errors use the ablation_compare.py construction (Sim(3) on the six
  input camera centres; mean per-camera rotation angle; median centre residual in scene radii). Focal error =
  median |f_arm / f_ref − 1| over cameras and both axes. Rendering deltas are read from #029's 40-scene JSON
  (foreground, raw). GT intrinsics are mapped to 518 px with `original_to_518_affine`.
- **Scenes used:** (1)+(2) the frozen 40-scene manifest × 7 W4A4 arms = 280 scene-arm pairs; (3) all 1330
  view-groups of the disagreement dataset, which are drawn from those same 40 scenes.

**1. Focal-length error beside pose error, vs full precision (n = 40):**

| arm | cam rot (deg) | centre (radii) | focal error |
|---|---:|---:|---:|
| w4a4 | 1.2754 | 0.02063 | 2.614 % |
| w4a4_local | 1.1014 | 0.01896 | 2.335 % |
| w4a4_no_lwc | 1.0799 | 0.01662 | 2.875 % |
| w4a4_no_smooth | 1.1390 | 0.02034 | 2.734 % |
| w4a4_no_lac | 1.3160 | 0.02286 | 3.862 % |
| w4a4_no_rot | 2.0323 | 0.03665 | 6.693 % |
| w4a4_rtn | 14.6565 | 0.12093 | 18.652 % |

**2. Spearman ρ between a scene's geometry error and its rendering loss vs full** (`*` p < 0.05):

| slice | target | cam rot | centre | focal | depth | points |
|---|---|---|---|---|---|---|
| all arms, n = 280 | ΔPSNR | +0.199* | +0.216* | +0.270* | +0.186* | +0.072 |
| all arms, n = 280 | ΔLPIPS | +0.399* | +0.348* | +0.301* | +0.277* | +0.286* |
| w4a4 only, n = 40 | ΔPSNR | −0.067 | −0.032 | −0.090 | −0.022 | −0.222 |
| w4a4 only, n = 40 | ΔLPIPS | **+0.409\* (p 0.0088)** | +0.192 | −0.098 | +0.067 | +0.106 |
| mild arms (no rtn, no_rot), n = 200 | ΔPSNR | −0.048 | −0.045 | +0.054 | −0.044 | **−0.231\*** |
| mild arms (no rtn, no_rot), n = 200 | ΔLPIPS | +0.258* | +0.188* | +0.057 | +0.086 | +0.062 |

**3. Against CO3D ground truth, 1330 view-groups, 0 unreadable:**

| quantity | mean | median | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| full vs GT rotation (deg) | 1.5905 | 1.0842 | 2.5521 | 10.2940 | 27.5783 |
| w4a4 vs GT rotation (deg) | 2.0784 | 1.4269 | 3.1320 | 11.6597 | 168.2326 |
| w4a4 vs full rotation (deg) | 1.3383 | 0.9149 | 1.9233 | 6.2651 | 149.0445 |
| full vs GT focal error | 0.0345 | 0.0237 | 0.0781 | 0.1742 | 0.2231 |
| w4a4 vs GT focal error | 0.0466 | 0.0287 | 0.1058 | 0.2191 | 0.2892 |

Paired, w4a4 minus full rotation error vs GT: mean +0.4879°, median +0.3113°, t-test p = 1.29e-05, Wilcoxon
p = 6.95e-115, w4a4 worse in 1073/1330. W4A4 departs from full by > 2° in 124/1330 groups, > 5° in 23, > 10° in 6.
**19 of the 23 groups above 5° are one scene, `bowl/70_5792_13401`**, where full precision is itself 8–28° off GT
(the other four: remote/195_20989_41543, hydrant/167_18184_34441, toytruck/190_20494_39385, vase/380_44863_89631).

**PASS/FAIL:**
- Target: **measure whether the swap's camera condition carried a large focal-length error.** `w4a4_rtn` focal is
  off by 18.652 %. **Yes — the swap's "camera damage" is confounded with focal damage and must be split** (queued
  as `geometry_probes.py camsplit`).
- Target: **does any per-scene geometry error predict W4A4's rendering loss (n = 40)?** Camera rotation vs ΔLPIPS
  ρ = +0.409, p = 0.0088; ten correlations were tested in that slice, Bonferroni ≈ 0.088. **Suggestive, not
  established.**
- Target: **does W4A4 make cameras measurably worse against ground truth?** Yes, in 1073/1330 groups,
  p = 6.95e-115, by a median +0.31° on top of VGGT's own median 1.08°. **PASS.**

**Interpretation.** Quantization makes W4A4's cameras reliably but slightly worse against ground truth — about a
third larger than VGGT's own error — which is small next to the error VGGT already makes and consistent with
W4A4 rendering like full precision. The apparent rare catastrophic W4A4 failures are **not** a quantization
phenomenon: they concentrate on one rotationally symmetric bowl where full precision is already unstable, so they
must not be reported as W4A4 tail risk. Among per-scene predictors, camera rotation is the one that tracks
perceptual loss even within the mild arms, consistent with the swap probe's interim finding that cameras, not
points, carry the damage. **Unexplained, flagged:** in the mild arms, larger point error goes with *smaller*
PSNR loss (ρ = −0.231, p = 0.00099); no mechanism is proposed and none should be assumed.

---

### #032 — Robustness re-test of every headline claim (pass 1); the dose-response is weaker than reported

- **Date/time:** 2026-09-11 13:56–13:57 CEST (CPU)
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/cpu_probes.py`
- **Command executed:**
  ```
  nice -n 15 env OMP_NUM_THREADS=1 /home/utn/luli38se/cv/.venv/bin/python -u \
      code/quantization/cpu_probes.py robustness --workers 8
  ```
- **Config:** no training. Inputs: #029 40-scene per-scene JSON (foreground, raw, 7000 it), #030 confidence-oracle
  JSON (n = 36), swap-probe JSON **as it stood at 13:56 (n = 29 of 40, interim)**, #031 per-scene camera errors.
  Per contrast: bootstrap 95 % CI (10,000 resamples, seed 0), sign-flip permutation p (20,000 flips), t and
  Wilcoxon p, leave-one-scene-out mean range and worst p, and the contrast with `bowl/70_5792_13401` removed.
  Dose-response: Spearman over the six locally calibrated arms, **exact** permutation p over all 720 orderings.
- **Scenes used:** frozen 40-scene manifest (swap: its first 29; confidence oracle: its first 36).

**Contrasts (positive = second arm renders worse):**

| contrast | n | mean | boot CI95 | t p | perm p | worst LOO p | no-bowl mean / p |
|---|---:|---:|---|---:|---:|---:|---|
| PSNR full − w4a4 | 40 | +0.1359 | [−0.0164, +0.2895] | 0.0988 | 0.0972 | 0.17 | +0.1408 / 0.095 |
| LPIPS w4a4 − full | 40 | +0.0108 | [+0.0033, +0.0182] | 0.00797 | 0.0077 | 0.0151 | +0.0111 / 0.008 |
| SSIM full − w4a4 | 40 | +0.0013 | [−0.0030, +0.0059] | 0.575 | 0.58 | 0.939 | +0.0015 / 0.53 |
| PSNR full − w4a4_local | 40 | +0.1638 | [+0.0184, +0.3160] | 0.043 | 0.0417 | 0.0799 | +0.1614 / 0.051 |
| LPIPS w4a4_local − full | 40 | +0.0079 | [+0.0019, +0.0141] | 0.0171 | 0.0176 | 0.032 | +0.0084 / 0.013 |
| PSNR full − w4a4_no_rot | 40 | +0.4316 | [+0.2374, +0.6371] | 0.000169 | 0.00025 | 0.000341 | +0.4073 / 0.00034 |
| LPIPS w4a4_no_rot − full | 40 | +0.0145 | [+0.0057, +0.0234] | 0.00285 | 0.0026 | 0.00544 | +0.0147 / 0.0033 |
| PSNR full − w4a4_rtn | 40 | +0.9848 | [+0.6222, +1.3575] | 7.51e-06 | 5e-05 | 1.62e-05 | +0.8877 / 5.7e-06 |
| LPIPS w4a4_rtn − full | 40 | +0.0563 | [+0.0363, +0.0778] | 6.61e-06 | 5e-05 | 1.43e-05 | +0.0512 / 6.5e-06 |
| PSNR conf-oracle C − R | 36 | −0.1276 | [−0.3028, +0.0372] | 0.154 | 0.162 | 0.301 | −0.1312 / 0.15 |
| LPIPS conf-oracle C − R | 36 | +0.0022 | [−0.0029, +0.0072] | 0.408 | 0.405 | 0.66 | +0.0022 / 0.41 |
| PSNR swap full − P (interim) | 29 | +0.1660 | [−0.1640, +0.5797] | 0.402 | 0.492 | 0.99 | +0.0014 / 0.99 |
| LPIPS swap full − P (interim) | 29 | +0.0061 | [−0.0090, +0.0276] | 0.541 | 0.739 | 0.639 | −0.0028 / 0.51 |
| PSNR swap full − K (interim) | 29 | +0.8771 | [+0.4934, +1.2767] | 0.000206 | 0.00015 | 0.000442 | +0.7985 / 0.00038 |
| LPIPS swap full − K (interim) | 29 | +0.0617 | [+0.0439, +0.0805] | 5.53e-07 | 5e-05 | 1.44e-06 | +0.0563 / 2.4e-07 |
| PSNR swap full − B (interim) | 29 | +1.1793 | [+0.7634, +1.6416] | 2.05e-05 | 5e-05 | 4.71e-05 | +1.0509 / 1.4e-05 |

(perm p of 5e-05 is the floor for 20,000 flips: no flip matched the observed mean.)

**Dose-response, exact p over 720 orderings:**

| metric | ρ | exact p | leave-one-arm-out ρ | no bowl: ρ / exact p |
|---|---:|---:|---|---|
| LPIPS | +0.9429 | 0.0167 | +0.900 … +1.000 | +0.8286 / 0.0583 |
| SSIM | −0.8857 | 0.0333 | −0.900 … −0.800 | −0.7143 / 0.1361 |
| PSNR | −0.8286 | 0.0583 | −1.000 … −0.700 | −0.8286 / 0.0583 |
| PSNR exposure-corrected | −0.4857 | 0.3556 | −0.700 … −0.100 | −0.4857 / 0.3556 |

**PASS/FAIL:**
- Target: **every headline pairwise claim survives permutation, bootstrap, leave-one-out and bowl removal.**
  W4A4 LPIPS (perm p 0.0077, CI excludes 0, worst LOO 0.0151, no-bowl 0.008): **PASS.** no_rot and rtn: **PASS.**
  W4A4 PSNR/SSIM null: **PASS (null robust).** Confidence-oracle null: **PASS (null robust).** Swap
  camera-vs-points (interim): **PASS**; points effect with bowl removed is +0.0014 dB, p 0.99.
  `w4a4_local` PSNR is fragile (worst LOO p 0.0799, no-bowl 0.051): **FAIL — not robust.**
- Target: **the dose-response survives an exact test.** LPIPS and SSIM yes (0.0167, 0.0333); PSNR no (0.0583);
  none of the three at p < 0.05 once the bowl scene is removed. **PARTIAL.**

**Correction to #029 and to statements made in conversation.** #029 reported the dose-response with scipy's
asymptotic Spearman p (LPIPS 0.004805, SSIM 0.01885, PSNR 0.04156). With n = 6 that approximation is too
optimistic. The exact values are LPIPS 0.0167, SSIM 0.0333, **PSNR 0.0583 (not significant)**, and none survive
removal of the bowl scene at 0.05. The claim "all three raw metrics track geometry error significantly" is
**withdrawn**; the licensed statement is that rendering quality is *consistent with* a monotone dependence on
geometry error (LPIPS, SSIM), resting on six arms.

**Interpretation.** The pairwise findings are robust by every test applied — W4A4's small LPIPS cost, the large
no_rot and rtn gaps, the confidence-oracle null, and (interim) cameras-not-points — and none depends on the
unstable bowl scene. The arm-level dose-response is the weak link: six points cannot carry a strong claim, which
is exactly why the synthetic damage sweep (running) exists.

---

### #033 — Depth accuracy against CO3D ground truth, full vs W4A4, all 1330 view-groups

- **Date/time:** 2026-09-11 13:57 CEST (CPU)
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/cpu_probes.py`
- **Command executed:**
  ```
  nice -n 15 env OMP_NUM_THREADS=1 /home/utn/luli38se/cv/.venv/bin/python -u \
      code/quantization/cpu_probes.py depthgt --workers 8
  ```
- **Config:** no training. GT depth = CO3D depth map resampled onto the 518×518 pad grid with
  `prep_depth.gt_depth_518` (the loader the depth-regularised oracle used), valid where CO3D's depth mask AND the
  object mask hold — i.e. **the object only**. Each predicted depth is aligned to GT by one per-image median scale
  (VGGT depth is up to scale). Metrics: median absolute relative error, fraction of pixels with
  max(pred/gt, gt/pred) < 1.25. Edge test: full-vs-w4a4 depth disagreement (after a median scale) in a 5-px band
  inside the object boundary, divided by the interior's. Sanity check before the run: full-precision absrel on one
  frame 0.0067, δ<1.25 = 1.0.
- **Scenes used:** all 1330 view-groups (drawn from the 40 frozen scenes); 1330 usable, 0 failed.

**Results:**

| subset | metric | full | w4a4 | w4a4 − full (median) | Wilcoxon p | w4a4 larger |
|---|---|---:|---:|---:|---:|---|
| all, n = 1330 | absrel | 0.00542 | 0.00645 | +0.00103 (+0.00084) | 2.01e-191 | 1214/1330 |
| all, n = 1330 | δ<1.25 | 0.99851 | 0.99871 | +0.00019 (0.00000) | 9.1e-17 | 527/1330 |
| no bowl, n = 1296 | absrel | 0.00539 | 0.00644 | +0.00105 (+0.00086) | 5.9e-191 | 1194/1296 |
| no bowl, n = 1296 | δ<1.25 | 0.99847 | 0.99868 | +0.00021 (0.00000) | 2.27e-24 | 527/1296 |

Edge band over interior disagreement: median ratio **1.594**, > 1 in **99.1 %** of 1330 groups, Wilcoxon on log
ratio p = 2.34e-218.

**PASS/FAIL:**
- Target: **is W4A4 object depth measurably worse than full against GT?** Yes, in 1214/1330 groups, p = 2.01e-191 —
  but by +0.00103 absolute relative error on a base of 0.00542. **PASS (real, tiny).**
- δ<1.25 is saturated (≥ 0.9985 for both) and its median difference is 0: **uninformative, not a finding.**
- Target: **does quantization disagreement concentrate at object edges?** 1.594× in 99.1 % of groups. **PASS.**

**Interpretation.** On the object, per-view depth survives W4A4 almost untouched — about 0.1 % of depth extra error,
consistent in sign but negligible in size — and what disagreement there is sits disproportionately on object
boundaries. The edge band is defined with the GT mask, which a deployed predictor would not have; whether a
mask-free proxy (depth gradient) recovers it is tested in the predictability run.

---

### #034 — Where W4A4's world-point error comes from: depth maps vs cameras

- **Date/time:** 2026-09-11 14:10 CEST (CPU)
- **Git commit:** `f0c8e32` + working tree (`cpu_probes.py decompose` added)
- **Command executed:**
  ```
  nice -n 15 env OMP_NUM_THREADS=1 /home/utn/luli38se/cv/.venv/bin/python -u \
      code/quantization/cpu_probes.py decompose --workers 6
  ```
- **Config:** no training. Per view-group: *total* = W4A4 world points vs full's after the Sim(3) on camera centres;
  *depth-only* = W4A4 depth, rescaled by one global median factor, unprojected through **full's** cameras, vs full's
  points — camera error removed by construction. Median over every 7th pixel of all six views (**all content
  pixels, object and background**), in scene radii. Convention check: unprojecting full's depth through full's
  cameras reproduces full's points to 3.85e-07.
- **Scenes used:** all 1330 view-groups; 1330 usable.

**Results:**

| subset | n | total (median) | depth-only (median) | depth-only / total (median per group) | total > depth-only | Wilcoxon p |
|---|---:|---:|---:|---:|---|---:|
| all | 1330 | 0.03624 | 0.02510 | 0.705 | 1169/1330 | 4.02e-181 |
| no bowl | 1296 | 0.03559 | 0.02469 | 0.712 | 1135/1296 | 1.1e-174 |

**PASS/FAIL:** Target: **attribute W4A4's world-point disagreement to depth vs cameras.** With cameras held exact,
the error is still 70.5 % of its full size. **Measured: most of W4A4's point disagreement is in its depth maps, not
its cameras.**

**Interpretation, and a correction to an expectation stated in conversation.** I expected the point error to be
mostly camera-driven, because #033 shows object depth nearly unchanged. It is not: depth alone reproduces ~70 % of
the point error. The two results reconcile only if W4A4's depth disagreement lives mainly in the **background**,
which #033 cannot see (GT depth exists only on the object) — **that is an inference, not yet measured.** It does not
contradict the swap probe: the swap measures *rendering* loss, and it shows point damage barely affects rendering
at all, so where the point error comes from matters less than that 3DGS absorbs it. The swap was on `w4a4_rtn`;
this decomposition is on `w4a4`, and they must not be merged into one sentence.

---

### #035 — W4A4 point error splits into thirds: depth shape, per-view scale drift, cameras (supersedes #034's inference)

- **Date/time:** 2026-09-11 14:15–14:25 CEST (CPU; two re-runs of the #034 command with added measurements)
- **Git commit:** `f0c8e32` + working tree (`cpu_probes.py decompose` extended)
- **Command executed (final):**
  ```
  nice -n 15 env OMP_NUM_THREADS=1 /home/utn/luli38se/cv/.venv/bin/python -u \
      code/quantization/cpu_probes.py decompose --workers 6
  ```
- **Config:** as #034, plus (a) the same medians restricted to the object mask and to background content pixels,
  and (b) *depth-shape*: W4A4 depth with **each view rescaled by its own median factor**, unprojected through
  full's cameras — removing per-view scale drift as well as camera error. Per-view drift = std of log per-view
  scale factors across the six views.
- **Scenes used:** all 1330 view-groups.

**Results (median over groups of per-group medians, scene radii):**

| region | total | depth, one global scale | share | depth shape, per-view scale | share |
|---|---:|---:|---:|---:|---:|
| object | 0.02163 | 0.01397 | 0.637 | 0.00646 | 0.316 |
| background | 0.03278 | 0.02175 | 0.627 | 0.01089 | 0.337 |

Per-view depth-scale drift of W4A4 relative to full: median 0.01308 (≈ 1.3 %), p90 0.02292.

**PASS/FAIL:**
- #034's inference — *"W4A4's depth disagreement lives mainly in the background"* — **FAIL, refuted**: the depth
  share is 0.637 on the object and 0.627 in the background.
- Target: **reconcile #033 (object depth almost unchanged per view) with #034 (depth explains ~70 %).** With per-view
  scaling, depth shape explains only ~0.32–0.34 of the error; the other ~0.3 is per-view scale drift of ~1.3 %.
  **Reconciled.**

**Interpretation.** W4A4's world-point disagreement with full precision is roughly one third each: per-view depth
**shape**, per-view depth **scale drift** (the six views disagreeing about scale by ~1.3 %), and **cameras**. Each
view's depth is individually almost unchanged — which is why #033, scaling every image separately, saw only +0.1 %.
The damage is mostly in how views fit *together*: scale drift and camera error are both inter-view inconsistencies,
which is precisely what a cross-view self-consistency check can observe without full precision. Whether it does
is the predictability run. **#034's background inference is withdrawn.**

---

### #036 — Can W4A4's error be predicted from W4A4's own output? Pixel level: partly; camera level: no

- **Date/time:** 2026-09-11; absolute-target run 13:57–14:01:11, relative-target run finished 14:07:58 CEST (CPU)
- **Timestamp correction for #034 and #035:** those entries give 14:10 and 14:15–14:25. Those times were written
  without reading the clock and are **wrong**. The measured file times are: #033 `cpu_depthgt.json` 13:57:57;
  #034 and #035 were successive runs of `cpu_probes.py decompose` whose final output `cpu_decompose.json` is
  stamped 14:04:02. The results in those entries are unaffected; only their stated times are.
- **Git commit:** `f0c8e32` + working tree (`cpu_probes.py predict`)
- **Commands executed:**
  ```
  nice -n 15 env OMP_NUM_THREADS=1 /home/utn/luli38se/cv/.venv/bin/python -u \
      code/quantization/cpu_probes.py predict --workers 8
  PREDICT_TARGET=relative nice -n 15 env OMP_NUM_THREADS=1 /home/utn/luli38se/cv/.venv/bin/python -u \
      code/quantization/cpu_probes.py predict --workers 8
  ```
- **Config:** no training. All 1330 view-groups; 300 random content pixels per view (seeded per group), 2,394,000
  pixels. **Label:** pixel in the top decile of W4A4-vs-full world-point error *within its own view-group* (the
  "where in this image" question a confidence map must answer). **Target** either absolute error (scene radii) or
  **relative** error (divided by the point's distance from full's camera). **Features — W4A4 output and RGB only,
  nothing from full precision:** log depth; log depth / view median; relative depth gradient; relative local depth
  std (5×5); image gradient; image texture (7×7 std); image intensity; radial image position; cross-view
  self-inconsistency (each pixel's W4A4 point reprojected into the other five views with W4A4's cameras, median
  relative disagreement with W4A4's depth there). **Models:** L2 logistic regression on standardised features
  (`linear`) and on 10 quantile bins per feature (`binned`). **Validation:** 5 folds holding out whole scenes, and
  5 folds holding out whole categories (20 categories). Foreground mask used only to subset evaluation, never as a
  feature.
- **Scenes used:** all 1330 view-groups of the 40-scene / 20-category set; 0 failed.

**Pixel level (AUROC; 0.5 = chance):**

| target | region | best single feature | scene-held-out linear / binned | category-held-out linear / binned |
|---|---|---|---|---|
| absolute | all content | log_depth_rel_median 0.821 | 0.8882 / 0.8979 | 0.8873 / 0.8970 |
| absolute | object only (1,600 positives) | depth_local_std_rel 0.828 | 0.8806 / 0.8678 | 0.8968 / 0.8808 |
| **relative** | all content | depth_local_std_rel 0.721 | **0.8165 / 0.8310** | **0.8171 / 0.8328** |
| **relative** | **object only** (6,169 positives) | depth_local_std_rel 0.641 | **0.6432 / 0.5888** | **0.6677 / 0.6314** |

Image-appearance features alone are at or below chance in every setting (img_grad 0.436–0.471, img_texture
0.432–0.451, img_intensity 0.489–0.563). Self-inconsistency alone: 0.770 / 0.691 absolute (all / object), 0.728 /
0.594 relative. Fold ranges are in `cpu_predict*.json`; object-only category folds span 0.606–0.787 (relative).

**Camera level (per view-group, Spearman, n = 1330; no-bowl n = 1296):**

| camera error | ~ self-inconsistency | ~ depth spread |
|---|---|---|
| w4a4 vs full rotation | ρ −0.032, p 0.248 (no bowl −0.024, p 0.395) | ρ −0.361, p 3.19e-42 |
| w4a4 vs GT rotation | ρ −0.090, p 0.00106 | ρ −0.352, p 4.22e-40 |
| full vs GT rotation | ρ −0.048, p 0.0808 | ρ −0.274, p 2.36e-24 |

**PASS/FAIL:**
- Target: **W4A4 pixel error is predictable from W4A4's own output, and the predictor transfers to unseen
  categories.** All content, relative target: AUROC 0.817–0.833, category-held-out equal to scene-held-out.
  **PASS.** On the object only: 0.589–0.668. **FAIL — weak.**
- The absolute-target numbers (≈ 0.89) **overstate predictability**: absolute point error grows with distance, and
  depth alone scores 0.811. The relative target is the one to cite.
- Target: **a cross-view self-consistency signal flags W4A4's camera error.** ρ = −0.032, p = 0.248. **FAIL.**

**Interpretation.** Where W4A4's geometry is wrong is predictable from its own output mainly at depth
discontinuities and image periphery — geometric cues, not appearance ones, so the 40-object appearance-diversity
concern raised in discussion does not bite: RGB features carry no signal at all. But on the object, where
reconstruction quality is decided, predictability falls to AUROC ~0.6–0.67. And at the camera level, where #030 and
the swap probe place the rendering damage, W4A4's outputs are **internally consistent even when their cameras are
wrong** — self-consistency does not see the error. Depth spread predicts camera error for full precision as well as
W4A4 (flatter view-groups have worse poses), so it is a property of VGGT pose estimation, not a quantization
signal. **Net for the proposal:** a confidence signal is available where it is least needed (points, background,
edges — which 3DGS absorbs anyway) and absent where it would matter (cameras).

---

### #037 — Why exposure correction erases the geometry signal: it removes ~half of the geometry-induced loss

- **Date/time:** 2026-09-11, `cpu_exposure.json` written before 14:11:30 CEST (CPU)
- **Git commit:** `f0c8e32` + working tree (`cpu_probes.py exposure`)
- **Command executed:**
  ```
  nice -n 15 /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/cpu_probes.py exposure
  ```
- **Config:** no training, no image loading. Per scene and arm, from #029's 40-scene JSON (foreground):
  *extra correction gain* = [PSNR after per-image gain+bias − raw PSNR] for the arm, minus the same quantity for
  `full` on the same scene. Correlated with that scene-arm's geometry error from #031.
- **Scenes used:** 40 scenes × 7 W4A4 arms = 280 pairs.

**Spearman ρ of extra correction gain with:**

| slice | cam rot | focal | point err | raw PSNR loss |
|---|---|---|---|---|
| all arms, n = 280 | +0.221 (p 0.00019) | +0.262 (p 8.9e-06) | +0.118 (p 0.049) | +0.766 (p 3.4e-55) |
| mild arms (no rtn, no_rot), n = 200 | +0.026 (p 0.71) | +0.030 (p 0.67) | −0.111 (p 0.12) | +0.705 (p 2.1e-31) |
| all arms, no bowl, n = 273 | +0.227 (p 0.00016) | +0.270 (p 5.8e-06) | +0.123 (p 0.043) | +0.782 (p 1.4e-57) |

**Per-arm mean extra correction gain (dB), Wilcoxon vs 0, n = 40:** w4a4 +0.1397 (p 0.277); w4a4_local +0.1234
(0.211); no_lwc +0.0383 (0.942); no_smooth +0.0347 (0.785); no_lac +0.0953 (0.476); **no_rot +0.4534 (0.000336)**;
**rtn +0.5246 (0.000402)**.

**PASS/FAIL:** Target: **test #029's claim that exposure correction removes geometry-driven signal.** For the two
arms with real geometry damage, the correction recovers +0.4534 and +0.5246 dB *more* than it does for full
precision on the same scenes (p ≤ 0.0004), and the extra gain tracks camera and focal error across arms. For
`rtn` that is 0.5246 of its 0.9848 dB raw gap. **PASS — mechanism supported.**

**Interpretation.** Roughly half of the rendering loss that bad geometry causes shows up as a global,
per-image brightness/contrast shift, and a per-image gain+bias fit removes exactly that half — which is why every
corrected metric loses its relationship to geometry. The images, training and evaluation cameras are identical
across arms, so this shift is caused by the geometry, not by auto-exposure. The *why* (e.g. misregistered views
averaged into lower-contrast Gaussians) is **not measured**. Caveat: the ρ ≈ 0.77 with raw PSNR loss is partly
mechanical — a worse raw render has more for a gain fit to recover — so the camera/focal correlations and the
per-arm tests are the evidence, not that column. In mild arms there is no geometry damage to remove and no
correlation.

---

### #038 — Power: how many scenes each observed effect needs

- **Date/time:** 2026-09-11, finished 14:11:30 CEST (CPU)
- **Git commit:** `f0c8e32` + working tree (`cpu_probes.py power`)
- **Command executed:**
  ```
  nice -n 15 /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/cpu_probes.py power
  ```
- **Config:** for each full-vs-arm contrast on #029's 40-scene per-scene data (foreground): the smallest n at which a
  two-sided paired t-test (α = 0.05) reaches 80 % power, by the noncentral t distribution, **taking the observed
  mean and sd as the true values.**
- **Scenes used:** 40-scene set, 7 arms.

| contrast (full vs …) | PSNR: mean / sd → n | SSIM → n | LPIPS: mean / sd → n | corrected PSNR → n |
|---|---|---|---|---|
| w4a4 | +0.1359 / 0.5083 → **112** | 986 | +0.0108 / 0.0245 → **43** | 170,555 |
| w4a4_local | +0.1638 / 0.4951 → 74 | 117 | → 53 | 689 |
| w4a4_no_lwc | −0.0344 / 0.5346 → 1901 | inf | → 2046 | 384 |
| w4a4_no_smooth | +0.0181 / 0.4390 → 4605 | 1897 | → 248 | 10,098 |
| w4a4_no_lac | +0.1000 / 0.4730 → 178 | 106 | → 54 | 115,208 |
| w4a4_no_rot | +0.4316 / 0.6561 → 21 | 108 | → 33 | 4774 |
| w4a4_rtn | +0.9848 / 1.2067 → 14 | 34 | → 14 | 174 |

**PASS/FAIL:** Target: **state, for the headline null, what sample size would resolve it.** Full vs W4A4 on PSNR
needs ~112 scenes at the observed effect (we have 40); on exposure-corrected PSNR the observed effect is
effectively zero (170,555). **Reported.**

**Interpretation.** The W4A4 PSNR null is an *underpowered* null at the observed +0.136 dB — roughly 3× the scenes
would be needed — whereas the corrected-PSNR null is a genuine absence. Caveat: these are post-hoc numbers built on
observed effects, so they are planning figures for a follow-up, not evidence in themselves; for contrasts already
significant (LPIPS 43, no_rot 21, rtn 14) they are circular and only confirm consistency.

---

### #039 — Camera/point swap, 40 scenes: the rendering damage is in the cameras, not the points

- **Date/time:** 2026-09-11 10:34 → 15:08 CEST (GPU)
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/geometry_probes.py`
- **Command executed:**
  ```
  /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/geometry_probes.py swap \
      --arm w4a4_rtn --scenes 40 --deadline 00:15
  ```
  (launched by `code/quantization/run_probes_day.sh`)
- **Config:**

| Field | Value |
|---|---|
| Frame | full precision's world frame; `w4a4_rtn` carried in by Sim(3) on the six input camera centres |
| Conditions | A full cams + full pts (existing `full`); P full cams + rtn pts; K rtn cams (pose **and intrinsics**) + full pts; B' rtn cams + rtn pts |
| Evaluation cameras | full's own held-out source for **all four** cells |
| Init | stride-4 VGGT point grid, 101,400 points |
| Iterations | 7000 |
| Densify | GraphDECO default |
| Input views | 6 input, 9 held-out |
| Mask type | foreground |
| Background handling | GS default black, unmasked full frames |
| Depth regularisation | off |

  Offline check before launch (apple/110): K changes only cameras (points identical to full), P only points
  (cameras identical); rtn camera error on that scene 14.875°, point error 0.5426 radii.
- **Scenes used:** all 40 of the frozen manifest; 0 failed.

**Per-scene foreground results:**

| scene | A PSNR | P PSNR | K PSNR | B' PSNR | A LPIPS | P LPIPS | K LPIPS | B' LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| apple/110_13051_23361 | 12.670 | 12.425 | 9.940 | 9.190 | 0.6001 | 0.6064 | 0.6958 | 0.7021 |
| apple/189_20393_38136 | 6.539 | 7.450 | 5.874 | 6.279 | 0.7393 | 0.7136 | 0.7832 | 0.7689 |
| ball/123_14363_28981 | 11.224 | 11.116 | 8.986 | 8.610 | 0.5655 | 0.5425 | 0.6489 | 0.6361 |
| ball/375_42693_85518 | 10.229 | 10.420 | 10.363 | 10.065 | 0.6800 | 0.6653 | 0.7044 | 0.7015 |
| bench/415_57112_110099 | 12.687 | 11.735 | 11.549 | 10.860 | 0.6278 | 0.6512 | 0.6499 | 0.6653 |
| bench/415_57121_110109 | 11.121 | 11.094 | 10.263 | 9.191 | 0.6127 | 0.6204 | 0.6818 | 0.7123 |
| book/119_13962_28926 | 8.465 | 8.051 | 8.601 | 7.525 | 0.6617 | 0.6323 | 0.7511 | 0.7194 |
| book/247_26469_51778 | 10.197 | 9.705 | 9.423 | 8.909 | 0.6969 | 0.7027 | 0.7622 | 0.7846 |
| bowl/69_5465_12831 | 10.959 | 11.671 | 8.865 | 8.872 | 0.6136 | 0.6109 | 0.7357 | 0.7315 |
| bowl/70_5792_13401 | 10.458 | 5.684 | 7.380 | 5.684 | 0.6549 | 0.9102 | 0.8652 | 0.9102 |
| broccoli/372_41112_81867 | 8.211 | 8.312 | 7.929 | 7.766 | 0.7040 | 0.6969 | 0.7345 | 0.7222 |
| broccoli/412_56288_108844 | 9.338 | 9.656 | 7.946 | 8.810 | 0.7163 | 0.6956 | 0.7737 | 0.7720 |
| cake/374_42274_84517 | 7.306 | 6.445 | 6.237 | 5.732 | 0.6285 | 0.6554 | 0.6972 | 0.7150 |
| cake/403_53094_103680 | 7.532 | 7.867 | 6.778 | 6.542 | 0.7464 | 0.7330 | 0.7726 | 0.7829 |
| donut/391_47032_93657 | 8.537 | 9.193 | 8.862 | 8.438 | 0.7418 | 0.7314 | 0.7868 | 0.7831 |
| donut/403_52964_103416 | 8.792 | 7.833 | 10.360 | 8.949 | 0.6310 | 0.6188 | 0.5779 | 0.5933 |
| hydrant/167_18184_34441 | 8.131 | 7.694 | 7.385 | 7.123 | 0.7404 | 0.7637 | 0.7828 | 0.7831 |
| hydrant/411_56064_108483 | 8.130 | 8.217 | 8.927 | 8.601 | 0.7223 | 0.7003 | 0.6835 | 0.6612 |
| mouse/107_12753_23606 | 6.719 | 7.835 | 6.827 | 7.745 | 0.7095 | 0.6623 | 0.7279 | 0.7212 |
| mouse/377_43416_86289 | 7.819 | 7.634 | 5.672 | 6.084 | 0.6685 | 0.6654 | 0.7758 | 0.7491 |
| orange/374_42196_84367 | 9.734 | 9.543 | 7.485 | 8.043 | 0.7331 | 0.7753 | 0.8629 | 0.8552 |
| orange/385_45386_90752 | 8.716 | 8.366 | 8.014 | 8.269 | 0.6867 | 0.7046 | 0.7170 | 0.7182 |
| plant/247_26441_50907 | 9.124 | 9.529 | 7.038 | 7.110 | 0.6091 | 0.6046 | 0.6972 | 0.6835 |
| plant/374_42005_84358 | 8.735 | 9.732 | 7.962 | 8.402 | 0.6161 | 0.5883 | 0.6667 | 0.6368 |
| remote/195_20989_41543 | 9.215 | 9.392 | 8.662 | 8.025 | 0.6238 | 0.6252 | 0.6786 | 0.6913 |
| remote/350_36761_68623 | 9.656 | 9.712 | 9.581 | 9.433 | 0.6703 | 0.6395 | 0.7047 | 0.6954 |
| skateboard/245_26182_52130 | 9.546 | 8.844 | 8.451 | 7.424 | 0.5752 | 0.6158 | 0.7035 | 0.7320 |
| skateboard/366_39266_76077 | 9.450 | 9.846 | 10.087 | 9.654 | 0.6357 | 0.6299 | 0.6878 | 0.6826 |
| suitcase/410_55734_107452 | 8.462 | 7.885 | 6.816 | 6.166 | 0.6277 | 0.6536 | 0.7180 | 0.7525 |
| suitcase/50_2928_8645 | 9.317 | 9.095 | 7.595 | 7.835 | 0.6498 | 0.6506 | 0.7191 | 0.6920 |
| teddybear/187_20215_38541 | 9.341 | 10.649 | 7.349 | 7.768 | 0.6020 | 0.5376 | 0.6700 | 0.6418 |
| teddybear/34_1479_4753 | 7.200 | 8.210 | 7.249 | 7.563 | 0.7102 | 0.7272 | 0.7433 | 0.7685 |
| toaster/372_41229_82130 | 8.949 | 8.183 | 7.141 | 6.954 | 0.6932 | 0.6656 | 0.7222 | 0.7344 |
| toaster/416_57389_110765 | 8.631 | 8.617 | 7.652 | 8.082 | 0.6369 | 0.6298 | 0.7075 | 0.7011 |
| toytrain/240_25394_51994 | 8.422 | 8.292 | 9.188 | 8.295 | 0.6894 | 0.9813 | 0.9453 | 0.9808 |
| toytrain/399_51323_100753 | 4.455 | 4.275 | 4.085 | 5.135 | 0.8379 | 0.8354 | 0.8447 | 0.8144 |
| toytruck/190_20494_39385 | 10.355 | 10.128 | 7.465 | 7.151 | 0.5123 | 0.5404 | 0.6021 | 0.6354 |
| toytruck/346_36113_66551 | 7.048 | 7.163 | 6.105 | 6.060 | 0.6621 | 0.6748 | 0.7204 | 0.7457 |
| vase/374_41862_83720 | 6.780 | 6.609 | 6.795 | 6.771 | 0.6182 | 0.6357 | 0.6581 | 0.6659 |
| vase/380_44863_89631 | 10.372 | 11.342 | 6.796 | 7.575 | 0.5649 | 0.5657 | 0.7163 | 0.6978 |

**Means and paired tests (n = 40; positive Δ = condition worse than A):**

| condition | PSNR | SSIM | LPIPS | Δ PSNR | p | Δ LPIPS | p | sign |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A full | 8.9642 | 0.2318 | 0.6604 | | | | | |
| P bad points | 8.8862 | 0.2277 | 0.6715 | +0.0780 | 0.61 | +0.0111 | 0.2893 | 22/40 |
| K bad cameras | 7.9922 | 0.2124 | 0.7269 | +0.9721 | 5.068e-06 | +0.0665 | 5.783e-09 | 30/40 |
| B' both | 7.8172 | 0.2090 | 0.7285 | +1.1470 | 7.852e-07 | +0.0681 | 9.53e-08 | 34/40 |

Additivity (PSNR): points +0.0780 + cameras +0.9721 vs both +1.1470; interaction +0.0969.

**Robustness (final pass of `cpu_probes.py robustness`, 15:09–15:10):** full − P: boot CI95 [−0.1781, +0.3999],
perm p 0.694, no-bowl −0.0424 / p 0.66. full − K: CI95 [+0.6206, +1.3219], perm p 5e-05 (floor), worst
leave-one-out p 1.1e-05, no-bowl +0.9181 / p 1e-05. full − B': CI95 [+0.7768, +1.5273], perm p 5e-05.

**PASS/FAIL:** Target: **attribute `w4a4_rtn`'s rendering loss to its cameras or its points, with both
effects measured under identical evaluation cameras.** Cameras alone: +0.9721 dB, p = 5.1e-06, robust to every
test. Points alone: +0.0780 dB, p = 0.61, CI spanning zero. Cameras account for 0.9721 / 1.1470 = **85 %** of the
joint loss. **PASS.**

**Interpretation.** The geometry prior's *points* are close to irrelevant to held-out rendering here; its
*cameras* carry the damage. This is why #030's perfect point-level confidence recovered nothing. Caveat carried
forward: K replaces pose and intrinsics together, and `rtn`'s focal is 18.652 % off (#031), so "cameras" is not
yet split into pose vs focal — `geometry_probes.py camsplit` is queued for that.

---

### #040 — Mechanism from the trained models: final 3D accuracy and render quality are decoupled

- **Date/time:** 2026-09-11; opacity > 0.5 run 15:09 (in `run_cpu_chain.sh`), opacity > 0.1 re-run finished 21:09:03 CEST (CPU)
- **Git commit:** `f0c8e32` + working tree (`cpu_probes.py gaussians`)
- **Commands executed:**
  ```
  nice -n 15 env OMP_NUM_THREADS=1 /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/cpu_probes.py gaussians --workers 8
  GS_OPACITY_MIN=0.1 nice -n 15 env OMP_NUM_THREADS=1 /home/utn/luli38se/cv/.venv/bin/python -u \
      code/quantization/cpu_probes.py gaussians --workers 8
  ```
- **Config:** no training. Reads every trained 7000-iteration PLY for full, w4a4, w4a4_rtn, conf50/rand50, and the
  three swap cells, plus their initial `points3D.txt`. CO3D's GT `pointcloud.ply` (≤ 60,000 points) carried into
  full's frame by Sim(3) from GT to full input camera centres (sanity: median GT → full-init distance 0.011 radii).
  Arm-frame models carried into full's frame by their own Sim(3). "Surface" Gaussians = sigmoid(opacity) above the
  threshold (≈ 2–3 % of Gaussians at 0.5). **Accuracy** = median distance surface → GT inside the GT bounding box
  (+10 %); **completeness** = median distance GT → surface; scene radii. Two scenes (`bowl/70_5792_13401`,
  `toytrain/240_25394_51994`) have no surface Gaussians inside the GT box for some models and are excluded from
  the accuracy pairs (n = 38). The first run's printed paired means were NaN for this reason; the tests below were
  recomputed NaN-aware from `cpu_gaussians*.json`.
- **Scenes used:** all 40 (swap cells 40; conf/rand 36).

**Paired tests (median difference, Wilcoxon p, positive = first worse):**

| comparison | opacity > 0.5 | opacity > 0.1 |
|---|---|---|
| swapP final accuracy − full final accuracy | +0.0320, p 1.51e-09, 36/38 | +0.0292, p 1.46e-11, 37/38 |
| swapK final accuracy − full final accuracy | +0.0022, p 0.335, 23/38 | +0.0028, p 0.0844, 24/38 |
| swapP final completeness − full | +0.0264, p 7.28e-12, 38/38 | +0.0161, p 7.28e-12, 38/38 |
| swapK final completeness − full | +0.0043, p 3.25e-05, 30/39 | +0.0020, p 4.82e-06, 31/39 |
| swapP final accuracy − swapP initial accuracy | −0.0008, p 0.688, 18/38 | −0.0047, p 0.22, 16/38 |

Initial-vs-final *completeness* is **not** compared: the final surface set is far sparser than the 101,400-point
initialisation, so that difference would measure density, not geometry.

**PASS/FAIL:**
- Hypothesis stated in conversation and in this file's `gaussians` docstring — *"3DGS moves bad points to the right
  place, which is why bad points do not hurt rendering"* — **FAIL, refuted at both thresholds**: training does not
  improve bad-point accuracy (p 0.688 / 0.22), and the bad-point model ends with clearly worse 3D geometry than full.
- Target: **does final 3D accuracy explain the rendering results of #039?** No. swapP: worse geometry, renders like
  full. swapK: geometry like full, renders ~1 dB worse. **Dissociation, robust to the threshold.**

**Interpretation.** In this 6-view CO3D setting, held-out rendering quality is **decoupled from the 3D accuracy
of the reconstruction**: it is governed by whether the training cameras are right, not by whether the Gaussians
are in the right place. That explains three earlier results at once — the oracle losing to a nearest-photograph
baseline (#022), point-level confidence recovering nothing (#030), and bad points costing nothing (#039) — and
it means rendering PSNR/SSIM/LPIPS on this benchmark cannot, by construction, reward better geometry priors. It
is the strongest single finding for the experimental report.

---

### #041 — Final robustness pass (swap at n = 40)

- **Date/time:** 2026-09-11 15:09–15:10 CEST (CPU, `run_cpu_chain.sh`)
- **Command executed:** `nice -n 15 env OMP_NUM_THREADS=1 /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/cpu_probes.py robustness --workers 8`
- **Config / scenes:** identical to #032, with the swap JSON now complete (n = 40). Output `cpu_robustness.json`
  (overwrites pass 1; pass-1 numbers are preserved in #032).
- **Results:** every non-swap row identical to #032. Swap rows as reported under #039.
- **PASS/FAIL:** Target: **the interim n = 29 swap conclusion survives at n = 40.** Points null (perm p 0.694),
  camera effect robust (worst leave-one-out p 1.1e-05; without the bowl +0.9181, p 1e-05). **PASS.**
- **Interpretation:** the interim swap numbers in #032 are superseded by #039; the conclusion is unchanged.

---

### #042 — Synthetic damage sweep: renders tolerate ~2° of camera error and a full scene-radius of point noise

- **Date/time:** 2026-09-11 15:08 → 21:11 CEST (GPU)
- **Git commit:** `f0c8e32` + working tree (`code/quantization/geometry_probes.py`)
- **Command executed:**
  ```
  /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/geometry_probes.py sweep --scenes 12 --deadline 00:15
  ```
  (launched by `code/quantization/run_probes_day.sh`)
- **Config:**

| Field | Value |
|---|---|
| Base geometry | full-precision VGGT (cameras, intrinsics, stride-4 points) |
| cameraR<deg> | every training camera rotated by exactly <deg> about its own centre, random axis, seeded per scene; centres, intrinsics, points untouched. Levels 0.5, 1, 2, 4, 8, 16° |
| points<σ> | isotropic Gaussian jitter on every initial point, median displacement σ scene radii (ablation_compare's point_err statistic); cameras untouched. Levels 0.03, 0.06, 0.12, 0.25, 0.5, 1.0 |
| Evaluation cameras | full's held-out source |
| Iterations / densify / views / mask / background / depth reg | 7000 / GraphDECO default / 6 in, 9 held-out / foreground / black, unmasked / off |

  Offline check before launch: realised rotations within 0.0006° of target, centre shift ≤ 7.8e-08; realised
  median point shifts 0.0300–0.9988 radii for targets 0.03–1.0.
- **Scenes used:** the first 12 of the frozen manifest: apple/110_13051_23361, apple/189_20393_38136,
  ball/123_14363_28981, ball/375_42693_85518, bench/415_57112_110099, bench/415_57121_110109,
  book/119_13962_28926, book/247_26469_51778, bowl/69_5465_12831, bowl/70_5792_13401,
  broccoli/372_41112_81867, broccoli/412_56288_108844. Per-scene values in `geometry_probe_sweep_it7000.json`.

**Results (n = 12; positive Δ = worse than full; per-level p are paired t, 12 levels tested):**

| condition | PSNR | Δ PSNR | p | LPIPS | Δ LPIPS | p | worse in |
|---|---:|---:|---:|---:|---:|---:|---|
| full | 10.1748 | | | 0.6561 | | | |
| cameraR 0.5° | 10.1223 | +0.0524 | 0.545 | 0.6571 | +0.0011 | 0.7728 | 7/12 |
| cameraR 1° | 10.0624 | +0.1124 | 0.4481 | 0.6581 | +0.0020 | 0.5142 | 7/12 |
| cameraR 2° | 10.1560 | +0.0187 | 0.8943 | 0.6583 | +0.0022 | 0.709 | 5/12 |
| cameraR 4° | 9.6575 | +0.5173 | 0.05027 | 0.6858 | +0.0297 | 0.005114 | 8/12 |
| cameraR 8° | 9.4156 | +0.7592 | 0.0264 | 0.7030 | +0.0469 | 0.001629 | 9/12 |
| cameraR 16° | 9.1731 | +1.0016 | 0.0003206 | 0.7268 | +0.0708 | 3.308e-07 | 12/12 |
| points 0.03 | 10.1386 | +0.0362 | 0.6909 | 0.6561 | +0.0000 | 0.9838 | 5/12 |
| points 0.06 | 10.2980 | −0.1232 | 0.3385 | 0.6549 | −0.0012 | 0.7478 | 6/12 |
| points 0.12 | 10.2274 | −0.0526 | 0.6571 | 0.6544 | −0.0016 | 0.6066 | 6/12 |
| points 0.25 | 10.2430 | −0.0682 | 0.5584 | 0.6642 | +0.0081 | 0.07343 | 5/12 |
| points 0.5 | 10.4574 | −0.2827 | 0.03602 | 0.6586 | +0.0026 | 0.6739 | 2/12 |
| points 1.0 | 10.3345 | −0.1597 | 0.4057 | 0.6665 | +0.0104 | 0.0373 | 5/12 |

**Trend across levels, one Spearman ρ per scene, Wilcoxon across the 12 scenes** (the pooled scenes × levels
Spearman printed by an ad-hoc check treated 72 correlated pairs as independent and is **not** cited):

| family | PSNR: median ρ, positive, p | LPIPS: median ρ, positive, p |
|---|---|---|
| camera rotation | +0.486, 11/12, 0.000977 | +0.743, 11/12, 0.000977 |
| point jitter | −0.229, 4/12, 0.212 | +0.514, 10/12, 0.0415 |

**PASS/FAIL:**
- Target: **locate the camera-error level at which rendering measurably degrades.** No effect at 0.5–2°
  (p ≥ 0.45); LPIPS significant from 4° (p 0.0051), PSNR from 8° (p 0.026); monotone trend p 0.000977. **Threshold
  lies between 2° and 4°.** Measured W4A4 camera error, 1.2754° (#029), is **below** it. **PASS.**
- Target: **the same for point error.** Up to 1.0 scene radius — more than twice `w4a4_rtn`'s 0.46422 — PSNR shows
  no degradation (trend p 0.212; one level, 0.5, is nominally *better* at p 0.036 of 12 tests, not significant after
  correction); LPIPS shows a weak upward trend (p 0.0415). **Points: effectively no threshold in range.**

**Interpretation.** This places every measured model on one axis and explains the W4A4 null from first principles:
rendering here is insensitive to camera rotation below ~2–4° and to point error up to a full scene radius, and
W4A4's cameras sit at 1.28° — inside the tolerance — so its rendering matches full precision. `w4a4_rtn`'s
14.66° sits past it, and pure 16° rotation costs +1.0016 dB, the same order as `rtn`'s camera-swap loss of +0.9721
dB (#039), consistent with — but not proof of — pose rather than focal length carrying `rtn`'s damage (the camsplit
probe tests that directly). Caveat: 12 scenes, the first 12 in manifest order, including the unstable bowl.

---

### #043 — Pose vs focal length: both hurt, pose more, and pose alone accounts for the perceptual loss

- **Date/time:** 2026-09-11 21:11 → 2026-09-12 00:11 CEST (GPU); robustness pass 2026-09-12 09:07
- **Git commit:** `f0c8e32` + working tree (`code/quantization/geometry_probes.py`, `run_camsplit_after.sh`,
  `run_camsplit_finish.sh`)
- **Commands executed:**
  ```
  /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/geometry_probes.py camsplit \
      --arm w4a4_rtn --scenes 40 --deadline 00:15          # via run_camsplit_after.sh
  nice -n 15 /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/cpu_probes.py robustness
  ```
  A resume path (`--resume`, `run_camsplit_finish.sh`) was queued in case the 00:15 guard cut the run short. It
  was **not needed**: the first run finished all 40 scenes at 00:11 and the continuation exited with nothing to do.
- **Config:** exactly #039's, except the two conditions, both built in full precision's frame and rendered from
  full's held-out source:

| condition | cameras | intrinsics | points |
|---|---|---|---|
| R (pose only) | `w4a4_rtn` pose, Sim(3)-carried | **full** | full |
| F (focal only) | **full** | `w4a4_rtn` | full |

  Offline check before launch: R leaves intrinsics equal to full's and changes pose; F leaves extrinsics equal to
  full's and changes intrinsics. `w4a4_rtn`'s focal error is 18.652 % and its camera rotation 14.6565° (#031, #029).
- **Scenes used:** all 40 of the frozen manifest; 0 failed.

**Per-scene foreground results:**

| scene | full PSNR | R PSNR | F PSNR | full LPIPS | R LPIPS | F LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| apple/110_13051_23361 | 12.670 | 10.427 | 10.297 | 0.6001 | 0.6906 | 0.6784 |
| apple/189_20393_38136 | 6.539 | 6.072 | 6.432 | 0.7393 | 0.8028 | 0.7755 |
| ball/123_14363_28981 | 11.224 | 8.957 | 10.294 | 0.5655 | 0.6427 | 0.5953 |
| ball/375_42693_85518 | 10.229 | 10.357 | 10.017 | 0.6800 | 0.7014 | 0.7161 |
| bench/415_57112_110099 | 12.687 | 11.776 | 12.067 | 0.6278 | 0.6377 | 0.6300 |
| bench/415_57121_110109 | 11.121 | 9.925 | 11.022 | 0.6127 | 0.6952 | 0.6079 |
| book/119_13962_28926 | 8.465 | 8.310 | 8.397 | 0.6617 | 0.7351 | 0.7036 |
| book/247_26469_51778 | 10.197 | 9.277 | 9.811 | 0.6969 | 0.7808 | 0.6997 |
| bowl/69_5465_12831 | 10.959 | 9.766 | 10.045 | 0.6136 | 0.7195 | 0.6167 |
| bowl/70_5792_13401 | 10.458 | 7.380 | 10.443 | 0.6549 | 0.8652 | 0.6576 |
| broccoli/372_41112_81867 | 8.211 | 7.351 | 8.458 | 0.7040 | 0.7352 | 0.7044 |
| broccoli/412_56288_108844 | 9.338 | 9.477 | 9.608 | 0.7163 | 0.7618 | 0.6909 |
| cake/374_42274_84517 | 7.306 | 7.398 | 6.197 | 0.6285 | 0.6711 | 0.6661 |
| cake/403_53094_103680 | 7.532 | 7.156 | 7.377 | 0.7464 | 0.7636 | 0.7461 |
| donut/391_47032_93657 | 8.537 | 7.830 | 8.683 | 0.7418 | 0.8178 | 0.7653 |
| donut/403_52964_103416 | 8.792 | 8.508 | 10.683 | 0.6310 | 0.6171 | 0.5718 |
| hydrant/167_18184_34441 | 8.131 | 7.456 | 8.087 | 0.7404 | 0.7666 | 0.7694 |
| hydrant/411_56064_108483 | 8.130 | 8.595 | 7.867 | 0.7223 | 0.6992 | 0.7057 |
| mouse/107_12753_23606 | 6.719 | 7.463 | 7.052 | 0.7095 | 0.7105 | 0.6990 |
| mouse/377_43416_86289 | 7.819 | 6.431 | 7.187 | 0.6685 | 0.7644 | 0.6685 |
| orange/374_42196_84367 | 9.734 | 8.645 | 7.985 | 0.7331 | 0.8520 | 0.8414 |
| orange/385_45386_90752 | 8.716 | 9.471 | 8.062 | 0.6867 | 0.6692 | 0.7227 |
| plant/247_26441_50907 | 9.124 | 7.676 | 7.518 | 0.6091 | 0.6893 | 0.6767 |
| plant/374_42005_84358 | 8.735 | 9.089 | 7.284 | 0.6161 | 0.6395 | 0.6628 |
| remote/195_20989_41543 | 9.215 | 8.790 | 8.568 | 0.6238 | 0.6962 | 0.6648 |
| remote/350_36761_68623 | 9.656 | 9.613 | 9.650 | 0.6703 | 0.7105 | 0.6491 |
| skateboard/245_26182_52130 | 9.546 | 8.396 | 9.509 | 0.5752 | 0.6954 | 0.6149 |
| skateboard/366_39266_76077 | 9.450 | 10.275 | 9.671 | 0.6357 | 0.6764 | 0.6381 |
| suitcase/410_55734_107452 | 8.462 | 6.700 | 8.508 | 0.6277 | 0.7267 | 0.6486 |
| suitcase/50_2928_8645 | 9.317 | 8.090 | 8.494 | 0.6498 | 0.6983 | 0.6880 |
| teddybear/187_20215_38541 | 9.341 | 6.897 | 8.963 | 0.6020 | 0.6739 | 0.5955 |
| teddybear/34_1479_4753 | 7.200 | 7.185 | 6.928 | 0.7102 | 0.7460 | 0.7220 |
| toaster/372_41229_82130 | 8.949 | 7.113 | 8.678 | 0.6932 | 0.7327 | 0.6935 |
| toaster/416_57389_110765 | 8.631 | 8.670 | 8.066 | 0.6369 | 0.7157 | 0.6689 |
| toytrain/240_25394_51994 | 8.422 | 9.238 | 7.620 | 0.6894 | 0.9396 | 0.6983 |
| toytrain/399_51323_100753 | 4.455 | 4.134 | 4.989 | 0.8379 | 0.8557 | 0.8247 |
| toytruck/190_20494_39385 | 10.355 | 8.350 | 6.695 | 0.5123 | 0.5662 | 0.6214 |
| toytruck/346_36113_66551 | 7.048 | 6.914 | 6.842 | 0.6621 | 0.7066 | 0.6722 |
| vase/374_41862_83720 | 6.780 | 7.589 | 5.876 | 0.6182 | 0.6286 | 0.6520 |
| vase/380_44863_89631 | 10.372 | 8.051 | 9.135 | 0.5649 | 0.6689 | 0.6168 |

**Means and paired tests (n = 40; positive Δ = worse than full):**

| condition | PSNR | SSIM | LPIPS | Δ PSNR | p | Δ LPIPS | p | sign |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| full | 8.9642 | 0.2318 | 0.6604 | | | | | |
| F focal only | 8.4767 | 0.2220 | 0.6810 | +0.4876 | 0.00125 | +0.0206 | 0.0004171 | 32/40 |
| R pose only | 8.2700 | 0.2199 | 0.7216 | +0.6943 | 0.0001121 | +0.0612 | 1.325e-08 | 29/40 |
| K both (from #039) | 7.9922 | 0.2124 | 0.7269 | +0.9721 | 5.068e-06 | +0.0665 | 5.783e-09 | 30/40 |

**Robustness (`cpu_probes.py robustness`, 2026-09-12 09:07):** R PSNR boot CI95 [+0.3875, +1.0094], perm p 0.00045,
worst leave-one-out p 0.000228, no-bowl +0.6331 / p 0.00019. F PSNR CI95 [+0.2265, +0.7641], perm p 0.00085,
worst LOO p 0.00239, no-bowl +0.4997 / p 0.0012. R LPIPS CI95 [+0.0454, +0.0784], perm p 5e-05 (floor).
F LPIPS CI95 [+0.0102, +0.0311], perm p 0.0003.

**PASS/FAIL:**
- Target: **split #039's camera damage into pose and focal length.** Both are real and robust. PSNR: pose +0.6943,
  focal +0.4876, jointly +0.9721 — **sub-additive** (sum 1.1819 vs 0.9721). LPIPS: pose +0.0612 vs joint +0.0665,
  focal only +0.0206 — **pose alone reproduces 92 % of the joint perceptual loss.** **PASS.**
- The caveat carried since #031 and #039 — *"cameras" is not yet split* — is now **resolved**.

**Interpretation.** Both parts of a wrong camera hurt, and neither is negligible: an 18.65 % focal error alone costs
0.49 dB even with perfect pose and perfect points. But pose is the larger term and dominates the perceptual metric.
Combined with #042 — where pure 16° rotation with correct centres, intrinsics and points cost +1.0016 dB, close to
the +0.9721 dB of `rtn`'s full camera swap — the picture is consistent: `rtn`'s rendering loss is mostly its camera
orientation. For the proposal this narrows the target further: if anything is worth predicting or correcting in a
quantized geometry prior, it is the **camera pose**, then intrinsics; points remain irrelevant (#039, #042).
Caveat: measured on one heavily damaged arm (`w4a4_rtn`); at W4A4's own 1.28° / 2.6 % the sweep (#042) says both
are below the detection threshold.

---

### #044 — Viewpoint sensitivity: geometry damage is largest on the CLOSEST held-out views, not the farthest

- **Date/time:** 2026-09-12, launched 09:16, `viewpoint_sensitivity.json` written before 09:33 CEST (CPU)
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/viewpoint_sensitivity.py`
- **Command executed:**
  ```
  nice -n 15 env OMP_NUM_THREADS=2 /home/utn/luli38se/cv/.venv/bin/python -u \
      code/quantization/viewpoint_sensitivity.py
  ```
- **Hypothesis under test (mine, stated in conversation):** *the benchmark's insensitivity to geometry is an
  artefact of interpolation; damage should grow on held-out views far from the training cameras, and evaluating on
  distant views would restore sensitivity.*
- **Config:** no training, no rendering — re-scores the existing 7000-iteration held-out renders per view.
  Foreground mask, PSNR and LPIPS (AlexNet on the mask bounding box). Distances from each held-out camera to the
  **nearest** training camera: `axis_angle` (between optical axes) and `orbit_angle` (subtended at the training-camera
  centroid). Conditions: w4a4, w4a4_rtn, and the swap cells P/K/B/R/F from #039 and #043, each compared with `full`.
  Test: per-scene Spearman between view distance and per-view damage, then Wilcoxon over the 40 scenes — never a
  pooled correlation over scene × view pairs (the pseudo-replication corrected in #042).
- **Scenes used:** all 40, 9 held-out views each. Distance spread: orbit angle median 16.43°, range 0.01–77.94°;
  axis angle median 10.39°.

**Results (median per-scene ρ; positive = damage grows with distance):**

| condition | PSNR: ρ, scenes positive, p | LPIPS: ρ, scenes positive, p |
|---|---|---|
| w4a4 | −0.108, 15/40, 0.076 | −0.192, 13/40, 0.0252 |
| w4a4_rtn | −0.342, 7/40, 4.25e-06 | −0.508, 6/40, 2.35e-06 |
| swapP (bad points) | +0.025, 21/40, 0.861 | +0.042, 21/40, 0.534 |
| swapK (bad cameras) | −0.425, 2/40, 7.02e-08 | −0.592, 4/40, 3.35e-07 |
| swapB (both) | −0.367, 3/40, 3.63e-07 | −0.525, 6/40, 3.13e-07 |
| swapR (pose only) | −0.408, 4/40, 3.59e-06 | −0.592, 4/40, 5.88e-07 |
| swapF (focal only) | −0.283, 14/40, 0.00411 | −0.252, 8/40, 0.000189 |

(orbit angle shown; the axis-angle column in `viewpoint_sensitivity.json` agrees throughout.)

**Floor-effect check.** `full` itself degrades with distance: PSNR ρ = −0.508 (2/40 positive, p 1.69e-07),
LPIPS ρ = +0.692 (40/40, p 3.54e-08). Normalising damage by full's own score does **not** reverse the trend:
relative PSNR damage for swapK ρ = −0.408 (p 1.32e-07), relative LPIPS damage ρ = −0.625 (p 4.36e-07).

**PASS/FAIL:** Target: **damage grows with held-out view distance (positive ρ).** Measured ρ is **negative** and
significant for every camera-damage condition, in both absolute and relative form. **FAIL — the hypothesis is
refuted, with the sign reversed.** Point damage remains flat at every distance (p 0.86).

**Interpretation.** Evaluating on more distant viewpoints would **reduce**, not increase, the benchmark's
sensitivity to geometry error. Distant views are poor for every arm — full precision included — so all arms
converge toward a common floor and the gap between them closes; the differences live in the near views, where a
correct model renders well and a camera-damaged one no longer can. This rules out "evaluate further away" as a fix
for the insensitivity documented in #039–#043, and it is a second, independent demonstration that the limitation is
in what sparse-view 3DGS can render at all, not in where the held-out cameras were placed. Caveat: CO3D held-out
views come from the same turntable orbit; a capture with genuinely novel elevations might behave differently, and
that is untested here.

---

### #045 — Input-order ensembling gives a deployable confidence signal for camera pose (ρ = 0.88)

- **Date/time:** 2026-09-12, w4a4 09:20–09:26, full 09:26–09:30, w4a4_rtn from 09:30 CEST (GPU)
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/pose_uncertainty.py`, `run_pose_uncertainty.sh`
- **Commands executed:**
  ```
  ./code/quantization/run_pose_uncertainty.sh
  #   -> pose_uncertainty.py --variant w4a4    --perms 4 --scenes all
  #   -> pose_uncertainty.py --variant full     --perms 4 --scenes all
  #   -> pose_uncertainty.py --variant w4a4_rtn --perms 4 --scenes all
  ```
- **Config:** inference only, no 3DGS. Per scene, the same six 518×518 pad-mode images are run through the model in
  4 orderings (identity + 3 seeded shuffles); outputs are un-permuted, then every ordering is carried into ordering
  0's frame by Sim(3) on the six camera centres (a different reference frame is not an error). *Spread* = mean over
  the six cameras of the mean pairwise rotation angle across orderings. *Error* = ordering 0's pose error against
  full precision's saved cameras, and against CO3D ground truth, by the same construction. Frozen manifest SHA
  verified at start. Cost: 4× inference, ~10 s/scene.
- **Scenes used:** all 40.

**Determinism check:** full precision, canonical ordering, vs the saved `full.npz` cameras — max 2.37e-02°,
median 8.30e-03°. The pipeline is reproducible, so the spread below is genuinely order-induced.

**Results (n = 40):**

| variant | spread median (range) | spread vs error-vs-full | vs GT error |
|---|---|---|---|
| w4a4 | 1.1389° (0.4452–11.1944) | **ρ +0.880, p 6.97e-14** (no bowl +0.871, p 5.59e-13) | ρ +0.764, p 9.73e-09 |
| full (control) | 0.6084° (0.2715–6.4497) | ρ −0.062, p 0.703 (this target is ~0 by construction) | ρ +0.709, p 3.02e-07 |

Quantization increases order-sensitivity: w4a4's spread exceeds full's in **39/40** scenes, Wilcoxon p = 4.32e-09.
Excess spread (w4a4 − full) vs w4a4's pose error: ρ = +0.751, p = 2.37e-08. Full's spread alone vs w4a4's pose
error: ρ = +0.391, p = 0.0125.

**PASS/FAIL:**
- Target: **a signal computable from the quantized model alone predicts its camera-pose error.** ρ = 0.880,
  p = 7e-14, robust to removing the bowl scene. **PASS** — and it is the first confidence signal in this project
  that works, after per-point true error (#030) and cross-view self-consistency (#036) both failed.
- Target: **is it specific to quantization, or just a scene-ambiguity detector?** Partly the latter: full
  precision's own spread predicts its GT error (ρ +0.709) and predicts w4a4's error (ρ +0.391). But the
  **excess** spread — the part quantization adds — predicts quantization's pose error at ρ = 0.751.
  **PASS with the caveat stated.**

**Interpretation.** VGGT's output depends on input order, and the disagreement across orderings is a usable
uncertainty estimate: it needs no ground truth, no full-precision reference, and no training, and it targets camera
pose — the quantity #039/#043 show carries the rendering damage. Two honest limits. It costs 4× inference, which
must be weighed against the proposal's efficiency clause (a cheaper 2-ordering variant is untested). And it is not
purely a quantization signal: unstable scenes are unstable for full precision too, so part of the correlation is
"this scene is ambiguous" rather than "quantization broke this scene" — the excess-spread result is the way to
state the quantization-specific part. **What it does not yet show is that acting on the signal helps**: at W4A4 the
pose error it predicts, 1.28° on average, is below the 2–4° rendering-damage threshold of #042, so there is nothing
to gain downstream at this bit-width. Its value would be at W3A3 and below.

---

### #046 — Order-ensembling on `w4a4_rtn`, and COLMAP refinement of a damaged prior: BLOCKED by SfM failure

- **Date/time:** 2026-09-12; `w4a4_rtn` ensembling 09:30–09:34; refinement final run finished before 09:38 CEST
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/pose_refine.py`
- **Commands executed:**
  ```
  /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/pose_uncertainty.py \
      --variant w4a4_rtn --perms 4 --scenes all
  nice -n 15 /home/utn/luli38se/cv/.venv/bin/python -u code/quantization/pose_refine.py \
      refine --arm w4a4_rtn --scenes 40
  ```

**Part A — order-ensembling extends to the heavily damaged arm (n = 40).** Spread median 6.6314°
(range 3.1576–130.6178), against w4a4's 1.1389° and full's 0.6084° (#045). Spread vs pose error against full:
ρ = +0.678, p = 1.56e-06 (no bowl +0.652, p 6.78e-06); vs GT error ρ = +0.637, p 9.99e-06. **PASS:** the signal
holds at 14.66° of damage, though less tightly than at W4A4 (ρ 0.880).

**Part B — can classical SfM refine the damaged cameras?**

- **Config:** COLMAP via pycolmap 4.2.0 on the six 518×518 input images per scene: SIFT extraction and exhaustive
  matching on **CPU** (`device=pycolmap.Device.cpu`, so as not to contend with GPU jobs), `PER_IMAGE` PINHOLE
  cameras with intrinsics **fixed to the quantized model's prediction**, then `incremental_mapping`. No ground
  truth and no full-precision model are used — only what is available at deployment. Pose error is scored on
  exactly the subset of cameras COLMAP registered, against the same subset of full/GT cameras, by the usual
  Sim(3)-on-centres construction. A scene counts as usable only if ≥ 4 of 6 images register.
- **Scenes used:** all 40.

| registered images | 0 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| scenes | **28** | 3 | 2 | 1 | 1 | 5 |

Usable: **7/40**. On those 7: pose error vs full, arm median 5.878° → refined **19.757°**, improved in **1/7**,
Wilcoxon p = 0.0781; vs GT, 4.842° → 19.332°, improved in 1/7, under 4° goes from 2 scenes to 0. Full precision's
own error vs GT on the same scenes: median 1.223°. Individual runs show both outcomes — one scene improved
14.87° → 3.30°, another degraded 5.38° → 128.66°.

**PASS/FAIL:** Target: **classical refinement pulls the arm's 14.66° pose error under the ~4° tolerance of #042.**
COLMAP fails to reconstruct at all on 28/40 scenes, is usable on 7, and on those it makes pose error worse in 6 of
7. **FAIL.** Stage 2 (`pose_refine.py build`, 3DGS on refined cameras) was **not run**: with 7 scenes and poses
mostly worse than the input, it could not produce a meaningful rendering comparison. Recording that as a decision,
not a result.

**Bugs found and fixed during this run, listed so the numbers above are traceable:** (1) `cam_from_world` is a
method in pycolmap 4.2, not a property — first attempt reported 0/40 successes for that reason alone; (2) requiring
all six images to register discarded the common 5/6 case, so scoring now uses the registered subset; (3) a
`use_gpu` attribute that does not exist in this version turned every scene into an exception; (4) the
"no reconstruction" return path returned two values where three were expected, which mislabelled 28 genuine
COLMAP failures as exceptions. The table above is from the run after all four fixes. COLMAP's mapper is not
deterministic — an earlier post-fix run gave 8/40 usable and a 16.487° refined median — so these counts should be
read as approximate, not exact.

**Interpretation.** On six wide-baseline views of a mostly textureless object, structure-from-motion usually
produces nothing, and when it does produce something it is typically worse than the quantized prior it was meant to
fix. This is the regime feed-forward models exist for, and it means "just refine the poses classically" is not
available as a correction step here. It also raises the bar for the confidence signal of #045: a detector is only
useful with a corrector, and the obvious corrector does not work. Untested alternatives that remain: photometric
pose refinement inside 3DGS training (needs a rasterizer with camera gradients, which GraphDECO's default does not
expose), and re-running the feed-forward model with more input views.

---

### #047 — Averaging input orderings corrects pose error: the corrector #046 was missing

- **Date/time:** 2026-09-12; w4a4_rtn 09:57–10:01, w4a4 10:01–10:08, full 10:08–10:11 CEST (GPU)
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/pose_ensemble.py`, `run_pose_ensemble.sh`
- **Command executed:**
  ```
  ./code/quantization/run_pose_ensemble.sh
  #   -> pose_ensemble.py --variant {w4a4_rtn,w4a4,full} --perms 4 --scenes all --save-extrinsics
  ```
- **Config:** inference only. Four orderings per scene (identity + 3 seeded shuffles) of the same six
  518×518 pad-mode images; outputs un-permuted; every ordering carried into ordering 0's frame by Sim(3) on the
  six camera centres. **Ensemble pose** = chordal mean rotation (arithmetic mean projected back to SO(3) by SVD,
  determinant forced positive) and mean camera centre, per camera. Intrinsics are **not** ensembled. Error =
  mean per-camera rotation angle against full precision's saved cameras and against CO3D ground truth. Frozen
  manifest SHA checked at start. Cost 4× inference, ~6.2 s/scene.
- **Scenes used:** all 40.

**Results (median over scenes; "reduction" is the median of the per-scene relative reduction):**

| variant | target | single | ensemble | improved | median reduction | Wilcoxon p |
|---|---|---:|---:|---|---:|---:|
| w4a4_rtn | vs full | 6.0741° | **4.8713°** | 31/40 | +15.8 % | 5.28e-07 |
| w4a4_rtn | vs full, no bowl | 6.0306° | 4.8254° | 30/39 | +16.7 % | 1.05e-06 |
| w4a4_rtn | vs GT | 6.2565° | 4.7472° | 32/40 | +12.3 % | 7.14e-07 |
| w4a4 | vs full | 1.1162° | **0.8557°** | 34/40 | +23.8 % | 7.71e-05 |
| w4a4 | vs GT | 1.7123° | 1.3210° | 32/40 | +14.4 % | 5.08e-05 |
| full (control) | vs full | 0.0083° | 0.4546° | 0/40 | −4687 % | 1.82e-12 |
| full (control) | vs GT | 1.1121° | 1.1193° | 27/40 | +6.4 % | 0.0278 |

**PASS/FAIL:**
- Target: **a corrector that reduces the pose error #045 detects, using only the quantized model.** Pose error
  falls for both quantized arms, on both references, at p ≤ 8e-05. **PASS.**
- Target: **the control behaves sensibly.** For `full` the "vs full" row is an artefact of the definition — its
  reference *is* ordering 0's own output, so any averaging moves away from it; the number is reported rather than
  hidden but carries no meaning. The meaningful control is `full` vs GT: +6.4 % median reduction, p = 0.0278,
  27/40 — far smaller than the quantized arms' 12–24 %. **PASS: the correction is mostly cancelling
  quantization noise, not VGGT's systematic error.**

**Correction to a number quoted in conversation.** The 8-scene smoke test gave a 40.9 % median reduction for
`w4a4_rtn`; at n = 40 it is **15.8 %**. The smoke-test figure was not representative and should not be cited.

**Interpretation.** Quantization noise in VGGT's pose head is substantially **zero-mean with respect to input
ordering**, so averaging four orderings removes part of it — a corrector that needs no ground truth, no
full-precision model, and no training, and that reuses exactly the 4× inference already spent on the #045
detector. Two honest limits. The median `w4a4_rtn` pose error lands at 4.87°, still above the ~2–4° tolerance of
#042, so full rendering recovery should **not** be expected. And intrinsics are not corrected, so the arm's
18.652 % focal error (#031), worth 0.4876 dB on its own (#043), survives untouched. Whether the pose correction
buys anything downstream is being measured now (`ensemble_downstream.py`), and must not be assumed either way.

---

### #048 — The corrected pose does NOT recover rendering; focal error is a bias, not order-noise

- **Date/time:** 2026-09-12; pose-only downstream 10:11–11:46, followups 11:46–13:31 CEST (GPU)
- **Git commit:** `f0c8e32` + working tree adding `code/quantization/ensemble_downstream.py`,
  `run_ensemble_downstream.sh`, `run_ensemble_followups.sh`
- **Commands executed:**
  ```
  ./code/quantization/run_ensemble_downstream.sh
  #   -> ensemble_downstream.py --arm w4a4_rtn --perms 4 --iterations 7000 --deadline 00:15
  ./code/quantization/run_ensemble_followups.sh
  #   -> pose_ensemble.py --variant w4a4_rtn --perms 4 --scenes all --save-extrinsics   (adds intrinsics)
  #   -> pose_ensemble.py --variant {w4a4_rtn,w4a4} --perms 2 --scenes all
  #   -> ensemble_downstream.py --arm w4a4_rtn --perms 4 --iterations 7000 --use-intrinsics
  ```
- **Config:** conditions built in full precision's frame, rendered from full's held-out source, 7000 iterations,
  6 input / 9 held-out views, foreground mask, GraphDECO defaults, no depth regularisation — identical to #039/#043
  so the numbers are directly comparable. A = `full`; K = `swapK_w4a4_rtn` (arm pose + arm intrinsics + full
  points); ensK = ensemble pose + arm intrinsics + full points; ensKF = ensemble pose + **ensemble intrinsics** +
  full points. Points are full precision everywhere, so only the camera channel varies.
- **Scenes used:** all 40.

**Downstream (n = 40, foreground):**

| condition | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| A full | 8.9642 | 0.2318 | 0.6604 |
| K uncorrected | 7.9922 | 0.2124 | 0.7269 |
| ensK pose-corrected | 8.0991 | 0.2135 | 0.7195 |
| ensKF pose+focal-corrected | 8.0339 | 0.2120 | 0.7183 |

| contrast | PSNR mean | p | LPIPS mean | p |
|---|---:|---:|---:|---:|
| ensK − K (what pose correction buys) | +0.1070 | 0.3807 | +0.0074 | 0.1458 |
| ensKF − K (pose + focal) | +0.0417 | 0.6726 | +0.0086 | 0.1338 |
| ensKF − ensK (focal on top) | −0.0653 | 0.5498 | +0.0012 | 0.7726 |
| A − ensK (what remains) | +0.8651 | 1.599e-08 | +0.0591 | 2.408e-11 |
| A − K (original gap) | +0.9721 | 5.068e-06 | +0.0665 | 5.783e-09 |

**Focal ensembling (n = 40):** focal error vs full, single median 0.17445 → ensemble 0.17700, improved in 18/40,
Wilcoxon p = 0.485 (w4a4: 0.02220 → 0.02153, 19/40, p = 0.493). **Averaging does not reduce focal error at all.**

**Ensemble size, pose error vs full (median, w4a4_rtn / w4a4):** 2 orderings 5.5751° (+7.3 %, p 0.0016) /
0.8373° (+13.1 %, p 0.00558); 4 orderings 5.1609° (+17.3 %, p 1.67e-05) / (#047: 0.8557°, +23.8 %).

**PASS/FAIL:**
- Target: **the pose correction of #047 recovers a measurable share of the 0.9721 dB rendering gap.** It recovers
  +0.1070 dB, 11 %, p = 0.3807. **FAIL — not significant.** With focal correction added, +0.0417 dB, p = 0.6726.
  **FAIL.**
- Target: **ensembling corrects intrinsics as it corrects pose.** p = 0.485, 18/40. **FAIL.**
- Target: **2 orderings retain most of the 4-ordering gain** (the efficiency question). They retain roughly half
  (7.3 % vs 17.3 % on rtn; 13.1 % vs 23.8 % on w4a4). **PARTIAL.**

**Reproducibility bug found, and its effect.** The ensemble seeded permutations with `abs(hash(name))`. Python
salts string hashing per interpreter, so **every process drew a different permutation set** and the ensemble was
not reproducible: the same command gave a 4.8713° median in #047 and 5.1609° here. Both runs show a significant
improvement and the conclusion is unchanged, but the exact reduction has run-to-run variance of a few tenths of a
degree and #047's 15.8 % should be read alongside this run's 17.3 %. Seeding is now `zlib.crc32(name)`, stable
across processes; all later runs are reproducible. Numbers produced before this fix are marked by the commands
above.

**Interpretation.** The detector works (#045) and the corrector reduces pose error (#047), but the correction is
not enough to matter downstream: it moves `w4a4_rtn` from ~6.07° to ~4.9–5.2°, and #042 puts the rendering-damage
threshold at ~2–4°, so the corrected prior is still inside the damaged regime. The focal result explains itself —
order-averaging can only cancel *order-dependent* noise, and quantization's focal error is a systematic bias that
every ordering shares, so averaging leaves it untouched. This is the honest state of the method: **a working
detector, a partial corrector, and no downstream gain at this damage level.** Whether more orderings cross the
threshold is being measured now (2/4/8/16, seeded); if the gain saturates above 4°, the ceiling is real.

---

### #049 — Ensemble size 2/4/8/16: the correction saturates above the damage threshold

- **Date/time:** 2026-09-12 13:39–14:58 CEST (GPU)
- **Git commit:** `f0c8e32` + working tree (`pose_ensemble.py` with crc32 seeding, `run_ensemble_scaling.sh`)
- **Command executed:**
  ```
  ./code/quantization/run_ensemble_scaling.sh
  #   -> pose_ensemble.py --variant {w4a4_rtn,w4a4} --perms {2,4,8,16} --scenes all --save-extrinsics
  ```
- **Config:** inference only, as #047, with permutations seeded by `zlib.crc32(scene)` so every run is
  reproducible (the `hash()`-based seeding of #045/#047/#048 was not — see #048). Pose error is the mean
  per-camera rotation angle against full precision's saved cameras after Sim(3) on the six camera centres.
- **Scenes used:** all 40, every configuration.

**Pose error vs full precision (median over 40 scenes):**

| arm | orderings | single | ensemble | reduction | improved | Wilcoxon p | scenes < 4° |
|---|---:|---:|---:|---:|---|---:|---|
| w4a4_rtn | 2 | 6.0741° | 5.4213° | 14.1 % | 30/40 | 0.0061 | 9/40 |
| w4a4_rtn | 4 | 6.0741° | 5.1255° | 26.2 % | 31/40 | 0.000538 | 16/40 |
| w4a4_rtn | 8 | 6.0741° | 4.6820° | 31.4 % | 35/40 | 2.27e-05 | 14/40 |
| w4a4_rtn | 16 | 6.0741° | **4.3743°** | 29.5 % | 35/40 | 1.16e-06 | 17/40 |
| w4a4 | 2 | 1.1162° | 0.8977° | 10.9 % | 25/40 | 0.0381 | 39/40 |
| w4a4 | 4 | 1.1162° | 0.6921° | 17.6 % | 30/40 | 0.00105 | 39/40 |
| w4a4 | 8 | 1.1162° | 0.7961° | 23.1 % | 31/40 | 8.25e-05 | 39/40 |
| w4a4 | 16 | 1.1162° | 0.7283° | 28.4 % | 31/40 | 0.00017 | 39/40 |

Marginal gain per doubling on `w4a4_rtn`: 5.4213 → 5.1255 → 4.6820 → 4.3743°, i.e. +0.296, +0.444, +0.308°.
The median relative reduction does not increase monotonically (26.2 → 31.4 → 29.5 %), and `w4a4`'s does not
either (17.6 → 23.1 → 28.4 % with a dip in the median error at 8), which bounds how precisely these can be read.

**PASS/FAIL:**
- Target: **does a larger ensemble cross the ~2–4° rendering-damage threshold of #042?** At 16 orderings
  `w4a4_rtn` reaches 4.3743°, still above 4°, with 17/40 scenes below it. **FAIL — the threshold is not
  crossed even at 4× the cost tested downstream in #048.**
- Target: **does the gain keep scaling?** From 4 to 16 orderings — a 4× cost increase — the median falls by
  0.75° (5.1255 → 4.3743). **Saturating; no ensemble size in reach fixes this arm.**
- Target (efficiency): **is 2 enough?** 2 orderings give 14.1 % of the 29.5 % available at 16 on rtn, and
  10.9 % vs 28.4 % on w4a4. **No — roughly half the gain at a quarter the extra cost.**

**Interpretation.** Order-ensembling reduces quantization pose error reliably and reproducibly, on both arms,
at every size tested — but it saturates around a 30 % reduction, which leaves `w4a4_rtn` at 4.37°, inside the
regime where rendering is damaged. That closes the loop opened in #045: the signal is real, the correction is
real, and neither is large enough to recover rendering at this damage level, which is exactly what #048 measured
downstream. The mechanism is consistent throughout: only the order-*dependent* part of quantization error can be
averaged away, and for this arm the order-independent remainder is itself above the threshold. For W4A4 the
correction takes 1.12° to 0.69–0.80°, both already inside tolerance, so there is nothing downstream to win there
either. **No further downstream run is justified for `w4a4_rtn`;** the case where this machinery could matter is
an arm whose single-run error sits just *above* the threshold and whose order-dependent share is large — W3A3 is
the candidate, and it is not yet available.

---

### #050 — W3A3 arrives (8 scenes, calibrated and inferred externally): pose 2.69°, focal 0.102, points 0.101

- **Date/time:** 2026-09-12 19:58–20:00 CEST (transfer), 20:05–20:12 CEST (verification + geometry, CPU)
- **Git commit:** `f0c8e32` + working tree
- **Provenance:** W3A3 was calibrated **and** inferred by Sharma Poudel / Prabin (UTN) on their machine.
  What arrived is `w3a3_8scenes_export.zip`, 1,065,794,236 bytes,
  sha256 `1f2c21c60f8f9504f80bd627f8eec6f97e91671d5dab55612fac76735e25b34a`, containing per scene one
  `w3a3.npz` (25,760,956 B) and one trained `point_cloud.ply`. **The QS checkpoints were NOT sent**, so
  nothing here can be re-inferred locally — see "What this does not unblock" below.
- **Command executed:**
  ```
  # contract check + geometry, both local, no GPU
  python - <<'…'   # shapes, frame numbers, finiteness vs our full.npz
  python - <<'…'   # pose_err / focal_err / point_err vs full and vs CO3D GT
  ```
- **Config:** pose error = mean per-camera rotation angle against full precision's saved cameras after the
  Sim(3) fitted on the six input camera centres — identical construction to #039/#043/#045. Focal error =
  median |K_arm/K_full − 1| over both axes and all six cameras. Point error = median ‖p_arm→full − p_full‖
  normalised by the scene radius, same Sim(3).
- **Scenes used:** the 8 `EXPENSIVE_SCENES`. These are **not** a random sample — they were selected in
  #002 to span categories *and both signs* of the full-minus-quant PSNR delta, which biases any effect
  size estimated on them. Stated here because every W3A3 number in #050–#052 inherits it.

**Contract check — all 8 pass.** Array shapes match the prediction contract
(`extrinsic` (6,3,4), `intrinsic` (6,3,3), `depth` (6,518,518,1), `world_points_from_depth` (6,518,518,3)),
all finite, and the frame numbers are **identical to our `full` arm** on every scene
(`[1,35,69,103,137,170]`, and `[1,35,69,102,135,168]` for hydrant).

**Geometry vs full precision:**

| scene | pose w4a4 | pose rtn | **pose w3a3** | focal w4a4 | focal rtn | **focal w3a3** | pts w4a4 | pts rtn | **pts w3a3** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| apple/110 | 0.718° | 14.875° | **2.865°** | 0.0449 | 0.3641 | **0.2909** | 0.0386 | 0.5421 | **0.1560** |
| ball/123 | 0.617° | 6.118° | **1.515°** | 0.0126 | 0.0536 | **0.0547** | 0.0218 | 0.2304 | **0.0388** |
| bowl/70 | 4.779° | 165.510° | **7.402°** | 0.1632 | 0.0445 | **0.0953** | 0.1879 | 3.4859 | **0.2271** |
| broccoli/412 | 0.681° | 4.473° | **1.934°** | 0.0173 | 0.1349 | **0.1128** | 0.0214 | 0.1782 | **0.0806** |
| hydrant/167 | 0.772° | 2.203° | **1.860°** | 0.0284 | 0.0695 | **0.0251** | 0.0345 | 0.2173 | **0.0680** |
| remote/350 | 1.335° | 4.198° | **2.011°** | 0.0089 | 0.1883 | **0.1444** | 0.0350 | 0.2591 | **0.1045** |
| teddybear/187 | 1.314° | 8.097° | **2.094°** | 0.0017 | 0.0989 | **0.0172** | 0.0434 | 0.3351 | **0.0714** |
| toaster/372 | 1.167° | 4.661° | **1.830°** | 0.0173 | 0.1478 | **0.0715** | 0.0286 | 0.2135 | **0.0592** |
| **mean** | 1.4230° | 26.2667° | **2.6887°** | 0.0368 | 0.1377 | **0.1015** | 0.0514 | 0.6827 | **0.1007** |
| **median** | 0.9699° | 5.3893° | **1.9725°** | 0.0173 | 0.1169 | **0.0834** | 0.0347 | 0.2447 | **0.0760** |

W3A3 is worse than W4A4 on **8/8** scenes for pose (Wilcoxon p 0.0078) and **8/8** for points (p 0.0078);
focal is worse on 6/8 (p 0.1094, not significant at n = 8). Full precision's own pose error against CO3D
ground truth on these scenes is 1.933° — i.e. **W3A3's 2.69° error relative to full is larger than full's
entire error relative to ground truth.**

**PASS/FAIL:**
- Target: **is W3A3 outside the ~2–4° rotation tolerance of #042?** Mean 2.69°, median 1.97°, max 7.40°;
  5/8 scenes above 2°. **PASS at the mean, marginal at the median — it straddles the threshold**, unlike
  W4A4 (1.42° mean) which sits below it.
- Target: **are W3A3's points in the damaging regime?** 0.101 scene radii, against #042's finding of no
  PSNR loss up to 1.0. **FAIL — an order of magnitude below the level at which points matter.**
- Target: **is the focal error in the damaging regime?** 0.1015, versus `rtn`'s 0.1377 which cost
  +0.4876 dB on its own (#043). **PASS — 74 % of a damage level already measured.**

**Interpretation.** W3A3 is the arm #049 said was needed: single-run error above the threshold where
rendering starts to care. The decomposition predicts, before any rendering was measured, that its damage
must be **camera-borne** — the points are far inside the harmless regime while pose straddles the tolerance
line and focal sits at three quarters of a known-damaging level. #051 tests that prediction downstream.

**What this does not unblock.** The QS checkpoints (`qs_frame_parameters_total.pth`,
`qs_global_parameters_total.pth`, `ablation_config.json`) were not transferred, so W3A3 cannot be run
locally. That blocks every experiment requiring *new* inference — specifically extending the
order-ensembling detector (#045) and corrector (#047) to W3A3, which #049 named as the case where the
machinery could plausibly pay off. It remains blocked.

**Threat to validity — not resolved.** `input_semantic_sha256` in the `w3a3_meta.json` files written here
is **copied from our own `full_meta.json`, not independently verified**: their meta files were not sent, so
we cannot confirm their preprocessing produced byte-identical 518×518 pad-mode inputs. Frame numbers match,
which is necessary but not sufficient. `w3a3_compare_contract.json` (40 scenes, sha256 of every train and
held-out GT PNG) was written for them to check against and has not been returned.

---

### #051 — W3A3 measurably degrades rendering: the first arm to separate from W4A4

- **Date/time:** 2026-09-12 20:01–20:20 CEST (GPU), 18.7 min
- **Git commit:** `f0c8e32` + working tree
- **Command executed:**
  ```
  python code/quantization/run_w2a4_downstream.py --scenes subset --iterations 7000 \
         --arms full w4a4 w3a3
  ```
- **Config:** their 8 `w3a3.npz` were installed into the group directories and the sources, 3DGS training
  and scoring were **all rebuilt locally** by `build_variant_sources()` — the same code path that produced
  `full` and `w4a4`. Their own renders and their own results JSON were deliberately **not** used. 7000
  iterations, `--resolution 1`, foreground mask primary.
- **Scenes used:** the 8 `EXPENSIVE_SCENES` (selection bias as stated in #050).

**Arm means (foreground mask):**

| arm | PSNR | SSIM | LPIPS | content PSNR |
|---|---:|---:|---:|---:|
| full | 9.9708 | 0.2403 | 0.6553 | 9.8990 |
| w4a4 | 9.6424 | 0.2369 | 0.6635 | 9.8931 |
| **w3a3** | **9.1880** | **0.2280** | **0.6787** | **9.2106** |

**Paired tests over scenes (raw, foreground):**

| pair | PSNR | p | 95 % CI | SSIM | p | LPIPS | p |
|---|---:|---:|---|---:|---:|---:|---:|
| full − w3a3 | **+0.7828** | **0.00629** | [+0.3021, +1.2636] | +0.0123 | 0.0630 | **−0.0234** | **0.0266** |
| w4a4 − w3a3 | **+0.4544** | **0.0263** | [+0.0715, +0.8373] | +0.0089 | 0.157 | −0.0152 | 0.142 |
| full − w4a4 | +0.3284 | 0.0935 | [−0.0719, +0.7287] | +0.0033 | 0.488 | −0.0082 | 0.0628 |

Per-scene full-minus-w3a3 PSNR: apple +1.011, ball +1.419, bowl +0.518, broccoli +0.858, hydrant **−0.226**,
remote +0.208, teddybear +1.132, toaster +1.342 — 7/8 positive.

**Exposure-corrected columns show nothing:** full − w3a3 corrected PSNR +0.0962, p 0.718. This is the
expected behaviour, not a contradiction: #037 measured that the correction removes real geometry signal.
Corrected LPIPS does separate (−0.0314, p 0.0196). **Raw is cited throughout.**

**PASS/FAIL:**
- Target: **does W3A3 degrade rendering vs full precision?** +0.7828 dB, p 0.00629, CI excludes zero, 7/8
  scenes. **PASS.** First arm in this project to do so at a bit-width nobody deliberately broke.
- Target: **does W3A3 separate from W4A4?** +0.4544 dB, p 0.0263. **PASS.**
- Target: **does the W4A4 null reproduce at n = 8?** +0.3284 dB, p 0.0935 — consistent with the n = 40
  result (+0.1359, p 0.099, #029), inflated as expected by the small biased subset. **PASS as a control.**

**Correcting the number their pipeline reported.** Their results JSON gave W3A3 foreground PSNR 8.8469 on
these scenes, which paired against our `full` implies a gap of **+1.1239 dB**. Rebuilt locally the gap is
**+0.7828 dB** — theirs is 0.3411 dB larger, i.e. **44 % above** the like-for-like value. Their per-scene
bowl figure (10.916) also *beat* our full precision (10.458), which does not reproduce here (9.940, gap
+0.518). The cross-machine comparison was therefore materially optimistic and **must not be cited**; the
discrepancy is unexplained and is a second reason the preprocessing verification in #050 matters. Their
JSON also carried only `psnr_exposure_corrected` and not the SSIM/LPIPS corrected variants, i.e. it came
from a pre-#037 version of `run_w2a4_downstream.py` — their code has drifted from ours.

**Interpretation.** The prediction of #050 holds: the arm whose *cameras* cross into the damaging regime
while its *points* stay far below it is the arm that loses rendering quality. Together with #039/#043 this
is now a consistent account across four arms — `w4a4` (cameras inside tolerance → no damage), `w3a3`
(cameras straddle it → damage), `no_rot` and `rtn` (cameras far outside → large damage).

---

### #052 — Perfect point confidence recovers −1.4 % of W3A3's gap: the proposal's mechanism fails on a second arm

- **Date/time:** 2026-09-12 21:44–22:15 CEST (GPU), 31 min
- **Git commit:** `f0c8e32` + working tree (`confidence_oracle.py` gains `--scene-set`)
- **Command executed:**
  ```
  python code/quantization/confidence_oracle.py --arm w3a3 --scene-set subset \
         --iterations 7000 --keep 0.5 --deadline 00:15
  ```
- **Config:** confidence = the **true** per-point error against full precision, i.e. a cheating predictor
  that no deployed system could have. Keep the best 50 % of initial points, retrain, re-render from the
  arm's own held-out source. The matched control drops an equally sized *random* half with a fixed seed,
  because halving the point count changes 3DGS behaviour on its own — **only (confidence − random) is
  attributable to confidence.**
- **Scenes used:** the same 8 (selection bias as stated in #050).

**The oracle's separation was real on every scene** (median true point error, kept / dropped): apple
0.1090/0.2140, ball 0.0256/0.1147, bowl 0.1604/0.3008, broccoli 0.0559/0.1253, hydrant 0.0488/0.1345,
remote 0.0706/0.1477, teddybear 0.0501/0.0998, toaster 0.0449/0.1386 — a 2–4× ratio throughout. Every
scene: 101,400 points → 50,700.

| condition | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| A — full precision | 9.9708 | 0.2403 | 0.6553 |
| B — w3a3, all points | 9.1880 | 0.2280 | 0.6787 |
| C — w3a3, confidence-kept 50 % | 9.1770 | 0.2163 | 0.6763 |
| R — w3a3, random-kept 50 % (control) | 9.2675 | 0.2204 | 0.6803 |

| paired delta (foreground) | PSNR | p | LPIPS | p |
|---|---:|---:|---:|---:|
| C − B  (confidence vs uniform) | −0.0109 | 0.930 | −0.0024 | 0.692 |
| **C − R  (confidence vs random — THE test)** | **−0.0904** | **0.635** | **−0.0040** | **0.507** |
| R − B  (point-count effect) | +0.0795 | 0.480 | +0.0016 | 0.174 |
| A − B  (the gap to close) | +0.7828 | 0.00629 | −0.0234 | 0.0266 |

**PASS/FAIL:**
- Target: **does perfect point confidence beat its matched random control?** −0.0904 dB, p 0.635.
  **FAIL** — the point estimate is on the wrong side and indistinguishable from random pruning.
- Target: **does it recover any of the +0.7828 dB gap?** C − B = −0.0109 dB, i.e. **−1.4 %**. **FAIL.**
- Target: **was the oracle strong enough for this to be informative?** Kept/dropped error separated 2–4×
  on all 8 scenes. **PASS — the null is not an artefact of a weak signal.**

**Interpretation.** This closes the proposal's mechanism by measurement on a **second, independent** arm:
`w4a4_rtn` recovered −12 % (#030), `w3a3` recovers −1.4 %. Because the oracle uses ground truth it is the
**ceiling** for any confidence predictor, so no learned head at any accuracy can help here — including the
AUROC 0.888 / 0.887 (scene- / category-held-out) pixel predictor already demonstrated in #036. The reason
is given by #050 and #040 together: W3A3's point error (0.101 scene radii) is an order of magnitude inside
the harmless regime, its damage is camera-borne, and 3DGS's appearance optimisation absorbs misplaced points
while camera poses are held fixed throughout training and never corrected.

**Consequence for the proposal.** *"Can a lightweight confidence predictor reduce novel-view rendering
degradation caused by QuantVGGT geometry errors?"* — answered **no** for point-level confidence, now on two
arms, with the ceiling measured rather than argued. The detector half (#036, #045) stands; the
intervention half does not. The only untested reading of the title is **soft weighting** rather than hard
pruning, which would require modifying the GraphDECO trainer to accept per-point weights; given that the
oracle shows no separation from random at all, it is not expected to change the conclusion and has not
been built.

**Threat to validity.** n = 8 with p 0.635 is a *null*, not evidence of harm; the confidence interval
includes zero in both directions. What makes it decisive is that the oracle bounds every predictor from
above, not the precision of this particular estimate.

---

### #053 — W3A3 swap probe: the damage is in the cameras (robust on LPIPS, not on PSNR at n = 8); points are null

- **Date/time:** 2026-09-12 23:13 – 2026-09-13 00:08 CEST (GPU), 55 min; finished before the 00:20 deadline
  (`stopped_at_deadline: false`) and the 00:35 shutdown.
- **Git commit:** `f0c8e32` + working tree (`geometry_probes.py` gains `--scene-set`)
- **Command executed:**
  ```
  python code/quantization/geometry_probes.py swap --arm w3a3 --scene-set subset \
         --iterations 7000 --deadline 00:20
  ```
- **Config:** the #039 construction applied to W3A3. All conditions are built in full precision's frame and
  rendered from full's held-out source: `swapP` = W3A3 points + full cameras, `swapK` = W3A3 cameras
  (pose + intrinsics) + full points, `swapB` = both from W3A3. Compared against the existing `full` renders.
- **Scenes used:** the 8 `EXPENSIVE_SCENES` (selection bias as stated in #050).

**Validity check — the swap construction reproduces the real arm.** `swapB` (both geometry channels from W3A3,
carried into full's frame) gives a mean gap of **+0.7691 dB**; the real W3A3 arm of #051 gives **+0.7828 dB**.
The `full` values used are identical to #051 on every scene (e.g. apple 12.6695, toaster 8.9490), so the two
runs share their reference renders. Per scene the two gaps agree in sign on 7/8; bowl diverges (+0.518 real,
−0.309 swapB). The probe therefore measures the same damage #051 measured.

**Per scene, full-minus-condition PSNR (positive = condition worse):**

| scene | full | swapP | swapK | swapB | Δ points | Δ cameras | Δ both |
|---|---:|---:|---:|---:|---:|---:|---:|
| apple/110 | 12.670 | 12.932 | 11.556 | 11.796 | −0.262 | **+1.114** | +0.873 |
| ball/123 | 11.224 | 10.577 | 9.545 | 9.477 | +0.647 | **+1.679** | +1.747 |
| bowl/70 | 10.458 | 11.181 | 9.403 | 10.767 | −0.723 | **+1.055** | −0.309 |
| broccoli/412 | 9.338 | 9.222 | 8.366 | 8.541 | +0.116 | **+0.972** | +0.797 |
| hydrant/167 | 8.131 | 8.028 | 8.343 | 8.496 | +0.103 | −0.212 | −0.365 |
| remote/350 | 9.656 | 9.741 | 9.669 | 9.398 | −0.085 | −0.013 | +0.259 |
| teddybear/187 | 9.341 | 9.465 | 8.098 | 8.048 | −0.124 | **+1.244** | +1.293 |
| toaster/372 | 8.949 | 7.537 | 9.187 | 7.092 | **+1.412** | −0.238 | +1.857 |

**Summary (condition vs full; positive = worse):**

| condition | PSNR | SSIM | LPIPS | Δ PSNR | t-test p | Wilcoxon p | worse | Δ LPIPS | t-test p | Wilcoxon p | worse |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| full | 9.9708 | 0.2403 | 0.6553 | | | | | | | | |
| swapP (points) | 9.8353 | 0.2388 | 0.6518 | +0.1355 | 0.571 | 0.945 | 4/8 | −0.0036 | 0.629 | 0.844 | 4/8 |
| **swapK (cameras)** | 9.2707 | 0.2299 | 0.6884 | **+0.7001** | **0.0319** | 0.109 | 5/8 | **+0.0330** | **0.00611** | **0.0078** | **8/8** |
| swapB (both) | 9.2017 | 0.2283 | 0.6860 | +0.7691 | 0.0387 | 0.0781 | 6/8 | +0.0307 | 0.0217 | 0.0391 | 7/8 |

SSIM: points +0.0015 (p 0.715), cameras +0.0104 (p 0.158), both +0.0119 (p 0.108) — no condition separates.
Decomposition of the joint loss: **cameras 91.0 % of PSNR, points 17.6 %**, interaction −0.0665 dB; on LPIPS
cameras account for 107.7 % (points marginally *improve* LPIPS). `swapB` 95 % CI on PSNR [+0.0529, +1.4853];
minimum detectable effect at 80 % power, n = 8: 0.849 dB PSNR, 0.0292 LPIPS.

**PASS/FAIL:**
- Target: **do W3A3's points damage rendering?** +0.1355 dB, p 0.571 (Wilcoxon 0.945), 4/8; LPIPS −0.0036,
  p 0.629. **FAIL — null on every metric under both tests**, as #050 predicted from the 0.101-radius point error.
- Target: **do W3A3's cameras damage rendering?** LPIPS +0.0330, **worse on 8/8 scenes**, t-test p 0.00611,
  Wilcoxon p 0.0078. **PASS on LPIPS, robust.** PSNR +0.7001, t-test p 0.0319 but **Wilcoxon p 0.109, worse
  on only 5/8. PASS on PSNR under the t-test only — not robust at n = 8.** The PSNR effect is carried by
  large drops (≥ +0.97 dB) on 5 scenes while 3 scenes are flat or slightly better.
- Target: **does W3A3 reproduce the `rtn` decomposition of #039?** Cameras 91.0 % of the joint PSNR loss here
  vs 85 % for `rtn` at n = 40. **PASS — same account, second arm.**

**Scene-level exceptions, reported rather than averaged away.** On **toaster** the pattern reverses: points
alone cost +1.412 dB and cameras alone *improve* PSNR by 0.238 — the only scene where the points carry the
damage. On **bowl** W3A3's points improve PSNR by 0.723 dB, and the combined condition is better than full
(−0.309) despite a +1.055 dB camera loss. On **hydrant** and **remote** neither channel does measurable damage,
consistent with W3A3 being close to full on those scenes in #051 (−0.226 and +0.208). With n = 8 these cannot
be separated from 3DGS training noise, and no claim is made about them.

**Interpretation.** This converts #050's inference into a direct measurement on W3A3 itself: its points are
harmless and its cameras carry the damage, reproducing #039's finding on a second, realistic arm. The strength
of that claim differs by metric and must be stated that way — **decisive on LPIPS** (8/8, both tests), and on
PSNR **supported by the mean and the t-test but not by the rank test**. Together with #052 it closes the case
against point-level intervention on W3A3 from both sides: pruning points by their true error recovers nothing
(#052), and replacing W3A3's points with full precision's recovers nothing either (+0.1355 dB, p 0.571). The
lever, if any exists, is the camera — which is the part 3DGS never optimises and which order-averaging could
not fix on `rtn` (#048). Whether it can on W3A3 remains blocked on the QS checkpoints (#050).

**Threat to validity.** n = 8, selected subset; the PSNR camera effect is not robust to a rank test, and the
MDE for PSNR at this n (0.849 dB) is larger than the effect itself. The 40-scene W3A3 predictions would
resolve this directly.
