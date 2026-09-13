# Render scoring code

The code that turns 3DGS renders into the PSNR / SSIM / LPIPS numbers in `ongoing_logs.md`.
Code only — no renders, no masks, no CO3D. Pair it with a render bundle (`renders7k_zips/` or
`renders7k_8scenes.zip`).

```
score_renders.py     standalone scorer. Runs anywhere, needs only the render bundle.
export_masks.py      dumps the foreground masks as PNGs. Needs the project + CO3D.
reference/           the original code that produced the logged numbers, for audit.
```

## Quick start

```bash
unzip apple.zip -d renders7k            # or: unzip renders7k_8scenes.zip -d renders7k
python score_renders.py --root renders7k --arms full w4a4 --ref full
```

That gives **content-mask** metrics, which reproduce the logs' `content` column exactly.
For the **primary** (foreground) numbers you also need the masks — see below.

Dependencies: `numpy scipy scikit-image pillow` and, unless you pass `--no-lpips`,
`torch lpips`. Versions used here: numpy 1.26.4, scipy 1.17.1, scikit-image 0.26.0,
pillow 11.3.0, torch 2.3.1+cu121, lpips 0.1.4. CPU is fine; no GPU needed.

## Which mask — this is the part that matters

Metrics are computed inside a mask, and the choice changes the answer a lot.

| `--mask` | what it is | needs | matches the logs? |
|---|---|---|---|
| `content` | the real image area before VGGT's 518×518 pad | nothing | yes, the `content` column |
| `foreground` | CO3D object mask ∩ content rectangle | `--masks-dir` | **yes, the primary column** |
| `full` | whole 518×518 frame, pad included | nothing | no — don't cite it |

The object covers a mean **15.3 %** of the content rectangle, so content-mask scores are mostly
measuring background that no arm reconstructs. That is why foreground is primary and why a
content-mask number will not match a foreground number.

`content` is recovered from the gt pixels: the pad is written as pure white `(255,255,255)` in
contiguous bands at the border, so stripping those bands gives the rectangle back. Verified
against the original `content_mask_from_record()` on all 360 held-out views — 360/360 exact.

`foreground` cannot be derived from these images; it comes from the CO3D annotations. Rather
than reimplement the warp onto the pad grid — where a subtly wrong version would produce
plausible numbers that quietly disagree with the logs — export the exact arrays:

```bash
# on the machine with the project + CO3D
python export_masks.py --out masks              # 360 PNGs, ~1.7 MB
# then anywhere
python score_renders.py --root renders7k --mask foreground --masks-dir masks
```

## Filename pairing — an earlier README got this wrong

`gt/` uses CO3D frame numbers, arm folders use 0-based indices:

```
apple/110_13051_23361/gt/frame000002.png  frame000024.png  frame000052.png ...
apple/110_13051_23361/full/00000.png      00001.png        00002.png ...
```

They correspond by **sorted order** — frame numbers ascend in the same order as the indices —
which is how the original scorer pairs them too. There is **no** `gt/00003.png`; the render
bundle README claimed there was, and that claim is wrong. `score_renders.py` pairs by sorted
order and refuses to score an arm whose render count differs from the gt count, because a
partial set would silently bias that scene's mean.

## Metric definitions, and one asymmetry worth knowing

- **PSNR** uses the mask **per-pixel**.
- **SSIM** and **LPIPS** use the mask's **bounding box**, because both need spatial
  neighbourhoods and a scattered pixel set is meaningless to them. The box therefore includes
  some background around the object. LPIPS is AlexNet, inputs scaled to [-1, 1].

## `--exposure`

Fits one per-channel gain+bias per image over the masked pixels before scoring. It exists
because the original has it, **not** because it is preferred: exposure correction was measured
to remove real geometry signal (it recovered an extra +0.45 / +0.52 dB for arms that were
deliberately damaged). **Cite the raw columns.** Note that `reference/run_w2a4_downstream.py`
still carries an old header line calling the corrected column "the one to trust" — that was
retracted; the file is shipped unedited so it matches what actually ran.

A channel that is near-constant inside the mask is left uncorrected rather than fitted, since
the least-squares system is degenerate there. This is load-bearing: dropping that guard moved
one of the 40 scenes by 1.58 dB.

## Statistics

`score_renders.py` reports the mean over scenes and a paired t-test of `--ref` minus each arm,
over **scenes**. Scenes are the independent unit — pooling the 9 views of a scene would treat
correlated views as independent. The minimum detectable effect at n = 40 is ≈ 0.23 dB.
p-values printed are uncorrected for multiple comparisons.

## Validation

`score_renders.py` is a reimplementation, so it was checked against the numbers it claims to
reproduce — the 40-scene run in `arms_full_w4a4_..._all_it7000.json`, restricted to the 8
scenes in `renders7k_8scenes.zip`:

| mask | comparisons | worst absolute difference |
|---|---|---|
| content, raw | 48 | 0.000e+00 |
| foreground, raw + exposure-corrected | 96 | 0.000e+00 |

Bit-exact on all six metrics for both arms. Reproduce with the commands above and diff against
`_results_json.zip`.

## `reference/`

The original code, copied unedited. It has absolute paths into this machine and will **not**
run elsewhere — it is here so you can check what `score_renders.py` mirrors.

| file | what to look at |
|---|---|
| `run_w2a4_downstream.py` | `score_scene()` — the per-image loop; `gain_bias()`; `paired_stats()` |
| `run_downstream_validation.py` | `psnr()`, `ssim_value()`, `bbox_from_mask()`, `content_mask_from_record()`, `foreground_mask_from_record()` (≈ lines 265–290 and 691–714) |
| `common.py` | scene list and path helpers the two above rely on |
