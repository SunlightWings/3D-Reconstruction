---
title: QuantSplat Explorer
emoji: "\U0001F9CA"
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.5.1
python_version: "3.11"
app_file: app.py
fullWidth: true
short_description: Compare precomputed VGGT to 3DGS reconstructions
pinned: false
---

# QuantSplat Explorer

A CPU-only, precomputed-results web demo for your **Full VGGT / W4A4 / W3A3 -> 3D Gaussian Splatting** project.

**No VGGT weights, calibration checkpoint, CUDA, NPZ, or Gaussian PLY is needed to serve this app.** It displays existing renders and imported measurements. It does not train, run inference, rescore images, or render a Gaussian scene interactively.

## What is included

- Real GT, Full, and W4A4 PNGs for all eight scenes, extracted unchanged from your uploaded `renders7k_8scenes.zip`.
- Synchronized scene/view controls, a ground-truth reference, two model panels, and an image-comparison slider.
- Per-scene and aggregate PSNR/SSIM/LPIPS, paired comparisons, and a scene-level difference chart.
- Geometry diagnostics and an optional table of recorded inference times.
- A W3A3 + confidence entry that remains visibly pending until you add its files.
- A data packer for your existing university-PC folder structure.

W3A3 images are **not** in the uploaded reference archive. Add them from your local downstream output using the command below. The app never substitutes Full/W4A4 images for a missing W3A3 result.

The starter metrics are a **labeled transcription of your console output**, not new measurements: per-scene PSNR is rounded to three decimals, SSIM/LPIPS to four. Aggregate and paired-test numbers are copied from the reported summary, not recomputed from those rounded values. Replace `data/metrics.json` with the original JSON through the packer before the final submission. Content/exposure metrics remain pending unless present in that JSON.

## 1. Add your completed W3A3 renders and exact metrics

Place this folder at `Clean/quantsplat_demo/`. From `Clean/`, with your existing `(cv)` environment active, run:

```bash
python quantsplat_demo/tools/pack_results.py \
  --archive /var/tmp/poli22wo/renders7k_8scenes \
  --downstream /var/tmp/poli22wo/quantsplat/downstream_validation_v1 \
  --metrics /var/tmp/poli22wo/quantsplat/oracle_sweep/results/w3a3_vs_existing_8scenes_9views.json \
  --predictions /var/tmp/poli22wo/quantsplat/disagreement_dataset_v1/run/predictions \
  --require-local
```

This only reads existing PNGs/JSONs and packages them. It does not install anything into `cv`, retrain, render, or change the original experiment files.

It checks:

- Exactly nine GT frames and nine renders per selected model/scene.
- Matching 518 x 518 dimensions.
- Local W3A3 held-out COLMAP image names against the archive's exact GT frame IDs.
- Local vs archived GT pixel equality where the local cache exists.
- Six training views and no input/held-out overlap when prediction metadata is supplied.
- The metrics JSON covers the same scene cohort as the render archive.

**A count of nine is not enough by itself.** If the camera source has 196 frames, a wrong split, or different IDs, packaging stops. Do not rename an arbitrary set of renders to bypass this check. If a copied reconstruction genuinely lacks its COLMAP source, `--trust-render-order` is available only after independent verification and is marked unverified in the catalog.

Previous packaged data is preserved in a sibling `data_backup_...` folder. Do not upload those backups. The original project results are untouched.

Optional: include the exact geometry JSON by adding:

```text
--geometry /var/tmp/poli22wo/quantsplat/oracle_sweep/results/geometry_ablation_w3a3_subset.json
```

Optional: include all available archive ablations by adding `--all-archive-arms`. This increases the asset size substantially. Some archived ablations exist for only four scenes; their missing images remain visible as pending. Scores are shown only when supplied, not inferred from renders.

## 2. Run locally (optional)

Use a separate small demo environment, not the working `cv` or `gs` environment:

```bash
cd quantsplat_demo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860` on the same computer. The app binds to port 7860; set `PORT` to use a different port. To limit listening to this computer, set `GRADIO_SERVER_NAME=127.0.0.1` before launching.

On Windows, use `.venv\Scripts\Activate.ps1` instead of `source .venv/bin/activate`.

## 3. Host on Hugging Face Spaces

Create a **Gradio** Space and upload the contents of this folder into the repository root:

```text
app.py
ui.py
demo_data.py
style.css
requirements.txt
README.md
data/
```

`README.md` includes the required Space configuration. Keep its YAML block intact. A GPU is not required by this app. The exact CPU hardware options shown by Hugging Face depend on your account and current availability.

Do not upload the parent folder as a nested directory: `app.py` and `README.md` must be at the root. Do not upload a ZIP and expect the Space to unpack it automatically.

To make a clean transfer archive after packaging your results:

```bash
python tools/make_upload_zip.py
```

It excludes virtual environments, caches, and old data backups. Unzip the result on the computer from which you will upload to the Space, then upload the contents.

The app reads its catalog at startup; restart/rebuild the Space after changing data. There are deliberately no public upload/admin controls.

## Add W3A3 + confidence later

Keep the same six input views, nine held-out camera IDs, preprocessing, and iteration budget. Place its artifacts in the same downstream convention:

```text
<downstream>/<category>/<sequence>/heldout/w3a3_conf_it7000/renders/00000.png ... 00008.png
<downstream>/<category>/<sequence>/sources/w3a3_conf/heldout/sparse/0/images.txt
```

Then rerun the packaging command with:

```text
--local-arms w3a3 w3a3_conf
--metrics /path/to/updated_comparison.json
```

The updated metrics should use the arm ID `w3a3_conf`, with the same schema as your current comparison JSON. Its images, score rows, and paired tests will appear without modifying the app. A new method must not inherit another arm's scores. Method availability and metric availability are tracked separately.

## Data and result scope

This bundle is the **eight-scene pilot**, not the 40-scene evaluation. Using 40 VGGT prediction files does not make the rendered evaluation a 40-scene result. To expand the demo, supply a render archive and matching metric JSON that cover all 40 scenes. The packer discovers scenes from the archive's GT folders; it does not invent missing views or scores.

See `docs/SCORING_PROTOCOL.md` for the supplied scorer's original definitions and `docs/DATA_FORMAT.md` for the file schema. Raw foreground is the primary view, following the latest supplied scoring README. Exposure-corrected values are supplementary when provided. Scene-level p-values shown in this demo are unadjusted for multiple comparisons.

A non-significant comparison is not equivalence. A difference in significance between two tests is not itself a test of the difference between their effects. The pilot results are not sufficient alone to establish a universal quantization threshold or an end-to-end speedup. Geometry spread is scale-sensitive and is not by itself a pose-collapse proof.

## Reproduction and provenance

The runtime only needs Gradio and Pillow. No inference libraries are imported by your app code. The packer copies render PNGs without resizing or recompressing them; it stores explicit view mappings in `data/catalog.json`. Metrics are imported, not recalculated. The screenshot/layout is independent of scoring.

The starter rendering assets are taken from the user-supplied archive. Confirm that your team/course and dataset terms allow public sharing before deploying publicly; choose a private Space when needed. No new license is asserted for the dataset images or third-party artifacts.

Implementation references:

- Hugging Face Space config: https://huggingface.co/docs/hub/spaces-config-reference
- Gradio ImageSlider: https://www.gradio.app/docs/gradio/imageslider
- Gradio styling: https://www.gradio.app/guides/custom-CSS-and-JS
