# LLFF downstream pipeline — BLOCKED

This directory is a placeholder. The LLFF downstream arm **cannot be built or run in this
environment**, and no code has been written that could not be verified. See
`ongoing_logs.md` entry #019.

## What was requested

Reuse `vggt_to_3dgs` unchanged; add an LLFF source builder (3/6/9 input views, every 8th
frame held out); reference arm = COLMAP poses + points; whole-image metrics (no masking);
gate = the COLMAP arm must land in published sparse-view range before running full/w4a4.

## Why it is blocked (updated 2026-09-08, entry #024)

**SfM tooling is no longer the blocker.** `pycolmap` 4.2.0 is installed and verified:
`run_colmap_sfm.py` reconstructs `llff_fern` at 20/20 images and 2,959 points in ~2 s (CPU).

**The blocker is dataset access.** Every canonical source for the 8-scene benchmark failed:

| source | result |
|---|---|
| `people.eecs.berkeley.edu/~bmild/nerf/nerf_llff_data.zip` | HTTP 404 |
| Google Drive `16VnMcF1KJYxN9QId6TClMsZRahHNMW5g` | `gdown` refused; Drive blocks programmatic access |
| HF mirrors (`nielsr`, `dsrivastavv`, `pcuenq`) | HTTP 401 |
| HF dataset search "llff" | only `poptree/LLFF` — `gated: manual`, and actually mipnerf360 data |

What is on disk is 2 VGGT demo scenes at 706x529, not the benchmark, and only one of them
reconstructs: `llff_flower` fails SfM outright ("no initial pair" on every attempt).
One demo scene cannot support a "published sparse-view range" comparison.

Without poses there is no reference arm; without a reference arm the gate cannot be
evaluated; and without the gate passing, running full/w4a4 is explicitly out of scope.
Fabricating poses (e.g. seeding from VGGT's own prediction) would make the "reference"
arm depend on the method under test and would produce a number that looks like a control
but measures nothing — the same failure mode D3 refused.

## How to unblock

1. Obtain the LLFF benchmark release (8 scenes: fern, flower, fortress, horns, leaves,
   orchids, room, trex). This needs a human with a browser (Google Drive's confirm page) or
   an accepted Hugging Face licence, or any mirror serving `nerf_llff_data.zip` over plain
   HTTP. `poses_bounds.npy` is not strictly required any more — `run_colmap_sfm.py` can
   produce `sparse/0/{cameras,images,points3D}` from images alone — but the released poses
   are preferable as a reference arm because they are the ones the published numbers use.
2. Then build here: an LLFF source builder writing GraphDECO COLMAP-text sources for
   3/6/9 input views with every 8th frame held out, whole-image metrics (LLFF has no
   object masks, so the foreground/content split in `run_downstream_validation.py` does
   not apply — pass the full-image mask for both).
3. Gate on the COLMAP arm reaching published sparse-view PSNR before running full/w4a4.

## Note on the metric work this depends on

`run_downstream_validation.py` now reports metrics under both a foreground and a content
mask. LLFF is unmasked, so its builder should pass an all-ones mask for both regions
rather than reusing `foreground_mask_from_record`, which is CO3D-specific.
