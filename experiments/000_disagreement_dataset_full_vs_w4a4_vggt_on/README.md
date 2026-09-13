# 000 — Disagreement dataset: full vs W4A4 VGGT on 40 CO3D scenes (before the log started)

_No `ongoing_logs.md` entry exists for this work (it predates the log)._

Work done before `ongoing_logs.md` existed: full-precision VGGT and QuantVGGT W4A4 run on the frozen 40-scene CO3D manifest, per-pixel disagreement between them analysed against ground-truth depth, and disagreement regions characterised.

The narrative write-ups for this stage are in the repository root on the `disagreement-dataset-validation` branch (`explanation.md`, `Readme.md`) and in `docs/` (copied here). `results/` holds the files the analysis scripts wrote.


---

## Files in this folder

- **code/** — the scripts this experiment ran. They are the versions in the working tree on 2026-09-13 (scripts were edited between experiments, so for exact reproduction check out the git commit named in the entry). Shared helpers they import (e.g. `code/downstream_3dgs/run_downstream_validation.py`, `code/quantization/quant_loader.py`) are in [../_shared](../_shared).

**code/**

- `code/analyze_full_dataset_disagreement.py`
- `code/characterize_disagreement_regions.py`
- `code/make_teaching_team_plots.py`
- `code/prepare_co3d_scene.py`
- `code/run_full_dataset_disagreement.py`
- `code/vggt_inference_core.py`
- `code/vggt_prediction_contract.py`
- `code/w4a4_memory_optimized_loader.py`

**results/**

- `docs/disagreement_analysis/DISAGREEMENT_ANALYSIS.md`
- `docs/disagreement_analysis/EXPERIMENT_SETUP.md`
- `docs/disagreement_analysis/PROVENANCE.md`
- `docs/disagreement_analysis/README.md`
- `docs/disagreement_analysis/figures/bench_frame_122_representative_panel.png`
- `docs/disagreement_analysis/figures/toaster_frame_202_representative_panel.png`
- `docs/disagreement_analysis/figures/toytruck_frame_123_representative_panel.png`
- `docs/full_vggt_teammate_handoff.md`
- `docs/vggt_prediction_schema.md`
- `results/dataset_summary.json`
- `results/disagreement_analysis/artifact_manifest.json`
- `results/disagreement_analysis/gt_disagreement_analysis.json`
- `results/disagreement_analysis/heatmap_localization_summary.json`
- `results/disagreement_analysis/heatmap_per_frame_summary.csv`
- `results/disagreement_analysis/heatmap_per_scene_summary.csv`
- `results/disagreement_analysis/primary_per_scene_summary.csv`
- `results/disagreement_quintiles.json`
- `results/figures/coverage_vs_high_excess_share.png`
- `results/figures/dataset_auc_distribution.png`
- `results/figures/dataset_rho_distribution.png`
- `results/figures/disagreement_quintiles.png`
- `results/figures/full_vs_quant_error.png`
- `results/figures/high_disagreement_enrichment.png`
- `results/figures/high_excess_enrichment.png`
- `results/figures/real_vs_shuffled.png`
- `results/figures/selected_examples/strong_hydrant_167_18184_34441_frame_000141.png`
- `results/figures/selected_examples/strong_remote_350_36761_68623_frame_000111.png`
- `results/figures/selected_examples/strong_teddybear_34_1479_4753_frame_000139.png`
- `results/figures/selected_examples/typical_broccoli_412_56288_108844_frame_000054.png`
- `results/figures/selected_examples/typical_plant_247_26441_50907_frame_000051.png`
- `results/figures/selected_examples/typical_skateboard_245_26182_52130_frame_000143.png`
- `results/figures/selected_examples/weak_apple_110_13051_23361_frame_000001.png`
- `results/figures/selected_examples/weak_apple_189_20393_38136_frame_000001.png`
- `results/figures/selected_examples/weak_toaster_372_41229_82130_frame_000149.png`
- `results/negative_control.json`
- `results/per_frame.csv`
- `results/per_scene.csv`
- `results/region_characterization/dataset_region_summary.csv`
- `results/region_characterization/dataset_region_summary.json`
- `results/region_characterization/per_scene_region_summary.csv`
- `results/teaching_team_plots/01_region_disagreement_enrichment.png`
- `results/teaching_team_plots/02_final_3dgs_full_vs_quant.png`
- `results/teaching_team_plots/03_per_scene_psnr_difference.png`
- `results/teaching_team_plots/04_per_scene_disagreement_ranked.png`
- `results/timing_experiment/broccoli_412_56288_108844/full/render_00000.png`
- `results/timing_experiment/broccoli_412_56288_108844/full/render_00001.png`
- `results/timing_experiment/broccoli_412_56288_108844/full/render_00002.png`
- `results/timing_experiment/broccoli_412_56288_108844/full/render_00003.png`
- `results/timing_experiment/broccoli_412_56288_108844/full/render_00004.png`
- `results/timing_experiment/broccoli_412_56288_108844/full/render_00005.png`
- `results/timing_experiment/broccoli_412_56288_108844/gt/gt_00000.png`
- `results/timing_experiment/broccoli_412_56288_108844/gt/gt_00001.png`
- `results/timing_experiment/broccoli_412_56288_108844/gt/gt_00002.png`
- `results/timing_experiment/broccoli_412_56288_108844/gt/gt_00003.png`
- `results/timing_experiment/broccoli_412_56288_108844/gt/gt_00004.png`
- `results/timing_experiment/broccoli_412_56288_108844/gt/gt_00005.png`

Large artifacts (prediction npz, 3DGS PLYs, renders) are not in git (GitHub 100 MB limit); they live under `/var/tmp/luli38se/quantsplat/` on the lab machine and, for the 40-scene full / W4A4 set, in OneDrive `CV/full & w4a4 outputs.zip`.
