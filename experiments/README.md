# Experiments

This directory is the research log for the QuantVGGT → 3DGS project.

Experiments `000`–`058` document the quantization, geometry, ablation, oracle, debugging, and downstream studies leading to the final system. Experiment `059` contains the final learned W3A3 reliability / residual-correction study, including code, frozen models, logs, figures, and machine-readable evaluation results.

- `000` is the disagreement-dataset stage that preceded the numbered experiment log.
- `001`–`053` are the entries of [ongoing_logs.md](ongoing_logs.md); their READMEs reproduce the original experiment entries.
- `054`–`058` are the 2026-09-13 post-debugging experiments and follow-up studies.
- `059` is the 2026-09-14 learned reliability + camera residual correction study and downstream evaluation.
- `_shared/` holds modules imported by multiple experiments plus the frozen 40-scene manifest.

> **⚠ Camera-conversion bug.** A quaternion-conjugation bug was discovered in Experiment 054.  
> Pre-fix **3DGS rendering metrics** (PSNR / SSIM / LPIPS) from affected experiments `001`–`053` should not be used as final quantitative evidence. Geometry-only analyses remain useful.  
> Post-fix downstream evidence includes Experiments `054`, `055`, and `059`; Experiment `057` also uses corrected cameras but its archived outputs were not scored in the original log.

Supporting documents: [PAPER_EVIDENCE.md](PAPER_EVIDENCE.md), [HANDOFF.md](HANDOFF.md), [explanation.md](explanation.md), and [CLAUDE.md](CLAUDE.md).

| # | experiment | note |
|---|---|---|
| 000 | [Disagreement dataset: full vs W4A4 VGGT on 40 CO3D scenes (before the log started)](000_disagreement_dataset_full_vs_w4a4_vggt_on/README.md) |  |
| 001 | [Backfill of pre-existing results (no new run)](001_backfill_of_pre_existing_results_no_new/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 002 | [Evaluation-region vs available-geometry audit (analysis, no training)](002_evaluation_region_vs_available_geometry_audit_analysis/README.md) |  |
| 003 | [Baseline oracle, iteration sweep (factor 2)](003_baseline_oracle_iteration_sweep_factor_2/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 004 | [Harness validation: A3 reproduction and foreground-mask re-scoring (analysis, no training)](004_harness_validation_a3_reproduction_and_foreground_mask/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 005 | [Densification disabled (factor 1)](005_densification_disabled_factor_1/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 006 | [Densification stopped at 3000 (factor 1)](006_densification_stopped_at_3000_factor_1/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 007 | [Opacity reset disabled (factor 3)](007_opacity_reset_disabled_factor_3/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 008 | [12 input views instead of 6 (factor 4)](008_12_input_views_instead_of_6_factor/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 009 | [Foreground-masked training images, constant background (factor 5)](009_foreground_masked_training_images_constant_background_factor/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 010 | [Depth regularisation against GT depth (factor 6)](010_depth_regularisation_against_gt_depth_factor_6/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 011 | [GT foreground points + random background shell (extra factor, not a pure oracle)](011_gt_foreground_points_random_background_shell_extra/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 012 | [24 input views (follow-up to factor 4)](012_24_input_views_follow_up_to_factor/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 013 | [COMBINED: 12 input views + GT depth regularisation (best config found)](013_combined_12_input_views_gt_depth_regularisation/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 014 | [C1 random-init control, re-scored on the foreground mask (analysis, no training)](014_c1_random_init_control_re_scored_on/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 015 | [A6 oracle self-fit (new diagnostic, no training)](015_a6_oracle_self_fit_new_diagnostic_no/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 016 | [A4 root cause: is the GT camera path self-consistent? (analysis, no training)](016_a4_root_cause_is_the_gt_camera/README.md) |  |
| 017 | [Sim(3) direction comparison, and a units bug in A5 (analysis, no training)](017_sim_3_direction_comparison_and_a_units/README.md) |  |
| 018 | [A5 re-run on all 40 scenes after the frame-consistency fix](018_a5_re_run_on_all_40_scenes/README.md) |  |
| 019 | [LLFF downstream pipeline: BLOCKED](019_llff_downstream_pipeline_blocked/README.md) |  |
| 020 | [40-scene Full-vs-W4A4 re-scored on the foreground mask; D1 recomputed](020_40_scene_full_vs_w4a4_re_scored/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 021 | [Pose vs geometry decomposition of the oracle-minus-VGGT gap](021_pose_vs_geometry_decomposition_of_the_oracle/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 022 | [Exposure check: how much of the oracle's 14.83 dB is photometric, not geometric?](022_exposure_check_how_much_of_the_oracle/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 023 | [Sim(3) with rotation constraints, and the similarity-model floor](023_sim_3_with_rotation_constraints_and_the/README.md) |  |
| 024 | [LLFF unblocking attempt: tooling resolved, benchmark data still BLOCKED](024_llff_unblocking_attempt_tooling_resolved_benchmark_data/README.md) |  |
| 025 | [W2A4 calibrated locally; the pose head collapses (unblocks D3 at 2 bits)](025_w2a4_calibrated_locally_the_pose_head_collapses/README.md) |  |
| 026 | [W4A4 ablation study: design, instrument, and the round-to-nearest arm (IN PROGRESS)](026_w4a4_ablation_study_design_instrument_and_the/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 027 | [W4A4 ablation, five arms measured: the Hadamard rotation is the load-bearing component](027_w4a4_ablation_five_arms_measured_the_hadamard/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 028 | [Sixth ablation arm measured; exposure correction extended to SSIM and LPIPS](028_sixth_ablation_arm_measured_exposure_correction_extended/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 029 | [40-scene, eight-arm run: pairwise gaps, dose-response, and the LPIPS clip test](029_40_scene_eight_arm_run_pairwise_gaps/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 030 | [Perfect-confidence oracle: pruning by TRUE error recovers none of a real gap](030_perfect_confidence_oracle_pruning_by_true_error/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 031 | [CPU camera analysis: focal-length error, per-scene predictors, and error against ground truth](031_cpu_camera_analysis_focal_length_error_per/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 032 | [Robustness re-test of every headline claim (pass 1); the dose-response is weaker than reported](032_robustness_re_test_of_every_headline_claim/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 033 | [Depth accuracy against CO3D ground truth, full vs W4A4, all 1330 view-groups](033_depth_accuracy_against_co3d_ground_truth_full/README.md) |  |
| 034 | [Where W4A4's world-point error comes from: depth maps vs cameras](034_where_w4a4_s_world_point_error_comes/README.md) |  |
| 035 | [W4A4 point error splits into thirds: depth shape, per-view scale drift, cameras (supersedes #034's inference)](035_w4a4_point_error_splits_into_thirds_depth/README.md) |  |
| 036 | [Can W4A4's error be predicted from W4A4's own output? Pixel level: partly; camera level: no](036_can_w4a4_s_error_be_predicted_from/README.md) |  |
| 037 | [Why exposure correction erases the geometry signal: it removes ~half of the geometry-induced loss](037_why_exposure_correction_erases_the_geometry_signal/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 038 | [Power: how many scenes each observed effect needs](038_power_how_many_scenes_each_observed_effect/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 039 | [Camera/point swap, 40 scenes: the rendering damage is in the cameras, not the points](039_camera_point_swap_40_scenes_the_rendering/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 040 | [Mechanism from the trained models: final 3D accuracy and render quality are decoupled](040_mechanism_from_the_trained_models_final_3d/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 041 | [Final robustness pass (swap at n = 40)](041_final_robustness_pass_swap_at_n_40/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 042 | [Synthetic damage sweep: renders tolerate ~2° of camera error and a full scene-radius of point noise](042_synthetic_damage_sweep_renders_tolerate_2_of/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 043 | [Pose vs focal length: both hurt, pose more, and pose alone accounts for the perceptual loss](043_pose_vs_focal_length_both_hurt_pose/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 044 | [Viewpoint sensitivity: geometry damage is largest on the CLOSEST held-out views, not the farthest](044_viewpoint_sensitivity_geometry_damage_is_largest_on/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 045 | [Input-order ensembling gives a deployable confidence signal for camera pose (ρ = 0.88)](045_input_order_ensembling_gives_a_deployable_confidence/README.md) |  |
| 046 | [Order-ensembling on `w4a4_rtn`, and COLMAP refinement of a damaged prior: BLOCKED by SfM failure](046_order_ensembling_on_w4a4_rtn_and_colmap/README.md) |  |
| 047 | [Averaging input orderings corrects pose error: the corrector #046 was missing](047_averaging_input_orderings_corrects_pose_error_the/README.md) |  |
| 048 | [The corrected pose does NOT recover rendering; focal error is a bias, not order-noise](048_the_corrected_pose_does_not_recover_rendering/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 049 | [Ensemble size 2/4/8/16: the correction saturates above the damage threshold](049_ensemble_size_2_4_8_16_the/README.md) |  |
| 050 | [W3A3 arrives (8 scenes, calibrated and inferred externally): pose 2.69°, focal 0.102, points 0.101](050_w3a3_arrives_8_scenes_calibrated_and_inferred/README.md) |  |
| 051 | [W3A3 measurably degrades rendering: the first arm to separate from W4A4](051_w3a3_measurably_degrades_rendering_the_first_arm/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 052 | [Perfect point confidence recovers −1.4 % of W3A3's gap: the proposal's mechanism fails on a second arm](052_perfect_point_confidence_recovers_1_4_of/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 053 | [W3A3 swap probe: the damage is in the cameras (robust on LPIPS, not on PSNR at n = 8); points are null](053_w3a3_swap_probe_the_damage_is_in/README.md) | contains pre-fix rendering numbers (invalid, camera bug) |
| 054 | [Camera rotation bug: quaternion conjugate in rotmat2qvec, fixed; corrected baselines](054_camera_rotation_bug_quaternion_conjugate_in_rotmat2qvec/README.md) | corrected-camera rendering baseline |
| 055 | [Point confidence on corrected cameras: oracle and learned predictor, W3A3, 8 scenes](055_point_confidence_on_corrected_cameras_oracle_and/README.md) | post-fix rendering |
| 056 | [Focal-length corrector and gsplat pose refinement: ABANDONED](056_focal_length_corrector_and_gsplat_pose_refinement/README.md) |  |
| 057 | [Old 40 scenes re-trained with corrected cameras (full and W4A4, 80 runs)](057_old_40_scenes_re_trained_with_corrected/README.md) | corrected-camera outputs; archived log notes they were not yet scored |
| 058 | [Extension to 48 more CO3D scenes: manifest built, run CANCELLED](058_extension_to_48_more_co3d_scenes_manifest/README.md) |  |
| **059** | **[Learned W3A3 reliability prediction + camera residual correction + downstream 3DGS evaluation](059_prabin_learned_confidence_residual_downstream/docs/EXPERIMENTS.md)** | **final learned-recovery study: confidence AUROC/AP, residual camera correction, custom cup ablation, N={3,6,9,12} view-count study, bottle replication, PSNR/SSIM/LPIPS/MAE, multi-view stability, and efficiency evaluation** |

## Experiment 059 at a glance

Experiment 059 is the final learned-recovery extension of the earlier quantization studies.

| Component | Result |
|---|---|
| Learned reliability vs analytic consistency | **AUROC 0.730 vs 0.371**, AP **0.706 vs 0.388** |
| W3A3 focal error | **13.48% → 4.01%** with residual correction |
| Captured cup, raw W3A3 → corrected W3A3 | **15.763 → 16.466 dB** PSNR |
| Confidence gating at N=9 | **13.122 → 14.485 dB** PSNR |
| Three-view W3A3 stability | combined method gives best W3A3 mean PSNR / SSIM |
| Independent bottle replication | recovery methods **do not improve** raw W3A3, retained as a negative result |

Detailed artifacts:

- [Scientific summary](059_prabin_learned_confidence_residual_downstream/docs/EXPERIMENTS.md)
- [Model / deployment pipeline](059_prabin_learned_confidence_residual_downstream/docs/MODEL_PIPELINE.md)
- [Results](059_prabin_learned_confidence_residual_downstream/results/)
- [Figures](059_prabin_learned_confidence_residual_downstream/figures/)
- [Logs](059_prabin_learned_confidence_residual_downstream/logs/)
- [Frozen models](059_prabin_learned_confidence_residual_downstream/models/)
- [Provenance](059_prabin_learned_confidence_residual_downstream/PROVENANCE.md)
- [SHA-256 checksums](059_prabin_learned_confidence_residual_downstream/SHA256SUMS.txt)
