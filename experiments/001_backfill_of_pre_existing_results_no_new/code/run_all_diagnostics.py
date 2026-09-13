#!/usr/bin/env python3
"""Orchestrator for the 15 downstream-3DGS diagnostic gate tests.

Runs every test exactly once (resumable: a test already marked "done" in
state.json is skipped unless --force is given), regenerates
results/downstream_3dgs/DIAGNOSTIC_RESULTS.md after each one, and never lets
one failing test stop the rest.

Cheap / CPU-only / render-reuse tests run first so useful signal appears
within the first hour or two; GPU-training-heavy controls (A3, C1, C2, C3) run
last since they are the only ones that need brand-new 3DGS training.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import gate_a  # noqa: E402
import gate_b  # noqa: E402
import gate_c  # noqa: E402
import gate_d  # noqa: E402
import gate_e  # noqa: E402

ORDER = [
    ("D3", gate_d.test_d3_harsher_quantization),
    ("D1", gate_d.test_d1_paired_statistics),
    ("A2", gate_a.test_a2_noise_floor),
    ("A1", gate_a.test_a1_train_view_psnr),
    ("A5", gate_a.test_a5_loo_sim3),
    ("B1", gate_b.test_b1_object_mask_metrics),
    ("B2", gate_b.test_b2_boundary_band_metrics),
    ("B3", gate_b.test_b3_object_background_split),
    ("D2", gate_d.test_d2_disagreement_correlation_rerun),
    ("E1", gate_e.test_e1_chamfer_distance),
    ("E3", gate_e.test_e3_floater_opacity_audit),
    ("A4", gate_a.test_a4_reprojection_overlay),
    ("E2", gate_e.test_e2_rendered_depth_error),
    ("A3", gate_a.test_a3_oracle_control),
    ("A6", gate_a.test_a6_oracle_selffit),
    ("C1", gate_c.test_c1_random_init_control),
    ("C2", gate_c.test_c2_densification_disabled),
    ("C3", gate_c.test_c3_iteration_sweep),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", help="run only these test IDs")
    p.add_argument("--force", action="store_true", help="rerun tests even if already marked done")
    args = p.parse_args()

    common.log("=" * 72)
    common.log("DOWNSTREAM-3DGS DIAGNOSTIC GATES: starting run")
    common.log(f"Expensive (GPU-training) scenes: {common.EXPENSIVE_SCENES}")
    common.log("=" * 72)

    for test_id, fn in ORDER:
        if args.only and test_id not in args.only:
            continue
        common.run_gate_test(test_id, fn, force=args.force)

    common.log("=" * 72)
    common.log(f"ALL DIAGNOSTIC TESTS ATTEMPTED. Report: {common.REPORT_PATH}")
    common.log("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        common.log("ORCHESTRATOR CRASHED:\n" + traceback.format_exc())
        raise
