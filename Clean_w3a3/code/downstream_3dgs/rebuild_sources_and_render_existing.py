#!/usr/bin/env python3

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_downstream_validation as dv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--sequence", required=True)
    p.add_argument("--variant", choices=["full", "w4a4"], required=True)
    p.add_argument("--npz", required=True)
    p.add_argument("--ply", required=True)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--iterations", type=int, default=7000)
    args = p.parse_args()

    category = args.category
    sequence = args.sequence
    variant = args.variant

    scene_root = (
        Path("/var/tmp/poli22wo/quantsplat/downstream_validation_v1")
        / category
        / sequence
    )

    npz_path = Path(args.npz)
    ply_path = Path(args.ply)

    print(f"Scene:   {category}/{sequence}")
    print(f"Variant: {variant}")
    print(f"NPZ:     {npz_path}")
    print(f"PLY:     {ply_path}")

    arrays = dict(np.load(npz_path))

    # ------------------------------------------------------------
    # Reuse already-prepared common images from the W3A3 scene.
    # These are the exact same six training images / held-out GT
    # images used by the controlled downstream experiment.
    # ------------------------------------------------------------

    common_train = scene_root / "common" / "train_images"
    common_heldout = scene_root / "common" / "heldout_images"

    if not common_train.is_dir():
        raise RuntimeError(f"Missing common train images: {common_train}")

    if not common_heldout.is_dir():
        raise RuntimeError(f"Missing common heldout images: {common_heldout}")

    train_names = sorted(
        x.name for x in common_train.glob("*.png")
    )

    heldout_names = sorted(
        x.name for x in common_heldout.glob("*.png")
    )

    if len(train_names) != 6:
        raise RuntimeError(
            f"Expected 6 training images, found {len(train_names)}"
        )

    if len(heldout_names) != 9:
        raise RuntimeError(
            f"Expected 9 held-out images, found {len(heldout_names)}"
        )

    # ------------------------------------------------------------
    # We need the same prepared training tensor that was used to
    # initialize the point cloud colors.
    # ------------------------------------------------------------

    train_paths = [
        common_train / name
        for name in train_names
    ]

    train_tensor = dv.preprocess_images(train_paths)

    # Remove stale Full/W4A4 sources so rotmat2qvec is definitely
    # called again using the corrected implementation.
    source_root = scene_root / "sources" / variant

    if source_root.exists():
        print(f"Removing stale source: {source_root}")
        shutil.rmtree(source_root)

    train_source = dv.build_train_source(
        scene_root,
        variant,
        arrays,
        train_tensor,
        train_names,
        common_train,
        args.stride,
    )

    # ------------------------------------------------------------
    # The held-out camera alignment must match the original
    # downstream protocol. Instead of approximating it here, read
    # the alignment/cameras from the existing W3A3 experiment
    # metadata is NOT sufficient because each arm has its own Sim3.
    #
    # So stop here after building train source if no matching
    # heldout camera preparation data are available.
    # ------------------------------------------------------------

    print()
    print("Train source rebuilt:")
    print(train_source)

    print()
    print(
        "The Full/W4A4 held-out source still needs the variant-specific "
        "Sim(3) alignment from the original scene preparation."
    )


if __name__ == "__main__":
    main()