#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_downstream_validation as dv


def qvec2rotmat(qvec):
    """COLMAP qvec qw qx qy qz -> rotation matrix."""

    w, x, y, z = np.asarray(
        qvec,
        dtype=np.float64,
    )

    return np.array(
        [
            [
                1 - 2 * y * y - 2 * z * z,
                2 * x * y - 2 * w * z,
                2 * x * z + 2 * w * y,
            ],
            [
                2 * x * y + 2 * w * z,
                1 - 2 * x * x - 2 * z * z,
                2 * y * z - 2 * w * x,
            ],
            [
                2 * x * z - 2 * w * y,
                2 * y * z + 2 * w * x,
                1 - 2 * x * x - 2 * y * y,
            ],
        ],
        dtype=np.float64,
    )


def axis_angle_rotation(axis, angle):
    axis = np.asarray(
        axis,
        dtype=np.float64,
    )

    axis /= np.linalg.norm(axis)

    x, y, z = axis

    K = np.array(
        [
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0],
        ],
        dtype=np.float64,
    )

    I = np.eye(3)

    return (
        I
        + np.sin(angle) * K
        + (1 - np.cos(angle)) * (K @ K)
    )


def check_rotation(name, R):
    q = dv.rotmat2qvec(R)

    recovered = qvec2rotmat(q)

    error = np.max(
        np.abs(
            recovered - R
        )
    )

    inverse_error = np.max(
        np.abs(
            recovered - R.T
        )
    )

    print(
        f"{name:16s} "
        f"roundtrip={error:.3e}  "
        f"inverse={inverse_error:.3e}"
    )

    if error > 1e-10:
        raise AssertionError(
            f"{name}: quaternion round-trip failed: "
            f"{error}"
        )


def main():
    check_rotation(
        "identity",
        np.eye(3),
    )

    check_rotation(
        "x rotation",
        axis_angle_rotation(
            [1, 0, 0],
            np.deg2rad(37),
        ),
    )

    check_rotation(
        "y rotation",
        axis_angle_rotation(
            [0, 1, 0],
            np.deg2rad(-51),
        ),
    )

    check_rotation(
        "z rotation",
        axis_angle_rotation(
            [0, 0, 1],
            np.deg2rad(90),
        ),
    )

    check_rotation(
        "arbitrary",
        axis_angle_rotation(
            [1, 2, 3],
            np.deg2rad(73),
        ),
    )

    print()
    print("PASS: rotmat2qvec preserves R rather than R.T")


if __name__ == "__main__":
    main()