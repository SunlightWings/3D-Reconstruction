#!/usr/bin/env python3

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_co3d_scene import (
    import_local_preprocessor,
    pad_transform,
)

CV_ROOT = Path("/home/utn/luli38se/cv")

REPO = CV_ROOT / "3D-Reconstruction"
QUANTVGGT = CV_ROOT / "QuantVGGT"
GS_ROOT = CV_ROOT / "gaussian-splatting"

V3_ROOT = (
    CV_ROOT
    / "data/co3d_pilot/"
      "quantsplat_co3dv2_known_pilot_v3"
)

MANIFEST = (
    CV_ROOT
    / "data/co3d_pilot/"
      "quantsplat_co3dv2_known_pilot_v3_manifest.json"
)

ANNOTATIONS = (
    V3_ROOT
    / "toilet/frame_annotations.jgz"
)

PREPARED = (
    REPO
    / "outputs/model_tests/quant_3dgs/prepared/"
      "quantsplat_co3dv2_known_pilot_v3/"
      "602_93503_186945/input_6"
)

FULL_PRED = Path(
    "/var/tmp/luli38se/quantsplat/predictions/"
    "602_93503_186945/input_6/"
    "full_controlled_20260813/full/predictions.npz"
)

QUANT_PRED = Path(
    "/var/tmp/luli38se/quantsplat/predictions/"
    "602_93503_186945/input_6/w4a4/predictions.npz"
)

FULL_TRAIN_SCENE = Path(
    "/var/tmp/luli38se/quantsplat/3dgs_inputs/"
    "602_93503_186945/input_6/"
    "full_controlled_friend518_20260813"
)

QUANT_TRAIN_SCENE = Path(
    "/var/tmp/luli38se/quantsplat/3dgs_inputs/"
    "602_93503_186945/input_6/"
    "w4a4_friend518_20260813"
)

OUT = Path(
    "/var/tmp/luli38se/quantsplat/"
    "heldout_eval/friend518_20260813"
)

SEQUENCE = "602_93503_186945"

INPUT_FRAMES = [
    1, 43, 91, 112, 174, 202
]

HELDOUT_FRAMES = [
    15, 22, 63, 84, 105,
    133, 153, 181, 195,
]

ALL_FRAMES = INPUT_FRAMES + HELDOUT_FRAMES

TARGET = 518

# PyTorch3D camera coordinates:
# +X left, +Y up, +Z forward
#
# OpenCV/COLMAP:
# +X right, +Y down, +Z forward
CAM_FLIP = np.diag(
    [-1.0, -1.0, 1.0]
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def camera_center(E):
    R = E[:3, :3]
    t = E[:3, 3]
    return -R.T @ t


def camera_centers(E):
    return np.stack(
        [camera_center(x) for x in E],
        axis=0,
    )


def rotation_angle_deg(R):
    value = np.clip(
        (np.trace(R) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return float(
        np.degrees(np.arccos(value))
    )


def umeyama(src, dst):
    """
    dst ~= s * A @ src + b

    Proper rotation only. No reflection.
    """

    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    X = src - src_mean
    Y = dst - dst_mean

    cov = (Y.T @ X) / len(src)

    U, S, Vt = np.linalg.svd(cov)

    D = np.eye(3)

    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1.0

    A = U @ D @ Vt

    var_src = (
        np.sum(X * X) / len(src)
    )

    if var_src <= 0:
        raise RuntimeError(
            "Degenerate camera-center geometry"
        )

    s = (
        np.sum(S * np.diag(D))
        / var_src
    )

    b = (
        dst_mean
        - s * (A @ src_mean)
    )

    return float(s), A, b


def co3d_to_opencv_camera(record):
    vp = record["viewpoint"]

    R_p3d = np.asarray(
        vp["R"],
        dtype=np.float64,
    )

    T_p3d = np.asarray(
        vp["T"],
        dtype=np.float64,
    )

    if R_p3d.shape != (3, 3):
        raise RuntimeError("Bad CO3D R")

    if T_p3d.shape != (3,):
        raise RuntimeError("Bad CO3D T")

    # Official CO3D contract:
    # row-vector:
    # X_cam = X_world @ R + T
    #
    # Column-vector equivalent:
    # X_cam = R.T @ X_world + T
    #
    # Then convert PyTorch3D camera axes
    # to OpenCV axes.
    R_cv = CAM_FLIP @ R_p3d.T
    t_cv = CAM_FLIP @ T_p3d

    det = np.linalg.det(R_cv)

    orth = np.linalg.norm(
        R_cv @ R_cv.T - np.eye(3)
    )

    if abs(det - 1.0) > 1e-4:
        raise RuntimeError(
            f"Invalid rotation determinant {det}"
        )

    if orth > 1e-4:
        raise RuntimeError(
            f"Non-orthogonal camera rotation {orth}"
        )

    H, W = [
        int(v)
        for v in record["image"]["size"]
    ]

    focal = np.asarray(
        vp["focal_length"],
        dtype=np.float64,
    )

    principal = np.asarray(
        vp["principal_point"],
        dtype=np.float64,
    )

    fmt = vp.get(
        "intrinsics_format",
        "ndc_norm_image_bounds",
    )

    if fmt == "ndc_norm_image_bounds":

        sx = W / 2.0
        sy = H / 2.0

    elif fmt == "ndc_isotropic":

        common = min(H, W) / 2.0
        sx = common
        sy = common

    else:
        raise RuntimeError(
            f"Unsupported CO3D intrinsics format: {fmt}"
        )

    fx = focal[0] * sx
    fy = focal[1] * sy

    # NDC principal point uses image-center origin
    # and PyTorch3D +X-left, +Y-up convention.
    cx = W / 2.0 - principal[0] * sx
    cy = H / 2.0 - principal[1] * sy

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    E = np.concatenate(
        [
            R_cv,
            t_cv[:, None],
        ],
        axis=1,
    )

    return E, K


def map_K_to_518(K, transform):
    affine = np.asarray(
        transform[
            "original_to_vggt_affine"
        ],
        dtype=np.float64,
    )

    return affine @ K


def transform_camera_to_variant(
    E_gt,
    scale,
    A,
    b,
):
    R_gt = E_gt[:, :3]
    C_gt = camera_center(E_gt)

    C_variant = (
        scale * (A @ C_gt)
        + b
    )

    R_variant = R_gt @ A.T
    t_variant = (
        -R_variant @ C_variant
    )

    return np.concatenate(
        [
            R_variant,
            t_variant[:, None],
        ],
        axis=1,
    )


def load_rotmat2qvec():
    source = (
        GS_ROOT
        / "scene/colmap_loader.py"
    )

    spec = importlib.util.spec_from_file_location(
        "gs_colmap_loader",
        source,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Cannot load GraphDECO colmap_loader.py"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module.rotmat2qvec


def write_colmap_eval_scene(
    root,
    heldout_E,
    heldout_K,
    image_names,
    image_paths,
    training_scene,
):
    root = Path(root)

    image_dir = root / "images"
    sparse = root / "sparse/0"

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sparse.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, source in zip(
        image_names,
        image_paths,
    ):
        destination = image_dir / name

        if destination.exists():
            destination.unlink()

        os.symlink(
            source.resolve(),
            destination,
        )

    rotmat2qvec = load_rotmat2qvec()

    with (
        sparse / "cameras.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# Camera list with one line "
            "of data per camera:\n"
        )

        for idx, K in enumerate(
            heldout_K,
            start=1,
        ):
            f.write(
                f"{idx} PINHOLE "
                f"{TARGET} {TARGET} "
                f"{K[0,0]:.12g} "
                f"{K[1,1]:.12g} "
                f"{K[0,2]:.12g} "
                f"{K[1,2]:.12g}\n"
            )

    with (
        sparse / "images.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# Image list with two lines "
            "of data per image:\n"
        )

        for idx, (E, name) in enumerate(
            zip(
                heldout_E,
                image_names,
            ),
            start=1,
        ):

            q = rotmat2qvec(
                E[:, :3]
            )

            t = E[:, 3]

            f.write(
                f"{idx} "
                + " ".join(
                    f"{x:.12g}"
                    for x in q
                )
                + " "
                + " ".join(
                    f"{x:.12g}"
                    for x in t
                )
                + f" {idx} {name}\n\n"
            )

    # Rendering a trained model does not use these
    # initialization points, but GraphDECO's scene
    # loader still expects a point cloud source.
    points_bin = (
        training_scene
        / "sparse/0/points3D.bin"
    )

    if not points_bin.is_file():
        raise FileNotFoundError(
            points_bin
        )

    shutil.copy2(
        points_bin,
        sparse / "points3D.bin",
    )


def alignment_report(
    name,
    E_gt_input,
    K_gt_input,
    E_pred,
    K_pred,
):
    C_gt = camera_centers(
        E_gt_input
    )

    C_pred = camera_centers(
        E_pred
    )

    centered = (
        C_gt - C_gt.mean(axis=0)
    )

    singular = np.linalg.svd(
        centered,
        compute_uv=False,
    )

    if (
        singular[0] <= 0
        or singular[1] / singular[0] < 1e-3
    ):
        raise RuntimeError(
            f"{name}: camera trajectory "
            "is too degenerate for Sim(3)"
        )

    scale, A, b = umeyama(
        C_gt,
        C_pred,
    )

    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(
            f"{name}: invalid Sim(3) scale"
        )

    if abs(
        np.linalg.det(A) - 1.0
    ) > 1e-5:
        raise RuntimeError(
            f"{name}: Sim(3) has reflection"
        )

    transformed = np.stack(
        [
            transform_camera_to_variant(
                E_gt_input[i],
                scale,
                A,
                b,
            )
            for i in range(
                len(E_gt_input)
            )
        ],
        axis=0,
    )

    C_trans = camera_centers(
        transformed
    )

    radius = math.sqrt(
        float(
            np.mean(
                np.sum(
                    (
                        C_pred
                        - C_pred.mean(axis=0)
                    )
                    ** 2,
                    axis=1,
                )
            )
        )
    )

    center_errors = np.linalg.norm(
        C_trans - C_pred,
        axis=1,
    )

    center_rmse = math.sqrt(
        float(
            np.mean(
                center_errors ** 2
            )
        )
    )

    center_rmse_pct = (
        center_rmse
        / max(radius, 1e-12)
        * 100.0
    )

    rotation_errors = []

    for i in range(len(E_pred)):
        R_expected = transformed[
            i, :, :3
        ]

        R_pred = E_pred[
            i, :, :3
        ]

        rotation_errors.append(
            rotation_angle_deg(
                R_pred @ R_expected.T
            )
        )

    rotation_errors = np.asarray(
        rotation_errors
    )

    focal_rel = np.concatenate(
        [
            np.abs(
                K_pred[:, 0, 0]
                - K_gt_input[:, 0, 0]
            )
            / np.maximum(
                np.abs(
                    K_gt_input[:, 0, 0]
                ),
                1e-12,
            ),
            np.abs(
                K_pred[:, 1, 1]
                - K_gt_input[:, 1, 1]
            )
            / np.maximum(
                np.abs(
                    K_gt_input[:, 1, 1]
                ),
                1e-12,
            ),
        ]
    )

    principal_error = np.sqrt(
        (
            K_pred[:, 0, 2]
            - K_gt_input[:, 0, 2]
        )
        ** 2
        +
        (
            K_pred[:, 1, 2]
            - K_gt_input[:, 1, 2]
        )
        ** 2
    )

    report = {
        "name": name,
        "scale": float(scale),
        "rotation": A.tolist(),
        "translation": b.tolist(),
        "trajectory_singular_values":
            singular.tolist(),
        "predicted_scene_radius":
            float(radius),
        "camera_center_rmse":
            float(center_rmse),
        "camera_center_rmse_percent_radius":
            float(center_rmse_pct),
        "rotation_error_deg":
            rotation_errors.tolist(),
        "rotation_error_median_deg":
            float(
                np.median(
                    rotation_errors
                )
            ),
        "rotation_error_max_deg":
            float(
                np.max(
                    rotation_errors
                )
            ),
        "focal_relative_error_median_percent":
            float(
                np.median(
                    focal_rel
                )
                * 100.0
            ),
        "principal_point_error_median_px":
            float(
                np.median(
                    principal_error
                )
            ),
    }

    # These are convention/alignment sanity gates,
    # NOT tuning criteria.
    if center_rmse_pct > 15.0:
        raise RuntimeError(
            f"{name}: camera-center alignment "
            f"too poor: {center_rmse_pct:.2f}%"
        )

    if report[
        "rotation_error_median_deg"
    ] > 10.0:
        raise RuntimeError(
            f"{name}: median rotation residual "
            "too large"
        )

    if report[
        "rotation_error_max_deg"
    ] > 20.0:
        raise RuntimeError(
            f"{name}: max rotation residual "
            "too large"
        )

    if report[
        "focal_relative_error_median_percent"
    ] > 50.0:
        raise RuntimeError(
            f"{name}: CO3D->pixel intrinsics "
            "conversion failed sanity check"
        )

    if report[
        "principal_point_error_median_px"
    ] > 100.0:
        raise RuntimeError(
            f"{name}: principal-point "
            "conversion failed sanity check"
        )

    return (
        scale,
        A,
        b,
        transformed,
        report,
    )


def save_mask_518(
    path,
    transform,
    destination,
):
    with Image.open(path) as img:
        img = img.convert("L")

        resized = img.resize(
            (
                int(
                    transform[
                        "resized_width"
                    ]
                ),
                int(
                    transform[
                        "resized_height"
                    ]
                ),
            ),
            resample=Image.Resampling.BILINEAR,
        )

    canvas = Image.new(
        "L",
        (TARGET, TARGET),
        color=0,
    )

    canvas.paste(
        resized,
        (
            int(
                transform[
                    "padding"
                ]["left"]
            ),
            int(
                transform[
                    "padding"
                ]["top"]
            ),
        ),
    )

    canvas.save(destination)


def main():

    print(
        "=============================================="
    )
    print(
        "HELDOUT EVALUATION PREPARATION"
    )
    print(
        "=============================================="
    )

    manifest = load_json(MANIFEST)

    scene = manifest["scenes"][0]

    if scene["sequence_name"] != SEQUENCE:
        raise RuntimeError(
            "Wrong manifest scene"
        )

    if scene["splits"]["input_6"] != INPUT_FRAMES:
        raise RuntimeError(
            "input_6 changed"
        )

    if scene["splits"]["heldout_9"] != HELDOUT_FRAMES:
        raise RuntimeError(
            "heldout_9 changed"
        )

    print(
        "Split contract: PASS"
    )

    print(
        "Annotation symlink:",
        ANNOTATIONS,
    )

    print(
        "Annotation resolves:",
        ANNOTATIONS.resolve(),
    )

    if not ANNOTATIONS.is_file():
        raise FileNotFoundError(
            ANNOTATIONS
        )

    with gzip.open(
        ANNOTATIONS,
        "rt",
        encoding="utf-8",
    ) as f:
        annotations = json.load(f)

    records = {
        int(record["frame_number"]):
            record
        for record in annotations
        if record.get(
            "sequence_name"
        ) == SEQUENCE
    }

    missing = [
        frame
        for frame in ALL_FRAMES
        if frame not in records
    ]

    if missing:
        raise RuntimeError(
            f"Missing CO3D frames: {missing}"
        )

    for frame in ALL_FRAMES:
        record = records[frame]

        if (
            record.get("meta", {})
            .get("frame_type")
            != "test_known"
        ):
            raise RuntimeError(
                f"Frame {frame} not test_known"
            )

        if not isinstance(
            record.get("viewpoint"),
            dict,
        ):
            raise RuntimeError(
                f"Frame {frame} has no viewpoint"
            )

    print(
        "15 CO3D camera annotations: PASS"
    )

    with np.load(
        FULL_PRED,
        allow_pickle=False,
    ) as f:
        full = {
            key: f[key]
            for key in f.files
        }

    with np.load(
        QUANT_PRED,
        allow_pickle=False,
    ) as q:
        quant = {
            key: q[key]
            for key in q.files
        }

    for name, pred in [
        ("Full", full),
        ("Quant", quant),
    ]:

        if pred[
            "frame_numbers"
        ].tolist() != INPUT_FRAMES:
            raise RuntimeError(
                f"{name} frame order changed"
            )

        if pred[
            "extrinsic"
        ].shape != (6, 3, 4):
            raise RuntimeError(
                f"{name} extrinsic shape"
            )

        if pred[
            "intrinsic"
        ].shape != (6, 3, 3):
            raise RuntimeError(
                f"{name} intrinsic shape"
            )

    print(
        "Full/Quant prediction contract: PASS"
    )

    common_dir = OUT / "common"

    images_dir = (
        common_dir / "images"
    )

    masks_dir = (
        common_dir / "masks"
    )

    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    masks_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_preprocess = load_json(
        PREPARED / "preprocess_meta.json"
    )

    first_contract = (
        input_preprocess["views"][0]
    )

    heldout_paths = []

    for frame in HELDOUT_FRAMES:
        path = (
            V3_ROOT
            / records[frame][
                "image"
            ]["path"]
        ).resolve()

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        heldout_paths.append(
            str(path)
        )

    preprocessor, _, signature = (
        import_local_preprocessor(
            QUANTVGGT
        )
    )

    kwargs = {
        "mode": "pad",
    }

    if (
        "target_size"
        in signature.parameters
    ):
        kwargs["target_size"] = TARGET

    heldout_tensor = preprocessor(
        heldout_paths,
        **kwargs,
    ).cpu().contiguous()

    if heldout_tensor.shape != (
        9,
        3,
        TARGET,
        TARGET,
    ):
        raise RuntimeError(
            "Heldout preprocessing shape mismatch"
        )

    if not torch.isfinite(
        heldout_tensor
    ).all():
        raise RuntimeError(
            "Non-finite heldout tensor"
        )

    transforms = {}
    image_names = []
    saved_image_paths = []
    saved_mask_paths = []

    E_gt = {}
    K_gt = {}

    for i, frame in enumerate(
        HELDOUT_FRAMES
    ):

        record = records[frame]

        image_path = (
            V3_ROOT
            / record["image"]["path"]
        ).resolve()

        with Image.open(
            image_path
        ) as img:
            W, H = img.size

        transform = pad_transform(
            W,
            H,
            TARGET,
        )

        transforms[frame] = transform

        # CO3Dv2 sequences should have a
        # fixed crop size; enforce the same
        # original geometry as the canonical
        # six-input preprocessing contract.
        if (
            int(
                transform[
                    "original_width"
                ]
            )
            != int(
                first_contract[
                    "original_width"
                ]
            )
            or
            int(
                transform[
                    "original_height"
                ]
            )
            != int(
                first_contract[
                    "original_height"
                ]
            )
        ):
            raise RuntimeError(
                f"Frame {frame}: source dimensions "
                "differ from canonical input views"
            )

        name = (
            f"frame{frame:06d}.png"
        )

        image_names.append(name)

        array = (
            heldout_tensor[i]
            .permute(1, 2, 0)
            .clamp(0, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .numpy()
        )

        destination = (
            images_dir / name
        )

        Image.fromarray(
            array,
            mode="RGB",
        ).save(destination)

        saved_image_paths.append(
            destination
        )

        mask_source = (
            V3_ROOT
            / record["mask"]["path"]
        ).resolve()

        mask_destination = (
            masks_dir / name
        )

        save_mask_518(
            mask_source,
            transform,
            mask_destination,
        )

        saved_mask_paths.append(
            mask_destination
        )

    print(
        "Heldout RGB/mask preprocessing: PASS"
    )

    # Build GT cameras for all 15 frames.
    all_E = {}
    all_K = {}

    for frame in ALL_FRAMES:

        record = records[frame]

        image_path = (
            V3_ROOT
            / record["image"]["path"]
        ).resolve()

        with Image.open(
            image_path
        ) as img:
            W, H = img.size

        transform = pad_transform(
            W,
            H,
            TARGET,
        )

        E, K_original = (
            co3d_to_opencv_camera(
                record
            )
        )

        K_518 = map_K_to_518(
            K_original,
            transform,
        )

        if (
            not np.isfinite(E).all()
            or not np.isfinite(
                K_518
            ).all()
        ):
            raise RuntimeError(
                f"Non-finite GT camera {frame}"
            )

        if (
            K_518[0, 0] <= 0
            or K_518[1, 1] <= 0
        ):
            raise RuntimeError(
                f"Invalid focal length {frame}"
            )

        all_E[frame] = E
        all_K[frame] = K_518

    E_gt_input = np.stack(
        [
            all_E[f]
            for f in INPUT_FRAMES
        ]
    )

    K_gt_input = np.stack(
        [
            all_K[f]
            for f in INPUT_FRAMES
        ]
    )

    E_gt_heldout = np.stack(
        [
            all_E[f]
            for f in HELDOUT_FRAMES
        ]
    )

    K_gt_heldout = np.stack(
        [
            all_K[f]
            for f in HELDOUT_FRAMES
        ]
    )

    print(
        "CO3D -> OpenCV -> 518 cameras: PASS"
    )

    (
        s_full,
        A_full,
        b_full,
        _,
        report_full,
    ) = alignment_report(
        "Full",
        E_gt_input,
        K_gt_input,
        full["extrinsic"].astype(
            np.float64
        ),
        full["intrinsic"].astype(
            np.float64
        ),
    )

    (
        s_quant,
        A_quant,
        b_quant,
        _,
        report_quant,
    ) = alignment_report(
        "Quant",
        E_gt_input,
        K_gt_input,
        quant["extrinsic"].astype(
            np.float64
        ),
        quant["intrinsic"].astype(
            np.float64
        ),
    )

    print()
    print(
        "=============================================="
    )
    print(
        "GT -> VGGT ALIGNMENT VALIDATION"
    )
    print(
        "=============================================="
    )

    for report in [
        report_full,
        report_quant,
    ]:
        print(
            f"{report['name']}: "
            f"scale={report['scale']:.6f}, "
            f"center_RMSE="
            f"{report['camera_center_rmse_percent_radius']:.3f}% radius, "
            f"rot_median="
            f"{report['rotation_error_median_deg']:.3f} deg, "
            f"rot_max="
            f"{report['rotation_error_max_deg']:.3f} deg, "
            f"focal_med="
            f"{report['focal_relative_error_median_percent']:.3f}%, "
            f"pp_med="
            f"{report['principal_point_error_median_px']:.3f}px"
        )

    heldout_full = np.stack(
        [
            transform_camera_to_variant(
                E_gt_heldout[i],
                s_full,
                A_full,
                b_full,
            )
            for i in range(9)
        ]
    )

    heldout_quant = np.stack(
        [
            transform_camera_to_variant(
                E_gt_heldout[i],
                s_quant,
                A_quant,
                b_quant,
            )
            for i in range(9)
        ]
    )

    # Same GT intrinsics for both branches.
    write_colmap_eval_scene(
        OUT / "sources/full",
        heldout_full,
        K_gt_heldout,
        image_names,
        saved_image_paths,
        FULL_TRAIN_SCENE,
    )

    write_colmap_eval_scene(
        OUT / "sources/quant",
        heldout_quant,
        K_gt_heldout,
        image_names,
        saved_image_paths,
        QUANT_TRAIN_SCENE,
    )

    bboxes = {}

    for frame in HELDOUT_FRAMES:
        tr = transforms[frame]

        left = int(
            tr["padding"]["left"]
        )

        top = int(
            tr["padding"]["top"]
        )

        right = (
            left
            + int(
                tr["resized_width"]
            )
        )

        bottom = (
            top
            + int(
                tr["resized_height"]
            )
        )

        bboxes[str(frame)] = [
            left,
            top,
            right,
            bottom,
        ]

    meta = {
        "sequence": SEQUENCE,
        "input_frames": INPUT_FRAMES,
        "heldout_frames":
            HELDOUT_FRAMES,
        "target_resolution":
            [TARGET, TARGET],
        "camera_conversion": {
            "co3d_extrinsic": (
                "official right-multiply PyTorch3D "
                "X_cam = X_world @ R + T"
            ),
            "opencv_conversion": (
                "R_cv=diag(-1,-1,1)@R_co3d.T; "
                "t_cv=diag(-1,-1,1)@T_co3d"
            ),
            "intrinsics": (
                "CO3D NDC -> original pixels -> "
                "same VGGT resize/pad affine"
            ),
        },
        "alignment": {
            "method":
                "one global proper Sim(3) from six CO3D GT input camera centers to variant predicted camera centers; no per-camera correction",
            "full": report_full,
            "quant": report_quant,
        },
        "content_bboxes":
            bboxes,
        "image_names":
            image_names,
        "full_source":
            str(
                OUT
                / "sources/full"
            ),
        "quant_source":
            str(
                OUT
                / "sources/quant"
            ),
    }

    (
        OUT
        / "heldout_eval_meta.json"
    ).write_text(
        json.dumps(
            meta,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    np.savez_compressed(
        OUT
        / "heldout_camera_bundle.npz",
        gt_input_extrinsic=
            E_gt_input,
        gt_input_intrinsic=
            K_gt_input,
        gt_heldout_extrinsic=
            E_gt_heldout,
        gt_heldout_intrinsic=
            K_gt_heldout,
        full_heldout_extrinsic=
            heldout_full,
        quant_heldout_extrinsic=
            heldout_quant,
        full_sim3_scale=
            np.asarray(s_full),
        full_sim3_rotation=
            A_full,
        full_sim3_translation=
            b_full,
        quant_sim3_scale=
            np.asarray(s_quant),
        quant_sim3_rotation=
            A_quant,
        quant_sim3_translation=
            b_quant,
        heldout_frames=
            np.asarray(
                HELDOUT_FRAMES,
                np.int64,
            ),
    )

    print()
    print(
        "=============================================="
    )
    print(
        "HELDOUT CAMERA PREPARATION: PASS"
    )
    print(
        "=============================================="
    )
    print(
        "Full source :",
        OUT / "sources/full",
    )
    print(
        "Quant source:",
        OUT / "sources/quant",
    )
    print(
        "Targets     : 9"
    )
    print(
        "No BA, no per-camera correction."
    )


if __name__ == "__main__":
    main()
