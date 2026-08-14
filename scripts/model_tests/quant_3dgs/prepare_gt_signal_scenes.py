#!/usr/bin/env python3

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree


CV = Path("/home/utn/luli38se/cv")

REPO = CV / "3D-Reconstruction"
QUANTVGGT = CV / "QuantVGGT"

CO3D = Path(
    "/var/tmp/luli38se/co3d_single_all"
)

SELECTION = Path(
    "/var/tmp/luli38se/quantsplat/"
    "gt_scene_selection/"
    "frozen_gt_scene_selection.json"
)

OUT = Path(
    "/var/tmp/luli38se/quantsplat/"
    "gt_signal_v1"
)

PREP_HELPER = (
    REPO
    / "scripts/model_tests/quant_3dgs/"
      "prepare_co3d_scene.py"
)

TARGET = 518

CAM_FLIP = np.diag(
    [-1.0, -1.0, 1.0]
)


def sha256_file(path: Path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                16 * 1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def semantic_tensor_sha(tensor):
    a = (
        tensor.detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )

    return hashlib.sha256(
        a.tobytes(order="C")
    ).hexdigest()


def load_jgz(path):
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_helper():
    if not PREP_HELPER.is_file():
        raise FileNotFoundError(
            PREP_HELPER
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            "quantsplat_prepare_helper",
            PREP_HELPER,
        )
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Cannot load prepare_co3d_scene.py"
        )

    module = (
        importlib.util
        .module_from_spec(spec)
    )

    spec.loader.exec_module(
        module
    )

    for name in [
        "import_local_preprocessor",
        "pad_transform",
    ]:
        if not hasattr(module, name):
            raise RuntimeError(
                f"Helper missing {name}"
            )

    return module


HELPER = load_helper()

PREPROCESSOR, _, PREPROCESS_SIGNATURE = (
    HELPER.import_local_preprocessor(
        QUANTVGGT
    )
)


def preprocess(paths):
    kwargs = {
        "mode": "pad",
    }

    if (
        "target_size"
        in PREPROCESS_SIGNATURE.parameters
    ):
        kwargs[
            "target_size"
        ] = TARGET

    tensor = PREPROCESSOR(
        [
            str(x)
            for x in paths
        ],
        **kwargs,
    )

    tensor = (
        tensor.detach()
        .cpu()
        .contiguous()
    )

    return tensor


def co3d_to_opencv(record):
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
        raise RuntimeError(
            "Bad CO3D rotation"
        )

    if T_p3d.shape != (3,):
        raise RuntimeError(
            "Bad CO3D translation"
        )

    # CO3D:
    # row-vector X_cam = X_world @ R + T
    #
    # Column representation:
    # X_cam = R.T @ X_world + T
    #
    # PyTorch3D camera axes -> OpenCV:
    # x: left -> right
    # y: up   -> down
    # z: forward unchanged
    R_cv = (
        CAM_FLIP
        @ R_p3d.T
    )

    t_cv = (
        CAM_FLIP
        @ T_p3d
    )

    if abs(
        np.linalg.det(R_cv)
        - 1.0
    ) > 1e-5:
        raise RuntimeError(
            "Camera rotation determinant failure"
        )

    if (
        np.linalg.norm(
            R_cv @ R_cv.T
            - np.eye(3)
        )
        > 1e-5
    ):
        raise RuntimeError(
            "Camera orthogonality failure"
        )

    H, W = [
        int(x)
        for x in
        record["image"]["size"]
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

        common = (
            min(H, W)
            / 2.0
        )

        sx = common
        sy = common

    else:
        raise RuntimeError(
            f"Unsupported intrinsics format "
            f"{fmt}"
        )

    fx = focal[0] * sx
    fy = focal[1] * sy

    cx = (
        W / 2.0
        - principal[0] * sx
    )

    cy = (
        H / 2.0
        - principal[1] * sy
    )

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


def K_to_518(
    K,
    transform,
):
    A = np.asarray(
        transform[
            "original_to_vggt_affine"
        ],
        dtype=np.float64,
    )

    return A @ K


def resize_depth_and_masks(
    depth,
    depth_mask,
    fg_mask,
    transform,
):

    rw = int(
        transform[
            "resized_width"
        ]
    )

    rh = int(
        transform[
            "resized_height"
        ]
    )

    left = int(
        transform[
            "padding"
        ]["left"]
    )

    top = int(
        transform[
            "padding"
        ]["top"]
    )

    depth_img = Image.fromarray(
        depth.astype(
            np.float32
        ),
        mode="F",
    )

    dm_img = Image.fromarray(
        (
            depth_mask.astype(
                np.uint8
            )
            * 255
        ),
        mode="L",
    )

    fg_img = Image.fromarray(
        fg_mask.astype(
            np.uint8
        ),
        mode="L",
    )

    depth_r = np.asarray(
        depth_img.resize(
            (rw, rh),
            resample=
                Image.Resampling.NEAREST,
        ),
        dtype=np.float32,
    )

    dm_r = (
        np.asarray(
            dm_img.resize(
                (rw, rh),
                resample=
                    Image.Resampling.NEAREST,
            )
        )
        > 0
    )

    fg_r = (
        np.asarray(
            fg_img.resize(
                (rw, rh),
                resample=
                    Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )
        / 255.0
    )

    out_depth = np.zeros(
        (TARGET, TARGET),
        dtype=np.float32,
    )

    out_dm = np.zeros(
        (TARGET, TARGET),
        dtype=bool,
    )

    out_fg = np.zeros(
        (TARGET, TARGET),
        dtype=np.float32,
    )

    out_depth[
        top:top+rh,
        left:left+rw,
    ] = depth_r

    out_dm[
        top:top+rh,
        left:left+rw,
    ] = dm_r

    out_fg[
        top:top+rh,
        left:left+rw,
    ] = fg_r

    valid = (
        out_dm
        & np.isfinite(out_depth)
        & (out_depth > 0)
    )

    return (
        out_depth,
        valid,
        out_fg,
    )


def unproject_to_world(
    depth,
    valid,
    K,
    E,
):
    H, W = depth.shape

    yy, xx = np.meshgrid(
        np.arange(
            H,
            dtype=np.float64,
        ),
        np.arange(
            W,
            dtype=np.float64,
        ),
        indexing="ij",
    )

    z = depth.astype(
        np.float64
    )

    x = (
        (xx - K[0, 2])
        / K[0, 0]
        * z
    )

    y = (
        (yy - K[1, 2])
        / K[1, 1]
        * z
    )

    cam = np.stack(
        [x, y, z],
        axis=-1,
    )

    R = E[:, :3]
    t = E[:, 3]

    # Column convention:
    # x_cam = R x_world + t
    #
    # For row-array points:
    # x_world = (x_cam - t) @ R
    world = (
        cam - t
    ) @ R

    world[
        ~valid
    ] = np.nan

    return world.astype(
        np.float32
    )


def read_point_cloud(path):
    ply = PlyData.read(
        str(path)
    )

    vertex = (
        ply["vertex"].data
    )

    names = (
        vertex.dtype.names
        or ()
    )

    for key in [
        "x",
        "y",
        "z",
    ]:
        if key not in names:
            raise RuntimeError(
                f"Point cloud missing {key}"
            )

    xyz = np.stack(
        [
            np.asarray(
                vertex["x"],
                dtype=np.float64,
            ),
            np.asarray(
                vertex["y"],
                dtype=np.float64,
            ),
            np.asarray(
                vertex["z"],
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    xyz = xyz[
        np.isfinite(
            xyz
        ).all(axis=1)
    ]

    if len(xyz) < 1000:
        raise RuntimeError(
            "Point cloud too small"
        )

    return xyz


def reference_radius(points):
    center = np.median(
        points,
        axis=0,
    )

    radius = float(
        np.quantile(
            np.linalg.norm(
                points - center,
                axis=1,
            ),
            0.90,
        )
    )

    if (
        not np.isfinite(radius)
        or radius <= 0
    ):
        raise RuntimeError(
            "Invalid point-cloud radius"
        )

    return radius


def validate_gt_vs_cloud(
    gt_world,
    point_cloud,
    seed,
):
    tree = cKDTree(
        point_cloud
    )

    radius = (
        reference_radius(
            point_cloud
        )
    )

    rng = np.random.default_rng(
        seed
    )

    errors = []

    per_frame = []

    for i in range(
        gt_world.shape[0]
    ):

        pts = (
            gt_world[i]
            .reshape(-1, 3)
        )

        pts = pts[
            np.isfinite(
                pts
            ).all(axis=1)
        ]

        if len(pts) > 10000:
            idx = rng.choice(
                len(pts),
                10000,
                replace=False,
            )

            pts = pts[idx]

        dist, _ = tree.query(
            pts,
            k=1,
            workers=-1,
        )

        errors.append(dist)

        per_frame.append(
            float(
                np.median(dist)
                / radius
            )
        )

    errors = np.concatenate(
        errors
    )

    median_norm = float(
        np.median(errors)
        / radius
    )

    return (
        radius,
        median_norm,
        per_frame,
    )


selection = json.loads(
    SELECTION.read_text()
)

selected = selection[
    "selected"
]

if len(selected) != 3:
    raise RuntimeError(
        f"Expected 3 frozen scenes, "
        f"got {len(selected)}"
    )


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

aggregate = {
    "version": 1,

    "purpose":
        (
            "Frozen GT-enabled scenes for "
            "Full-vs-W4A4 disagreement "
            "supervision validation"
        ),

    "preprocessing": {
        "target":
            TARGET,

        "mode":
            "pad",

        "rgb_range":
            "[0,1]",

        "depth_resize":
            "nearest",

        "depth_mask_resize":
            "nearest",

        "foreground_mask_resize":
            "bilinear",

        "padding_rgb":
            1.0,

        "padding_depth":
            0.0,
    },

    "scenes": [],
}


print()
print(
    "============================================================"
)
print(
    "PREPARING THREE GT SIGNAL-VALIDATION SCENES"
)
print(
    "============================================================"
)


for scene_idx, scene in enumerate(
    selected,
    start=1,
):

    category = scene[
        "category"
    ]

    sequence = scene[
        "sequence_name"
    ]

    input_frames = [
        int(x)
        for x in scene[
            "input_6"
        ]
    ]

    heldout_frames = [
        int(x)
        for x in scene[
            "heldout_9"
        ]
    ]

    if len(input_frames) != 6:
        raise RuntimeError(
            "input_6 != 6"
        )

    if len(heldout_frames) != 9:
        raise RuntimeError(
            "heldout_9 != 9"
        )

    scene_root = (
        CO3D
        / category
        / sequence
    )

    frame_ann = (
        CO3D
        / category
        / "frame_annotations.jgz"
    )

    seq_ann = (
        CO3D
        / category
        / "sequence_annotations.jgz"
    )

    records_all = load_jgz(
        frame_ann
    )

    records = {
        int(r["frame_number"]):
            r
        for r in records_all
        if r.get(
            "sequence_name"
        ) == sequence
    }

    seq_records = load_jgz(
        seq_ann
    )

    seq_match = [
        r
        for r in seq_records
        if r.get(
            "sequence_name"
        ) == sequence
    ]

    if len(seq_match) != 1:
        raise RuntimeError(
            "Sequence annotation mismatch"
        )

    seq_record = seq_match[0]

    pc_meta = seq_record.get(
        "point_cloud"
    )

    if not isinstance(
        pc_meta,
        dict,
    ):
        raise RuntimeError(
            f"{category}/{sequence}: "
            "point_cloud unavailable"
        )

    point_cloud_path = (
        CO3D
        / pc_meta["path"]
    )

    if not point_cloud_path.is_file():
        raise FileNotFoundError(
            point_cloud_path
        )

    all_selected = (
        input_frames
        + heldout_frames
    )

    for frame in all_selected:

        if frame not in records:
            raise RuntimeError(
                f"Missing frame {frame}"
            )

        r = records[frame]

        if not isinstance(
            r.get("viewpoint"),
            dict,
        ):
            raise RuntimeError(
                f"Frame {frame}: "
                "camera missing"
            )

        if not isinstance(
            r.get("depth"),
            dict,
        ):
            raise RuntimeError(
                f"Frame {frame}: "
                "depth metadata missing"
            )

        if not isinstance(
            r.get("mask"),
            dict,
        ):
            raise RuntimeError(
                f"Frame {frame}: "
                "foreground mask missing"
            )

    scene_out = (
        OUT
        / category
        / sequence
    )

    scene_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_paths = [
        (
            CO3D
            / records[f][
                "image"
            ]["path"]
        ).resolve()
        for f in input_frames
    ]

    tensor = preprocess(
        input_paths
    )

    if tensor.shape != (
        6,
        3,
        TARGET,
        TARGET,
    ):
        raise RuntimeError(
            f"{category}/{sequence}: "
            f"bad tensor shape "
            f"{tuple(tensor.shape)}"
        )

    if not torch.isfinite(
        tensor
    ).all():
        raise RuntimeError(
            "Input tensor non-finite"
        )

    tmin = float(
        tensor.min()
    )

    tmax = float(
        tensor.max()
    )

    if (
        tmin < -1e-6
        or tmax > 1.000001
    ):
        raise RuntimeError(
            "Unexpected RGB range"
        )

    tensor_path = (
        scene_out
        / "input_tensor.pt"
    )

    torch.save(
        tensor,
        tensor_path,
    )

    semantic_sha = (
        semantic_tensor_sha(
            tensor
        )
    )

    file_sha = sha256_file(
        tensor_path
    )

    # Save exact 518 RGB views too.
    rgb_dir = (
        scene_out
        / "input_rgb_518"
    )

    rgb_dir.mkdir(
        exist_ok=True
    )

    for i, frame in enumerate(
        input_frames
    ):
        arr = (
            tensor[i]
            .permute(1, 2, 0)
            .clamp(0, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .numpy()
        )

        Image.fromarray(
            arr,
            mode="RGB",
        ).save(
            rgb_dir
            / f"frame_{frame}.png"
        )

    depths = []
    depth_valids = []
    fg_masks = []
    intrinsics = []
    extrinsics = []
    world_points = []

    transforms = []

    valid_counts = []

    for frame in input_frames:

        record = records[
            frame
        ]

        image_path = (
            CO3D
            / record[
                "image"
            ]["path"]
        )

        with Image.open(
            image_path
        ) as img:
            W, H = img.size

        ann_H, ann_W = [
            int(x)
            for x in record[
                "image"
            ]["size"]
        ]

        if (
            W != ann_W
            or H != ann_H
        ):
            raise RuntimeError(
                f"Frame {frame}: "
                "annotation/image size mismatch"
            )

        transform = (
            HELPER.pad_transform(
                W,
                H,
                TARGET,
            )
        )

        transforms.append(
            transform
        )

        E, K_original = (
            co3d_to_opencv(
                record
            )
        )

        K_518 = K_to_518(
            K_original,
            transform,
        )

        depth_meta = (
            record["depth"]
        )

        depth_path = (
            CO3D
            / depth_meta[
                "path"
            ]
        )

        dm_path = (
            CO3D
            / depth_meta[
                "mask_path"
            ]
        )

        fg_path = (
            CO3D
            / record[
                "mask"
            ]["path"]
        )

        for p in [
            depth_path,
            dm_path,
            fg_path,
        ]:
            if not p.is_file():
                raise FileNotFoundError(
                    p
                )

        # CO3D_FLOAT16_BITPATTERN_FIX
        # Official PyTorch3D/CO3D PNG decoding:
        # the uint16 PNG stores FLOAT16 BIT PATTERNS,
        # not integer-valued depths.
        with Image.open(
            depth_path
        ) as img:
            depth_raw = (
                np.frombuffer(
                    np.array(
                        img,
                        dtype=np.uint16,
                    ),
                    dtype=np.float16,
                )
                .astype(np.float32)
                .reshape(
                    (
                        img.size[1],
                        img.size[0],
                    )
                )
            )

        with Image.open(
            dm_path
        ) as img:
            dm_raw = np.asarray(
                img
            )

        with Image.open(
            fg_path
        ) as img:
            fg_raw = np.asarray(
                img.convert("L")
            )

        if (
            depth_raw.shape
            != (H, W)
        ):
            raise RuntimeError(
                f"Frame {frame}: "
                f"depth shape "
                f"{depth_raw.shape} "
                f"!= {(H,W)}"
            )

        if (
            dm_raw.shape
            != (H, W)
        ):
            raise RuntimeError(
                f"Frame {frame}: "
                "depth-mask shape mismatch"
            )

        if (
            fg_raw.shape
            != (H, W)
        ):
            raise RuntimeError(
                f"Frame {frame}: "
                "foreground-mask shape mismatch"
            )

        scale_adjustment = float(
            depth_meta.get(
                "scale_adjustment",
                1.0,
            )
        )

        depth_actual = (
            depth_raw.astype(
                np.float32
            )
            * scale_adjustment
        )

        (
            depth_518,
            valid_518,
            fg_518,
        ) = resize_depth_and_masks(
            depth_actual,
            dm_raw != 0,
            fg_raw,
            transform,
        )

        # GT geometry region:
        # valid CO3D depth AND foreground.
        gt_valid = (
            valid_518
            & (fg_518 >= 0.5)
        )

        count = int(
            gt_valid.sum()
        )

        valid_counts.append(
            count
        )

        if count < 1000:
            raise RuntimeError(
                f"Frame {frame}: only "
                f"{count} valid foreground "
                "GT-depth pixels"
            )

        world = (
            unproject_to_world(
                depth_518,
                gt_valid,
                K_518,
                E,
            )
        )

        depths.append(
            depth_518
        )

        depth_valids.append(
            gt_valid
        )

        fg_masks.append(
            fg_518
        )

        intrinsics.append(
            K_518
        )

        extrinsics.append(
            E
        )

        world_points.append(
            world
        )

    depths = np.stack(
        depths
    )

    depth_valids = np.stack(
        depth_valids
    )

    fg_masks = np.stack(
        fg_masks
    )

    intrinsics = np.stack(
        intrinsics
    )

    extrinsics = np.stack(
        extrinsics
    )

    world_points = np.stack(
        world_points
    )

    point_cloud = (
        read_point_cloud(
            point_cloud_path
        )
    )

    (
        pc_radius,
        gt_pc_median,
        gt_pc_per_frame,
    ) = validate_gt_vs_cloud(
        world_points,
        point_cloud,
        seed=
            20260813 + scene_idx,
    )

    # Loose convention sanity gate.
    # We are not tuning the metric to this.
    if gt_pc_median > 0.10:
        raise RuntimeError(
            f"{category}/{sequence}: "
            "GT depth/camera unprojection "
            "does not agree with the "
            "official sequence point cloud. "
            f"Median NN="
            f"{100*gt_pc_median:.3f}% "
            "of reference radius."
        )

    bundle_path = (
        scene_out
        / "gt_bundle.npz"
    )

    np.savez_compressed(
        bundle_path,

        frame_numbers=
            np.asarray(
                input_frames,
                dtype=np.int64,
            ),

        depth=
            depths.astype(
                np.float32
            ),

        depth_valid=
            depth_valids,

        foreground_mask=
            fg_masks.astype(
                np.float32
            ),

        intrinsic=
            intrinsics.astype(
                np.float64
            ),

        extrinsic=
            extrinsics.astype(
                np.float64
            ),

        world_points=
            world_points.astype(
                np.float32
            ),
    )

    scene_meta = {
        "category":
            category,

        "sequence_name":
            sequence,

        "input_6":
            input_frames,

        "heldout_9":
            heldout_frames,

        "set_list":
            scene[
                "set_list"
            ],

        "point_cloud": {
            "path":
                str(
                    point_cloud_path
                ),

            "sha256":
                sha256_file(
                    point_cloud_path
                ),

            "quality_score":
                pc_meta.get(
                    "quality_score"
                ),

            "n_points_annotation":
                pc_meta.get(
                    "n_points"
                ),

            "n_points_loaded":
                int(
                    len(
                        point_cloud
                    )
                ),

            "robust_radius":
                pc_radius,
        },

        "viewpoint_quality_score":
            seq_record.get(
                "viewpoint_quality_score"
            ),

        "input_tensor": {
            "path":
                str(
                    tensor_path
                ),

            "shape":
                list(
                    tensor.shape
                ),

            "dtype":
                str(
                    tensor.dtype
                ),

            "range":
                [
                    tmin,
                    tmax,
                ],

            "file_sha256":
                file_sha,

            "semantic_sha256":
                semantic_sha,
        },

        "gt_bundle": {
            "path":
                str(
                    bundle_path
                ),

            "sha256":
                sha256_file(
                    bundle_path
                ),

            "valid_foreground_depth_pixels_per_frame":
                valid_counts,

            "gt_depth_to_reference_cloud_median_nn_fraction_radius":
                gt_pc_median,

            "gt_depth_to_reference_cloud_per_frame_fraction_radius":
                gt_pc_per_frame,
        },

        "preprocess_transforms":
            transforms,
    }

    meta_path = (
        scene_out
        / "scene_meta.json"
    )

    meta_path.write_text(
        json.dumps(
            scene_meta,
            indent=2,
        )
        + "\n"
    )

    aggregate[
        "scenes"
    ].append(
        scene_meta
    )

    print()
    print(
        f"[{scene_idx}/3] "
        f"{category} / {sequence}"
    )

    print(
        "  frames:",
        input_frames
    )

    print(
        "  tensor:",
        tuple(
            tensor.shape
        )
    )

    print(
        "  tensor file SHA:",
        file_sha
    )

    print(
        "  tensor semantic SHA:",
        semantic_sha
    )

    print(
        "  valid GT pixels:",
        valid_counts
    )

    print(
        "  GT depth -> PC median NN:",
        f"{100*gt_pc_median:.4f}% radius"
    )

    print(
        "  GT bundle:",
        bundle_path
    )


aggregate_path = (
    OUT
    / "gt_signal_manifest.json"
)

aggregate_path.write_text(
    json.dumps(
        aggregate,
        indent=2,
    )
    + "\n"
)


print()
print(
    "============================================================"
)
print(
    "GT-SIGNAL PREPARATION SUMMARY"
)
print(
    "============================================================"
)

for scene in aggregate[
    "scenes"
]:

    print(
        f"{scene['category']} / "
        f"{scene['sequence_name']}"
    )

    print(
        "  input SHA:",
        scene[
            "input_tensor"
        ][
            "file_sha256"
        ]
    )

    print(
        "  semantic SHA:",
        scene[
            "input_tensor"
        ][
            "semantic_sha256"
        ]
    )

    print(
        "  valid pixels min/max:",
        min(
            scene[
                "gt_bundle"
            ][
                "valid_foreground_depth_pixels_per_frame"
            ]
        ),
        "/",
        max(
            scene[
                "gt_bundle"
            ][
                "valid_foreground_depth_pixels_per_frame"
            ]
        ),
    )

    print(
        "  GT->PC median:",
        f"{100*scene['gt_bundle']['gt_depth_to_reference_cloud_median_nn_fraction_radius']:.4f}% radius"
    )

print()
print(
    "Aggregate manifest:",
    aggregate_path
)

print(
    "Aggregate manifest SHA:",
    sha256_file(
        aggregate_path
    )
)

print()
print(
    "============================================================"
)
print(
    "GT-SIGNAL PREPARATION: PASS"
)
print(
    "============================================================"
)
