#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


GT_ROOT = Path(
    "/var/tmp/luli38se/quantsplat/"
    "gt_signal_v1"
)

PRED_ROOT = Path(
    "/var/tmp/luli38se/quantsplat/"
    "gt_signal_v1_predictions"
)

OUT = Path(
    "/var/tmp/luli38se/quantsplat/"
    "gt_signal_v1_analysis"
)

SCENES = [
    (
        "toytruck",
        "190_20494_39385",
    ),
    (
        "bench",
        "415_57112_110099",
    ),
    (
        "toaster",
        "372_41229_82130",
    ),
]

EXPECTED_FRAMES = {
    "toytruck/190_20494_39385":
        [1, 41, 75, 123, 162, 202],

    "bench/415_57112_110099":
        [1, 41, 81, 122, 162, 202],

    "toaster/372_41229_82130":
        [1, 41, 81, 122, 162, 202],
}

SAMPLE_PER_FRAME = 5000
RNG_SEED = 20260813


def camera_center(E):
    R = E[:, :3]
    t = E[:, 3]
    return -R.T @ t


def centers(E):
    return np.stack(
        [
            camera_center(x)
            for x in E
        ]
    )


def umeyama(src, dst):
    src = np.asarray(
        src,
        np.float64,
    )

    dst = np.asarray(
        dst,
        np.float64,
    )

    ms = src.mean(axis=0)
    md = dst.mean(axis=0)

    X = src - ms
    Y = dst - md

    cov = (
        Y.T @ X
        / len(src)
    )

    U, S, Vt = np.linalg.svd(
        cov
    )

    D = np.eye(3)

    if np.linalg.det(
        U @ Vt
    ) < 0:
        D[-1, -1] = -1.0

    A = U @ D @ Vt

    var = (
        np.sum(X * X)
        / len(src)
    )

    if var <= 0:
        raise RuntimeError(
            "Degenerate camera trajectory"
        )

    scale = (
        np.sum(
            S * np.diag(D)
        )
        / var
    )

    b = (
        md
        - scale
        * (A @ ms)
    )

    return (
        float(scale),
        A,
        b,
    )


def pred_to_gt(
    P,
    s,
    A,
    b,
):
    return (
        (P - b)
        @ A
    ) / s


def angle_deg(R):
    c = np.clip(
        (
            np.trace(R)
            - 1.0
        )
        / 2.0,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(c)
        )
    )


def average_ranks(x):
    x = np.asarray(
        x,
        np.float64,
    )

    order = np.argsort(
        x,
        kind="mergesort",
    )

    xs = x[order]

    ranks = np.empty(
        len(x),
        np.float64,
    )

    i = 0

    while i < len(x):
        j = i + 1

        while (
            j < len(x)
            and xs[j] == xs[i]
        ):
            j += 1

        ranks[
            order[i:j]
        ] = (
            i + j - 1
        ) / 2.0

        i = j

    return ranks


def pearson(x, y):
    x = np.asarray(
        x,
        np.float64,
    )

    y = np.asarray(
        y,
        np.float64,
    )

    x -= x.mean()
    y -= y.mean()

    d = np.sqrt(
        np.sum(x*x)
        * np.sum(y*y)
    )

    if d <= 0:
        return float("nan")

    return float(
        np.sum(x*y) / d
    )


def spearman(x, y):
    return pearson(
        average_ranks(x),
        average_ranks(y),
    )


def roc_auc(score, positive):
    score = np.asarray(
        score,
        np.float64,
    )

    positive = np.asarray(
        positive,
        bool,
    )

    np_ = int(
        positive.sum()
    )

    nn = (
        len(positive)
        - np_
    )

    if np_ == 0 or nn == 0:
        return float("nan")

    ranks = (
        average_ranks(score)
        + 1.0
    )

    return float(
        (
            ranks[
                positive
            ].sum()
            - np_
            * (np_ + 1)
            / 2.0
        )
        / (
            np_
            * nn
        )
    )


def top_precision(
    score,
    target,
    frac=0.10,
):
    st = np.quantile(
        score,
        1.0-frac,
    )

    tt = np.quantile(
        target,
        1.0-frac,
    )

    chosen = score >= st

    if chosen.sum() == 0:
        return float("nan")

    return float(
        np.mean(
            target[chosen]
            >= tt
        )
    )


def metric_pack(
    disagreement,
    quant_error,
    full_error,
):
    excess = np.maximum(
        quant_error
        - full_error,
        0.0,
    )

    q90 = np.quantile(
        quant_error,
        0.90,
    )

    e90 = np.quantile(
        excess,
        0.90,
    )

    return {
        "n":
            int(
                len(disagreement)
            ),

        "full_error_median":
            float(
                np.median(
                    full_error
                )
            ),

        "quant_error_median":
            float(
                np.median(
                    quant_error
                )
            ),

        "disagreement_median":
            float(
                np.median(
                    disagreement
                )
            ),

        "excess_error_median":
            float(
                np.median(
                    excess
                )
            ),

        "rho_disagreement_quant_error":
            spearman(
                disagreement,
                quant_error,
            ),

        "rho_disagreement_full_error":
            spearman(
                disagreement,
                full_error,
            ),

        "rho_disagreement_excess_error":
            spearman(
                disagreement,
                excess,
            ),

        "auc_top10_quant_error":
            roc_auc(
                disagreement,
                quant_error >= q90,
            ),

        "auc_top10_excess_error":
            roc_auc(
                disagreement,
                excess >= e90,
            ),

        "top10_precision_quant_error":
            top_precision(
                disagreement,
                quant_error,
            ),

        "top10_precision_excess_error":
            top_precision(
                disagreement,
                excess,
            ),
    }


def directory_hash(root):
    h = hashlib.sha256()

    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
    )

    entries = []

    for p in files:
        fh = hashlib.sha256(
            p.read_bytes()
        ).hexdigest()

        rel = str(
            p.relative_to(root)
        )

        size = p.stat().st_size

        text = (
            f"{rel}\0{size}\0{fh}\n"
        )

        h.update(
            text.encode()
        )

        entries.append({
            "relative_path":
                rel,

            "size_bytes":
                size,

            "sha256":
                fh,
        })

    return {
        "root":
            str(root.resolve()),

        "content_sha256":
            h.hexdigest(),

        "files":
            entries,
    }


all_dis = []
all_qerr = []
all_ferr = []

all_depth_q = []
all_depth_f = []

scene_results = []

for si, (
    category,
    sequence,
) in enumerate(SCENES):

    key = (
        f"{category}/{sequence}"
    )

    print()
    print(
        "============================================================"
    )
    print(key)
    print(
        "============================================================"
    )

    scene_root = (
        GT_ROOT
        / category
        / sequence
    )

    meta = json.loads(
        (
            scene_root
            / "scene_meta.json"
        ).read_text()
    )

    radius = float(
        meta[
            "point_cloud"
        ][
            "robust_radius"
        ]
    )

    with np.load(
        scene_root
        / "gt_bundle.npz",
        allow_pickle=False,
    ) as g:

        frames = (
            g[
                "frame_numbers"
            ].tolist()
        )

        Egt = (
            g["extrinsic"]
            .astype(np.float64)
        )

        Kgt = (
            g["intrinsic"]
            .astype(np.float64)
        )

        Pgt = (
            g["world_points"]
            .astype(np.float64)
        )

        Dgt = (
            g["depth"]
            .astype(np.float64)
        )

        valid = (
            g[
                "depth_valid"
            ].astype(bool)
        )

    if (
        frames
        != EXPECTED_FRAMES[key]
    ):
        raise RuntimeError(
            f"{key}: GT frame order changed"
        )

    fp = (
        PRED_ROOT
        / category
        / sequence
        / "full"
        / "predictions.npz"
    )

    qp = (
        PRED_ROOT
        / category
        / sequence
        / "w4a4"
        / "predictions.npz"
    )

    if not fp.is_file():
        raise FileNotFoundError(fp)

    if not qp.is_file():
        raise FileNotFoundError(qp)

    with np.load(
        fp,
        allow_pickle=False,
    ) as f:

        if (
            f[
                "frame_numbers"
            ].tolist()
            != frames
        ):
            raise RuntimeError(
                f"{key}: Full frames differ"
            )

        Ef = (
            f["extrinsic"]
            .astype(np.float64)
        )

        Kf = (
            f["intrinsic"]
            .astype(np.float64)
        )

        Pf = (
            f[
                "world_points_from_depth"
            ].astype(np.float64)
        )

        Df = (
            f["depth"][
                ..., 0
            ].astype(np.float64)
        )

    with np.load(
        qp,
        allow_pickle=False,
    ) as q:

        if (
            q[
                "frame_numbers"
            ].tolist()
            != frames
        ):
            raise RuntimeError(
                f"{key}: Quant frames differ"
            )

        Eq = (
            q["extrinsic"]
            .astype(np.float64)
        )

        Kq = (
            q["intrinsic"]
            .astype(np.float64)
        )

        Pq = (
            q[
                "world_points_from_depth"
            ].astype(np.float64)
        )

        Dq = (
            q["depth"][
                ..., 0
            ].astype(np.float64)
        )

    Cgt = centers(Egt)
    Cf = centers(Ef)
    Cq = centers(Eq)

    sf, Af, bf = umeyama(
        Cgt,
        Cf,
    )

    sq, Aq, bq = umeyama(
        Cgt,
        Cq,
    )

    Pf_gt = pred_to_gt(
        Pf,
        sf,
        Af,
        bf,
    )

    Pq_gt = pred_to_gt(
        Pq,
        sq,
        Aq,
        bq,
    )

    # Camera validation after scene-level Sim(3).
    def camera_report(
        Epred,
        Kpred,
        s,
        A,
        b,
    ):
        center_err = []
        rot_err = []

        for i in range(6):

            C_expected = (
                s
                * (
                    A
                    @ Cgt[i]
                )
                + b
            )

            R_expected = (
                Egt[
                    i, :, :3
                ]
                @ A.T
            )

            center_err.append(
                np.linalg.norm(
                    camera_center(
                        Epred[i]
                    )
                    - C_expected
                )
                / (
                    s * radius
                )
            )

            rot_err.append(
                angle_deg(
                    Epred[
                        i, :, :3
                    ]
                    @ R_expected.T
                )
            )

        focal_rel = np.concatenate(
            [
                np.abs(
                    Kpred[:,0,0]
                    - Kgt[:,0,0]
                )
                / np.maximum(
                    np.abs(
                        Kgt[:,0,0]
                    ),
                    1e-12,
                ),

                np.abs(
                    Kpred[:,1,1]
                    - Kgt[:,1,1]
                )
                / np.maximum(
                    np.abs(
                        Kgt[:,1,1]
                    ),
                    1e-12,
                ),
            ]
        )

        return {
            "scale":
                float(s),

            "center_error_median_fraction_gt_radius":
                float(
                    np.median(
                        center_err
                    )
                ),

            "center_error_max_fraction_gt_radius":
                float(
                    np.max(
                        center_err
                    )
                ),

            "rotation_error_median_deg":
                float(
                    np.median(
                        rot_err
                    )
                ),

            "rotation_error_max_deg":
                float(
                    np.max(
                        rot_err
                    )
                ),

            "focal_relative_error_median":
                float(
                    np.median(
                        focal_rel
                    )
                ),
        }

    cam_f = camera_report(
        Ef,
        Kf,
        sf,
        Af,
        bf,
    )

    cam_q = camera_report(
        Eq,
        Kq,
        sq,
        Aq,
        bq,
    )

    # Hard sanity gate before geometry analysis.
    if (
        cam_f[
            "center_error_median_fraction_gt_radius"
        ] > 0.25
        or
        cam_q[
            "center_error_median_fraction_gt_radius"
        ] > 0.25
        or
        cam_f[
            "rotation_error_median_deg"
        ] > 20
        or
        cam_q[
            "rotation_error_median_deg"
        ] > 20
    ):
        raise RuntimeError(
            f"{key}: camera alignment "
            "sanity gate failed"
        )

    scene_dis = []
    scene_qerr = []
    scene_ferr = []

    scene_dq = []
    scene_df = []

    per_frame = []

    for i, frame in enumerate(
        frames
    ):

        v = (
            valid[i]
            & np.isfinite(
                Pgt[i]
            ).all(axis=2)
            & np.isfinite(
                Pf_gt[i]
            ).all(axis=2)
            & np.isfinite(
                Pq_gt[i]
            ).all(axis=2)
        )

        idx = np.flatnonzero(
            v.reshape(-1)
        )

        if len(idx) < SAMPLE_PER_FRAME:
            raise RuntimeError(
                f"{key}/frame {frame}: "
                f"only {len(idx)} valid pixels; "
                f"need {SAMPLE_PER_FRAME}"
            )

        rng = np.random.default_rng(
            RNG_SEED
            + si * 100
            + i
        )

        idx = rng.choice(
            idx,
            SAMPLE_PER_FRAME,
            replace=False,
        )

        pg = (
            Pgt[i]
            .reshape(-1, 3)[idx]
        )

        pf = (
            Pf_gt[i]
            .reshape(-1, 3)[idx]
        )

        pq = (
            Pq_gt[i]
            .reshape(-1, 3)[idx]
        )

        ferr = (
            np.linalg.norm(
                pf - pg,
                axis=1,
            )
            / radius
        )

        qerr = (
            np.linalg.norm(
                pq - pg,
                axis=1,
            )
            / radius
        )

        dis = (
            np.linalg.norm(
                pq - pf,
                axis=1,
            )
            / radius
        )

        # Secondary scalar depth comparison.
        gt_d = (
            Dgt[i]
            .reshape(-1)[idx]
        )

        full_d = (
            Df[i]
            .reshape(-1)[idx]
            / sf
        )

        quant_d = (
            Dq[i]
            .reshape(-1)[idx]
            / sq
        )

        positive = (
            np.isfinite(gt_d)
            & np.isfinite(full_d)
            & np.isfinite(quant_d)
            & (gt_d > 0)
            & (full_d > 0)
            & (quant_d > 0)
        )

        denom = np.maximum(
            np.abs(
                gt_d[positive]
            ),
            1e-12,
        )

        df_absrel = (
            np.abs(
                full_d[positive]
                - gt_d[positive]
            )
            / denom
        )

        dq_absrel = (
            np.abs(
                quant_d[positive]
                - gt_d[positive]
            )
            / denom
        )

        pack = metric_pack(
            dis,
            qerr,
            ferr,
        )

        pack.update({
            "frame":
                int(frame),

            "full_scalar_depth_absrel_median":
                float(
                    np.median(
                        df_absrel
                    )
                ),

            "quant_scalar_depth_absrel_median":
                float(
                    np.median(
                        dq_absrel
                    )
                ),
        })

        per_frame.append(
            pack
        )

        scene_dis.append(dis)
        scene_qerr.append(qerr)
        scene_ferr.append(ferr)

        scene_df.append(
            df_absrel
        )

        scene_dq.append(
            dq_absrel
        )

    scene_dis = np.concatenate(
        scene_dis
    )

    scene_qerr = np.concatenate(
        scene_qerr
    )

    scene_ferr = np.concatenate(
        scene_ferr
    )

    scene_pack = metric_pack(
        scene_dis,
        scene_qerr,
        scene_ferr,
    )

    scene_pack[
        "full_scalar_depth_absrel_median"
    ] = float(
        np.median(
            np.concatenate(
                scene_df
            )
        )
    )

    scene_pack[
        "quant_scalar_depth_absrel_median"
    ] = float(
        np.median(
            np.concatenate(
                scene_dq
            )
        )
    )

    all_dis.append(
        scene_dis
    )

    all_qerr.append(
        scene_qerr
    )

    all_ferr.append(
        scene_ferr
    )

    all_depth_f.extend(
        scene_df
    )

    all_depth_q.extend(
        scene_dq
    )

    scene_results.append({
        "category":
            category,

        "sequence":
            sequence,

        "reference_radius":
            radius,

        "camera_full":
            cam_f,

        "camera_quant":
            cam_q,

        "geometry":
            scene_pack,

        "per_frame":
            per_frame,
    })

    print(
        "Full camera center med:",
        f"{100*cam_f['center_error_median_fraction_gt_radius']:.3f}% radius"
    )

    print(
        "Quant camera center med:",
        f"{100*cam_q['center_error_median_fraction_gt_radius']:.3f}% radius"
    )

    print(
        "Full/Quant rot med:",
        f"{cam_f['rotation_error_median_deg']:.3f} / "
        f"{cam_q['rotation_error_median_deg']:.3f} deg"
    )

    print(
        "3D median Full/Quant/dis:",
        f"{100*scene_pack['full_error_median']:.3f}% / "
        f"{100*scene_pack['quant_error_median']:.3f}% / "
        f"{100*scene_pack['disagreement_median']:.3f}% radius"
    )

    print(
        "rho(dis,Qerr/excess):",
        f"{scene_pack['rho_disagreement_quant_error']:.4f} / "
        f"{scene_pack['rho_disagreement_excess_error']:.4f}"
    )

    print(
        "AUC(Qerr/excess):",
        f"{scene_pack['auc_top10_quant_error']:.4f} / "
        f"{scene_pack['auc_top10_excess_error']:.4f}"
    )


DIS = np.concatenate(
    all_dis
)

QERR = np.concatenate(
    all_qerr
)

FERR = np.concatenate(
    all_ferr
)

pooled = metric_pack(
    DIS,
    QERR,
    FERR,
)

pooled[
    "full_scalar_depth_absrel_median"
] = float(
    np.median(
        np.concatenate(
            all_depth_f
        )
    )
)

pooled[
    "quant_scalar_depth_absrel_median"
] = float(
    np.median(
        np.concatenate(
            all_depth_q
        )
    )
)


# Disagreement quintiles.
EXCESS = np.maximum(
    QERR - FERR,
    0.0,
)

edges = np.quantile(
    DIS,
    [
        0,
        .2,
        .4,
        .6,
        .8,
        1,
    ],
)

quintiles = []

for i in range(5):

    if i == 4:
        m = (
            (DIS >= edges[i])
            & (DIS <= edges[i+1])
        )
    else:
        m = (
            (DIS >= edges[i])
            & (DIS < edges[i+1])
        )

    quintiles.append({
        "quintile":
            i + 1,

        "n":
            int(m.sum()),

        "disagreement_median":
            float(
                np.median(
                    DIS[m]
                )
            ),

        "quant_error_median":
            float(
                np.median(
                    QERR[m]
                )
            ),

        "full_error_median":
            float(
                np.median(
                    FERR[m]
                )
            ),

        "excess_error_median":
            float(
                np.median(
                    EXCESS[m]
                )
            ),
    })


w4root = Path(
    "/home/utn/luli38se/cv/"
    "QuantVGGT/evaluation/outputs/"
    "w4a4/"
    "a44_model_tracker_fixed_e20.pt_sym"
)

w4prov = (
    directory_hash(
        w4root.resolve()
    )
)


result = {
    "version":
        1,

    "primary_signal":
        (
            "Euclidean distance between "
            "globally-aligned Full and W4A4 "
            "world_points_from_depth at the "
            "same GT-valid 518-space pixel"
        ),

    "primary_target":
        (
            "W4A4 Euclidean 3D error against "
            "CO3Dv2 GT depth unprojected with "
            "GT cameras"
        ),

    "sampling": {
        "pixels_per_frame":
            SAMPLE_PER_FRAME,

        "seed":
            RNG_SEED,

        "frames_per_scene":
            6,

        "scenes":
            3,
    },

    "scenes":
        scene_results,

    "pooled":
        pooled,

    "quintiles":
        quintiles,

    "w4a4_provenance":
        w4prov,
}


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

path = (
    OUT
    / "gt_disagreement_analysis.json"
)

path.write_text(
    json.dumps(
        result,
        indent=2,
    )
    + "\n"
)


print()
print(
    "============================================================"
)
print(
    "POOLED DISAGREEMENT SUPERVISION RESULT"
)
print(
    "============================================================"
)

print(
    "Samples:",
    pooled["n"],
)

print()
print(
    "Median 3D Full GT error :",
    f"{100*pooled['full_error_median']:.3f}% radius"
)

print(
    "Median 3D Quant GT error:",
    f"{100*pooled['quant_error_median']:.3f}% radius"
)

print(
    "Median Full-Quant disagreement:",
    f"{100*pooled['disagreement_median']:.3f}% radius"
)

print()
print(
    "rho(disagreement, Quant GT error):",
    f"{pooled['rho_disagreement_quant_error']:.4f}"
)

print(
    "rho(disagreement, Full GT error):",
    f"{pooled['rho_disagreement_full_error']:.4f}"
)

print(
    "rho(disagreement, Quant excess error):",
    f"{pooled['rho_disagreement_excess_error']:.4f}"
)

print()
print(
    "AUC -> top10 Quant GT error:",
    f"{pooled['auc_top10_quant_error']:.4f}"
)

print(
    "AUC -> top10 excess error:",
    f"{pooled['auc_top10_excess_error']:.4f}"
)

print()
print(
    "Top10 disagreement precision -> "
    "top10 Quant error:",
    f"{100*pooled['top10_precision_quant_error']:.2f}%"
)

print(
    "Top10 disagreement precision -> "
    "top10 excess error:",
    f"{100*pooled['top10_precision_excess_error']:.2f}%"
)

print()
print(
    "Secondary scalar depth AbsRel:"
)

print(
    "  Full :",
    f"{100*pooled['full_scalar_depth_absrel_median']:.3f}%"
)

print(
    "  Quant:",
    f"{100*pooled['quant_scalar_depth_absrel_median']:.3f}%"
)

print()
print(
    "Disagreement quintiles:"
)

print(
    "Q   QuantErr%   FullErr%   ExcessErr%"
)

for q in quintiles:
    print(
        f"{q['quintile']}   "
        f"{100*q['quant_error_median']:9.3f}   "
        f"{100*q['full_error_median']:8.3f}   "
        f"{100*q['excess_error_median']:10.3f}"
    )

print()
print(
    "W4A4 directory content SHA256:"
)

print(
    w4prov[
        "content_sha256"
    ]
)

print()
print(
    "Saved:",
    path
)

print()
print(
    "============================================================"
)
print(
    "GT DISAGREEMENT MILESTONE: PASS"
)
print(
    "============================================================"
)
