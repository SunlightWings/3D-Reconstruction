from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Category-disjoint split fixed BEFORE model selection.
TRAIN_CATS = {
    "apple", "ball", "bench", "bowl", "broccoli", "hydrant", "mouse",
    "orange", "skateboard", "suitcase", "teddybear", "toaster",
    "toytrain", "vase",
}
VAL_CATS = {"plant", "remote", "toytruck"}
TEST_CATS = {"book", "cake", "donut"}


def pick(fields, exact=(), contains=()):
    lookup = {f.lower(): f for f in fields}

    for x in exact:
        if x.lower() in lookup:
            return lookup[x.lower()]

    for f in fields:
        fl = f.lower()
        if all(x.lower() in fl for x in contains):
            return f

    return None


def load_labels(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        required = {
            "category": "category",
            "sequence": "sequence",
            "frame": "frame_number",
            "rot": "rotation_error_deg",
            "center": "center_error_percent_scene_radius",
            "focal": "focal_error_percent",
        }

        missing = [
            col for col in required.values()
            if col not in fields
        ]

        if missing:
            raise RuntimeError(
                f"Missing required teacher-label columns: {missing}; "
                f"Available columns: {fields}"
            )

        cols = required
        labels = {}

        for row in reader:
            key = (
                row[cols["category"]],
                row[cols["sequence"]],
                int(float(row[cols["frame"]])),
            )

            labels[key] = np.array(
                [
                    float(row[cols["rot"]]),
                    float(row[cols["center"]]),
                    float(row[cols["focal"]]),
                ],
                dtype=np.float32,
            )

    return labels, cols

def rot_angle(Ra, Rb):
    R = Ra.T @ Rb
    x = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(x)))


def camera_centers(E):
    ans = []

    for e in E:
        R = e[:3, :3].astype(np.float64)
        t = e[:3, 3].astype(np.float64)
        ans.append(-R.T @ t)

    return np.stack(ans)


def scene_features(npz_path, labels):
    category = npz_path.parents[1].name
    sequence = npz_path.parent.name

    with np.load(npz_path, allow_pickle=False) as z:
        depth = z["depth"][..., 0].astype(np.float32)
        K = z["intrinsic"].astype(np.float32)
        E = z["extrinsic"].astype(np.float32)
        P = z["world_points_from_depth"].astype(np.float32)
        frames = z["frame_numbers"].reshape(-1).astype(int)

    T, H, W = depth.shape

    C = camera_centers(E)

    pair_distances = [
        np.linalg.norm(C[i] - C[j])
        for i in range(T)
        for j in range(i + 1, T)
    ]

    baseline = max(float(np.median(pair_distances)), 1e-6)

    c2w = [
        E[i, :3, :3].T.astype(np.float64)
        for i in range(T)
    ]

    fx_scene = np.array(
        [K[i, 0, 0] for i in range(T)],
        dtype=np.float64,
    )

    fx_median = max(float(np.median(fx_scene)), 1e-6)

    features = []
    targets = []

    for i in range(T):
        d = depth[i]

        valid = np.isfinite(d) & (d > 0)
        dv = d[valid]

        if dv.size < 1000:
            raise RuntimeError(
                f"Too few valid depth pixels: {npz_path}, frame {i}"
            )

        dmed = max(float(np.median(dv)), 1e-6)
        q = np.percentile(dv, [5, 10, 25, 75, 90, 95])

        filled = np.where(valid, d, dmed)
        gy, gx = np.gradient(filled)
        grad = np.sqrt(gx * gx + gy * gy)[valid]

        # Robust point-cloud shape statistics.
        pts = P[i].reshape(-1, 3)[::64]
        pts = pts[np.isfinite(pts).all(axis=1)]

        pc = np.median(pts, axis=0)
        radius = np.linalg.norm(pts - pc, axis=1)

        r50 = max(float(np.median(radius)), 1e-6)
        r90 = float(np.percentile(radius, 90))

        center_offset = (
            np.linalg.norm(C[i] - np.median(C, axis=0))
            / baseline
        )

        trajectory_dist = [
            float(np.linalg.norm(C[i] - C[j]) / baseline)
            for j in range(T)
            if j != i
        ]

        trajectory_angles = [
            rot_angle(c2w[i], c2w[j]) / 180.0
            for j in range(T)
            if j != i
        ]

        fx = float(K[i, 0, 0])
        fy = float(K[i, 1, 1])
        cx = float(K[i, 0, 2])
        cy = float(K[i, 1, 2])

        f = np.array(
            [
                fx / W,
                fy / H,
                cx / W,
                cy / H,
                fx / max(fy, 1e-6),

                fx / fx_median,
                abs(fx - fx_median) / fx_median,

                float(valid.mean()),

                *list(q / dmed),

                float(np.std(dv) / dmed),
                float(np.mean(grad) / dmed),
                float(np.percentile(grad, 90) / dmed),

                r90 / r50,

                float(center_offset),

                min(trajectory_dist),
                float(np.mean(trajectory_dist)),
                max(trajectory_dist),

                min(trajectory_angles),
                float(np.mean(trajectory_angles)),
                max(trajectory_angles),

                i / max(T - 1, 1),
                float(frames[i] / max(frames.max(), 1)),
            ],
            dtype=np.float32,
        )

        key = (category, sequence, int(frames[i]))

        if key not in labels:
            raise KeyError(f"Missing teacher label for {key}")

        features.append(f)
        targets.append(labels[key])

    return (
        category,
        sequence,
        frames,
        np.stack(features),
        np.stack(targets),
    )


class ErrorConfidenceNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(n_features, 96),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Regression:
        # rotation residual, center residual, focal residual
        self.regression = nn.Linear(32, 3)

        # Binary pose-risk head.
        self.pose_bad = nn.Linear(32, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.regression(h), self.pose_bad(h).squeeze(-1)


def binary_auc(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score)

    pos = score[y == 1]
    neg = score[y == 0]

    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    return float(
        (pos[:, None] > neg[None, :]).mean()
        + 0.5 * (pos[:, None] == neg[None, :]).mean()
    )


def evaluate_metrics(y, prediction, probability):
    mae = np.mean(np.abs(prediction - y), axis=0)

    # Focal is handled by our independent calibrated correction.
    # Confidence gate therefore represents POSE reliability.
    bad = (
        (y[:, 0] > 5.0)
        | (y[:, 1] > 5.0)
    ).astype(int)

    pred_bad = (probability >= 0.5).astype(int)

    tp = ((pred_bad == 1) & (bad == 1)).sum()
    tn = ((pred_bad == 0) & (bad == 0)).sum()
    fp = ((pred_bad == 1) & (bad == 0)).sum()
    fn = ((pred_bad == 0) & (bad == 1)).sum()

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    return {
        "mae_rotation_deg": float(mae[0]),
        "mae_center_pct": float(mae[1]),
        "mae_focal_pct": float(mae[2]),
        "pose_bad_prevalence": float(bad.mean()),
        "pose_bad_auc": binary_auc(bad, probability),
        "pose_bad_balanced_accuracy": float(
            (sensitivity + specificity) / 2
        ),
        "confusion_TN_FP_FN_TP": [
            int(tn),
            int(fp),
            int(fn),
            int(tp),
        ],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--seeds", type=int, default=5)

    args = parser.parse_args()

    root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    labels, label_columns = load_labels(Path(args.labels))

    print("Teacher label columns:", label_columns, flush=True)

    paths = sorted(
        (root / "w3a3").glob("*/*/w3a3.npz")
    )

    print("W3A3 scenes:", len(paths), flush=True)

    if len(paths) != 40:
        raise RuntimeError(
            f"Expected 40 W3A3 NPZ files, found {len(paths)}"
        )

    scenes = []

    for i, p in enumerate(paths, 1):
        s = scene_features(p, labels)
        scenes.append(s)

        print(
            f"loaded {i:02d}/40 "
            f"{s[0]}/{s[1]}",
            flush=True,
        )

    X = np.concatenate(
        [s[3] for s in scenes],
        axis=0,
    )

    Y = np.concatenate(
        [s[4] for s in scenes],
        axis=0,
    )

    scene_index = np.concatenate(
        [
            np.full(len(s[2]), i)
            for i, s in enumerate(scenes)
        ]
    )

    categories = np.array(
        [scenes[i][0] for i in scene_index]
    )

    train_idx = np.where(
        np.isin(categories, list(TRAIN_CATS))
    )[0]

    val_idx = np.where(
        np.isin(categories, list(VAL_CATS))
    )[0]

    test_idx = np.where(
        np.isin(categories, list(TEST_CATS))
    )[0]

    print(
        "frames train/val/test = "
        f"{len(train_idx)}/"
        f"{len(val_idx)}/"
        f"{len(test_idx)}",
        flush=True,
    )

    if (
        len(train_idx),
        len(val_idx),
        len(test_idx),
    ) != (168, 36, 36):
        raise RuntimeError(
            "Expected category-disjoint frame split 168/36/36"
        )

    # Normalize strictly using TRAIN only.
    x_mean = X[train_idx].mean(axis=0)
    x_std = np.maximum(
        X[train_idx].std(axis=0),
        1e-6,
    )

    Xn = (X - x_mean) / x_std

    # Positive errors, so regression in log domain.
    Z = np.log1p(np.maximum(Y, 0))

    z_mean = Z[train_idx].mean(axis=0)
    z_std = np.maximum(
        Z[train_idx].std(axis=0),
        1e-6,
    )

    Zn = (Z - z_mean) / z_std

    pose_bad = (
        (Y[:, 0] > 5.0)
        | (Y[:, 1] > 5.0)
    ).astype(np.float32)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device, flush=True)

    TX = torch.from_numpy(
        Xn.astype(np.float32)
    ).to(device)

    TZ = torch.from_numpy(
        Zn.astype(np.float32)
    ).to(device)

    TB = torch.from_numpy(
        pose_bad
    ).to(device)

    positives = max(
        float(pose_bad[train_idx].sum()),
        1.0,
    )

    negatives = max(
        float(
            len(train_idx)
            - pose_bad[train_idx].sum()
        ),
        1.0,
    )

    pos_weight = torch.tensor(
        negatives / positives,
        device=device,
    )

    print(
        "train pose_bad prevalence="
        f"{pose_bad[train_idx].mean():.3f}, "
        f"pos_weight={pos_weight.item():.3f}",
        flush=True,
    )

    def evaluate(model, indices):
        model.eval()

        with torch.no_grad():
            rz, logit = model(TX[indices])

        rz = rz.cpu().numpy()

        probability = (
            torch.sigmoid(logit)
            .cpu()
            .numpy()
        )

        prediction = np.expm1(
            rz * z_std + z_mean
        ).clip(min=0)

        return (
            evaluate_metrics(
                Y[indices],
                prediction,
                probability,
            ),
            prediction,
            probability,
        )

    best_global = None
    seed_results = []

    for seed in range(
        20260913,
        20260913 + args.seeds,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = ErrorConfidenceNet(
            X.shape[1]
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=2e-3,
        )

        best_val_score = float("inf")
        best_state = None
        stale = 0

        shuffled = train_idx.copy()

        for epoch in range(
            1,
            args.epochs + 1,
        ):
            np.random.shuffle(shuffled)

            model.train()

            for start in range(
                0,
                len(shuffled),
                32,
            ):
                batch = shuffled[
                    start:start + 32
                ]

                rz, logit = model(TX[batch])

                regression_loss = (
                    F.smooth_l1_loss(
                        rz,
                        TZ[batch],
                    )
                )

                classification_loss = (
                    F.binary_cross_entropy_with_logits(
                        logit,
                        TB[batch],
                        pos_weight=pos_weight,
                    )
                )

                loss = (
                    regression_loss
                    + 0.35
                    * classification_loss
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()
                optimizer.step()

            val_metrics, _, _ = evaluate(
                model,
                val_idx,
            )

            # Validation-only model selection.
            val_score = (
                val_metrics[
                    "mae_rotation_deg"
                ] / 5.0
                + val_metrics[
                    "mae_center_pct"
                ] / 5.0
                + val_metrics[
                    "mae_focal_pct"
                ] / 10.0
            )

            if val_score < (
                best_val_score - 1e-4
            ):
                best_val_score = val_score
                stale = 0

                best_state = {
                    k: v.detach()
                    .cpu()
                    .clone()
                    for k, v
                    in model.state_dict().items()
                }
            else:
                stale += 1

            if (
                epoch == 1
                or epoch % 100 == 0
            ):
                print(
                    f"seed={seed} "
                    f"epoch={epoch:04d} "
                    f"val_score={val_score:.4f} "
                    f"val={val_metrics}",
                    flush=True,
                )

            if stale >= 120:
                print(
                    f"seed={seed} "
                    f"early stop at {epoch}",
                    flush=True,
                )
                break

        model.load_state_dict(
            best_state
        )

        val_metrics, _, _ = evaluate(
            model,
            val_idx,
        )

        checkpoint = {
            "state_dict":
                model.state_dict(),
            "x_mean": x_mean,
            "x_std": x_std,
            "z_mean": z_mean,
            "z_std": z_std,
            "n_features": X.shape[1],
            "seed": seed,
        }

        torch.save(
            checkpoint,
            out / f"seed_{seed}.pt",
        )

        seed_results.append(
            {
                "seed": seed,
                "validation_score":
                    best_val_score,
                "validation":
                    val_metrics,
            }
        )

        print(
            f"seed={seed} DONE "
            f"best_val_score="
            f"{best_val_score:.4f}",
            flush=True,
        )

        if (
            best_global is None
            or best_val_score
            < best_global[0]
        ):
            best_global = (
                best_val_score,
                seed,
                checkpoint,
                model,
            )

    (
        _,
        best_seed,
        best_checkpoint,
        best_model,
    ) = best_global

    # Only now evaluate untouched TEST.
    val_metrics, _, _ = evaluate(
        best_model,
        val_idx,
    )

    test_metrics, test_pred, test_prob = (
        evaluate(
            best_model,
            test_idx,
        )
    )

    torch.save(
        best_checkpoint,
        out / "best_model.pt",
    )

    # Constant predictor baseline from TRAIN only.
    baseline = np.median(
        Y[train_idx],
        axis=0,
    )

    baseline_mae = np.mean(
        np.abs(
            Y[test_idx] - baseline
        ),
        axis=0,
    )

    summary = {
        "best_seed": best_seed,

        "split": {
            "train_categories":
                sorted(TRAIN_CATS),
            "validation_categories":
                sorted(VAL_CATS),
            "test_categories":
                sorted(TEST_CATS),

            "train_frames":
                len(train_idx),
            "validation_frames":
                len(val_idx),
            "test_frames":
                len(test_idx),
        },

        "validation":
            val_metrics,

        "test":
            test_metrics,

        "constant_baseline_test_mae": {
            "rotation_deg":
                float(baseline_mae[0]),
            "center_pct":
                float(baseline_mae[1]),
            "focal_pct":
                float(baseline_mae[2]),
        },

        "focal_multiplier_prior":
            0.9235193794837057,

        "seed_selection":
            seed_results,
    }

    (
        out / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    contract = {
        "name":
            "w3a3_camera_error_confidence_v1",

        "deployment_input":
            "W3A3 outputs only",

        "required_npz_arrays": [
            "depth",
            "intrinsic",
            "extrinsic",
            "world_points_from_depth",
            "frame_numbers",
        ],

        "teacher_training_labels":
            "Full-VGGT versus W3A3 residuals",

        "output_per_frame": [
            "pred_rotation_error_deg",
            "pred_center_error_pct",
            "pred_focal_error_pct",
            "pose_failure_probability",
            "confidence = 1 - pose_failure_probability",
        ],

        "pose_failure_definition":
            "rotation_error_deg > 5 OR center_error_pct > 5",

        "focal_policy":
            "Apply independent W3A3 focal multiplier "
            "0.9235193794837057. "
            "Predicted focal error remains diagnostic.",

        "test_policy":
            "book/cake/donut untouched until final evaluation",
    }

    (
        out / "model_contract.json"
    ).write_text(
        json.dumps(
            contract,
            indent=2,
        )
    )

    with (
        out / "test_predictions.csv"
    ).open("w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "category",
                "sequence",
                "frame_number",

                "true_rotation_deg",
                "true_center_pct",
                "true_focal_pct",

                "pred_rotation_deg",
                "pred_center_pct",
                "pred_focal_pct",

                "pose_failure_probability",
                "confidence",
            ]
        )

        k = 0

        for scene in scenes:
            category, sequence, frames, _, y = scene

            if category not in TEST_CATS:
                continue

            for j in range(len(frames)):
                pred = test_pred[k]
                probability = float(
                    test_prob[k]
                )

                writer.writerow(
                    [
                        category,
                        sequence,
                        int(frames[j]),

                        *map(float, y[j]),
                        *map(float, pred),

                        probability,
                        1.0 - probability,
                    ]
                )

                k += 1

    print()
    print(
        "=== FINAL SUMMARY ===",
        flush=True,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        ),
        flush=True,
    )

    print(
        "TRAINING COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
