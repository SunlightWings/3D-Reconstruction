from pathlib import Path
import argparse, csv, json, math
import numpy as np
import torch
import torch.nn as nn


FOCAL_SCALE = 0.9235193794837057


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


def extract_features(npz_path):
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

    for i in range(T):
        d = depth[i]
        valid = np.isfinite(d) & (d > 0)
        dv = d[valid]

        if dv.size < 1000:
            raise RuntimeError(
                f"Too few valid depth pixels in frame {i}: {dv.size}"
            )

        dmed = max(float(np.median(dv)), 1e-6)
        q = np.percentile(dv, [5, 10, 25, 75, 90, 95])

        filled = np.where(valid, d, dmed)
        gy, gx = np.gradient(filled)
        grad = np.sqrt(gx * gx + gy * gy)[valid]

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

        features.append(f)

    return frames, np.stack(features)


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

        self.regression = nn.Linear(32, 3)
        self.pose_bad = nn.Linear(32, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.regression(h), self.pose_bad(h).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    npz_path = Path(args.npz)
    checkpoint_path = Path(args.checkpoint)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames, X = extract_features(npz_path)

    ck = torch.load(checkpoint_path, map_location="cpu")

    if X.shape[1] != int(ck["n_features"]):
        raise RuntimeError(
            f"Feature mismatch: extracted {X.shape[1]}, "
            f"checkpoint expects {ck['n_features']}"
        )

    model = ErrorConfidenceNet(int(ck["n_features"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    x_mean = np.asarray(ck["x_mean"], dtype=np.float32)
    x_std = np.asarray(ck["x_std"], dtype=np.float32)
    z_mean = np.asarray(ck["z_mean"], dtype=np.float32)
    z_std = np.asarray(ck["z_std"], dtype=np.float32)

    Xn = (X - x_mean) / x_std

    with torch.no_grad():
        rz, logits = model(torch.from_numpy(Xn.astype(np.float32)))

    pred = np.expm1(
        rz.numpy() * z_std + z_mean
    ).clip(min=0)

    probability = torch.sigmoid(logits).numpy()
    confidence = 1.0 - probability

    rows = []

    for i, frame in enumerate(frames):
        rows.append({
            "frame_number": int(frame),
            "pred_rotation_error_deg": float(pred[i, 0]),
            "pred_center_error_pct": float(pred[i, 1]),
            "pred_focal_error_pct": float(pred[i, 2]),
            "pose_failure_probability": float(probability[i]),
            "confidence": float(confidence[i]),
            "predicted_unreliable": bool(probability[i] >= 0.5),
        })

    csv_path = out / "confidence_predictions.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    unreliable = sum(r["predicted_unreliable"] for r in rows)

    summary = {
        "input_npz": str(npz_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_seed": int(ck["seed"]),
        "frames": len(rows),
        "feature_dimension": int(X.shape[1]),

        "mean_pred_rotation_error_deg":
            float(np.mean(pred[:, 0])),

        "mean_pred_center_error_pct":
            float(np.mean(pred[:, 1])),

        "mean_pred_focal_error_pct":
            float(np.mean(pred[:, 2])),

        "mean_pose_failure_probability":
            float(np.mean(probability)),

        "mean_confidence":
            float(np.mean(confidence)),

        "predicted_unreliable_frames":
            int(unreliable),

        # Diagnostic scene-level rule only.
        # It does NOT trigger Full-VGGT in this E2E demo.
        "scene_low_confidence":
            bool(unreliable >= 2),

        "deployment_action":
            "continue_with_w3a3_to_3dgs",

        "focal_multiplier_for_3dgs":
            FOCAL_SCALE,

        "note":
            "Confidence model predicts error magnitude/risk. "
            "It does not estimate pose correction vectors. "
            "Focal calibration is applied separately."
    }

    summary_path = out / "confidence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("=== PER-FRAME PREDICTIONS ===")
    for r in rows:
        print(
            f"frame={r['frame_number']:>4} "
            f"rot={r['pred_rotation_error_deg']:.3f}deg "
            f"center={r['pred_center_error_pct']:.3f}% "
            f"focal={r['pred_focal_error_pct']:.3f}% "
            f"risk={r['pose_failure_probability']:.3f} "
            f"confidence={r['confidence']:.3f}"
        )

    print("\n=== SCENE SUMMARY ===")
    print(json.dumps(summary, indent=2))

    print("\nPREDICTOR DEPLOYMENT PASS")
    print("CSV:", csv_path)
    print("JSON:", summary_path)


if __name__ == "__main__":
    main()
