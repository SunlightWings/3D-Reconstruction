#!/usr/bin/env python3

from pathlib import Path
import argparse, sys, json
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path("/var/tmp/atep12oh/QuantSplat_Integrated")
sys.path.insert(0, str(ROOT / "scripts"))

import infer_camera_confidence as confmod


def extract_features(path):
    obj = confmod.extract_features(str(path))

    if isinstance(obj, np.ndarray):
        cand = [obj]
    elif isinstance(obj, (tuple, list)):
        cand = [x for x in obj if isinstance(x, np.ndarray)]
    elif isinstance(obj, dict):
        cand = [x for x in obj.values() if isinstance(x, np.ndarray)]
    else:
        cand = []

    cand = [
        np.asarray(x, np.float64)
        for x in cand
        if x.ndim == 2 and x.shape[1] == 27
    ]

    if not cand:
        raise RuntimeError(
            f"Could not extract 27-D features from {path}"
        )

    return cand[0]


def camera_centers(E):
    R = E[:, :3, :3]
    t = E[:, :3, 3]
    return -np.einsum(
        "nij,nj->ni",
        np.transpose(R, (0,2,1)),
        t,
    )


def unproject(depth, K, E):
    if depth.ndim == 4:
        z = depth[..., 0]
    else:
        z = depth

    T, H, W = z.shape

    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float64),
        np.arange(W, dtype=np.float64),
        indexing="ij",
    )

    out = np.empty((T, H, W, 3), dtype=np.float32)

    for i in range(T):
        fx = K[i,0,0]
        fy = K[i,1,1]
        cx = K[i,0,2]
        cy = K[i,1,2]

        zz = z[i].astype(np.float64)

        xc = (xx - cx) / fx * zz
        yc = (yy - cy) / fy * zz

        cam = np.stack(
            [xc, yc, zz],
            axis=-1,
        )

        R = E[i,:3,:3]
        t = E[i,:3,3]

        world = np.einsum(
            "ij,hwj->hwi",
            R.T,
            cam - t[None,None,:],
        )

        out[i] = world.astype(np.float32)

    return out


ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument(
    "--model",
    default=str(
        ROOT /
        "outputs/camera_corrector_40scene/"
        "camera_residual_corrector.npz"
    ),
)
ap.add_argument("--pt-output", default=None)

args = ap.parse_args()

inp = Path(args.input)
out = Path(args.output)
model_path = Path(args.model)

out.parent.mkdir(parents=True, exist_ok=True)

with np.load(inp, allow_pickle=False) as z:
    raw = {k: z[k] for k in z.files}

with np.load(model_path, allow_pickle=False) as z:
    model = {k: z[k] for k in z.files}

X = extract_features(inp)

x_mean = model["x_mean"].astype(np.float64)
x_scale = model["x_scale"].astype(np.float64)
y_mean = model["y_mean"].astype(np.float64)
y_scale = model["y_scale"].astype(np.float64)
coef = model["coef"].astype(np.float64)
intercept = model["intercept"].astype(np.float64)

shrink = float(model["shrink"])

Xn = (X - x_mean) / x_scale
Yn = Xn @ coef.T + intercept
pred = Yn * y_scale + y_mean

E = raw["extrinsic"].astype(np.float64)
K = raw["intrinsic"].astype(np.float64).copy()

Rw = E[:,:3,:3]
Cw = camera_centers(E)

radius = float(
    np.max(
        np.linalg.norm(
            Cw - Cw.mean(axis=0),
            axis=1,
        )
    )
)

if not np.isfinite(radius) or radius <= 1e-8:
    raise RuntimeError(f"Invalid camera radius {radius}")

# Rotation residual.
dR = Rotation.from_rotvec(
    shrink * pred[:,:3]
).as_matrix()

Rcorr = np.einsum(
    "nij,njk->nik",
    dR,
    Rw,
)

# Camera-local center residual -> world coordinates.
delta_local = (
    shrink *
    pred[:,3:6] *
    radius
)

delta_world = np.einsum(
    "nij,nj->ni",
    np.transpose(Rw, (0,2,1)),
    delta_local,
)

Ccorr = Cw + delta_world

# Focal correction.
K[:,0,0] *= np.exp(
    shrink * pred[:,6]
)

K[:,1,1] *= np.exp(
    shrink * pred[:,7]
)

Ecorr = E.copy()
Ecorr[:,:3,:3] = Rcorr
Ecorr[:,:3,3] = -np.einsum(
    "nij,nj->ni",
    Rcorr,
    Ccorr,
)

Pcorr = unproject(
    raw["depth"],
    K,
    Ecorr,
)

corrected = dict(raw)
corrected["intrinsic"] = K.astype(np.float32)
corrected["extrinsic"] = Ecorr.astype(np.float32)
corrected["world_points_from_depth"] = Pcorr.astype(np.float32)

np.savez_compressed(out, **corrected)

if args.pt_output:
    pt = Path(args.pt_output)
    pt.parent.mkdir(parents=True, exist_ok=True)

    torch_obj = {}
    for k, v in corrected.items():
        if isinstance(v, np.ndarray):
            torch_obj[k] = torch.from_numpy(v)

    torch.save(torch_obj, pt)

summary = {
    "input": str(inp),
    "model": str(model_path),
    "output": str(out),
    "frames": int(len(E)),
    "shrink": shrink,
    "camera_radius": radius,
    "mean_pred_rotation_correction_deg":
        float(
            np.degrees(
                np.linalg.norm(
                    shrink * pred[:,:3],
                    axis=1,
                )
            ).mean()
        ),
    "mean_pred_center_correction_pct_radius":
        float(
            100.0 *
            np.linalg.norm(
                shrink * pred[:,3:6],
                axis=1,
            ).mean()
        ),
    "mean_fx_multiplier":
        float(np.exp(shrink * pred[:,6]).mean()),
    "mean_fy_multiplier":
        float(np.exp(shrink * pred[:,7]).mean()),
}

summary_path = out.with_suffix(".summary.json")
summary_path.write_text(json.dumps(summary, indent=2))

print(json.dumps(summary, indent=2))
print()
print("CORRECTED NPZ:", out)
if args.pt_output:
    print("CORRECTED PT :", args.pt_output)
print("SUMMARY      :", summary_path)
print("CAMERA CORRECTOR APPLICATION: PASS")
