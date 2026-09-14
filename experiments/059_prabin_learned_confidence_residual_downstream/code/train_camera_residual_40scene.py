from pathlib import Path
import sys, csv, json
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path("/var/tmp/atep12oh/QuantSplat_Integrated")
BUNDLE = Path("/var/tmp/atep12oh/confidence40_train_bundle")
REF = Path("/var/tmp/atep12oh/paired_40scene_reference")
OUT = ROOT / "outputs/camera_corrector_40scene"
OUT.mkdir(parents=True, exist_ok=True)

SPLIT_FILE = BUNDLE / "camera_residual_split_40scene.json"
LABEL_FILE = BUNDLE / "camera_residual_labels_40scene.csv"

sys.path.insert(0, str(ROOT / "scripts"))
import infer_camera_confidence as confmod

SEED = 20260914
np.random.seed(SEED)


# ============================================================
# Basic geometry
# ============================================================

def camera_centers(E):
    R = E[:, :3, :3]
    t = E[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(R, (0,2,1)), t)


def umeyama(src, dst):
    """dst ~= s * R @ src + t"""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)

    mx = src.mean(0)
    my = dst.mean(0)

    X = src - mx
    Y = dst - my

    cov = (Y.T @ X) / len(src)

    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt

    var = np.mean(np.sum(X * X, axis=1))
    if var <= 1e-12:
        raise RuntimeError("Degenerate camera geometry")

    s = np.sum(D * np.diag(S)) / var
    t = my - s * (R @ mx)

    return float(s), R, t


def extract_features(path):
    obj = confmod.extract_features(str(path))

    if isinstance(obj, np.ndarray):
        candidates = [obj]

    elif isinstance(obj, (tuple, list)):
        candidates = [
            x for x in obj
            if isinstance(x, np.ndarray)
        ]

    elif isinstance(obj, dict):
        candidates = [
            x for x in obj.values()
            if isinstance(x, np.ndarray)
        ]

    else:
        candidates = []

    candidates = [
        np.asarray(x, np.float64)
        for x in candidates
        if x.ndim == 2 and x.shape[1] == 27
    ]

    if not candidates:
        raise RuntimeError(
            f"Could not recover 27-D features from {path}; type={type(obj)}"
        )

    X = candidates[0]

    if not np.isfinite(X).all():
        raise RuntimeError(f"Non-finite feature vector: {path}")

    return X


def rot_error(Rp, Rt):
    rel = np.einsum(
        "nij,nkj->nik",
        Rt,
        Rp,
    )
    return np.degrees(
        Rotation.from_matrix(rel).magnitude()
    )


def focal_error(fp, ft):
    return 100.0 * np.mean(
        np.abs(fp - ft)
        / np.maximum(np.abs(ft), 1e-8),
        axis=1,
    )


# ============================================================
# Load frozen protocol
# ============================================================

split = json.loads(SPLIT_FILE.read_text())

scene_split = {}

for name in ["train", "val", "test"]:
    for s in split[f"{name}_scenes"]:
        scene_split[s] = name

if (
    len(split["train_scenes"]) != 28
    or len(split["val_scenes"]) != 6
    or len(split["test_scenes"]) != 6
):
    raise RuntimeError("Unexpected frozen split")


# Existing magnitude labels: used ONLY as implementation sanity check.
label_rows = defaultdict(list)

with LABEL_FILE.open(newline="") as f:
    for r in csv.DictReader(f):
        key = f"{r['category']}/{r['sequence']}"
        label_rows[key].append(r)

for k in label_rows:
    label_rows[k].sort(key=lambda x: int(x["frame_index"]))


# ============================================================
# Scene loader + signed target construction
# ============================================================

def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def build_scene(scene):
    cat, seq = scene.split("/")

    w3p = BUNDLE / "w3a3" / cat / seq / "w3a3.npz"
    fp = REF / cat / seq / "full" / "full.npz"
    w4p = REF / cat / seq / "w4a4" / "w4a4.npz"

    for p in [w3p, fp, w4p]:
        if not p.is_file():
            raise FileNotFoundError(p)

    w = load_npz(w3p)
    f = load_npz(fp)
    w4 = load_npz(w4p)

    fw_frames = np.asarray(w["frame_numbers"])
    ff_frames = np.asarray(f["frame_numbers"])
    f4_frames = np.asarray(w4["frame_numbers"])

    if not np.array_equal(fw_frames, ff_frames):
        raise RuntimeError(f"W3/Full frame mismatch: {scene}")

    if not np.array_equal(fw_frames, f4_frames):
        raise RuntimeError(f"W3/W4 frame mismatch: {scene}")

    X = extract_features(w3p)

    if X.shape != (6, 27):
        raise RuntimeError(f"{scene}: bad feature shape {X.shape}")

    Ew = w["extrinsic"].astype(np.float64)
    Ef = f["extrinsic"].astype(np.float64)

    Rw = Ew[:, :3, :3]
    Rf = Ef[:, :3, :3]

    Cw = camera_centers(Ew)
    Cf = camera_centers(Ef)

    # W3 world -> Full world gauge.
    s, Rg, tg = umeyama(Cw, Cf)

    Cwa = (
        s * np.einsum("ij,nj->ni", Rg, Cw)
        + tg
    )

    Rwa = np.einsum(
        "nij,jk->nik",
        Rw,
        Rg.T,
    )

    # Camera trajectory radius COMPUTABLE from W3 alone.
    r_w3 = np.max(
        np.linalg.norm(
            Cw - Cw.mean(axis=0),
            axis=1,
        )
    )

    if not np.isfinite(r_w3) or r_w3 <= 1e-8:
        raise RuntimeError(f"Invalid W3 radius: {scene}")

    # --------------------------------------------------------
    # Signed rotation correction.
    #
    # R_full = DeltaR * R_w3_aligned
    # DeltaR is camera-coordinate/gauge invariant.
    # --------------------------------------------------------
    DeltaR = np.einsum(
        "nij,nkj->nik",
        Rf,
        Rwa,
    )

    y_rot = Rotation.from_matrix(
        DeltaR
    ).as_rotvec()

    # --------------------------------------------------------
    # Signed center correction.
    #
    # First compute desired correction in Full world gauge,
    # transform back into W3 world units, then into each
    # camera-local coordinate frame.
    # --------------------------------------------------------
    delta_full = Cf - Cwa

    delta_w3_world = (
        np.einsum(
            "ij,nj->ni",
            Rg.T,
            delta_full,
        )
        / s
    )

    y_center = (
        np.einsum(
            "nij,nj->ni",
            Rw,
            delta_w3_world,
        )
        / r_w3
    )

    fw = np.stack(
        [
            w["intrinsic"][:, 0, 0],
            w["intrinsic"][:, 1, 1],
        ],
        axis=1,
    ).astype(np.float64)

    ff = np.stack(
        [
            f["intrinsic"][:, 0, 0],
            f["intrinsic"][:, 1, 1],
        ],
        axis=1,
    ).astype(np.float64)

    y_focal = np.log(
        np.maximum(ff, 1e-8)
        / np.maximum(fw, 1e-8)
    )

    Y = np.concatenate(
        [y_rot, y_center, y_focal],
        axis=1,
    )

    # W4 baseline.
    E4 = w4["extrinsic"].astype(np.float64)
    R4 = E4[:, :3, :3]
    C4 = camera_centers(E4)

    s4, Rg4, tg4 = umeyama(C4, Cf)

    C4a = (
        s4 * np.einsum("ij,nj->ni", Rg4, C4)
        + tg4
    )

    R4a = np.einsum(
        "nij,jk->nik",
        R4,
        Rg4.T,
    )

    f4 = np.stack(
        [
            w4["intrinsic"][:, 0, 0],
            w4["intrinsic"][:, 1, 1],
        ],
        axis=1,
    ).astype(np.float64)

    return {
        "scene": scene,
        "split": scene_split[scene],
        "frames": fw_frames.astype(int),

        "X": X,
        "Y": Y,

        "Rw": Rw,
        "Cw": Cw,
        "fw": fw,

        "Rf": Rf,
        "Cf": Cf,
        "ff": ff,

        "s": s,
        "Rg": Rg,
        "tg": tg,
        "r_w3": r_w3,

        "Rwa": Rwa,
        "Cwa": Cwa,

        "R4a": R4a,
        "C4a": C4a,
        "f4": f4,

        "sim3_scale": s,
    }


all_scene_names = (
    split["train_scenes"]
    + split["val_scenes"]
    + split["test_scenes"]
)

print("==============================================")
print("BUILD 40-SCENE SIGNED TARGET DATASET")
print("==============================================")

scenes = {}

for i, name in enumerate(all_scene_names, 1):
    sc = build_scene(name)
    scenes[name] = sc

    print(
        f"{i:02d}/40 "
        f"{sc['split']:5s} "
        f"{name:34s} "
        f"scale={sc['sim3_scale']:.5f}"
    )


# ============================================================
# Sanity-check our gauge alignment against old magnitude labels.
# This does NOT train/tune anything.
# ============================================================

rot_diffs = []
scale_rel = []

for name, sc in scenes.items():
    lr = label_rows[name]

    old_rot = np.asarray(
        [float(r["rotation_error_deg"]) for r in lr]
    )

    new_rot = rot_error(
        sc["Rwa"],
        sc["Rf"],
    )

    rot_diffs.extend(
        np.abs(old_rot - new_rot).tolist()
    )

    old_s = float(lr[0]["sim3_scale_w3_to_full"])

    scale_rel.append(
        abs(sc["sim3_scale"] - old_s)
        / max(abs(old_s), 1e-12)
    )

print()
print("==============================================")
print("ALIGNMENT CONTRACT SANITY CHECK")
print("==============================================")
print(
    "rotation label MAE:",
    f"{np.mean(rot_diffs):.6f} deg"
)
print(
    "rotation label max diff:",
    f"{np.max(rot_diffs):.6f} deg"
)
print(
    "Sim3 scale mean relative diff:",
    f"{100*np.mean(scale_rel):.4f}%"
)

# Do not abort unless convention is catastrophically inconsistent.
if np.mean(rot_diffs) > 1.0:
    print(
        "WARNING: alignment convention differs noticeably "
        "from old unsigned-label implementation."
    )


# ============================================================
# Stack train / val
# ============================================================

def stack(names):
    X = np.concatenate(
        [scenes[n]["X"] for n in names],
        axis=0,
    )
    Y = np.concatenate(
        [scenes[n]["Y"] for n in names],
        axis=0,
    )
    return X, Y


Xtr, Ytr = stack(split["train_scenes"])
Xva, Yva = stack(split["val_scenes"])

print()
print("train:", Xtr.shape, Ytr.shape)
print("val:  ", Xva.shape, Yva.shape)
print("test scenes:", len(split["test_scenes"]))


# ============================================================
# Evaluation
# ============================================================

def predict_model(model, xs, ys, X):
    Yn = model.predict(xs.transform(X))
    return ys.inverse_transform(Yn)


def eval_scene(sc, pred, shrink):
    pred = np.asarray(pred, np.float64)

    # Baseline W3 after frozen gauge alignment.
    raw_rot = rot_error(
        sc["Rwa"],
        sc["Rf"],
    )

    raw_ctr = (
        100.0
        * np.linalg.norm(
            sc["Cwa"] - sc["Cf"],
            axis=1,
        )
        / max(
            np.max(
                np.linalg.norm(
                    sc["Cf"] - sc["Cf"].mean(0),
                    axis=1,
                )
            ),
            1e-8,
        )
    )

    raw_foc = focal_error(
        sc["fw"],
        sc["ff"],
    )

    # --------------------------------------------------------
    # Apply learned correction in ORIGINAL W3 gauge.
    # --------------------------------------------------------
    drot = Rotation.from_rotvec(
        shrink * pred[:, :3]
    ).as_matrix()

    Rcorr = np.einsum(
        "nij,njk->nik",
        drot,
        sc["Rw"],
    )

    local_offset = (
        shrink
        * pred[:, 3:6]
        * sc["r_w3"]
    )

    world_offset = np.einsum(
        "nij,nj->ni",
        np.transpose(sc["Rw"], (0,2,1)),
        local_offset,
    )

    Ccorr = sc["Cw"] + world_offset

    fcorr = (
        sc["fw"]
        * np.exp(
            shrink * pred[:, 6:8]
        )
    )

    # Compare under the SAME raw W3->Full scene gauge.
    Rcorr_a = np.einsum(
        "nij,jk->nik",
        Rcorr,
        sc["Rg"].T,
    )

    Ccorr_a = (
        sc["s"]
        * np.einsum(
            "ij,nj->ni",
            sc["Rg"],
            Ccorr,
        )
        + sc["tg"]
    )

    corr_rot = rot_error(
        Rcorr_a,
        sc["Rf"],
    )

    corr_ctr = (
        100.0
        * np.linalg.norm(
            Ccorr_a - sc["Cf"],
            axis=1,
        )
        / max(
            np.max(
                np.linalg.norm(
                    sc["Cf"] - sc["Cf"].mean(0),
                    axis=1,
                )
            ),
            1e-8,
        )
    )

    corr_foc = focal_error(
        fcorr,
        sc["ff"],
    )

    return {
        "raw_rot": raw_rot,
        "corr_rot": corr_rot,
        "raw_ctr": raw_ctr,
        "corr_ctr": corr_ctr,
        "raw_foc": raw_foc,
        "corr_foc": corr_foc,

        "Rcorr": Rcorr,
        "Ccorr": Ccorr,
        "fcorr": fcorr,
    }


def evaluate(names, model, xs, ys, shrink):
    vals = defaultdict(list)

    for name in names:
        sc = scenes[name]
        pred = predict_model(
            model,
            xs,
            ys,
            sc["X"],
        )

        e = eval_scene(
            sc,
            pred,
            shrink,
        )

        for k in [
            "raw_rot", "corr_rot",
            "raw_ctr", "corr_ctr",
            "raw_foc", "corr_foc",
        ]:
            vals[k].extend(
                e[k].tolist()
            )

    return {
        k: np.asarray(v, np.float64)
        for k, v in vals.items()
    }


def selection_score(v):
    ratios = []

    for raw, corr in [
        ("raw_rot", "corr_rot"),
        ("raw_ctr", "corr_ctr"),
        ("raw_foc", "corr_foc"),
    ]:
        ratios.append(
            np.mean(v[corr])
            / max(np.mean(v[raw]), 1e-12)
        )

    return float(np.mean(ratios))


# ============================================================
# Validation-only hyperparameter selection
# ============================================================

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
SHRINKS = [0.0, 0.25, 0.50, 0.75, 1.0]

best = None

print()
print("==============================================")
print("VALIDATION HYPERPARAMETER SEARCH")
print("==============================================")

for alpha in ALPHAS:
    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(Ytr)

    m = Ridge(alpha=alpha)

    m.fit(
        xs.transform(Xtr),
        ys.transform(Ytr),
    )

    for shrink in SHRINKS:
        ev = evaluate(
            split["val_scenes"],
            m,
            xs,
            ys,
            shrink,
        )

        score = selection_score(ev)

        print(
            f"alpha={alpha:8g} "
            f"shrink={shrink:.2f} "
            f"score={score:.5f}"
        )

        candidate = (
            score,
            alpha,
            shrink,
        )

        if best is None or candidate < best:
            best = candidate


best_score, best_alpha, best_shrink = best

print()
print(
    "SELECTED:",
    f"alpha={best_alpha}",
    f"shrink={best_shrink}",
    f"val_score={best_score:.6f}",
)


# ============================================================
# Refit on TRAIN + VAL, then touch TEST exactly once.
# ============================================================

trainval = (
    split["train_scenes"]
    + split["val_scenes"]
)

Xtv, Ytv = stack(trainval)

xs = StandardScaler().fit(Xtv)
ys = StandardScaler().fit(Ytv)

model = Ridge(alpha=best_alpha)

model.fit(
    xs.transform(Xtv),
    ys.transform(Ytv),
)

test_ev = evaluate(
    split["test_scenes"],
    model,
    xs,
    ys,
    best_shrink,
)


# ============================================================
# W4A4 test baseline
# ============================================================

w4 = {
    "rot": [],
    "ctr": [],
    "foc": [],
}

for name in split["test_scenes"]:
    sc = scenes[name]

    r = rot_error(
        sc["R4a"],
        sc["Rf"],
    )

    full_radius = max(
        np.max(
            np.linalg.norm(
                sc["Cf"] - sc["Cf"].mean(0),
                axis=1,
            )
        ),
        1e-8,
    )

    c = (
        100.0
        * np.linalg.norm(
            sc["C4a"] - sc["Cf"],
            axis=1,
        )
        / full_radius
    )

    f = focal_error(
        sc["f4"],
        sc["ff"],
    )

    w4["rot"].extend(r.tolist())
    w4["ctr"].extend(c.tolist())
    w4["foc"].extend(f.tolist())

for k in w4:
    w4[k] = np.asarray(w4[k], np.float64)


# ============================================================
# Save per-frame proper held-out predictions
# ============================================================

test_rows = []

for name in split["test_scenes"]:
    sc = scenes[name]

    pred = predict_model(
        model,
        xs,
        ys,
        sc["X"],
    )

    e = eval_scene(
        sc,
        pred,
        best_shrink,
    )

    for i, frame in enumerate(sc["frames"]):
        test_rows.append({
            "scene": name,
            "frame_number": int(frame),

            "w3_rotation_deg":
                float(e["raw_rot"][i]),

            "corrected_rotation_deg":
                float(e["corr_rot"][i]),

            "w3_center_pct":
                float(e["raw_ctr"][i]),

            "corrected_center_pct":
                float(e["corr_ctr"][i]),

            "w3_focal_pct":
                float(e["raw_foc"][i]),

            "corrected_focal_pct":
                float(e["corr_foc"][i]),
        })


test_csv = OUT / "test_predictions.csv"

with test_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=test_rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(test_rows)


# ============================================================
# Save deployable linear model
# ============================================================

np.savez_compressed(
    OUT / "camera_residual_corrector.npz",

    x_mean=xs.mean_.astype(np.float32),
    x_scale=xs.scale_.astype(np.float32),

    y_mean=ys.mean_.astype(np.float32),
    y_scale=ys.scale_.astype(np.float32),

    coef=model.coef_.astype(np.float32),
    intercept=model.intercept_.astype(np.float32),

    alpha=np.asarray(best_alpha),
    shrink=np.asarray(best_shrink),

    seed=np.asarray(SEED),
)


# ============================================================
# Summary
# ============================================================

def summarize(raw, corr):
    return {
        "raw_median": float(np.median(raw)),
        "corrected_median": float(np.median(corr)),
        "raw_mean": float(np.mean(raw)),
        "corrected_mean": float(np.mean(corr)),
        "improved_frames": int(np.sum(corr < raw)),
        "n_frames": int(len(raw)),
    }


metrics = {
    "rotation_deg": summarize(
        test_ev["raw_rot"],
        test_ev["corr_rot"],
    ),
    "center_pct": summarize(
        test_ev["raw_ctr"],
        test_ev["corr_ctr"],
    ),
    "focal_pct": summarize(
        test_ev["raw_foc"],
        test_ev["corr_foc"],
    ),
}

summary = {
    "protocol":
        "Frozen category-disjoint 28 train / 6 validation / 6 test scenes",

    "train_categories":
        split["train_categories"],

    "validation_categories":
        split["val_categories"],

    "test_categories":
        split["test_categories"],

    "selected_alpha":
        best_alpha,

    "selected_shrink":
        best_shrink,

    "validation_selection_score":
        best_score,

    "test_corrector":
        metrics,

    "test_w4a4": {
        "rotation_deg_median":
            float(np.median(w4["rot"])),
        "rotation_deg_mean":
            float(np.mean(w4["rot"])),

        "center_pct_median":
            float(np.median(w4["ctr"])),
        "center_pct_mean":
            float(np.mean(w4["ctr"])),

        "focal_pct_median":
            float(np.median(w4["foc"])),
        "focal_pct_mean":
            float(np.mean(w4["foc"])),
    },

    "note":
        "Hyperparameters selected on validation only. "
        "Final model refit on train+validation. "
        "Untouched book/cake/donut test evaluated once."
}

summary_path = OUT / "summary.json"

summary_path.write_text(
    json.dumps(summary, indent=2)
)


# ============================================================
# Console report
# ============================================================

print()
print("====================================================")
print("FINAL CATEGORY-DISJOINT TEST RESULT")
print("====================================================")

for name, m in metrics.items():
    print()
    print(name)
    print(
        f"  W3A3 raw median : {m['raw_median']:.5f}"
    )
    print(
        f"  corrected median: {m['corrected_median']:.5f}"
    )
    print(
        f"  W3A3 raw mean   : {m['raw_mean']:.5f}"
    )
    print(
        f"  corrected mean  : {m['corrected_mean']:.5f}"
    )
    print(
        f"  frames improved : "
        f"{m['improved_frames']}/{m['n_frames']}"
    )

print()
print("====================================================")
print("W4A4 ON SAME UNTOUCHED TEST")
print("====================================================")
print(
    "rotation:",
    f"median={np.median(w4['rot']):.5f}",
    f"mean={np.mean(w4['rot']):.5f}",
)
print(
    "center:  ",
    f"median={np.median(w4['ctr']):.5f}",
    f"mean={np.mean(w4['ctr']):.5f}",
)
print(
    "focal:   ",
    f"median={np.median(w4['foc']):.5f}",
    f"mean={np.mean(w4['foc']):.5f}",
)

print()
print("MODEL :", OUT / "camera_residual_corrector.npz")
print("CSV   :", test_csv)
print("JSON  :", summary_path)
print()
print("CAMERA RESIDUAL 40-SCENE STUDY: COMPLETE")
