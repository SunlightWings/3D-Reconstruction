import os
import random
import hashlib

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


SEED = 42

FRAME_NUMBERS = [1, 43, 91, 112, 174, 202]

# Put the file your teammate sends you here
INPUT_TENSOR_PATH = "inputs/input_tensor.pt"

OUTPUT_PATH = "results/full_vggt_raw.pt"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_sha256(x):
    x = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(x.tobytes(order="C")).hexdigest()


def main():
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    print("Device:", device)
    print("dtype:", dtype)

    # ---------------------------------------------------------
    # LOAD THE EXACT CANONICAL INPUT TENSOR FROM YOUR TEAMMATE
    # ---------------------------------------------------------

    if not os.path.isfile(INPUT_TENSOR_PATH):
        raise FileNotFoundError(
            f"Could not find input tensor: {INPUT_TENSOR_PATH}"
        )

    images = torch.load(
        INPUT_TENSOR_PATH,
        map_location="cpu",
    )

    # In case the .pt contains a dict rather than the tensor directly.
    if isinstance(images, dict):
        print("Loaded dictionary keys:", images.keys())

        if "images" in images:
            images = images["images"]
        elif "input_tensor" in images:
            images = images["input_tensor"]
        else:
            raise RuntimeError(
                "The .pt file contains a dictionary, but I cannot "
                "find an 'images' or 'input_tensor' key."
            )

    print("\nCanonical input loaded")
    print("Input tensor shape:", tuple(images.shape))
    print("Input tensor dtype:", images.dtype)
    print("Input tensor min:", images.min().item())
    print("Input tensor max:", images.max().item())

    input_hash = tensor_sha256(images)

    print("Tensor-content SHA256:", input_hash)

    # ---------------------------------------------------------
    # SANITY CHECKS
    # ---------------------------------------------------------

    if tuple(images.shape) != (6, 3, 518, 518):
        raise RuntimeError(
            f"Wrong input shape: {tuple(images.shape)}. "
            "Expected (6, 3, 518, 518)."
        )

    if images.dtype != torch.float32:
        raise RuntimeError(
            f"Wrong input dtype: {images.dtype}. "
            "Expected torch.float32."
        )

    if images.min().item() < 0 or images.max().item() > 1:
        raise RuntimeError(
            "Input values are outside the expected [0, 1] range."
        )

    # IMPORTANT:
    # When your teammate gives you the FINAL tensor-content SHA,
    # add the comparison here.
    #
    # EXPECTED_TENSOR_SHA = "..."
    #
    # if input_hash != EXPECTED_TENSOR_SHA:
    #     raise RuntimeError("Tensor-content SHA mismatch.")

    # Move exact input tensor to GPU only AFTER hashing/checking it.
    images = images.to(device)

    # ---------------------------------------------------------
    # FULL VGGT
    # ---------------------------------------------------------

    print("\nLoading pretrained Full VGGT...")

    model = VGGT.from_pretrained(
        "facebook/VGGT-1B"
    ).to(device)

    model.eval()

    print("Running RAW VGGT inference...")

    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=dtype,
            enabled=(device == "cuda"),
        ):
            predictions = model(images)

    print("\nRaw prediction keys:")

    for key, value in predictions.items():
        if torch.is_tensor(value):
            print(
                f"  {key}: "
                f"shape={tuple(value.shape)}, "
                f"dtype={value.dtype}"
            )
        else:
            print(f"  {key}: {type(value)}")

    # ---------------------------------------------------------
    # RAW VGGT OUTPUTS
    # ---------------------------------------------------------

    pose_enc = predictions["pose_enc"]
    depth = predictions["depth"]
    depth_conf = predictions["depth_conf"]
    world_points = predictions["world_points"]
    world_points_conf = predictions["world_points_conf"]

    # Deterministic conversion of VGGT pose representation.
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        pose_enc,
        images.shape[-2:],
    )

    # Deterministically construct world points from
    # VGGT-predicted depth + VGGT-predicted cameras.
    world_points_from_depth = unproject_depth_map_to_point_map(
        depth[0].detach().cpu().numpy(),
        extrinsic[0].detach().cpu().numpy(),
        intrinsic[0].detach().cpu().numpy(),
    )

    # ---------------------------------------------------------
    # SAVE OUTPUTS
    # ---------------------------------------------------------

    output = {
        "pose_enc": pose_enc.detach().cpu(),
        "depth": depth.detach().cpu(),
        "depth_conf": depth_conf.detach().cpu(),
        "intrinsic": intrinsic.detach().cpu(),
        "extrinsic": extrinsic.detach().cpu(),
        "world_points": world_points.detach().cpu(),
        "world_points_conf": world_points_conf.detach().cpu(),
        "world_points_from_depth":
            torch.as_tensor(world_points_from_depth).cpu(),

        "frame_numbers": torch.tensor(
            FRAME_NUMBERS,
            dtype=torch.int64,
        ),

        # Useful experiment metadata
        "input_tensor_sha256": input_hash,
        "seed": SEED,
    }

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    torch.save(
        output,
        OUTPUT_PATH,
    )

    print("\nSaved:", OUTPUT_PATH)

    print("\nSaved contents:")

    for key, value in output.items():
        if torch.is_tensor(value):
            print(
                f"  {key}: "
                f"{tuple(value.shape)} "
                f"{value.dtype}"
            )
        else:
            print(
                f"  {key}: {value}"
            )


if __name__ == "__main__":
    main()