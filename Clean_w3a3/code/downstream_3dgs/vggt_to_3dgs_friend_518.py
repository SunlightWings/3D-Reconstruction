import argparse
import os

import numpy as np
import torch
from PIL import Image

from vggt.dependency.np_to_pycolmap import (
    batch_np_matrix_to_pycolmap_wo_track,
)


DEFAULT_STRIDE = 4
IMAGE_SIZE = 518


def parse_args():
    parser = argparse.ArgumentParser(
        description="Shared VGGT -> COLMAP/3DGS adapter"
    )

    parser.add_argument(
        "--raw",
        required=True,
        help="Raw Full/Quant VGGT .pt output",
    )

    parser.add_argument(
        "--input_tensor",
        required=True,
        help="Canonical [6,3,518,518] input tensor",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output 3DGS scene directory",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Deterministic point sampling stride",
    )

    return parser.parse_args()


def load_input_tensor(path):
    data = torch.load(
        path,
        map_location="cpu",
    )

    if isinstance(data, dict):
        if "images" in data:
            data = data["images"]
        elif "input_tensor" in data:
            data = data["input_tensor"]
        else:
            raise RuntimeError(
                "Input .pt contains a dictionary but no "
                "'images' or 'input_tensor' key."
            )

    data = data.float()

    if tuple(data.shape) != (6, 3, 518, 518):
        raise RuntimeError(
            f"Unexpected input shape: {tuple(data.shape)}"
        )

    return data


def save_input_images(images, image_dir):
    """
    Save canonical model inputs as actual 518x518 RGB PNGs.

    This ensures 3DGS sees images that correspond exactly
    to the VGGT camera intrinsics.
    """

    os.makedirs(
        image_dir,
        exist_ok=True,
    )

    for frame_idx in range(images.shape[0]):
        img = (
            images[frame_idx]
            .permute(1, 2, 0)
            .clamp(0, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

        path = os.path.join(
            image_dir,
            f"image_{frame_idx + 1}.png",
        )

        Image.fromarray(
            img,
            mode="RGB",
        ).save(path)

        print("Saved image:", path)


def write_ascii_ply(path, xyz, rgb):
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for point, color in zip(xyz, rgb):
            f.write(
                f"{point[0]:.8f} "
                f"{point[1]:.8f} "
                f"{point[2]:.8f} "
                f"{int(color[0])} "
                f"{int(color[1])} "
                f"{int(color[2])}\n"
            )


def main():
    args = parse_args()

    print("========================================")
    print("Shared VGGT -> COLMAP/3DGS Adapter")
    print("========================================")

    print("\nRaw input:")
    print(args.raw)

    raw = torch.load(
        args.raw,
        map_location="cpu",
    )

    required_keys = [
        "intrinsic",
        "extrinsic",
        "world_points_from_depth",
        "frame_numbers",
    ]

    for key in required_keys:
        if key not in raw:
            raise RuntimeError(
                f"Missing required key: {key}"
            )

    # ---------------------------------------------------------
    # LOAD CANONICAL IMAGES
    # ---------------------------------------------------------

    images = load_input_tensor(
        args.input_tensor
    )

    print(
        "\nCanonical input:",
        tuple(images.shape),
        images.dtype,
    )

    # Save EXACT images corresponding to VGGT cameras.
    image_dir = os.path.join(
        args.output,
        "images",
    )

    save_input_images(
        images,
        image_dir,
    )

    # ---------------------------------------------------------
    # CAMERA MATRICES
    # ---------------------------------------------------------

    intrinsics = raw["intrinsic"]
    extrinsics = raw["extrinsic"]

    # Remove batch dimension: [1,6,...] -> [6,...]
    if intrinsics.ndim == 4:
        intrinsics = intrinsics[0]

    if extrinsics.ndim == 4:
        extrinsics = extrinsics[0]

    intrinsics = (
        intrinsics
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    extrinsics = (
        extrinsics
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    print(
        "\nIntrinsics:",
        intrinsics.shape,
    )

    print(
        "Extrinsics:",
        extrinsics.shape,
    )

    # ---------------------------------------------------------
    # POINTS + COLORS
    # ---------------------------------------------------------

    points = raw[
        "world_points_from_depth"
    ]

    if not torch.is_tensor(points):
        points = torch.as_tensor(points)

    points = points.float()

    if tuple(points.shape) != (
        6,
        518,
        518,
        3,
    ):
        raise RuntimeError(
            f"Unexpected point shape: "
            f"{tuple(points.shape)}"
        )

    colors = (
        images
        .permute(0, 2, 3, 1)
        .contiguous()
    )

    stride = args.stride

    # Same exact pixel positions in every experiment.
    ys = torch.arange(
        0,
        IMAGE_SIZE,
        stride,
    )

    xs = torch.arange(
        0,
        IMAGE_SIZE,
        stride,
    )

    grid_y, grid_x = torch.meshgrid(
        ys,
        xs,
        indexing="ij",
    )

    sampled_points = points[
        :,
        ::stride,
        ::stride,
        :,
    ]

    sampled_colors = colors[
        :,
        ::stride,
        ::stride,
        :,
    ]

    print(
        "\nSampled per-view shape:",
        tuple(sampled_points.shape),
    )

    all_points = []
    all_colors = []
    all_xyf = []

    for frame_idx in range(6):
        frame_points = sampled_points[
            frame_idx
        ].reshape(-1, 3)

        frame_colors = sampled_colors[
            frame_idx
        ].reshape(-1, 3)

        frame_xy = torch.stack(
            [
                grid_x.reshape(-1),
                grid_y.reshape(-1),
            ],
            dim=1,
        ).float()

        frame_ids = torch.full(
            (
                frame_xy.shape[0],
                1,
            ),
            frame_idx,
            dtype=torch.float32,
        )

        frame_xyf = torch.cat(
            [
                frame_xy,
                frame_ids,
            ],
            dim=1,
        )

        # Only remove non-finite geometry.
        finite = torch.isfinite(
            frame_points
        ).all(dim=1)

        frame_points = frame_points[
            finite
        ]

        frame_colors = frame_colors[
            finite
        ]

        frame_xyf = frame_xyf[
            finite
        ]

        all_points.append(
            frame_points
        )

        all_colors.append(
            frame_colors
        )

        all_xyf.append(
            frame_xyf
        )

    points = torch.cat(
        all_points,
        dim=0,
    )

    colors = torch.cat(
        all_colors,
        dim=0,
    )

    points_xyf = torch.cat(
        all_xyf,
        dim=0,
    )

    print(
        "Final point count:",
        len(points),
    )

    # ---------------------------------------------------------
    # CONVERT TO NUMPY
    # ---------------------------------------------------------

    points_np = (
        points
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    xyf_np = (
        points_xyf
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    rgb_np = (
        colors
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )

    # ---------------------------------------------------------
    # SAVE PLY FOR VISUAL DEBUGGING
    # ---------------------------------------------------------

    os.makedirs(
        args.output,
        exist_ok=True,
    )

    ply_path = os.path.join(
        args.output,
        "initial_points.ply",
    )

    write_ascii_ply(
        ply_path,
        points_np.astype(np.float32),
        rgb_np,
    )

    print(
        "\nSaved debug PLY:",
        ply_path,
    )

    # ---------------------------------------------------------
    # CREATE COLMAP RECONSTRUCTION
    # ---------------------------------------------------------

    print(
        "\nBuilding COLMAP reconstruction..."
    )

    reconstruction = (
        batch_np_matrix_to_pycolmap_wo_track(
            points3d=points_np,
            points_xyf=xyf_np,
            points_rgb=rgb_np,
            extrinsics=extrinsics,
            intrinsics=intrinsics,

            # Helper expects [width, height].
            image_size=np.array(
                [IMAGE_SIZE, IMAGE_SIZE]
            ),

            shared_camera=False,

            # Preserve fx, fy, cx, cy separately.
            camera_type="PINHOLE",
        )
    )

    sparse_dir = os.path.join(
        args.output,
        "sparse",
        "0",
    )

    os.makedirs(
        sparse_dir,
        exist_ok=True,
    )

    # Make COLMAP image names match the files in output/images/
    for image_id, pyimage in reconstruction.images.items():
        pyimage.name = f"image_{image_id}.png"

    reconstruction.write(
        sparse_dir
    )

    print(
        "Saved COLMAP model:",
        sparse_dir,
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\n========================================")
    print("Adapter complete")
    print("========================================")

    print(
        "Frames:",
        raw["frame_numbers"].tolist(),
    )

    print(
        "Exported points:",
        len(points_np),
    )

    print(
        "Camera model: PINHOLE"
    )

    print(
        "Shared camera: False"
    )

    print(
        "Sampling stride:",
        stride,
    )

    print("\nNo confidence threshold.")
    print("No visibility threshold.")
    print("No reprojection filtering.")
    print("No tracking.")
    print("No bundle adjustment.")
    print("No random sampling.")

    print(
        "\n3DGS scene directory:"
    )

    print(
        args.output
    )


if __name__ == "__main__":
    main()