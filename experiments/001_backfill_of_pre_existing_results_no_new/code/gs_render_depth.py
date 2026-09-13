#!/usr/bin/env python3
"""Standalone GraphDECO render pass that also dumps per-view depth as .npy.

Mirrors gaussian-splatting/render.py's train-split render loop exactly (same
Scene/GaussianModel loading, same camera list), but additionally saves the
rasterizer's own "depth" output (gaussian_renderer/__init__.py already returns
it; the stock render.py just discards it). Must be invoked with the GraphDECO
python (GS_PY) and PYTHONPATH pointed at GS_ROOT (see run_downstream_validation.gs_env()),
exactly like the stock render.py.

Usage: GS_PY gs_render_depth.py -m <model> -s <source> --iteration <N> --out <out_dir>
"""

import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision
from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args


def main() -> None:
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset = model.extract(args)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        pipe = pipeline.extract(args)

        views = scene.getTrainCameras()
        out_dir = args.out
        renders_dir = os.path.join(out_dir, "renders")
        depth_dir = os.path.join(out_dir, "depth")
        os.makedirs(renders_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        for idx, view in enumerate(views):
            out = render(view, gaussians, pipe, background)
            rendering = out["render"]
            depth = out["depth"]
            torchvision.utils.save_image(rendering, os.path.join(renders_dir, f"{idx:05d}.png"))
            np.save(os.path.join(depth_dir, f"{idx:05d}.npy"), depth.detach().cpu().numpy())

    print(f"Rendered {len(views)} views with depth to {out_dir}")


if __name__ == "__main__":
    main()
