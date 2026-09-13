#!/usr/bin/env python3
"""QuantSplat: CPU-only explorer of precomputed research results."""
from __future__ import annotations

import base64
import html
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import gradio as gr
from PIL import Image

from demo_data import DemoData
import ui


# ---------------------------------------------------------------------
# Existing project data and styling
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA = DemoData(
    Path(os.environ.get("QUANTSPLAT_DATA", ROOT / "data"))
)

CSS = (ROOT / "style.css").read_text(encoding="utf-8")

THEME = gr.themes.Base(
    primary_hue="teal",
    secondary_hue="slate",
    neutral_hue="slate",
    font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=["ui-monospace", "Consolas", "monospace"],
).set(
    body_background_fill="#f4f6f8",
    block_background_fill="#ffffff",
    block_border_color="#dde5e9",
    block_radius="14px",
    button_primary_background_fill="#107e79",
    button_primary_text_color="#ffffff",
    input_background_fill="#f8fafb",
    input_border_color="#d9e3e8",
    body_text_color="#1c303b",
    body_text_color_subdued="#627380",
)


# Supplements your existing style.css.
CSS += """
.qs-panel {
    background: #fff;
    border: 1px solid #dde5e9;
    color: #1c303b;
}

.qs-panel-title {
    color: #1c303b;
}

.qs-panel-subtitle {
    color: #627380;
    opacity: 1;
}

.qs-render-grid .qs-render {
    max-height: 340px;
    object-fit: contain;
}

.qs-view-header {
    flex-wrap: wrap;
    gap: 8px;
}

.qs-input-grid {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 8px;
}

.qs-input-grid .qs-panel {
    padding: 7px;
}

.qs-input-grid .qs-panel-title {
    font-size: 11px;
}

@media(max-width: 760px) {
    .qs-input-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }
}
"""
# ---------------------------------------------------------------------
# Image loading: embedded PNGs, no dynamic Gradio image-file outputs
# ---------------------------------------------------------------------

@lru_cache(maxsize=256)
def _encode_image(path: str, modified_ns: int, size: int) -> str:
    """Cache display data; changed files get a new cache key."""
    with Image.open(path) as image:
        with BytesIO() as buffer:
            image.convert("RGB").save(buffer, format="PNG")
            encoded = base64.b64encode(
                buffer.getvalue()
            ).decode("ascii")

    return "data:image/png;base64," + encoded


def image_uri(path) -> str:
    if not path:
        return ""

    p = Path(path)
    stat = p.stat()

    return _encode_image(
        str(p),
        stat.st_mtime_ns,
        stat.st_size,
    )


def image_panel(
    title: str,
    uri: str,
    subtitle: str = "",
) -> str:
    title = html.escape(str(title))
    subtitle = html.escape(str(subtitle))

    if uri:
        body = (
            f'<img src="{uri}" alt="{title}" '
            'class="qs-render" draggable="false">'
        )
    else:
        body = (
            '<div class="qs-missing">'
            'Render not packaged yet / Pending'
            '</div>'
        )

    return (
        '<section class="qs-panel">'
        f'<div class="qs-panel-title">{title}</div>'
        f'{body}'
        f'<div class="qs-panel-subtitle">{subtitle}</div>'
        '</section>'
    )


def view_index(scene_id: str, view) -> int:
    if scene_id not in DATA.by_scene:
        raise gr.Error(f"Unknown scene: {scene_id}")

    count = len(DATA.by_scene[scene_id]["frames"])

    if count == 0:
        raise gr.Error(f"No frames packaged for {scene_id}")

    return max(
        0,
        min(int(view or 1) - 1, count - 1),
    )

def render_slider(scene_id, view, arm_a, arm_b):
    idx = view_index(scene_id, view)

    path_a = DATA.image(scene_id, arm_a, idx)
    path_b = DATA.image(scene_id, arm_b, idx)

    if not path_a or not path_b:
        return None

    with Image.open(path_a) as im_a:
        img_a = im_a.convert("RGB").copy()

    with Image.open(path_b) as im_b:
        img_b = im_b.convert("RGB").copy()

    return (img_a, img_b)

TURN_ROOT = DATA.root / "turntable"


def available_turntable_scenes():
    scenes = []

    if not TURN_ROOT.is_dir():
        return scenes

    for category_dir in sorted(TURN_ROOT.iterdir()):
        if not category_dir.is_dir():
            continue

        for sequence_dir in sorted(category_dir.iterdir()):
            if not sequence_dir.is_dir():
                continue

            scene_id = f"{category_dir.name}/{sequence_dir.name}"

            # Only show scenes that have all three reconstructions.
            if not all(
                (sequence_dir / arm).is_dir()
                for arm in ["full", "w4a4", "w3a3"]
            ):
                continue

            if scene_id in DATA.by_scene:
                label = DATA.by_scene[scene_id].get(
                    "label",
                    scene_id,
                )
            else:
                label = scene_id

            scenes.append((label, scene_id))

    return scenes


def render_turntable_html(scene_id: str, angle):
    if not scene_id:
        return """
        <div class="notice">
            No turntable reconstructions are packaged yet.
        </div>
        """

    # We rendered 36 frames:
    # 000 = 0°, 001 = 10°, ... 035 = 350°.
    angle = int(angle or 0)
    angle = max(0, min(angle, 350))

    idx = angle // 10

    category, sequence = scene_id.split("/", 1)
    scene_root = TURN_ROOT / category / sequence

    panels = []

    for arm in ["full", "w4a4", "w3a3"]:
        path = scene_root / arm / f"{idx:03d}.png"

        uri = image_uri(path) if path.is_file() else ""

        panels.append(
            image_panel(
                DATA.label(arm),
                uri,
                f"Novel orbit view · {angle}°",
            )
        )

    scene_label = html.escape(
        DATA.by_scene.get(
            scene_id,
            {},
        ).get("label", scene_id)
    )

    return (
        '<div class="qs-view-header">'
        f'<strong>{scene_label}</strong>'
        f'<span>Rotation {angle}°</span>'
        '</div>'
        '<div class="qs-render-grid">'
        + "".join(panels)
        + "</div>"
        '<div class="fineprint">'
        'These are synthetic novel-camera views rendered from the '
        'trained 3D Gaussian representations. They are intended for '
        'qualitative inspection and are not used for PSNR, SSIM, '
        'or LPIPS evaluation.'
        '</div>'
    )


def render_view_html(
    scene_id: str,
    view,
    arm_a: str,
    arm_b: str,
) -> str:
    """Return the three held-out render panels as one HTML string."""
    idx = view_index(scene_id, view)
    scene = DATA.by_scene[scene_id]

    frame = int(scene["frames"][idx])
    count = len(scene["frames"])

    gt_uri = image_uri(DATA.image(scene_id, "gt", idx))
    a_uri = image_uri(DATA.image(scene_id, arm_a, idx))
    b_uri = image_uri(DATA.image(scene_id, arm_b, idx))

    label_a = DATA.label(arm_a)
    label_b = DATA.label(arm_b)

    panels = (
        image_panel("Ground truth", gt_uri, f"Frame {frame}")
        + image_panel(label_a, a_uri, f"Frame {frame}")
        + image_panel(label_b, b_uri, f"Frame {frame}")
    )

    scene_label = html.escape(
        scene.get("label", scene_id)
    )

    return (
        '<div class="qs-view-header">'
        f'<strong>{scene_label}</strong>'
        f'<span>Held-out view {idx + 1} / {count}'
        f' &middot; Frame {frame}</span>'
        '</div>'
        f'<div class="qs-render-grid">{panels}</div>'
    )


def input_images_html(scene_id: str) -> str:
    scene = DATA.by_scene[scene_id]
    paths = scene.get("inputs", [])
    frames = scene.get("input_frames", [])

    if not paths:
        return (
            '<div class="fineprint">'
            'Input images have not been packaged. '
            'This does not affect the held-out renders.'
            '</div>'
        )

    panels = []

    for i, path in enumerate(paths):
        subtitle = (
            f"Frame {frames[i]}"
            if i < len(frames)
            else ""
        )

        panels.append(
            image_panel(
                f"Input {i + 1}",
                image_uri(DATA.path(path)),
                subtitle,
            )
        )

    return (
        '<div class="qs-input-grid">'
        + "".join(panels)
        + "</div>"
    )


def render_comparison(
    scene_id: str,
    view,
    arm_a: str,
    arm_b: str,
):
    """Return the five comparison-view outputs."""
    idx = view_index(scene_id, view)

    return (
        ui.view_status(
            DATA,
            scene_id,
            idx + 1,
            arm_a,
            arm_b,
        ),

        render_view_html(
            scene_id,
            idx + 1,
            arm_a,
            arm_b,
        ),

        render_slider(
            scene_id,
            idx + 1,
            arm_a,
            arm_b,
        ),

        ui.scene_scores(
            DATA,
            scene_id,
            [arm_a, arm_b],
        ),

        input_images_html(scene_id),
    )


def render_results(region, correction, metric):
    return (
        ui.summary_table(DATA, region, correction),
        ui.pairs_table(DATA, region, correction),
        ui.delta_plot(DATA, region, metric, correction),
    )


# ---------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------

def build_demo():
    first = DATA.scenes[0]["id"]

    models = [
        (model["label"], model["id"])
        for model in DATA.models.values()
    ]

    view_models = [("Ground truth", "gt")] + models

    initial_a = "full"
    initial_b = (
        "w3a3"
        if DATA.availability("w3a3") == len(DATA.scenes)
        else "w4a4"
    )

    initial = render_comparison(
        first, 1, initial_a, initial_b
    )

    max_views = max(
        len(scene["frames"])
        for scene in DATA.scenes
    )

    with gr.Blocks(
        title="QuantSplat | Precision vs. Reconstruction",
        analytics_enabled=False,
    ) as app:

        gr.HTML(ui.hero(DATA))
        gr.HTML(ui.availability(DATA))

        with gr.Tabs(elem_id="main-tabs"):

            # ---------------------------------------------------------
            # Compare views
            # ---------------------------------------------------------

            with gr.Tab("Compare views", id="compare"):
                gr.HTML("""
                <div class="section-heading">
                    <div>
                        <span class="eyebrow small">
                            01 / VISUAL EXPLORER
                        </span>
                        <h2>Same camera. Different precision.</h2>
                    </div>
                    <span>
                        Choose a scene and move through its held-out views.
                    </span>
                </div>
                """)

                with gr.Row(elem_classes="control-row"):
                    scene = gr.Dropdown(
                        [
                            (s["label"], s["id"])
                            for s in DATA.scenes
                        ],
                        value=first,
                        label="Scene",
                        scale=2,
                        filterable=False,
                        elem_id="qs-scene-control",
                    )

                    view = gr.Slider(
                        minimum=1,
                        maximum=max_views,
                        value=1,
                        step=1,
                        label="Held-out view",
                        scale=2,
                        elem_id="qs-view-control",
                    )

                    a = gr.Dropdown(
                        view_models,
                        value=initial_a,
                        label="Model A",
                        scale=2,
                        filterable=False,
                    )

                    b = gr.Dropdown(
                        view_models,
                        value=initial_b,
                        label="Model B",
                        scale=2,
                        filterable=False,
                    )

                status = gr.HTML(initial[0])

                viewer = gr.HTML(
                    value=initial[1],
                    elem_id="qs-html-viewer",
                )

                comparison_slider = gr.ImageSlider(
                    value=initial[2],
                    label="Direct comparison",
                    interactive=True,
                    height=420,
                )

                scene_table = gr.HTML(initial[3])

                gr.HTML("""
                <div class="explain-card">
                    <b>How to read this</b>
                    <p>
                        The images share one held-out frame ID.
                        The table averages all held-out views of this
                        scene, not just the view on screen.
                        Confidence-predictor renders remain pending
                        until supplied.
                    </p>
                </div>
                """)

                with gr.Accordion(
                    "The six input images",
                    open=False,
                ):
                    input_gallery = gr.HTML(initial[4])

                # One shared listener for all four controls.
                gr.on(
                    triggers=[
                        scene.change,
                        view.change,
                        a.change,
                        b.change,
                    ],
                    fn=render_comparison,
                    inputs=[scene, view, a, b],
                    outputs=[
                        status,
                        viewer,
                        comparison_slider,
                        scene_table,
                        input_gallery,
                    ],
                    trigger_mode="always_last",
                    concurrency_limit=1,
                    show_progress="minimal",
                    api_name="compare_views",
                )

            # ---------------------------------------------------------
            # 3D reconstruction viewer
            # ---------------------------------------------------------

            with gr.Tab("3D Reconstruction", id="reconstruction"):
                gr.HTML("""
                <div class="section-heading">
                    <div>
                        <span class="eyebrow small">
                            02 / 3D RECONSTRUCTION
                        </span>
                        <h2>Rotate the reconstructed scene.</h2>
                    </div>
                    <span>
                        Same virtual camera trajectory across all model variants.
                    </span>
                </div>
                """)

                turntable_scenes = available_turntable_scenes()

                if turntable_scenes:
                    first_turntable = turntable_scenes[0][1]

                    with gr.Row(elem_classes="control-row"):
                        turntable_scene = gr.Dropdown(
                            choices=turntable_scenes,
                            value=first_turntable,
                            label="Scene",
                            scale=2,
                            filterable=False,
                        )

                        turntable_angle = gr.Slider(
                            minimum=0,
                            maximum=350,
                            value=0,
                            step=10,
                            label="Rotation",
                            scale=3,
                        )

                    turntable_viewer = gr.HTML(
                        render_turntable_html(
                            first_turntable,
                            0,
                        )
                    )

                    gr.on(
                        triggers=[
                            turntable_scene.change,
                            turntable_angle.change,
                        ],
                        fn=render_turntable_html,
                        inputs=[
                            turntable_scene,
                            turntable_angle,
                        ],
                        outputs=turntable_viewer,
                        trigger_mode="always_last",
                        concurrency_limit=1,
                        show_progress="hidden",
                    )

                else:
                    gr.HTML("""
                    <div class="notice">
                        No turntable reconstructions have been packaged yet.
                    </div>
                    """)

            # ---------------------------------------------------------
            # Results
            # ---------------------------------------------------------

            with gr.Tab("Results", id="results"):
                gr.HTML("""
                <div class="section-heading">
                    <div>
                        <span class="eyebrow small">
                            03 / MEASURED RESULTS
                        </span>
                        <h2>Look beyond a single view.</h2>
                    </div>
                    <span>
                        Scene-level averages and paired differences.
                    </span>
                </div>
                """)

                gr.HTML(ui.results_cards(DATA))

                gr.HTML(
                    '<div class="notice">'
                    + ui.e(DATA.source_note())
                    + "</div>"
                )

                with gr.Row():
                    region = gr.Radio(
                        [
                            ("Foreground (primary)", "foreground"),
                            ("Content (secondary)", "content"),
                        ],
                        value="foreground",
                        label="Evaluation region",
                    )

                    correction = gr.Radio(
                        ["Raw", "Exposure-corrected"],
                        value="Raw",
                        label="Scoring variant",
                    )

                    metric = gr.Dropdown(
                        [
                            ("PSNR", "psnr"),
                            ("SSIM", "ssim"),
                            ("LPIPS", "lpips"),
                        ],
                        value="psnr",
                        label="Per-scene plot",
                        filterable=False,
                    )

                summary = gr.HTML(ui.summary_table(DATA))

                plot = gr.HTML(
                    ui.delta_plot(
                        DATA, "foreground", "psnr", "Raw"
                    )
                )

                gr.HTML("""
                <div class="section-heading">
                    <h3>Paired comparisons</h3>
                    <span>
                        Reference minus variant; supplied results only.
                    </span>
                </div>
                """)

                pairs = gr.HTML(
                    ui.pairs_table(
                        DATA, "foreground", "Raw"
                    )
                )

                for control in [region, correction, metric]:
                    control.change(
                        render_results,
                        inputs=[region, correction, metric],
                        outputs=[summary, pairs, plot],
                        queue=False,
                    )

                gr.Markdown(
                    "**Interpretation:** the displayed pilot results "
                    "show a larger raw-metric gap for W3A3 than W4A4. "
                    "They do not, by themselves, prove a universal "
                    "bit-width threshold, equivalence, or an efficiency "
                    "gain. The two quantization configurations can also "
                    "differ in calibration provenance."
                )

                metrics_path = DATA.root / "metrics.json"

                gr.File(
                    value=(
                        str(metrics_path)
                        if metrics_path.is_file()
                        else None
                    ),
                    label="Results JSON used by this demo",
                    interactive=False,
                )

            # ---------------------------------------------------------
            # Geometry and timing
            # ---------------------------------------------------------

            with gr.Tab("Geometry & timing", id="geometry"):
                gr.HTML("""
                <div class="section-heading">
                    <div>
                        <span class="eyebrow small">
                            04 / UPSTREAM DIAGNOSTICS
                        </span>
                        <h2>What changed before splatting?</h2>
                    </div>
                </div>
                """)

                gr.HTML(ui.geometry_table(DATA))

                gr.HTML("""
                <div class="section-heading">
                    <h3>Recorded inference timing</h3>
                    <span>
                        Lower precision is not automatically faster execution.
                    </span>
                </div>
                """)

                gr.HTML(ui.runtime_table(DATA))

                gr.HTML("""
                <div class="notice">
                    End-to-end timing is pending until matched measurements
                    include model loading, VGGT inference, any confidence
                    module, 3DGS training, and rendering. Post-training
                    calibration is a separate, amortized setup cost.
                </div>
                """)

            # ---------------------------------------------------------
            # Method and data
            # ---------------------------------------------------------

            with gr.Tab("Method & data", id="method"):
                gr.HTML("""
                <div class="section-heading">
                    <div>
                        <span class="eyebrow small">
                            05 / EXPERIMENT PROTOCOL
                        </span>
                        <h2>One downstream pipeline.</h2>
                    </div>
                </div>

                <div class="pipeline">
                    <div>
                        <small>01</small>
                        <b>Six RGB views</b>
                        <span>518 x 518, pad mode</span>
                    </div>
                    <div>
                        <small>02</small>
                        <b>VGGT variant</b>
                        <span>Cameras, depth, points</span>
                    </div>
                    <div>
                        <small>03</small>
                        <b>Shared adapter</b>
                        <span>Camera conversion + Sim(3)</span>
                    </div>
                    <div>
                        <small>04</small>
                        <b>3DGS</b>
                        <span>7,000 training iterations</span>
                    </div>
                    <div>
                        <small>05</small>
                        <b>Nine held-out views</b>
                        <span>Compare with real photos</span>
                    </div>
                </div>
                """)

                gr.Markdown("""
### Evaluation protocol

This viewer follows your supplied **render_scoring/README.md**:
raw metrics are the headline; foreground is primary and content secondary.
PSNR uses the exact mask. SSIM and AlexNet LPIPS use the mask bounding box,
so they can include nearby background. LPIPS inputs in the upstream
scorer are in [-1, 1].

Each scene averages its nine views before scene-level aggregation and
paired testing. Exposure-corrected metrics are supplementary, only shown
when supplied. Missing results are shown as pending, never zero.

### Pairing and provenance

The archive uses `gt/frameXXXXXX.png` and render indices `00000.png`
through `00008.png`, paired by sorted order as documented by your scorer.
The packaging script checks local W3A3 camera-source names against those
exact nine GT IDs. It rejects 196-view folders or mismatched ordering.
The app does not recalculate metrics from compressed previews.

### Confidence predictor: work in progress

The intended extension uses a lightweight confidence predictor to guide
W3A3-based reconstruction. No improvement, speedup, or Full-equivalent
quality is claimed before its renders and evaluation results are supplied.
The eight inspected scenes are an exploratory pilot, not an untouched
final test set.

### Scope

This is a viewer for precomputed images and measurements. It does not run
VGGT, train Gaussians, or provide a true interactive Gaussian-splat renderer.
No CUDA, PyTorch checkpoint, `.npz`, or `.ply` is required to serve this app.
                """)

                gr.HTML(ui.data_status(DATA))

                gr.File(
                    value=str(DATA.root / "catalog.json"),
                    label="Packaged view mapping and provenance",
                    interactive=False,
                )

        gr.HTML("""
        <footer class="footer">
            <span>
                <b>QUANTSPLAT</b> / Computer Vision Final Project
            </span>
            <span>
                Precomputed results &middot; Missing data stays visible
            </span>
        </footer>
        """)

    return app


demo = build_demo()

if __name__ == "__main__":
    demo.queue().launch(
        server_name=os.environ.get(
            "GRADIO_SERVER_NAME",
            "0.0.0.0",
        ),
        server_port=int(
            os.environ.get(
                "GRADIO_SERVER_PORT",
                os.environ.get("PORT", "7860"),
            )
        ),
        share=False,
        theme=THEME,
        css=CSS,
        allowed_paths=[str(DATA.root)],
        footer_links=[],
        show_error=True,
    )