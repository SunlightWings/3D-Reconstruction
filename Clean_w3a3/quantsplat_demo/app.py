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

.qs-wipe {
    --qs-wipe: 50%;
    position: relative;
    width: 100%;
    max-width: 740px;
    aspect-ratio: 1;
    margin: 12px auto;
    overflow: hidden;
    border-radius: 12px;
    border: 1px solid #d9e3e8;
    background: #fff;
    isolation: isolate;
}

.qs-wipe img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
    margin: 0;
    user-select: none;
    pointer-events: none;
}

.qs-wipe .qs-wipe-top {
    clip-path: inset(0 calc(100% - var(--qs-wipe)) 0 0);
}

.qs-wipe-line {
    position: absolute;
    top: 0;
    bottom: 0;
    left: var(--qs-wipe);
    width: 3px;
    transform: translateX(-50%);
    background: #fff;
    box-shadow: 0 0 5px #132c38;
    pointer-events: none;
}

.qs-wipe-line::after {
    content: '\\2194';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    display: grid;
    place-items: center;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: #fff;
    color: #132c38;
    box-shadow: 0 1px 8px #132c3840;
    font-size: 24px;
}

.qs-wipe input[type=range] {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    opacity: 0;
    margin: 0;
    cursor: ew-resize;
    z-index: 3;
    touch-action: none;
}

.qs-wipe:focus-within {
    outline: 3px solid #107e79;
    outline-offset: 3px;
}

.qs-wipe-label {
    position: absolute;
    top: 12px;
    padding: 5px 9px;
    border-radius: 6px;
    background: #132c38df;
    color: #fff;
    z-index: 2;
    pointer-events: none;
    font-size: 12px;
}

.qs-wipe-label.left {
    left: 12px;
}

.qs-wipe-label.right {
    right: 12px;
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


# This listener stays on the HTML component when its contents change.
# Moving the divider is browser-only: no Python call or image reload.
WIPE_JS = """
if (!element.dataset.qsWipeBound) {
    element.dataset.qsWipeBound = "true";

    element.addEventListener("input", (event) => {
        const control = event.target;

        if (
            !(control instanceof HTMLInputElement) ||
            !control.matches("[data-qs-divider]")
        ) {
            return;
        }

        const stage = control.closest(".qs-wipe");

        if (stage) {
            stage.style.setProperty(
                "--qs-wipe",
                `${control.value}%`
            );
        }
    });
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


def render_view_html(
    scene_id: str,
    view,
    arm_a: str,
    arm_b: str,
) -> str:
    """Return the three panels and divider as one HTML string."""
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

    if a_uri and b_uri:
        comparison = f"""
        <div class="qs-wipe" data-frame="{frame}">
            <img
                src="{b_uri}"
                alt="{html.escape(label_b)}"
                draggable="false"
            >

            <img
                class="qs-wipe-top"
                src="{a_uri}"
                alt="{html.escape(label_a)}"
                draggable="false"
            >

            <span class="qs-wipe-label left">
                {html.escape(label_a)}
            </span>

            <span class="qs-wipe-label right">
                {html.escape(label_b)}
            </span>

            <span class="qs-wipe-line"></span>

            <input
                data-qs-divider
                type="range"
                min="0"
                max="100"
                step="1"
                value="50"
                aria-label="Comparison divider position"
            >
        </div>

        <p class="fineprint">
            Drag the divider, or focus it and use the arrow keys.
            Both images use frame {frame}.
            Left: {html.escape(label_a)}.
            Right: {html.escape(label_b)}.
        </p>
        """
    else:
        comparison = """
        <div class="notice">
            The divider becomes available when both selected models
            have a packaged render.
        </div>
        """

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
        '<h3 class="qs-compare-title">Direct comparison</h3>'
        + comparison
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
    """Return four HTML values to four existing HTML components."""
    idx = view_index(scene_id, view)

    return (
        ui.view_status(
            DATA, scene_id, idx + 1, arm_a, arm_b
        ),
        render_view_html(
            scene_id, idx + 1, arm_a, arm_b
        ),
        ui.scene_scores(
            DATA, scene_id, [arm_a, arm_b]
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

                # This replaces gr.Image and gr.ImageSlider.
                viewer = gr.HTML(
                    value=initial[1],
                    elem_id="qs-html-viewer",
                    js_on_load=WIPE_JS,
                    apply_default_css=False,
                )

                scene_table = gr.HTML(initial[2])

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
                    input_gallery = gr.HTML(
                        initial[3],
                        apply_default_css=False,
                    )

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
                        scene_table,
                        input_gallery,
                    ],
                    trigger_mode="always_last",
                    concurrency_limit=1,
                    show_progress="minimal",
                    api_name="compare_views",
                )

            # ---------------------------------------------------------
            # Results
            # ---------------------------------------------------------

            with gr.Tab("Results", id="results"):
                gr.HTML("""
                <div class="section-heading">
                    <div>
                        <span class="eyebrow small">
                            02 / MEASURED RESULTS
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
                            03 / UPSTREAM DIAGNOSTICS
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
                            04 / EXPERIMENT PROTOCOL
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