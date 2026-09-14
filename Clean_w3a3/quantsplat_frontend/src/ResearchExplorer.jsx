import { useEffect, useState } from "react";
import Experiments from "./Experiments";
import { focalCalibration } from "./focalCalibration";

const asset = (path) => `/${path.replace(/^\//, "")}`;
const number = (value, digits = 4) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
const arms = ["full", "w4a4", "w3a3", focalCalibration.id];
const labels = { full: "Full VGGT", w4a4: "W4A4", w3a3: "W3A3", [focalCalibration.id]: focalCalibration.label };
const correctedMetricIds = { full: "full_qfix", w3a3: "w3a3_qfix", [focalCalibration.id]: "w3a3_qfix_focal" };

function useResearchData() {
  const [data, setData] = useState(null); const [error, setError] = useState("");
  useEffect(() => {
    Promise.all(["catalog.json", "geometry.json", "runtime.json", "focal_calibration_metrics.json", "final_model_metrics.json"].map((name) =>
      fetch(asset(name)).then((response) => response.ok ? response.json() : Promise.reject(new Error(name))),
    )).then(([catalog, geometry, runtimes, focalMetrics, finalMetrics]) => setData({ catalog, geometry, runtimes, focalMetrics, finalMetrics }))
      .catch(() => setError("Packaged research data could not be loaded. Build the site from this repository so Vite can include quantsplat_demo/data."));
  }, []);
  return { data, error };
}

function ImageCard({ title, src, subtitle }) {
  return <article className="research-image-card"><b>{title}</b>{src ? <img src={asset(src)} alt={`${title}: ${subtitle}`} /> : <div className="missing-render">Render pending</div>}<small>{subtitle}</small></article>;
}

const finalModelScenes = [
  ["fern", "Fern", [0, 8, 16]], ["flower", "Flower", [0, 8, 16, 24, 32]],
  ["fortress", "Fortress", [0, 8, 16, 24, 32, 40]], ["horns", "Horns", [0, 8, 16, 24, 32, 40, 48, 56]],
  ["leaves", "Leaves", [0, 8, 16, 24]], ["orchids", "Orchids", [0, 8, 16, 24]],
  ["room", "Room", [0, 8, 16, 24, 32, 40]], ["trex", "T-Rex", [0, 8, 16, 24, 32, 40, 48]],
].map(([id, label, views]) => ({ id, label, views }));

function FinalModelComparison({ finalMetrics }) {
  const [sceneId, setSceneId] = useState(finalModelScenes[0].id); const [view, setView] = useState(0);
  const scene = finalModelScenes.find((item) => item.id === sceneId) || finalModelScenes[0]; const frame = scene.views[view] ?? scene.views[0];
  const image = (arm) => `final_scenes/${scene.id}/${arm}/view${String(frame).padStart(3, "0")}.png`;
  const means = finalMetrics.summary?.means?.w3a3_corrector || {}; const delta = finalMetrics.summary?.corrector_minus_w3a3?.psnr || {};
  return <section className="final-model-preview" aria-label="Final corrected confidence-aware model comparison">
    <div className="final-model-header"><div><p className="eyebrow">FINAL MODEL</p><h3>W3A3+ Resolve</h3><p>Corrector + confidence-aware reconstruction</p></div><span>Final evaluation set</span></div>
    <div className="final-model-controls"><label>Scene<select value={scene.id} onChange={(e) => { setSceneId(e.target.value); setView(0); }}>{finalModelScenes.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label><label>View<select value={view} onChange={(e) => setView(Number(e.target.value))}>{scene.views.map((item, index) => <option key={item} value={index}>View {index + 1} · frame {item}</option>)}</select></label></div>
    <div className="final-model-body"><div className="final-model-images"><ImageCard title="Ground truth" src={image("ground_truth")} subtitle={`${scene.label} · frame ${frame}`} /><ImageCard title="W3A3+ Resolve" src={image("after_corrector")} subtitle="Corrector + confidence-aware output" /></div><div className="final-model-scores" aria-label="Final model eight-scene mean scores"><div><small>PSNR ↑</small><b>{number(means.psnr, 3)}</b><span>8-scene mean</span></div><div><small>SSIM ↑</small><b>{number(means.ssim)}</b><span>8-scene mean</span></div><div><small>LPIPS ↓</small><b>{number(means.lpips)}</b><span>8-scene mean</span></div></div></div><p className="final-model-note">+{number(delta.mean, 3)} dB mean PSNR vs. W3A3 · improved in {delta.improved}/{delta.n} scenes.</p>
  </section>;
}

function Compare({ catalog, focalMetrics, finalMetrics }) {
  const [sceneId, setSceneId] = useState(catalog.scenes[0].id); const [view, setView] = useState(0);
  const [left, setLeft] = useState("full"); const [right, setRight] = useState(focalCalibration.id);
  const scene = catalog.scenes.find((item) => item.id === sceneId) || catalog.scenes[0];
  const row = focalMetrics.per_scene?.find((item) => item.scene === scene.id) || {};
  const image = (arm) => arm === "gt" ? scene.gt[view] : arm === focalCalibration.id ? focalCalibration.pathFor(scene.id, view) : scene.renders?.[arm]?.[view];
  const renderFrame = (arm) => arm === focalCalibration.id ? focalCalibration.frameFor(scene.id, view) : scene.frames[view];
  const metricRow = (arm) => { const metricId = correctedMetricIds[arm]; return <tr key={arm}><th>{labels[arm]}</th><td>{number(row[`${metricId}_psnr`], 3)}</td><td>{number(row[`${metricId}_ssim`])}</td><td>{number(row[`${metricId}_lpips`])}</td></tr>; };
  return <><FinalModelComparison finalMetrics={finalMetrics} /><section className="research-section benchmark-comparison"><h2>Earlier benchmark comparison.</h2><p className="small-note">This separate reference set contains Full VGGT, W4A4, W3A3, and focal calibration. It is shown below the final model because it uses different scenes.</p><div className="research-controls"><label>Scene<select value={sceneId} onChange={(e) => { setSceneId(e.target.value); setView(0); }}>{catalog.scenes.map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}</select></label><label>Held-out view<select value={view} onChange={(e) => setView(Number(e.target.value))}>{scene.frames.map((frame, index) => <option value={index} key={frame}>View {index + 1} · frame {frame}</option>)}</select></label><label>Model A<select value={left} onChange={(e) => setLeft(e.target.value)}>{["gt", ...arms].map((arm) => <option value={arm} key={arm}>{arm === "gt" ? "Ground truth" : labels[arm]}</option>)}</select></label><label>Model B<select value={right} onChange={(e) => setRight(e.target.value)}>{["gt", ...arms].map((arm) => <option value={arm} key={arm}>{arm === "gt" ? "Ground truth" : labels[arm]}</option>)}</select></label></div><p className="frame-note"><b>{scene.label}</b> · held-out frame {scene.frames[view]} · models use the same reference view.</p><div className="research-images"><ImageCard title="Ground truth" src={image("gt")} subtitle={`Frame ${scene.frames[view]}`} /><ImageCard title={left === "gt" ? "Ground truth" : labels[left]} src={image(left)} subtitle={`Render frame ${renderFrame(left)}`} /><ImageCard title={right === "gt" ? "Ground truth" : labels[right]} src={image(right)} subtitle={`Render frame ${renderFrame(right)}`} /></div><h3>Scene scores</h3><p className="small-note">Corrected foreground scores from the supplied 40-scene export; each scene score averages nine held-out views. W4A4 is not included in that export.</p><MetricTable headers={["Model", "PSNR ↑", "SSIM ↑", "LPIPS ↓"]}>{[...new Set([left, right])].filter((arm) => arm !== "gt").map(metricRow)}</MetricTable></section></>;
}

function MetricTable({ headers, children }) { return <div className="table-wrap"><table><thead><tr>{headers.map((header) => <th key={header}>{header}</th>)}</tr></thead><tbody>{children}</tbody></table></div>; }

function Results({ focalMetrics }) {
  const means = focalMetrics.arm_means_foreground || {};
  const focalVsW3 = focalMetrics.focal_vs_baseline?.all40 || {};
  const rows = [["full_qfix", "Full VGGT"], ["w3a3_qfix", "W3A3"], ["w3a3_qfix_focal", focalCalibration.label]];
  return <section className="research-section"><h2>Corrected 40-scene results.</h2><p>The focal-calibration model improves on W3A3 under the corrected camera evaluation. This table uses the supplied 40-scene foreground summary; it deliberately does not mix it with the earlier 8-scene result export.</p><MetricTable headers={["Model", "Coverage", "PSNR ↑ (dB)", "SSIM ↑", "LPIPS ↓"]}>{rows.map(([id, label]) => <tr key={id}><th>{label}</th><td>{means[id]?.n || "—"} scenes</td><td>{number(means[id]?.psnr, 3)}</td><td>{number(means[id]?.ssim)}</td><td>{number(means[id]?.lpips)}</td></tr>)}</MetricTable><h3>W3A3 + focal calibration vs. W3A3</h3><MetricTable headers={["Metric", "Mean change", "Improved scenes", "95% CI", "Wilcoxon p"]}><tr><th>PSNR ↑</th><td>+{number(focalVsW3.psnr?.mean, 3)} dB</td><td>{focalVsW3.psnr?.improved}/40</td><td>[{number(focalVsW3.psnr?.bootstrap_ci95?.[0], 3)}, {number(focalVsW3.psnr?.bootstrap_ci95?.[1], 3)}]</td><td>{number(focalVsW3.psnr?.wilcoxon_p, 5)}</td></tr><tr><th>SSIM ↑</th><td>+{number(focalVsW3.ssim?.mean)}</td><td>{focalVsW3.ssim?.improved}/40</td><td>[{number(focalVsW3.ssim?.bootstrap_ci95?.[0])}, {number(focalVsW3.ssim?.bootstrap_ci95?.[1])}]</td><td>{number(focalVsW3.ssim?.wilcoxon_p, 5)}</td></tr><tr><th>LPIPS ↓</th><td>−{number(focalVsW3.lpips?.mean)}</td><td>{focalVsW3.lpips?.improved}/40</td><td>[{number(focalVsW3.lpips?.bootstrap_ci95?.[0])}, {number(focalVsW3.lpips?.bootstrap_ci95?.[1])}]</td><td>{number(focalVsW3.lpips?.wilcoxon_p, 5)}</td></tr></MetricTable><p className="small-note">For LPIPS, the reported value is the reduction from W3A3, so a positive reduction favours focal calibration.</p></section>;
}

function Diagnostics({ geometry, runtimes }) {
  const rows = arms.filter((arm) => geometry[arm]?.mean).map((arm) => ({ arm, ...geometry[arm].mean }));
  return <section className="research-section"><h2>Geometry & timing</h2><h3>Geometry diagnostics</h3>{rows.length ? <MetricTable headers={["Model", "n", "Rotation (deg)", "Centre error", "Spread ratio", "Relative depth error", "Point error"]}>{rows.map((row) => <tr key={row.arm}><th>{labels[row.arm]}</th><td>{row.n_scenes}</td><td>{number(row.cam_rot_deg)}</td><td>{number(row.cam_centre_err)}</td><td>{number(row.cam_spread_ratio)}</td><td>{number(row.depth_rel_err)}</td><td>{number(row.point_err)}</td></tr>)}</MetricTable> : <p className="notice">Geometry diagnostics have not been supplied.</p>}<h3>Recorded inference timing</h3>{runtimes.records?.length ? <MetricTable headers={["Scene", "Model", "Seconds", "Peak allocated (MiB)"]}>{runtimes.records.map((row, index) => <tr key={index}><th>{row.scene.split("/")[0]}</th><td>{labels[row.variant] || row.variant}</td><td>{number(row.seconds, 3)}</td><td>{number(row.peak_allocated_mib, 1)}</td></tr>)}</MetricTable> : <p className="notice">Timing data has not been packaged. No end-to-end speedup is assumed from bit-width.</p>}</section>;
}

export default function ResearchExplorer() {
  const { data, error } = useResearchData(); const [tab, setTab] = useState("compare");
  const tabs = [["compare", "Compare views"], ["results", "Results"], ["experiments", "Experiments"], ["diagnostics", "Geometry & timing"], ["method", "Method & data"]];
  if (error) return <section className="research-section"><p className="notice">{error}</p></section>; if (!data) return <section className="research-section"><p>Loading research explorer…</p></section>;
  return <section className="research-shell"><nav className="research-tabs">{tabs.map(([id, label]) => <button className={tab === id ? "active" : ""} onClick={() => setTab(id)} key={id}>{label}</button>)}</nav>{tab === "compare" && <Compare catalog={data.catalog} focalMetrics={data.focalMetrics} finalMetrics={data.finalMetrics} />}{tab === "results" && <Results focalMetrics={data.focalMetrics} />}{tab === "experiments" && <Experiments />}{tab === "diagnostics" && <Diagnostics geometry={data.geometry} runtimes={data.runtimes} />}{tab === "method" && <section className="research-section"><h2>One downstream pipeline.</h2><div className="pipeline-steps"><span>01<br /><b>Six RGB views</b></span><span>02<br /><b>VGGT variant</b></span><span>03<br /><b>Camera conversion + Sim(3)</b></span><span>04<br /><b>3DGS, 7k iterations</b></span><span>05<br /><b>Nine held-out views</b></span></div><h3>Evaluation protocol</h3><p>Foreground raw metrics are primary. Each scene averages nine held-out views before scene-level aggregation and paired testing. The corrected 40-scene results are shown separately from the earlier 8-scene export.</p><p>This viewer displays precomputed images and supplied measurements; it does not run VGGT or train Gaussian Splats.</p></section>}</section>;
}
