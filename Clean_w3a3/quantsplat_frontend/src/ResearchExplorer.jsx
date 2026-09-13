import { useEffect, useState } from "react";

const asset = (path) => `/${path.replace(/^\//, "")}`;
const number = (value, digits = 4) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
const arms = ["full", "w4a4", "w3a3", "w3a3_conf"];
const labels = { full: "Full VGGT", w4a4: "W4A4", w3a3: "W3A3", w3a3_conf: "W3A3 + confidence" };

function useResearchData() {
  const [data, setData] = useState(null); const [error, setError] = useState("");
  useEffect(() => {
    Promise.all(["catalog.json", "metrics.json", "geometry.json", "runtime.json", "turntable/manifest.json"].map((name) =>
      fetch(asset(name)).then((response) => response.ok ? response.json() : name.includes("turntable") ? {} : Promise.reject(new Error(name))),
    )).then(([catalog, metrics, geometry, runtimes, turntable]) => setData({ catalog, metrics, geometry, runtimes, turntable }))
      .catch(() => setError("Packaged research data could not be loaded. Build the site from this repository so Vite can include quantsplat_demo/data."));
  }, []);
  return { data, error };
}

function ImageCard({ title, src, subtitle }) {
  return <article className="research-image-card"><b>{title}</b>{src ? <img src={asset(src)} alt={`${title}: ${subtitle}`} /> : <div className="missing-render">Render pending</div>}<small>{subtitle}</small></article>;
}

function Compare({ catalog, metrics }) {
  const [sceneId, setSceneId] = useState(catalog.scenes[0].id); const [view, setView] = useState(0);
  const [left, setLeft] = useState("full"); const [right, setRight] = useState("w3a3");
  const scene = catalog.scenes.find((item) => item.id === sceneId) || catalog.scenes[0];
  const row = metrics.per_scene?.find((item) => item.scene === scene.id) || {};
  const image = (arm) => arm === "gt" ? scene.gt[view] : scene.renders?.[arm]?.[view];
  const metricRow = (arm) => <tr key={arm}><th>{labels[arm]}</th><td>{number(row[`${arm}_foreground_psnr`], 3)}</td><td>{number(row[`${arm}_foreground_ssim`])}</td><td>{number(row[`${arm}_foreground_lpips`])}</td></tr>;
  return <section className="research-section"><h2>Same camera. Different precision.</h2><div className="research-controls"><label>Scene<select value={sceneId} onChange={(e) => { setSceneId(e.target.value); setView(0); }}>{catalog.scenes.map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}</select></label><label>Held-out view<select value={view} onChange={(e) => setView(Number(e.target.value))}>{scene.frames.map((frame, index) => <option value={index} key={frame}>View {index + 1} · frame {frame}</option>)}</select></label><label>Model A<select value={left} onChange={(e) => setLeft(e.target.value)}>{["gt", ...arms].map((arm) => <option value={arm} key={arm}>{arm === "gt" ? "Ground truth" : labels[arm]}</option>)}</select></label><label>Model B<select value={right} onChange={(e) => setRight(e.target.value)}>{["gt", ...arms].map((arm) => <option value={arm} key={arm}>{arm === "gt" ? "Ground truth" : labels[arm]}</option>)}</select></label></div><p className="frame-note"><b>{scene.label}</b> · held-out frame {scene.frames[view]} · models use the same reference view.</p><div className="research-images"><ImageCard title="Ground truth" src={image("gt")} subtitle={`Frame ${scene.frames[view]}`} /><ImageCard title={left === "gt" ? "Ground truth" : labels[left]} src={image(left)} subtitle={`Frame ${scene.frames[view]}`} /><ImageCard title={right === "gt" ? "Ground truth" : labels[right]} src={image(right)} subtitle={`Frame ${scene.frames[view]}`} /></div><h3>Scene scores</h3><p className="small-note">Foreground, raw; averages over all nine held-out views.</p><MetricTable headers={["Model", "PSNR ↑", "SSIM ↑", "LPIPS ↓"]}>{[...new Set([left, right])].filter((arm) => arm !== "gt").map(metricRow)}</MetricTable></section>;
}

function MetricTable({ headers, children }) { return <div className="table-wrap"><table><thead><tr>{headers.map((header) => <th key={header}>{header}</th>)}</tr></thead><tbody>{children}</tbody></table></div>; }

function Turntable({ catalog, manifest }) {
  const ids = Object.keys(manifest.scenes || {}).filter((id) => catalog.scenes.some((scene) => scene.id === id));
  const [id, setId] = useState(ids[0] || ""); const [angle, setAngle] = useState(0); const scene = catalog.scenes.find((item) => item.id === id);
  if (!id) return <section className="research-section"><h2>Precomputed 3D reconstruction</h2><p className="notice">No turntable images are packaged.</p></section>;
  const index = String(Math.floor(angle / 10)).padStart(3, "0"); const category = id.split("/")[0];
  return <section className="research-section"><h2>Rotate the reconstructed scene.</h2><div className="research-controls"><label>Scene<select value={id} onChange={(e) => setId(e.target.value)}>{ids.map((sceneId) => <option key={sceneId} value={sceneId}>{catalog.scenes.find((item) => item.id === sceneId)?.label || sceneId}</option>)}</select></label><label>Rotation <input type="range" min="0" max="350" step="10" value={angle} onChange={(e) => setAngle(Number(e.target.value))} /><span>{angle}°</span></label></div><div className="research-images">{["full", "w4a4", "w3a3"].map((arm) => <ImageCard key={arm} title={labels[arm]} src={`turntable/${category}/${id.split("/")[1]}/${arm}/${index}.png`} subtitle={`Novel orbit view · ${angle}°`} />)}</div><p className="small-note">These synthetic orbit views are for qualitative inspection only; they are not PSNR, SSIM, or LPIPS evaluation images.</p></section>;
}

function Results({ catalog, metrics }) {
  const [region, setRegion] = useState("foreground"); const [metric, setMetric] = useState("psnr");
  const reported = (arm) => metrics.arms?.[arm]?.[region] || {}; const suffix = metric.toUpperCase();
  const pairs = Object.entries(metrics.pairs || {}).filter(([key]) => key.includes(`__${region}__`));
  return <section className="research-section"><h2>Look beyond a single view.</h2><div className="research-controls"><label>Evaluation region<select value={region} onChange={(e) => setRegion(e.target.value)}><option value="foreground">Foreground (primary)</option><option value="content">Content (secondary)</option></select></label><label>Per-scene metric<select value={metric} onChange={(e) => setMetric(e.target.value)}><option value="psnr">PSNR</option><option value="ssim">SSIM</option><option value="lpips">LPIPS</option></select></label></div><MetricTable headers={["Model", "Coverage", "PSNR ↑ (dB)", "SSIM ↑", "LPIPS ↓"]}>{arms.map((arm) => <tr key={arm}><th>{labels[arm]}</th><td>{reported(arm).psnr == null ? "Pending" : `${catalog.scenes.filter((scene) => metrics.per_scene?.some((row) => row.scene === scene.id && row[`${arm}_${region}_psnr`] != null)).length} scenes`}</td><td>{number(reported(arm).psnr)}</td><td>{number(reported(arm).ssim)}</td><td>{number(reported(arm).lpips)}</td></tr>)}</MetricTable><h3>Paired comparisons</h3><MetricTable headers={["Comparison", "Metric", "n", "Mean difference", "95% CI", "p (unadjusted)"]}>{pairs.map(([key, value]) => { const match = /^full_minus_(.+?)__[^_]+__(.+)$/.exec(key); const ci = value.ci95 || []; return <tr key={key}><th>Full − {labels[match?.[1]] || match?.[1]}</th><td>{match?.[2]?.toUpperCase() || suffix}</td><td>{value.n}</td><td>{number(value.mean_delta)}</td><td>[{number(ci[0])}, {number(ci[1])}]</td><td>{number(value.p, 5)}</td></tr>; })}</MetricTable><p className="small-note">Positive Full-minus-variant values favour Full for PSNR/SSIM; negative values favour Full for LPIPS. A non-significant test is not equivalence.</p></section>;
}

function Diagnostics({ geometry, runtimes }) {
  const rows = arms.filter((arm) => geometry[arm]?.mean).map((arm) => ({ arm, ...geometry[arm].mean }));
  return <section className="research-section"><h2>Geometry & timing</h2><h3>Geometry diagnostics</h3>{rows.length ? <MetricTable headers={["Model", "n", "Rotation (deg)", "Centre error", "Spread ratio", "Relative depth error", "Point error"]}>{rows.map((row) => <tr key={row.arm}><th>{labels[row.arm]}</th><td>{row.n_scenes}</td><td>{number(row.cam_rot_deg)}</td><td>{number(row.cam_centre_err)}</td><td>{number(row.cam_spread_ratio)}</td><td>{number(row.depth_rel_err)}</td><td>{number(row.point_err)}</td></tr>)}</MetricTable> : <p className="notice">Geometry diagnostics have not been supplied.</p>}<h3>Recorded inference timing</h3>{runtimes.records?.length ? <MetricTable headers={["Scene", "Model", "Seconds", "Peak allocated (MiB)"]}>{runtimes.records.map((row, index) => <tr key={index}><th>{row.scene.split("/")[0]}</th><td>{labels[row.variant] || row.variant}</td><td>{number(row.seconds, 3)}</td><td>{number(row.peak_allocated_mib, 1)}</td></tr>)}</MetricTable> : <p className="notice">Timing data has not been packaged. No end-to-end speedup is assumed from bit-width.</p>}</section>;
}

export default function ResearchExplorer() {
  const { data, error } = useResearchData(); const [tab, setTab] = useState("compare");
  const tabs = [["compare", "Compare views"], ["turntable", "Precomputed 3D"], ["results", "Results"], ["diagnostics", "Geometry & timing"], ["method", "Method & data"]];
  if (error) return <section className="research-section"><p className="notice">{error}</p></section>; if (!data) return <section className="research-section"><p>Loading research explorer…</p></section>;
  return <section className="research-shell"><nav className="research-tabs">{tabs.map(([id, label]) => <button className={tab === id ? "active" : ""} onClick={() => setTab(id)} key={id}>{label}</button>)}</nav>{tab === "compare" && <Compare catalog={data.catalog} metrics={data.metrics} />}{tab === "turntable" && <Turntable catalog={data.catalog} manifest={data.turntable} />}{tab === "results" && <Results catalog={data.catalog} metrics={data.metrics} />}{tab === "diagnostics" && <Diagnostics geometry={data.geometry} runtimes={data.runtimes} />}{tab === "method" && <section className="research-section"><h2>One downstream pipeline.</h2><div className="pipeline-steps"><span>01<br /><b>Six RGB views</b></span><span>02<br /><b>VGGT variant</b></span><span>03<br /><b>Camera conversion + Sim(3)</b></span><span>04<br /><b>3DGS, 7k iterations</b></span><span>05<br /><b>Nine held-out views</b></span></div><h3>Evaluation protocol</h3><p>Foreground raw metrics are primary; content is secondary. Each scene averages nine held-out views before scene-level aggregation and paired testing. Missing results remain pending and are never inferred from images.</p><p>This viewer displays precomputed images and supplied measurements; it does not run VGGT or train Gaussian Splats.</p></section>}</section>;
}
