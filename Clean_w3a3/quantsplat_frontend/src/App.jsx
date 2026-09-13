import { useEffect, useState } from "react";
import { createJob, getJob, health, jobModelUrl } from "./api/quantsplat";
import ImageSlots from "./components/ImageSlots";
import Progress from "./components/Progress";
import GaussianViewer from "./components/GaussianViewer";
import ResearchExplorer from "./ResearchExplorer";
import "./styles.css";

const KEY = "quantsplat_job_id"; const MAX_SIZE = 10 * 1024 * 1024;
const valid = (file) => file && ["image/jpeg", "image/png", "image/webp"].includes(file.type) && file.size <= MAX_SIZE;

function CreateReconstruction() {
  const [files, setFiles] = useState(Array(6).fill(null)); const [job, setJob] = useState(null);
  const [state, setState] = useState("selecting"); const [error, setError] = useState(""); const [online, setOnline] = useState(null);
  const modelUrl = jobModelUrl(job);
  useEffect(() => { health().then(() => setOnline(true)).catch(() => setOnline(false)); const saved = localStorage.getItem(KEY); if (saved) resume(saved); }, []);
  useEffect(() => { if (state !== "processing" || !job?.job_id) return undefined; const id = setInterval(() => poll(job.job_id), 2000); return () => clearInterval(id); }, [state, job?.job_id]);
  async function poll(id) { try { const next = await getJob(id); setJob(next); if (next.status === "failed") { setError(next.error || "Reconstruction failed. Please try again."); setState("error"); } else if (next.status === "complete") setState(jobModelUrl(next) ? "viewing" : "complete"); } catch (e) { setError(e.message); setState("error"); } }
  function resume(id) { setState("processing"); poll(id); }
  async function submit() { setError(""); if (files.some((file) => !valid(file))) { setError("Use six JPEG, PNG, or WebP images, each no larger than 10 MB."); return; } try { setState("uploading"); const created = await createJob(files); localStorage.setItem(KEY, created.job_id); setJob(created); setState("processing"); await poll(created.job_id); } catch (e) { setError(e.message); setState("error"); } }
  function restart() { localStorage.removeItem(KEY); setJob(null); setFiles(Array(6).fill(null)); setError(""); setState("selecting"); }
  return <><section className="hero"><p className="eyebrow">CONFIDENCE-WEIGHTED 3D RECONSTRUCTION</p><h1>Six photos.<br /><em>One explorable scene.</em></h1><p>Upload six overlapping views of one object. QuantSplat estimates geometry now and will deliver an interactive Gaussian Splat as soon as reconstruction export is available.</p></section>
    {(state === "selecting" || state === "uploading" || state === "error") && <section className="card"><div className="section-title"><div><p className="eyebrow">01 / YOUR PHOTOS</p><h2>Show the object from six angles.</h2></div><span>{files.filter(Boolean).length} / 6 selected</span></div><ImageSlots files={files} onChange={setFiles} disabled={state === "uploading"} />
      <aside><b>For the best reconstruction</b><p>Move around the object; keep it visible and consistently lit; make adjacent photos overlap; avoid blur.</p></aside>{error && <div className="error">{error}</div>}<button className="primary" onClick={submit} disabled={state === "uploading" || files.some((file) => !file)}>{state === "uploading" ? "Uploading…" : "Reconstruct in 3D"}</button></section>}
    {state === "processing" && <Progress job={job} />}
    {state === "complete" && <section className="card"><p className="eyebrow">GEOMETRY INFERENCE COMPLETE</p><h2>Your reconstruction is being prepared.</h2><p>The backend has completed W3A3 geometry inference. Interactive Gaussian Splat export is being integrated; no result model URL is available yet.</p><button className="primary" onClick={restart}>Reconstruct another object</button></section>}
    {state === "viewing" && <section className="card"><p className="eyebrow">INTERACTIVE RECONSTRUCTION</p><h2>Your Gaussian Splat scene</h2><GaussianViewer modelUrl={modelUrl} onReset={restart} /></section>}
    <section className="method"><b>About the method</b><p>QuantSplat uses W3A3 QuantVGGT for camera and geometry prediction, followed by Gaussian Splatting. Research evaluation metrics are intentionally not shown for arbitrary uploads.</p></section></>;
}

export default function App() {
  const [page, setPage] = useState("create");
  return <main><header><a className="brand" href="/">QUANTSPLAT <i>/</i> VISION LAB</a><span className="app-nav"><button className={page === "create" ? "active" : ""} onClick={() => setPage("create")}>Create reconstruction</button><button className={page === "research" ? "active" : ""} onClick={() => setPage("research")}>Research explorer</button></span></header>{page === "create" ? <CreateReconstruction /> : <ResearchExplorer />}</main>;
}
