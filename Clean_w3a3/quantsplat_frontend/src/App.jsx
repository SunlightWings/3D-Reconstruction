import { lazy, Suspense, useEffect, useState } from "react";
import { createJob, health, jobModelUrl, pollJob } from "./api/quantsplat";
import ImageSlots from "./components/ImageSlots";
import Progress from "./components/Progress";
import ResearchExplorer from "./ResearchExplorer";
import "./styles.css";

const GaussianViewer = lazy(() => import("./components/GaussianViewer"));

const JOB_KEY = "quantsplat_job_id";
const MAX_SIZE = 10 * 1024 * 1024;
const ACCEPTED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const TEST_MODEL_URL = import.meta.env.VITE_TEST_MODEL_URL || "";

function validateFiles(files) {
  if (files.filter(Boolean).length !== 6) return "Exactly six images are required.";
  const oversized = files.findIndex((file) => file.size > MAX_SIZE);
  if (oversized >= 0) return `Image ${oversized + 1} is larger than 10 MB.`;
  const unsupported = files.findIndex((file) => !ACCEPTED_TYPES.has(file.type));
  if (unsupported >= 0) return `Image ${unsupported + 1} has an unsupported format. Use JPEG, PNG, or WebP.`;
  return "";
}

function friendlyError(error, fallback = "Something went wrong.") {
  const technical = error instanceof Error ? error.message : String(error || fallback);
  const lower = technical.toLowerCase();
  let message = fallback;
  if (error?.status === 409 || lower.includes("already running")) message = "A reconstruction is already running.";
  else if (error?.status === 429 || lower.includes("budget")) message = "The monthly demo budget has been reached.";
  else if (lower.includes("exactly") && lower.includes("six")) message = "Exactly six images are required.";
  else if (lower.includes("10 mb") || lower.includes("too large")) message = "One of the images is larger than 10 MB.";
  else if (lower.includes("unsupported") || lower.includes("content type")) message = "One of the images has an unsupported format.";
  else if (lower.includes("reach") || lower.includes("network") || lower.includes("fetch")) message = "Unable to reach the reconstruction server.";
  else if ((error?.status || 0) >= 500) message = "The GPU service is temporarily unavailable.";
  return { message, technical };
}

function ErrorPanel({ error, onRetry, onRestart }) {
  return <section className="error-card" role="alert">
    <p className="eyebrow">RECONSTRUCTION INTERRUPTED</p>
    <h2>{error.message}</h2>
    <p>Your selected photos are still available unless you start over.</p>
    {error.technical && error.technical !== error.message && <details><summary>Show details</summary><pre>{error.technical}</pre></details>}
    <div className="error-actions">{onRetry && <button className="primary" onClick={onRetry}>Try again</button>}<button className="secondary" onClick={onRestart}>Start over</button></div>
  </section>;
}

function CreateReconstruction() {
  const savedJobId = localStorage.getItem(JOB_KEY);
  const [files, setFiles] = useState(Array(6).fill(null));
  const [job, setJob] = useState(savedJobId ? { job_id: savedJobId, status: "waiting", stage: "waiting", progress: 0 } : null);
  const [activeJobId, setActiveJobId] = useState(savedJobId);
  const [pollAttempt, setPollAttempt] = useState(0);
  const [state, setState] = useState(savedJobId ? "processing" : "selecting");
  const [error, setError] = useState(null);
  const [online, setOnline] = useState(null);
  const modelUrl = jobModelUrl(job);

  useEffect(() => {
    const controller = new AbortController();
    health({ signal: controller.signal }).then(() => setOnline(true)).catch((reason) => {
      if (reason?.name !== "AbortError") setOnline(false);
    });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!activeJobId) return undefined;
    const controller = new AbortController();
    setState("processing");
    setError(null);
    pollJob(activeJobId, { signal: controller.signal, onUpdate: setJob }).then((finalJob) => {
      if (finalJob.status === "failed") {
        localStorage.removeItem(JOB_KEY);
        setActiveJobId(null);
        setError({ message: "Reconstruction failed.", technical: finalJob.error || `Failed during ${finalJob.stage || "processing"}.` });
        setState("error");
      } else {
        setState(jobModelUrl(finalJob) ? "viewing" : "complete_without_model");
      }
    }).catch((reason) => {
      if (reason?.name === "AbortError") return;
      setError(friendlyError(reason, "Unable to check reconstruction progress."));
      setState("error");
    });
    return () => controller.abort();
  }, [activeJobId, pollAttempt]);

  async function submit() {
    setError(null);
    const validation = validateFiles(files);
    if (validation) { setError({ message: validation, technical: "" }); return; }
    try {
      setState("uploading");
      const created = await createJob(files);
      localStorage.setItem(JOB_KEY, created.job_id);
      setJob(created);
      setActiveJobId(created.job_id);
    } catch (reason) {
      setError(friendlyError(reason, "The images could not be uploaded."));
      setState("selecting");
    }
  }

  function restart() {
    localStorage.removeItem(JOB_KEY);
    setActiveJobId(null); setJob(null); setFiles(Array(6).fill(null)); setError(null); setState("selecting");
  }

  function retry() {
    if (activeJobId) setPollAttempt((value) => value + 1);
    else setState("selecting");
  }

  function showTestModel() {
    setJob({ status: "complete", stage: "complete", progress: 1, result: { model_url: TEST_MODEL_URL } });
    setState("viewing");
  }

  return <>
    <section className="hero">
      <span className={`service-pill ${online ? "online" : online === false ? "offline" : ""}`}>{online === null ? "Checking backend" : online ? "Backend online" : "Backend unavailable"}</span>
      <p className="eyebrow">INTERACTIVE RECONSTRUCTION</p>
      <p className="hero-instruction">Upload six overlapping views of one object. QuantSplat estimates its geometry, trains a Gaussian Splat, and returns an interactive 3D reconstruction.</p>
    </section>

    {(state === "selecting" || state === "uploading") && <section className="card">
      <div className="section-title"><div><p className="eyebrow">01 / YOUR PHOTOS</p><h2>Show the object from six angles.</h2></div><span>{files.filter(Boolean).length} / 6 selected</span></div>
      <ImageSlots files={files} onChange={setFiles} disabled={state === "uploading"} />
      <aside><b>For the best reconstruction</b><p>Move around the object rather than moving it. Keep the whole object visible, preserve overlap between neighboring views, avoid blur, and keep lighting and background consistent.</p></aside>
      {error && <div className="inline-error" role="alert">{error.message}</div>}
      <div className="submit-actions"><button className="primary" onClick={submit} disabled={state === "uploading" || files.some((file) => !file)}>{state === "uploading" ? "Uploading six photos…" : "Reconstruct in 3D"}</button>{TEST_MODEL_URL && <button className="secondary" onClick={showTestModel}>Open completed test model</button>}</div>
    </section>}
    {state === "processing" && <Progress job={job} />}
    {state === "error" && <ErrorPanel error={error} onRetry={activeJobId ? retry : null} onRestart={restart} />}
    {state === "complete_without_model" && <section className="card"><p className="eyebrow">COMPLETED JOB</p><h2>No interactive model is available for this job.</h2><p>This is likely an older geometry-only job or an expired result. Start a new reconstruction to receive the current Gaussian Splat output.</p><button className="primary" onClick={restart}>Reconstruct another object</button></section>}
    {state === "viewing" && <section className="card result-card"><p className="eyebrow">INTERACTIVE RECONSTRUCTION</p><h2>Your Gaussian Splat scene</h2><Suspense fallback={<div className="viewer viewer-loading">Loading 3D viewer…</div>}><GaussianViewer modelUrl={modelUrl} onReset={restart} /></Suspense></section>}
    <section className="method"><b>About the method</b><p>QuantSplat uses W3A3 QuantVGGT for camera and geometry prediction, followed by 7,000-iteration Gaussian Splatting. Research metrics are intentionally not shown for arbitrary uploads.</p></section>
  </>;
}

export default function App() {
  const [page, setPage] = useState("research");
  return <main>
    <header><a className="brand" href="/"><span>QUANTSPLAT</span><small>Arham Shahzad · Prabin Sharma Poudel · Mohan Manideep Danda</small></a><nav className="app-nav" aria-label="Main navigation"><button className={page === "research" ? "active" : ""} onClick={() => setPage("research")}>Research explorer</button><button className={page === "create" ? "active action" : ""} onClick={() => setPage("create")}>Create reconstruction</button></nav></header>
    {page === "research" ? <ResearchExplorer /> : <CreateReconstruction />}
  </main>;
}
