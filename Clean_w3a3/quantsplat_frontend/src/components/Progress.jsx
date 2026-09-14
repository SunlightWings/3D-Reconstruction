const STAGES = {
  waiting: "Preparing reconstruction…",
  uploaded: "Preparing reconstruction…",
  loading_inputs: "Preparing images…",
  loading_model: "Loading reconstruction model…",
  preprocessing: "Preparing images for reconstruction…",
  running_inference: "Estimating cameras and geometry…",
  saving_predictions: "Saving geometry…",
  inference_complete: "Geometry inference complete…",
  preparing_scene: "Preparing Gaussian Splatting…",
  preparing_3dgs: "Preparing Gaussian Splatting…",
  training_3dgs: "Building your 3D reconstruction…",
  exporting_model: "Preparing interactive model…",
};

export const stageLabel = (stage) => STAGES[stage] || "Processing reconstruction…";

function iterationValues(job) {
  const detail = job?.stage_detail || job?.details || job?.result || {};
  const current = job?.current_iteration ?? job?.iteration ?? detail.current_iteration ?? detail.iteration;
  const total = job?.total_iterations ?? detail.total_iterations ?? detail.iterations ?? 7000;
  return { current: Number(current), total: Number(total) };
}

export default function Progress({ job }) {
  const raw = Number(job?.progress || 0);
  const progress = Math.max(0, Math.min(100, Math.round(raw <= 1 ? raw * 100 : raw)));
  const { current, total } = iterationValues(job);
  const hasIterations = job?.stage === "training_3dgs" && Number.isFinite(current) && Number.isFinite(total);

  return <section className="progress-card" aria-live="polite">
    <p className="eyebrow">RECONSTRUCTION IN PROGRESS</p>
    <h2>{job?.stage === "training_3dgs" ? "Building your 3D reconstruction" : "Preparing your reconstruction"}</h2>
    {hasIterations && <p className="iteration-count"><b>{current.toLocaleString()}</b> / {total.toLocaleString()} iterations</p>}
    <div className="progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow={progress}>
      <i style={{ width: `${progress}%` }} />
    </div>
    <div className="progress-row"><span>{stageLabel(job?.stage)}</span><b>{progress}%</b></div>
    {job?.stage === "waiting" || job?.stage === "loading_model"
      ? <p className="muted">A cold GPU worker can take a minute or two to start. Progress is reported by the backend and is not estimated locally.</p>
      : <p className="muted">You can leave or refresh this page; this job will be recovered automatically.</p>}
  </section>;
}
