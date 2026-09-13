const STAGES = {
  loading_inputs: "Preparing images…", loading_model: "Loading reconstruction model…",
  preprocessing: "Preparing images for reconstruction…", running_inference: "Estimating geometry and cameras…",
  saving_predictions: "Saving geometry…", preparing_3dgs: "Preparing Gaussian Splatting…",
  training_3dgs: "Building 3D reconstruction…", exporting_model: "Preparing interactive model…",
};
export const stageLabel = (stage) => STAGES[stage] || "Processing reconstruction…";

export default function Progress({ job }) {
  const progress = Math.max(0, Math.min(100, Math.round((job?.progress || 0) * 100)));
  return <section className="progress-card"><p className="eyebrow">RECONSTRUCTION IN PROGRESS</p>
    <h2>Building your reconstruction</h2><div className="progress-track"><i style={{ width: `${progress}%` }} /></div>
    <div className="progress-row"><span>{stageLabel(job?.stage)}</span><b>{progress}%</b></div>
    <p className="muted">This can take time. Keep this tab open; the job will also resume after a refresh.</p>
  </section>;
}
