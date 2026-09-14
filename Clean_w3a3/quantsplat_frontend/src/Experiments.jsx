import { useState } from "react";

const phases = [
  {
    number: "01", title: "Validating the benchmark", range: "000–024",
    summary: "Before comparing models, we audited masks, convergence, alignment, exposure, and the downstream 3DGS harness.",
    takeaway: "Foreground evaluation, a fixed 7,000-iteration budget, and reproducibility checks became prerequisites for later comparisons.",
    experiments: [
      ["000", "Full vs W4A4 disagreement dataset", "40-scene CO3D benchmark built"],
      ["002", "Evaluation-region audit", "Content mask found to be background-dominated"],
      ["003", "Oracle iteration sweep", "7k iterations sufficient"],
      ["008", "12 input views", "Improved oracle reconstruction"],
      ["012", "24-view follow-up", "No further improvement"],
      ["017", "Sim(3) audit", "Units/direction issue identified"],
      ["020", "Foreground re-score", "Earlier Full/W4A4 conclusion changed"],
      ["022", "Exposure correction", "Photometry is a substantial confound"],
    ],
  },
  {
    number: "02", title: "Stress-testing quantization", range: "025–038",
    summary: "We measured quantization damage directly in cameras, depth, and points, then ablated QuantVGGT components.",
    takeaway: "W2A4 causes a structured pose collapse; direct geometry metrics expose damage more clearly than costly downstream rendering.",
    experiments: [
      ["025", "W2A4 characterization", "Pose head collapse; camera spread ≈1% of Full"],
      ["026–029", "W4A4 component ablations", "Hadamard rotation especially important"],
      ["030", "Perfect-confidence oracle pruning", "Upper bound for confidence-driven selection"],
      ["033", "Depth accuracy across 1,330 groups", "Quantization effect quantified at scale"],
      ["034–035", "World-point error decomposition", "Depth, camera, and scale contributions separated"],
      ["036", "Predicting W4A4 error", "Tested predictability from W4A4 outputs"],
      ["038", "Statistical power", "Scene-count requirements estimated"],
    ],
  },
  {
    number: "03", title: "Finding the failure mechanism", range: "039–049",
    summary: "Camera/point swaps, synthetic damage, focal tests, viewpoint sensitivity, and input-order ensembling isolated likely sources of downstream damage.",
    takeaway: "Camera and focal errors matter more than raw point error; input-order instability became a useful confidence signal.",
    experiments: [
      ["039", "Camera / point swap", "Cameras implicated more strongly than points"],
      ["045", "Input-order confidence", "Reported camera-error correlation ρ ≈ 0.88"],
      ["047–049", "Input-order ensembling", "Averaging orderings reduced pose error"],
    ],
  },
  {
    number: "04", title: "Pushing to W3A3", range: "050–053",
    summary: "W3A3 was characterized as the middle-bit regime, followed by point-confidence and camera/point-swap experiments.",
    takeaway: "W3A3 offers a more useful degradation regime than W2A4, but its pre-fix rendering results are historical only.",
    experiments: [
      ["050", "First W3A3 characterization", "Geometry regime established"],
      ["051", "Initial W3A3 downstream arm", "Historical rendering separation observed"],
      ["052", "Perfect point-confidence", "Confidence upper-bound experiment"],
      ["053", "W3A3 camera / point swap", "Failure mechanism probed"],
    ],
  },
  {
    number: "05", title: "Corrected downstream results", range: "054–058",
    summary: "A camera-rotation conversion bug was fixed, downstream rendering was re-established, and confidence/focal interventions were tested on corrected cameras.",
    takeaway: "Confidence can rank unreliable points, but point pruning/weighting did not materially close the W3A3 reconstruction gap. Camera-level correction remains the more promising path.",
    experiments: [
      ["054", "Camera rotation conversion fix", "All prior rendering metrics invalidated"],
      ["055", "Point-confidence downstream test", "Little/no reconstruction recovery"],
      ["056", "Focal-length correction", "Camera focal error reduced"],
      ["057", "Corrected 40-scene rerun", "80 3DGS runs completed; scoring pending at log time"],
      ["058", "48-scene extension", "Valid manifest built; extension cancelled"],
    ],
  },
];

const selected = [
  ["000", "Full vs W4A4 disagreement dataset", "Dataset built"],
  ["002", "Evaluation-region audit", "Metric flaw found"],
  ["003", "7k vs 30k iteration sweep", "7k sufficient"],
  ["008", "12 input views", "Improved oracle"],
  ["012", "24 input views", "No further gain"],
  ["025", "W2A4", "Pose collapse"],
  ["027", "W4A4 component ablation", "Hadamard rotation critical"],
  ["033", "Depth accuracy, 1,330 groups", "Quantified"],
  ["039", "Camera / point swap", "Cameras implicated"],
  ["045", "Input-order confidence", "Strong signal"],
  ["050", "W3A3", "Middle-bit regime"],
  ["054", "Camera quaternion bug", "Critical fix"],
  ["055", "Point-confidence downstream test", "Little/no recovery"],
  ["056", "Focal correction", "Camera error reduced"],
  ["057", "Corrected 40-scene rerun", "Completed"],
  ["058", "48-scene extension", "Cancelled"],
];

export default function Experiments() {
  const [showIndex, setShowIndex] = useState(false);
  return <section className="experiments-section">
    <div className="experiment-heading">
      <div><p className="eyebrow">RESEARCH JOURNEY</p><h2>What we tested</h2></div>
      <span>59 documented experiments · 000–058</span>
    </div>

    <aside className="experiment-warning">
      <b>Important interpretation note</b>
      <p>Rendering metrics from experiments 001–053 used the pre-fix camera conversion and are superseded. Their geometry-only findings remain useful. Corrected rendering begins at experiment 054.</p>
    </aside>

    <p className="experiment-intro">We systematically asked what quantizing a feed-forward geometry model changes, where the resulting reconstruction damage lives, and whether confidence or correction can recover it.</p>

    <div className="experiment-flow" aria-label="Research progression">
      <span>Quantization</span><i>↓</i><span>Metric audit</span><i>↓</i><span>Camera mechanism</span><i>↓</i><span>Confidence tests</span><i>↓</i><span>Corrected evaluation</span>
    </div>

    <div className="phase-list">
      {phases.map((phase, index) => <details className={`phase-card ${index === phases.length - 1 ? "corrected" : ""}`} key={phase.number} open={index === phases.length - 1}>
        <summary><span className="phase-number">{phase.number}</span><span className="phase-copy"><b>{phase.title} <em>— Summary</em></b><small>Experiments {phase.range} · {phase.summary}</small></span><span className="phase-toggle">Details</span></summary>
        <div className="phase-body"><p>{phase.summary}</p><p className="phase-takeaway"><b>Takeaway:</b> {phase.takeaway}</p><table><thead><tr><th>#</th><th>Experiment</th><th>Outcome</th></tr></thead><tbody>{phase.experiments.map(([id, name, outcome]) => <tr key={id}><td>{id}</td><td>{name}</td><td>{outcome}</td></tr>)}</tbody></table></div>
      </details>)}
    </div>

    <section className="corrected-results">
      <p className="eyebrow">CORRECTED CAMERA EVALUATION</p>
      <h3>Experiment 055: confidence identified bad points, but did not rescue 3DGS.</h3>
      <p>With corrected cameras, point confidence ranked reliability reasonably well (top-decile-error AUROC 0.843; Spearman confidence vs. negative error 0.607). Yet neither learned confidence pruning nor weighting materially recovered W3A3’s downstream gap.</p>
      <div className="table-wrap"><table><thead><tr><th>Method</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th></tr></thead><tbody><tr><td>Full</td><td>14.574</td><td>0.3195</td><td>0.3464</td></tr><tr><td>W3A3</td><td>12.983</td><td>0.2485</td><td>0.4390</td></tr><tr><td>Oracle prune</td><td>13.142</td><td>0.2551</td><td>0.4407</td></tr><tr><td>Learned prune</td><td>12.533</td><td>0.2477</td><td>0.4478</td></tr><tr><td>Learned weight</td><td>12.931</td><td>0.2491</td><td>0.4379</td></tr></tbody></table></div>
    </section>

    <section className="experiment-index">
      <div><h3>Experiment index</h3><p>Selected milestones are shown by default. The project archive contains 59 documented experiments with motivations, configurations, code, results, and logs.</p></div>
      <button className="secondary index-toggle" onClick={() => setShowIndex((value) => !value)}>{showIndex ? "Show selected milestones" : "View all listed milestones"}</button>
      <div className="table-wrap"><table><thead><tr><th>#</th><th>Experiment</th><th>Outcome</th></tr></thead><tbody>{(showIndex ? phases.flatMap((phase) => phase.experiments) : selected).map(([id, name, outcome]) => <tr key={`${id}-${name}`}><td>{id}</td><td>{name}</td><td>{outcome}</td></tr>)}</tbody></table></div>
    </section>
  </section>;
}
