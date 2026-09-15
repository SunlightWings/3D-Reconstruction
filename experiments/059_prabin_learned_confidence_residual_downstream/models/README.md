# Frozen learned modules

Final deployment artifacts for Experiment 059.

| Artifact | Role |
|---|---|
| `confidence_predictor_best_model.pt` | Learned W3A3 camera-error / reliability predictor |
| `camera_residual_corrector.npz` | Ridge residual corrector for rotation, camera center, and focal parameters |

Both operate downstream of W3A3 geometry. Training and inference code is in `../code/`.
Exact SHA-256 hashes are recorded in `../SHA256SUMS.txt`.
