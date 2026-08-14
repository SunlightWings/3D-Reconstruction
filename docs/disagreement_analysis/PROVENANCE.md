# Provenance

The canonical, complete machine-readable record is [`artifact_manifest.json`](../../results/disagreement_analysis/artifact_manifest.json). It records local absolute source paths and SHA-256 values for frozen selection, corrected GT manifest, all input tensors and GT bundles, checkpoint and W4A4 directory contents, six predictions, analysis outputs, and scripts.

Key frozen identities:

- Frozen selection: `cb0aad3e784d30457681369a9fc7fbd9e656f8aba9eed8decfb53cc7c2deb514`
- Corrected GT manifest: `bc0e84502fd9d49d5a1d20bc7ead1444934c2104f89ef92dc72ff994d905c357`
- Base VGGT checkpoint: `b08a43baa2db1aad9718e71e098831b8ad32f6f6826c802e9eb714aa34420969`
- W4A4 directory-content hash: `c3add7d27ad7a34a8c754bdc55dbe8708b05e953ec9fd0c1dcc199761d653341`

The branch intentionally contains compact summaries and representative figures only. Predictions, tensors, GT bundles, checkpoints, datasets, heatmap trees, logs, and training outputs remain local and are referenced—not committed.
