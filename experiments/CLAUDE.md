# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Experiment logging (standing rule)

**Maintain `ongoing_logs.md` at the repo root. Append an entry for EVERY experiment run. Never overwrite, never edit a past entry.**

The log is append-only because its value is the record of what was believed when, and what changed that belief. Correcting a past entry destroys that. If an earlier entry turns out to be wrong, append a new entry that says so and supersedes it.

### Every entry must contain

- **Date/time, git commit hash, and the exact command executed** — the full command line, copy-pasteable, not a paraphrase.
- **Full config:**
  - init source (`full` / `w4a4` / `oracle` / `random`)
  - iterations
  - densify settings (e.g. `--densify_until_iter`, or `OFF`)
  - number of input views (stride and held-out count)
  - mask type (`content` / `foreground` / `boundary_band` / `background`)
  - background handling (white vs black, masked vs not)
  - depth regularisation on/off
- **Scenes used** — list them, or name the exact scene set and its size.
- **Results:** mean **and** per-scene PSNR / SSIM / LPIPS, plus where relevant Chamfer F-score, floater mass, Gaussian count.
- **PASS/FAIL against the stated target, with the target written out** — not "passed the gate" but "target: oracle >= 20 dB; measured 14.06 dB; FAIL".
- **One-line interpretation:** what this run rules in or rules out.

### DECISION STATE

Keep a `DECISION STATE` section pinned at the top of `ongoing_logs.md`, rewritten in place after each entry (this is the one section that is rewritten rather than appended). It contains:

- Which diagnostic gates currently pass and which fail
- Which conclusion is currently licensed by the evidence
- What single experiment would most change the picture
- Open blockers

### Never write a number you did not measure

No estimates, no numbers carried over from memory, no plausible-looking placeholders. If a value was not produced by the run being logged, cite where it came from or leave it out.

If something cannot be run, write **BLOCKED** with the reason — follow the D3 entry in [DIAGNOSTIC_RESULTS.md](results/downstream_3dgs/DIAGNOSTIC_RESULTS.md), which states what artifact is missing, why it was not faked, and what the concrete next step is. A blocked entry is a real result; a fabricated one is not.
