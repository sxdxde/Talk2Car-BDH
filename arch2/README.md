# arch2 — Distractor-Contrastive Loss (planned, not yet built)

Future home for the "second architecture round" work — currently just the
distractor-contrastive loss idea (see FINDINGS.md for full context and
rationale). Nothing here is implemented yet; this folder is a placeholder
so the eventual files (data-prep extension, loss module, tests) have a
home without touching the existing `analysis/`, `bdh_grounding/`, or
`train.py` structure prematurely.

Planned contents, mirroring the existing project layout conventions:
- `build_distractor_index.py` — Step 6 extension: projects nuScenes 3D
  distractor boxes into 2D pixel coordinates (train + val, not just val),
  alongside the existing count-only `analysis/build_scene_index.py`.
- distractor-hinge-loss module (CPU-testable, same convention as
  `bdh_grounding.pipeline.TverskyFocalLoss`).
- wiring into `train.py` (new `--distractor-index` CLI arg, `lambda_distractor`
  config knob, added to the training loss).
- a local unit test under `tests/`, same convention as
  `test_tversky_focal_loss` in `tests/plumbing_test.py`.

See FINDINGS.md's "Distractor-Contrastive Loss (planned)" section for the
full Step A-E scoping and the honest effort/risk assessment before
starting this.
