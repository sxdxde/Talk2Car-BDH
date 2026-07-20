# Findings & Research Log — BDH-Swapped AttnGrounder

Living document. Append to this as we iterate — don't rewrite history, add
dated entries. STATUS.md is the operational "how to resume" doc (commands,
environment, current blocker); this file is the research narrative: what we
tried, what we learned, what we're still deciding.

## Hypothesis (unchanged since project start)

"Grounding = in-context retrieval": architectures with an explicit,
growing associative memory (BDH's outer-product `rho = K^T @ V`) should show
an AP50 advantage over standard softmax attention specifically on scenes
with multiple same-class candidate objects ("ambiguous" scenes) — because
disambiguation is a retrieval problem, not a compression problem.

## Results so far (Talk2Car val, n=1163; 830 ambiguous / 333 unambiguous)

| Variant | AP50 (all) | AP50 (ambiguous) | AP50 (unambiguous) | Params |
|---|---|---|---|---|
| Baseline (AttnGrounder, unmodified) | 64.92 | 63.37 | 68.77 | 75.84M |
| BDH Mode C (symmetric, non-causal) | 64.06 | 62.17 | 68.77 | 76.63M |
| BDH Mode A (causal text-context, then region query) | 65.09 | 63.98 | 67.87 | 76.63M |
| BDH Mode B | 60.50 (train best) | pending stratified | pending stratified | 76.63M |

**Mode C**: hypothesis NOT supported — uniformly worse than baseline,
concentrated in the ambiguous bucket but in the wrong direction.

**Mode A**: first result pointing in the hypothesis's predicted direction —
beats baseline on ambiguous (+0.61) while ceding ground on unambiguous
(-0.90). A real trade-off toward disambiguation, not a uniform shift. Still
one run/one seed.

**Mode B**: DONE, best AP50 = 60.50 (train), 60.53 (val eval) — clearly the
weakest of the three (train loss/acc looked healthy at ~0.70/~0.97, so this
is a genuine kernel-formulation weakness, not a training collapse).
Single-causal-sequence-with-RoPE, sharing one stream between words and
regions, appears to lose useful structure relative to keeping them
separate (modes A/C). Stratified: ambiguous 58.19 (-5.18 vs baseline),
unambiguous 66.37 (-2.40 vs baseline).

**Full final table:**

| Variant | AP50 (all) | AP50 (ambiguous, n=830) | AP50 (unambiguous, n=333) |
|---|---|---|---|
| Baseline | 64.92 | 63.37 | 68.77 |
| BDH Mode C | 64.06 (-0.86) | 62.17 (-1.20) | 68.77 (0.00) |
| **BDH Mode A** | **65.09 (+0.17)** | **63.98 (+0.61)** | 67.87 (-0.90) |
| BDH Mode B | 60.53 (-4.39) | 58.19 (-5.18) | 66.37 (-2.40) |

**Interpretive insight**: mode B's deficit is *also* concentrated more
heavily on ambiguous than unambiguous (-5.18 vs -2.40), same qualitative
direction as mode C, just much larger. So **two of three BDH formulations
(B and C) do worse specifically on ambiguous scenes** despite both having
an explicit outer-product associative memory — meaning the win isn't just
"has associative memory," it's specifically **mode A's causal
text-self-contextualization step** (words attend to each other before
regions query them) that produces the hypothesis-predicted trade-off. This
directly motivates the BDH->attention hybrid idea (queued above), which
reuses exactly that piece of mode A (`Q_ctx`) as its starting point.

**DECISION (2026-07-20): Mode A wins the ablation study.** All further
improvements (Tversky+Focal loss, BDH stacking, BDH->attention hybrid,
depth prior) build on Mode A going forward.

## SOTA comparison — ThinkDeeper (papers/thinkdeeper.md)

ThinkDeeper (Liao et al., Nov 2025): ViT+BERT (BLIP-pretrained) backbone,
Spatial-Aware World Model (latent future-state rollout from depth maps),
cross-modal hypergraph decoder, two-stage training (world-model rollout +
grounding supervision, Tversky+Focal loss), LLM-annotated DrivePilot
dataset. Reports **76.64 AP50 on Talk2Car test**, #1 on the leaderboard.

**Key finding: their Table 1 also lists same-class-bracket baselines** —
AttnGrounder itself at 61.32 (test), CMSVG (EfficientNet, non-transformer)
at 68.61 (test). Transformer/VLM-pretrained models (VL-BERT 70.03,
RSD-LXMERT 72.64, CAVG 74.62, ThinkDeeper 76.64) form a distinctly higher
bracket that correlates with pretrained-transformer backbones, not just
fusion-mechanism cleverness.

**Reframed target**: not "reach ThinkDeeper's 76.64" (would require
replacing the entire backbone — a different project) but **"match/beat
CMSVG's 68.61, the best same-backbone-class (CNN, non-transformer,
non-VLM) result"**. This is the scientifically honest comparison bracket
for a fusion-mechanism study on a Darknet-53 + BiLSTM/GloVe backbone.

**Test-set evaluation finding (important, changes what "SOTA-comparable"
can mean for us)**: Talk2Car test-set GT has never been publicly released.
The only historical evaluation path was submitting predictions to an
AIcrowd-hosted server (ECCV 2020 C4AV challenge) — **confirmed closed to
submissions since August 1, 2020**. Later papers (RSD-LXMERT 2022,
CAVG/ThinkDeeper 2024-2025) still point to that same dead link as "the"
evaluation mechanism in their own repos, with no evidence of a second live
server. Their test-set numbers are therefore most likely **self-reported,
not independently verified** — the same infrastructure anyone would need
has been defunct for six years. Conclusion: **we cannot obtain a
legitimately-verified test-set number, and neither, as far as we can tell,
can anyone else right now.** We report val-set AP50 only (as we've been
doing), explicitly caveated against other papers' self-reported test
numbers — directionally informative, not strict apples-to-apples.

## Ideas evaluated and rejected / deprioritized

- **Edge/Laplacian boundary loss** (from medical imaging segmentation
  background): rejected. Earns its keep on dense, high-res pixel masks
  where boundary sharpness matters; our aux supervision is a coarse
  13x13/26x26/52x52 grid (`attn_map`/beta) and the primary loss is YOLO box
  regression, not a mask — there's no meaningful "boundary" at that
  resolution for a gradient filter to find.
- **BDH inside the YOLO head**: rejected. The head is a regression module
  (conv -> box offsets), not an attention/retrieval module — no natural
  insertion point for an associative-memory mechanism, and swapping it in
  wouldn't be motivated by the hypothesis.
- **Vision-BDH** (ViT-style patch BDH replacing the visual backbone):
  deprioritized (user's own instinct, agreed). Much bigger, riskier lift —
  redesigns the visual backbone, not just fusion — and doesn't obviously
  serve the memory hypothesis specifically.

## Ideas queued (not yet implemented)

1. **Tversky + Focal loss** for the `attn_map`/beta aux supervision
   (replaces plain `BCELoss`), from ThinkDeeper's Stage-1 loss (Appendix C).
   **IMPLEMENTED (2026-07-20)**: `TverskyFocalLoss` added to
   `bdh_grounding/pipeline.py` (CPU-testable, standard Salehi et al. 2017 /
   Lin et al. 2017 formulas with tunable lambda_tve/lambda_foc — ThinkDeeper's
   exact combination weights weren't recoverable from the PDF extraction, so
   this uses the well-established reference formulas rather than guessing).
   Wired into `train.py` via `--map-loss {bce,tversky_focal}` CLI override
   (default stays `bce`, existing runs unaffected); `ckpt_tag` extended with
   a `_tve` suffix so this doesn't collide with the existing `bdh_A_best`
   checkpoint. Local sanity test added (`tests/plumbing_test.py`): finite,
   gradients flow, lower loss for closer predictions — passes. **Launching
   on Mode A now** (`--bdh-mode A --map-loss tversky_focal`). NOT YET
   RESULTS.
2. **BDH stacking** (2-3 layers): current `BDHVisualTextAttention` is a
   single lift -> memory-read -> gate -> decode pass, no layer stacking.
   Proposed: iteratively refine region features by re-querying the same
   word memory across N residual-connected layers (DETR-decoder-style).
   Directly tests "does more memory read/write depth help" — a direct
   probe of the "memory is the key factor" framing.
3. **BDH -> attention hybrid ("Mode D")**: BDH builds a memory-
   contextualized word representation (reusing Mode A's causal self-BDH
   pass over text, `Q_ctx`), then the *original* AttnGrounder softmax
   cross-attention does region-word alignment using `Q_ctx` instead of raw
   `lang_feat`. Tests "memory feeding attention" vs. "memory replacing
   attention" as a distinct manipulation of the same hypothesis.
4. **Depth/distance-aware prior**: reuse the 3D translation data already
   pulled from nuScenes for Step 6 (no new depth-estimation model needed)
   to weight/disambiguate "closer" vs. "farther" same-class objects.

## Rough AP50 gain estimates (hedged — order-of-magnitude, not precise)

| Change | Estimated effect | Confidence | Why |
|---|---|---|---|
| Tversky+Focal loss swap | +0.3 to +1.0 | Low-moderate | Only reweights the auxiliary loss term (`lambda_map=0.1` of total loss) — small direct lever on the primary box loss |
| BDH stacking (2-3 layers) | -1 to +2 (could hurt) | Low | More capacity on a small train set (8,349 images) risks overfitting without added regularization; direction genuinely uncertain until tried |
| BDH -> attention hybrid | unknown | Very low | Exploratory/framing experiment more than a guaranteed-gain change |
| Depth prior (existing nuScenes data) | +0.5 to +1.5 | Low-moderate | Capped by how often Talk2Car commands actually use distance cues ("the closer car") vs. other disambiguators |
| Pretrained transformer backbone (BERT/ViT swap) | +3 to +8 | Moderate | The single biggest lever based on the SOTA-bracket pattern (CNN-class ~61-68 vs. transformer-class ~70-76) — but this is a full project-scope change, would need applying to *both* baseline and BDH variant to keep the mechanism comparison fair, and stops being "just" a BDH study |

None of these are validated yet — treat as planning priors, update this
table with real numbers as each change is actually tried.

## Params comparison across the field (2026-07-20)

ThinkDeeper's own Table 1 has no params column — had to check each model's
own paper/repo individually; several aren't publicly stated, marked below.

| Model | Params | Confidence |
|---|---|---|
| Ours (baseline / BDH) | 75.84M / 76.63M | High (measured) |
| VL-BERT | ~118M | Moderate (secondary source) |
| RSD-LXMERT (LXMERT-based) | ~210-230M | Low-moderate (vanilla LXMERT, not the RSD variant specifically) |
| MiniGPT-v2 / LLaVA-NeXT / Qwen-VL variants | 7B-72B | High (named by size) |
| CAVG | not comparable | calls GPT-4 via API, no disclosed size |
| TransVG / CMSVG / CMRT / MDETR / VLTVG / UNINEXT / ThinkDeeper | not found | not publicly stated in an easily-verifiable place |

**Takeaway**: we're meaningfully smaller than the transformer/LXMERT-class
models (60-70% of VL-BERT, ~1/3 of RSD-LXMERT) and trivially smaller than
the VLM-based ones. Legitimate framing: "~76M params, 1-year-old memory
mechanism, competitive with models 1.5-3x the size" for the CNN-class
bracket.

## Idea evaluated: YOLOv7 / YOLO26 backbone+head swap (2026-07-20)

Considered replacing Darknet-53 + YOLOv3 head with a modern detector
(YOLO26: Sept 2025/2026 Ultralytics release, anchor-free, NMS-free,
Small-Target-Aware Label Assignment; or YOLOv7: E-ELAN backbone,
re-parameterized). YOLOv3 itself: COCO AP50=57.9. YOLO26 x-variant: 58.99M
params (smaller than Darknet-53-based YOLOv3's ~62M), COCO mAP@[.5:.95]
40.9-57.5 across scales — stronger and smaller than our current backbone.

**Scoped out for now, not rejected** — this is a legitimate lever but a
different research question than the BDH/memory hypothesis, and NOT a
drop-in swap like the BDH fusion module was:
- Different channel counts per FPN scale -> `mapping_visu` needs rewriting
- Anchor-free/NMS-free head is architecturally unlike the current
  anchor-based `build_target`/`yolo_loss` in `train.py` -> loss logic needs
  a substantial rewrite, not a swap
- Would need applying to BOTH baseline and BDH arms to keep the mechanism
  comparison fair -> effectively a second full mini-study
- AP50 gain estimate: +2 to +6, wide uncertainty — COCO mAP gains don't
  linearly transfer to single-box grounding-from-fused-features AP50, no
  validated extrapolation exists

Revisit after the current BDH-focused round (Tversky+Focal, stacking,
BDH->attention hybrid) if there's appetite for a second, parallel
engineering track.

## Open questions / decisions needed

- Which of A/B/C to build on once B finishes (currently A leads).
- Whether to pursue the backbone-upgrade lever at all, given it would
  broaden scope beyond the core BDH-mechanism question (if pursued, must
  apply to both variants for a fair comparison).
- Whether val-only reporting (with the test-set caveat above) is
  sufficient for the eventual write-up, or whether to attempt contacting
  Talk2Car maintainers directly for research-use test GT access (slow,
  uncertain, not on the critical path).
