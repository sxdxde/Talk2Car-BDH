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
server. We report val-set AP50 only (as we've been doing), explicitly
caveated against other papers' self-reported test numbers —
directionally informative, not strict apples-to-apples.

**REFINEMENT (2026-07-20, user pushback — "are you sure they all
reported test and not val, how do they get test?")**: checked further,
found a genuine two-tier split, not one uniform "don't trust any of it":
- **CMSVG's own paper explicitly states**: "Our work comes as an
  official entry to the Commands 4 Autonomous Vehicles (C4AV) competition
  2020" — published Sept 2020, inside the live evaluation window (server
  closed Aug 1 2020). Its 68.7% test score is plausibly a genuine,
  server-verified number. Same timing logic applies to AttnGrounder
  (also ECCV 2020 workshop) — its 61.32 is plausibly legitimate too.
- **Everything published 2021+** (TransVG, VL-BERT-on-Talk2Car,
  RSD-LXMERT, CAVG, ThinkDeeper, likely CMRT/VLTVG/MDETR/UNINEXT) —
  the server was already closed, so these could not have gotten a
  live-verified test score through the official mechanism. Most
  plausibly third-party reproductions (possibly run on val, since
  that's the only split with public GT, then propagated as "test"
  through citation-chain table-copying without re-verification).
- Confirmed via the raw `leaderboard.md` (fetched directly, not
  summarized): **zero provenance metadata** — no dates, no verification
  status, no val/test distinction beyond a bare number. Explicitly
  PR-maintained ("pull requests with new results always welcome"), not
  centrally verified. ThinkDeeper's own paper has no methodology
  statement on where its Table 1 baseline numbers came from either.

**Implication for how we report/compare**: (1) our own numbers stay on
val, unavoidable and correct; (2) the comparison that's actually
bulletproof is our own baseline-vs-BDH result — same split, same code,
same everything but the fusion mechanism, no provenance problem since we
control both sides; (3) weight cross-paper numbers by era — CMSVG/
AttnGrounder (2020) reasonably trustworthy, everything 2021+ cited with
explicit skepticism, not treated as ground truth; (4) if an airtight
secondary comparison is wanted, the methodologically strongest move is
reproducing one more baseline ourselves on our own val split (as already
done for AttnGrounder: got 64.92 vs. their claimed 63.30 — a real
sanity check that self-reported numbers do drift) rather than trusting
any 2021+ paper's number at face value.

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
   gradients flow, lower loss for closer predictions — passes.

   **RESULT (2026-07-20): CONFIRMED FINAL, negative.** Best AP50 = 64.03
   (training completed all 100 epochs; epoch 98=63.51, epoch 99=63.60,
   both below the peak — no late-training recovery) — *below*
   both Mode A's original BCE result (65.09) and even baseline (64.92).
   The foreground/distractor-imbalance reweighting did not help here,
   plausibly because the aux `attn_map`/beta loss is a small auxiliary
   term (`lambda_map=0.1` of total loss) with limited leverage over the
   primary box-regression objective. Not pursuing further (no
   hyperparameter sweep on tversky_alpha/beta/focal_gamma planned — low
   expected payoff for the tuning cost). Stratified eval not run (no
   reason to chase a losing aggregate result further).
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
| CMSVG (best: RoBERTa-large + EfficientNet-B2) | **~364M** (355M + 9.2M, both standard published sizes) | Moderate — paper doesn't state a total, this is a component-sum estimate |
| CMSVG (lightweight: DistilBERT-base + EfficientNet-B0) | **~71-72M** (66M + 5.3M) | Moderate, same caveat |
| TransVG / CMRT / MDETR / VLTVG / UNINEXT / ThinkDeeper | not found | not publicly stated in an easily-verifiable place |

**Takeaway**: we're meaningfully smaller than the transformer/LXMERT-class
models (60-70% of VL-BERT, ~1/3 of RSD-LXMERT) and trivially smaller than
the VLM-based ones. Legitimate framing: "~76M params, 1-year-old memory
mechanism, competitive with models 1.5-3x the size" for the CNN-class
bracket.

## CORRECTION (2026-07-20): the "beat CMSVG" bracket-1 target was wrong

Checked CMSVG's actual architecture (had only checked its AP50 before, not
its components) — **CMSVG is not a same-class comparison for our current
model at all**:
- CMSVG uses a pretrained **Sentence-BERT** text encoder (STS RoBERTa-large
  for its best result, or STS DistilBERT-base for a lighter variant) +
  EfficientNet (ImageNet-pretrained) for vision. It is NOT a "no
  pretrained language" model — it already has exactly the kind of
  pretrained-text advantage our BiLSTM+GloVe setup lacks.
- The headline 68.61-68.7 AP50 comes from the **~364M-parameter**
  RoBERTa-large + EfficientNet-B2 configuration — ~4.8x our current
  ~76M — not a comparable size at all.
- CMSVG's own lightweight variant (DistilBERT-base + EfficientNet-B0,
  ~71-72M, genuinely size-comparable to us) scores **67.8% test AP50**,
  "quite close" to the best config per the paper.
- Similarly, **TransVG (65.83) also uses a pretrained BERT text
  encoder** (confirmed via its own paper) — also not a clean "no
  pretrained language" comparison. VLTVG/CMRT's text encoders not
  independently verified — treat with the same suspicion until checked.

**Revised, more honest bracket structure**:
- **True "bracket 1"** (no pretrained language at all, size-comparable to
  us): as far as verified, just **AttnGrounder (61.32)**. Our current
  best (Mode A, 65.09) already beats this by +3.77 — we may already be
  ahead of the only genuinely apples-to-apples comparison point.
- **"Bracket 1.5"** (pretrained text encoder, size-comparable ~70-80M,
  non-VLM): CMSVG-lightweight (67.8%, ~71-72M), possibly TransVG
  (65.83, size unconfirmed) — **this is the real target for our planned
  text-encoder-swap experiment (previously mislabeled "bracket 2")**,
  not VL-BERT/RSD-LXMERT.
- **"Bracket 2"** (larger pretrained-language models, ~110-370M):
  CMSVG-best (68.7%, ~364M), VL-BERT (70.03%, ~118M), RSD-LXMERT
  (72.64%, ~210-230M).

**Implication for "is 66-67 competitive enough" (user question,
2026-07-20)**: yes, genuinely strong — in the *true*, no-pretrained-text,
size-matched bracket, that would put us +5 to +6 over the only confirmed
comparable model (AttnGrounder). The earlier framing ("beat CMSVG's
68.61 in bracket 1") set an unfairly high, size-mismatched bar; the
corrected target for the current (pre-text-swap) phase of the two-phase
plan is closer to "solidly beat AttnGrounder, approach TransVG's 65.83
despite TransVG's BERT advantage" — already essentially achieved at
65.09, would be clearly achieved at 66-67. The CMSVG-lightweight number
(67.8%, ~71-72M) becomes the target for the *second* phase (after our
own text-encoder swap), where it's a fair, size-matched comparison.

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

**Sharper gain anchor (2026-07-20)**: same-benchmark data point beats COCO
extrapolation — AttnGrounder (Darknet, 61.32) vs CMSVG (EfficientNet,
68.61) is a ~7-point AP50 gap from backbone modernization alone, no
language/fusion changes. Revises the YOLO-swap estimate to roughly +2 to +8.

## Idea queued: pretrained text encoder (2026-07-20)

Ranked against the YOLO swap and a pretrained vision backbone swap as the
best gain-per-effort lever: our text side (BiLSTM + GloVe, non-contextual,
2014-era embeddings, no large-scale pretraining) is the most dated
component in the stack, and every SOTA-bracket model above ~68 AP50 uses a
pretrained language or joint vision-language backbone (BERT in
VL-BERT/TransVG, LXMERT's joint pretraining, ViT+BERT+BLIP-init in
ThinkDeeper) — plausibly a bigger bottleneck than the vision backbone.

**Why this ranks above the YOLO/vision-backbone swaps**: contained scope —
only replaces `RNNEncoder` and the GloVe-vocab tokenizer/`corpus.pth`
pipeline with a pretrained model + its own tokenizer (e.g. HuggingFace
`AutoTokenizer`). `mapping_lang`, the BDH fusion module, vision backbone,
and detection head are all untouched — BDH just consumes whatever
`lang_feat` it's handed, dimension-agnostic via `mapping_lang`. Real but
scoped caveat: still a genuine tokenizer/data-pipeline change, not a
one-line swap.

**REVISED PLAN (2026-07-20) — offline/frozen, not live**: user's own prior
research (brain MRI + radiology report segmentation) used a RadBERT text
encoder run OFFLINE, once, with results cached — the segmentation model
itself never runs the text encoder live. Directly applicable here and
strictly better than a live swap:

1. Talk2Car has a small, fully enumerable command set (~12k across
   train/val/test — test commands' text is available even without box
   labels) -> embeddings can be precomputed once for every command.
2. Because it's offline, model size is a non-issue — could use
   DistilBERT/MiniLM or even something much larger (checked: NX-AI
   released a real pretrained **xLSTM-7B** checkpoint, 2.3T tokens,
   HuggingFace `NX-AI/xLSTM-7b` — usable here specifically because it
   never touches the live model).
3. Optional domain-adaptive step mirroring the RadBERT recipe: continue
   masked-LM pretraining on Talk2Car's own ~12k commands before extracting
   features (cheap, one-time). Checked: no existing "driving-command
   BERT" equivalent to RadBERT exists publicly (radiology has huge public
   report corpora like MIMIC-CXR enabling RadBERT; Talk2Car's ~12k
   commands are nowhere near enough to pretrain from scratch, and nobody
   has published a driving-domain equivalent) — would need to replicate
   the recipe ourselves rather than reuse an existing checkpoint.
4. Cache per-token hidden states (not pooled — fusion needs the full
   `(T, hidden_dim)` sequence, same contract as today's `lang_feat`) to
   disk, keyed by command. Storage trivial (~12k x ~30 tokens x 768-dim,
   well under 1GB even fp32).
5. `RNNEncoder` becomes a lookup, not a forward pass. `mapping_lang` and
   everything downstream (BDH, fusion, YOLO head) unchanged.

**Why this beats the live-swap plan**: zero runtime parameter/compute
cost (the "~76M, competitive with 1.5-3x larger models" framing stays
fully intact, arguably strengthens), no tokenizer integration into the
training loop, no architecture cascade. Real tradeoff: frozen means no
joint fine-tuning of the text encoder against the grounding loss —
domain-adaptive continued pretraining (step 3) is what compensates for
that, same role it played for RadBERT.

NOT YET IMPLEMENTED — queued behind the Tversky+Focal result (avoid
changing two things before reading one result).

**Bracket clarification (2026-07-20, user question)**: does adding a
pretrained text encoder push us into the VLM bracket, breaking the
"beat CMSVG" comparison? No — applying the same VLM criteria used
elsewhere in this doc (built around a large pretrained LM as the core
reasoning engine + web-scale paired image-text pretraining + generalist/
prompt-driven capability), none apply here: the encoder stays a frozen
input feature extractor, text-only pretraining (no paired image-text
data), still single-task/single-dataset. Directly analogous to VL-BERT
and RSD-LXMERT, neither labeled "VLM" in ThinkDeeper's own table. Their
table implies a 3-bracket structure: CNN-only (AttnGrounder/CMSVG/us
currently, 61-69), **CNN + pretrained language** (VL-BERT 70.03,
RSD-LXMERT 72.64), VLM/LLM-based (42-77, wide). Adding the text encoder
would move us from bracket 1 to bracket 2, not bracket 3 — a real but
more attainable target shift (VL-BERT/RSD-LXMERT territory, not
ThinkDeeper's). **Reporting plan if implemented**: keep as a separate,
clearly-labeled second result (pure-mechanism study vs. CMSVG stays
clean; text-encoder result compared honestly against the bracket-2
range), not folded into one comparison. Note the offline/frozen design
means we'd pay the encoder's cost once, not per-inference like VL-BERT/
RSD-LXMERT do — cheaper at inference even within the same bracket, worth
highlighting if pursued.

## Idea queued: Distractor-Contrastive Loss (2026-07-20)

Motivated by the user's explicit ask for a genuinely *original* loss —
not another borrowed-from-literature trick (Tversky+Focal was tried and
was a clean negative, see above) — built specifically on infrastructure
nobody else has for Talk2Car: the nuScenes-joined distractor data from
Step 6. Consensus literature search (2026-07-20) found no existing work
using ground-truth same-class distractor locations as training-time hard
negatives for this dataset — the closest adjacent idea found was
[Relationship-Embedded Representation Learning for Grounding Referring
Expressions](https://consensus.app/papers/details/993bd93cf36d582c90c3f634033c34b2)
(Yang et al., IEEE TPAMI 2019), which builds a language-guided relation
graph between candidates — conceptually adjacent but a much heavier
hypergraph-style rebuild, not what's proposed here.

**Idea**: during training, on ambiguous images, add a margin/hinge loss
term that explicitly pushes the model's confidence (`attn_map`/beta) at
the true target's location higher than at known same-class distractors'
locations:
```
L_distractor = mean[ max(0, margin - (beta_at_target - beta_at_distractor)) ]
```
Hard-negative-mining is well-established in metric learning generally,
but using nuScenes-derived, ground-truth-verified same-class distractor
locations as the hard negatives, specifically for Talk2Car grounding, is
the novel part — directly targets the disambiguation hypothesis rather
than a generic imbalance fix.

**Implementation scope (Steps A-E), destination: `arch2/` (created, see
`arch2/README.md`), NOT YET BUILT**:
- **Step A**: extend `analysis/build_scene_index.py`'s approach (as a new
  `arch2/build_distractor_index.py`, not modifying the existing
  count-only script) to (1) keep the `camera_intrinsic` currently
  discarded in `nusc.get_sample_data(...)` and use
  `nuscenes.utils.geometry_utils.view_points` to project each
  distractor's 3D box corners to 2D pixel coordinates (native
  1600x900), and (2) run on **train** as well as val (currently
  val-only, 8,349 vs 1,163 commands). New correctness-sensitive geometry
  code — a silently-wrong projection would corrupt the loss without
  erroring, needs careful testing before trusting it.
- **Step B**: feed distractor boxes into training via a plain lookup
  dict keyed by image filename in `train.py` (same pattern
  `stratified_eval.py` already uses for the scene index) — deliberately
  NOT modifying `Talk2CarDataset`/the AttnGrounder loader itself, to
  keep this low-risk and isolated.
- **Step C**: map distractor box centers to grid cells at each of the 3
  FPN scales (same style as AttnGrounder's own `build_target` for GT
  boxes), compute the hinge loss — contributes 0 naturally on
  unambiguous images.
- **Step D**: wire in via new `lambda_distractor` config knob (~0.1,
  matching `lambda_map`'s scale) and `--distractor-index` CLI arg, added
  to `train_epoch`'s total loss.
- **Step E**: CPU-testable unit test for the hinge-loss math (dummy
  tensors, same convention as `test_tversky_focal_loss`), before any GPU
  run.

**Honest effort/risk assessment**: bigger lift than anything built so
far — new geometry code, a new data-prep pass over the full train split,
a new loss term threaded through the training loop. Realistically a few
hours of careful work, not a quick patch. Explicitly deferred — user
wants this scoped and logged now, built later, not immediately.

## Mode A seed confirmation — RESULT (2026-07-21): concerning, not conclusive

Seed=1 (same config as the winning Mode A run, `--seed 1`): **best AP50
= 64.20** (epoch 98=62.91, epoch 99=62.82, both below peak — final,
matches the no-late-improvement pattern seen in every prior run).

```
                AP50 (all)
baseline           64.92
mode A, seed=0     65.09
mode A, seed=1     64.20
```

Seed=1 lands *below* baseline, not above — the two seeds bracket
baseline rather than both beating it. On the aggregate number alone,
this is a real yellow flag: can't currently rule out that Mode A's true
average effect on aggregate AP50 is near zero, and seed=0 was on the
fortunate side of normal run-to-run variance (~0.9 point spread between
seeds, comparable in magnitude to the ambiguous-scene effect itself).

**Not conclusive on its own** — the aggregate isn't the real test. Next
step: run `stratified_eval.py` on `bdh_A_seed1_best.pth.tar` to check
whether the *qualitative* pattern replicates (better than baseline on
ambiguous specifically, even if the aggregate is less flattering) or
fails there too (which would suggest seed=0's ambiguous-scene win was
itself partly noise). Result pending.

## Mode A seed confirmation — STRATIFIED RESULT (2026-07-21): does NOT replicate

```
                     AP50 (all)   AP50 (ambiguous, n=830)   AP50 (unambiguous, n=333)
baseline               64.92             63.37                      68.77
mode C                 64.06             62.17                      68.77
mode A, seed=0         65.09             63.98                      67.87
mode A, seed=1         64.14             62.89                      67.27
mode B                 60.53             58.19                      66.37
```

**Seed=1 is worse than baseline on ambiguous scenes too** (62.89 vs
63.37, delta -0.48) — not just the aggregate. Its pattern (uniform
deficit across both buckets) looks like modes B/C's failure mode, not
like seed=0's specific disambiguation trade-off. The two Mode A seeds
disagree on the ambiguous delta by ~1.1 points (+0.61 vs -0.48) —
comparable to or larger than the effect itself. Averaged across both
seeds, the ambiguous-scene delta is ≈ +0.065 — indistinguishable from
zero.

**Honest conclusion as of now: no BDH kernel formulation tested so far
(A across 2 seeds, B, C) has robustly replicated an ambiguous-scene
advantage over baseline.** Seed=0's result, taken alone, overstated the
case — exactly the failure mode seed confirmation exists to catch.

**Gap exposed**: baseline has only been run once (seed=0 equivalent).
We don't actually know baseline's own seed-to-seed variance, so every
delta computed so far assumes a stable baseline reference that hasn't
itself been verified. Not previously flagged as a design gap until now.

**Real options going forward, not yet decided**:
1. Run a 3rd Mode A seed to break the tie — 2-of-3 same-direction would
   be informative (either "real but noisy, mostly positive" or "no
   effect, seed=0 was the outlier"); 3rd seed disagreeing with both
   would mean high variance, more seeds needed regardless.
2. Run a 2nd baseline seed to establish its own variance — currently a
   real gap, not just an omission on the BDH side.
3. Reframe the paper's honest current finding as: a rigorously seed-
   tested negative result across all three tested BDH kernel
   formulations — still a real, publishable contribution (a well-tested
   negative result was always on the table as a legitimate outcome, see
   "Why Keep Pursuing This" framing), just not the "Mode A wins" story
   the two-phase plan below was built around.
4. Treat this as motivation to prioritize the higher-potential
   mechanism changes (BDH->attention hybrid, distractor-contrastive
   loss) over declaring Mode A "the winner" prematurely — the ablation
   study's real conclusion right now is "none of A/B/C robustly show
   the hypothesized effect," which argues for trying a structurally
   different mechanism (the hybrid) rather than more seeds of a
   formulation with no confirmed effect to begin with.

**DECIDED (2026-07-21): options 1+2 (tie-breaking Mode A seed=2, and
baseline seed=1)**, run sequentially before touching the hybrid or
distractor-contrastive loss. No new code needed — same `--seed` CLI
flag already built and confirmed working from the seed=1 run.

**CORRECTION (2026-07-21)**: a run of epoch logs (best AP50=64.89) was
initially misattributed to "Mode A seed=2" without confirming which job
it actually was — no `[model]` line was ever seen for it, and the
assumption turned out to be wrong. Checked `checkpoints/full/` directly:
`bdh_A_seed2_best.pth.tar` / `_last.pth.tar` do not exist anywhere (Mode
A seed=2 was never actually launched), while `baseline_seed1_best.pth.tar`
/ `_last.pth.tar` both exist with real timestamps (Jul 21 09:11/09:50) —
**the 64.89 result belongs to baseline seed=1, not Mode A seed=2.**
Retracting the earlier "Mode A seed=2 CONFIRMED FINAL" entry and the
"concurrent-GPU-jobs corrupted the checkpoint" theory built on top of
it — there was no corruption, just a misattributed log. Lesson: always
confirm the `[model]` line before logging a result, don't infer which
job a pasted log belongs to from conversation context alone.

**Corrected current state**: baseline seed=1 ran successfully, checkpoint
exists, best AP50 (train-logged) = 64.89 — stratified eval not yet run
on it. **Mode A seed=2 has not been run yet** — still needed to complete
the 3-seed picture:
```
                     AP50 (all)
baseline (seed=0)       64.92
baseline (seed=1)       64.89 (train-logged; stratified pending)
mode A, seed=0          65.09
mode A, seed=1          64.14
mode A, seed=2          NOT YET RUN
```
Interesting early signal on baseline's own variance: two baseline seeds
(64.92, 64.89) are much tighter than Mode A's two seeds (65.09, 64.14) —
suggestive that baseline is more stable and Mode A's spread is real
variance from the architecture/mechanism, not just generic training
noise, though n=2 per arm is still thin evidence.

**Next steps**: (1) run `stratified_eval.py` on
`baseline_seed1_best.pth.tar` to get baseline's second stratified data
point; (2) actually launch Mode A seed=2 (for real this time) to
complete the 3-seed comparison — run it alone, not concurrently with
anything else on the GPU.

## Baseline seed=1 stratified — RESULT (2026-07-21): baseline's own split is noisy too

```
                      AP50 (all)   AP50 (ambiguous, n=830)   AP50 (unambiguous, n=333)
baseline, seed=0        64.92             63.37                      68.77
baseline, seed=1        64.92             64.70                      65.47
mode A, seed=0           65.09             63.98                      67.87
mode A, seed=1           64.14             62.89                      67.27
mode C                   64.06             62.17                      68.77
mode B                   60.53             58.19                      66.37
```

Baseline's aggregate is coincidentally identical across seeds (64.92
both times), but its **stratified split is not stable**: ambiguous
ranges 63.37-64.70 (spread 1.33), unambiguous swings 65.47-68.77 (spread
3.30 — plausibly partly explained by smaller sample size there, n=333 vs
n=830, but still substantial).

**This further undercuts the original Mode A seed=0 result**: its
ambiguous score (63.98) now falls *inside* baseline's own observed
range (63.37-64.70) — no longer clearly a win once baseline's own
variance is accounted for. Mode A seed=1's ambiguous score (62.89) is
the only BDH result that falls *below* baseline's full observed range
on either seed — a more consistent (if unfavorable) signal than seed=0
provided.

**Revised honest conclusion**: the case for Mode A's ambiguous-scene
advantage is weaker than it looked even after the seed=1 disappointment
alone — part of what looked like a BDH effect could just as easily be
baseline noise. This is exactly why the baseline reseed mattered as much
as the BDH reseed did; a single-baseline-seed comparison was never
well-powered enough to support the original claim. Mode A seed=2 and
ideally a 3rd baseline seed are needed before drawing a final conclusion
on this hypothesis.

## DECIDED (2026-07-20): two-phase paper structure

Goal explicitly reframed by user: not beating SOTA, but demonstrating
BDH's potential as a novel, viable mechanism for grounding — "a brand-new
architecture shows promise" is the thesis, not a leaderboard claim.

**Phase 1**: maximize competitiveness with the current architecture
(BiLSTM+GloVe, no pretrained text) against the true, size-matched,
no-pretrained-language bracket (realistically just AttnGrounder, 61.32 —
see the CMSVG/TransVG correction below). This isolates the BDH mechanism
contribution cleanly, no confounds.

**Phase 2**: swap in the offline/frozen pretrained text encoder (already
queued), and re-run baseline-vs-BDH as a **paired comparison again** at
that tier — not just report an isolated post-swap number. Condition for
this to stay scientifically coherent (not scope creep / not "picking up
scraps"): phase 2 must ask the same question phase 1 did — does the
ambiguous-scene advantage persist once text representation quality
improves — rather than becoming an unrelated bolt-on score-chasing
experiment. Target bracket for phase 2: CMSVG-lightweight (67.8%,
~71-72M) and similar pretrained-text/moderate-size models — see
correction below, not VL-BERT/RSD-LXMERT (those are a further tier up).

## Open questions / decisions needed

- Which of A/B/C to build on once B finishes (currently A leads).
- Whether to pursue the backbone-upgrade lever at all, given it would
  broaden scope beyond the core BDH-mechanism question (if pursued, must
  apply to both variants for a fair comparison).
- Whether val-only reporting (with the test-set caveat above) is
  sufficient for the eventual write-up, or whether to attempt contacting
  Talk2Car maintainers directly for research-use test GT access (slow,
  uncertain, not on the critical path).
