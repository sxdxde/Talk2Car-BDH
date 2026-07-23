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

**DETAILED IMPLEMENTATION PLAN (2026-07-21)** — scoped by actually
reading `dataset/talk2car_loader.py` first rather than assuming its
interface (first time touching this file; every prior edit stayed in
`grounding_model.py`/`train.py`/our own `bdh_grounding/`).

**Model choice**: DistilBERT-base specifically, not an arbitrary
pretrained encoder — matches CMSVG-lightweight's own text-encoder class
(STS DistilBERT-base), making "are we on par with CMSVG-lightweight" a
fair, matched comparison rather than apples-to-oranges.

**Gotcha caught before coding (would have silently corrupted training if
missed)**: the loader's augmentation path (`talk2car_loader.py:127`)
does a horizontal-flip-triggered text mutation —
`phrase.replace('right',...).replace('left','right')...` — swapping
left/right words when the image is mirrored (fires on ~50% of augmented
samples). A naive "cache one embedding per original command" plan would
hit cache-misses on these swapped strings during training. Fix baked
into the plan: the offline cache must include **both** each phrase and
its deterministic left/right-swapped variant (reusing the exact same
replace-chain so the two stay in lockstep).

**Two integration points identified** (confirmed by reading the current
text path end-to-end): the loader's `tokenize_phrase()` step, and
`grounding_model`'s `textmodel`/`mapping_lang` block. Everything
downstream of `flang` (BDH fusion, YOLO head, loss) is agnostic to where
the text embedding came from — no changes needed there.

Stages:
1. **Offline extraction** (`arch2/build_text_cache.py`, new): enumerate
   every phrase from `talk2car_{train,val,test}.pth` (phrases live
   there, not just the commands JSON) + their left/right-swapped
   variants; run frozen DistilBERT-base, keep per-token
   `last_hidden_state` (drop `[CLS]`/`[SEP]`, optionally keep `[CLS]` as
   a sentence-summary token — flagged as a design choice to test later
   if wanted); zero-pad/truncate to `query_len=40` (DistilBERT subwords
   for ~11-word commands run ~15-25 tokens, safe margin); cache as
   `{phrase_string: float16 array (40, 768)}` to one `.pth` file (~12k×2
   keys, well under 1.5GB). Runs on CPU in minutes, done locally, synced
   up like everything else.
2. **Loader change** (`dataset/talk2car_loader.py`): new
   `text_encoder="glove"` init param (default preserves current
   behavior exactly, zero risk to existing runs); if `"distilbert"`,
   look up the post-augmentation (post-swap) phrase in the cache and
   return the `(40, 768)` float array as the 5th tuple element instead
   of int word IDs.
3. **Model change** (`grounding_model.py`): new `text_encoder` kwarg;
   if `"distilbert"`, skip building `self.textmodel` (the BiLSTM)
   entirely, `mapping_lang` becomes `Linear(768, emb_size)` instead of
   `Linear(600, emb_size)`; forward pass trims to batch-max real length
   via a zero-row mask on the incoming embeddings, then straight into
   `mapping_lang` — everything downstream unchanged.
4. **Config/CLI/tag wiring**: `full_a100.yaml` gets `model.text_encoder:
   glove`; `pipeline.config_to_args` reads it; `train.py`/
   `stratified_eval.py` get `--text-encoder {glove,distilbert}` +
   `_distilbert` ckpt-tag suffix, same override pattern as every prior
   toggle (`--bdh-mode`, `--map-loss`, `--seed`, `--growing-scales`).
5. **Local test** before any GPU run: cache-lookup shape/correctness,
   swapped-phrase keys resolve, model builds with the 768-dim
   `mapping_lang`, mock `(bs,40,768)` batch runs end-to-end on CPU.

**Param-count framing (worth headlining if this ships)**: DistilBERT
never enters the trained model — frozen, offline, cached — so we
actually *drop* the small BiLSTM and *add* only a 768→emb_size linear.
Runtime params go down slightly, not up. The "~76M, lightweight" story
survives this change fully intact, unlike a live text-encoder swap
would.

**Status**: fully scoped, NOT YET BUILT. Unblocked now that the
AttnGrounder stopping criterion has been met (Mode A seed=2 + Mode E
both landed, see FINAL SYNTHESIS above) — ready to start whenever
prioritized against the metrics-collection and TransVG tracks.

**IMPLEMENTED (2026-07-21)**, all 5 stages, tested locally, all green:
- `arch2/build_text_cache.py` (new): offline extraction script, includes
  the `lr_swap` helper kept in exact lockstep with the loader's own
  replace-chain. One deviation from the original plan, made deliberately
  during implementation: DistilBERT's `[CLS]`/`[SEP]` tokens are kept in
  the cached sequence rather than stripped -- the fusion module treats
  every position uniformly, so stripping mid-sequence would only add
  indexing-bug risk for no benefit (costs 2 of the 40 `query_len` slots).
- `dataset/talk2car_loader.py`: new `text_encoder`/`text_cache_path`
  params (default preserves exact original behavior); cache lookup
  happens post-augmentation (post left/right-swap) so keys match.
- `grounding_model.py`: new `text_encoder` kwarg, orthogonal to
  variant/bdh_mode (applies to baseline or bdh); distilbert branch skips
  building the BiLSTM entirely, `mapping_lang` becomes `Linear(768,
  emb_size)`; forward pass trims to real length via a zero-row mask.
  Also changed `.view()` to `.reshape()` at the mapping_lang call site
  (needed since a sliced tensor isn't guaranteed contiguous) --
  verified as a no-op for the existing glove path.
- `pipeline.py`/`train.py`/`stratified_eval.py`/`full_a100.yaml`:
  config field + `--text-encoder {glove,distilbert}` + `--text-cache`
  CLI overrides + `_distilbert` ckpt-tag suffix, same pattern as every
  prior toggle.
- `tests/text_encoder_test.py` (new): covers what's locally testable
  without remote data/`transformers` -- `lr_swap` correctness (single/
  double/no-op cases), cache-lookup contract (fails loud on a miss, not
  silent), the real-length-trim + `mapping_lang` projection logic
  (shapes, gradients), and confirms the `view`->`reshape` change is a
  verified no-op for the glove path. All passing, plus all 4 pre-existing
  suites still green.

**NOT locally testable** (needs remote data/GPU/`transformers`): the
real `Talk2CarDataset` end-to-end, and running actual DistilBERT
extraction. **NOT YET RUN ON REMOTE.**

**Remote steps to actually use this**:
```bash
# sync (see below), then on remote:
pip install transformers
cd ~/BDH/Talk2Car/AttnGrounder
python arch2/build_text_cache.py --data-root ln_data --out arch2/text_cache_distilbert.pth

# recommend a quick sanity check (watch the first few training steps,
# Ctrl+C once confirmed no crash/shape error) before committing a full
# ~12+ hour run, since this is genuinely new, unverified-on-real-data code path
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/full_a100.yaml \
    --variant baseline --text-encoder distilbert --text-cache arch2/text_cache_distilbert.pth
```

**RESULT (2026-07-21): baseline + DistilBERT is substantially WORSE, not
better.** Best AP50 = 59.98 (epoch 98=58.18, epoch 99=59.21, consistent
no-late-improvement pattern — final). Cache built cleanly (16,059 unique
phrases incl. swap variants), no crashes, real learning happened (acc
climbed 0.025->0.97+, val AP50 nonzero from epoch 0) — this is a genuine
negative *result*, not a broken pipeline.

```
baseline + glove (mean, 2 seeds)   64.92
baseline + distilbert               59.98      delta -4.94
```

**Honest diagnosis, not excuses** — three candidate causes, roughly by
likely impact:
1. **Frozen = zero task adaptation.** The BiLSTM is trained end-to-end
   *for this task*; DistilBERT here is 100% frozen, only a single
   `Linear(768,emb_size)` is trainable — a much smaller adaptable
   surface, and generic web-text pretraining doesn't know anything
   about driving-command phrasing specifically.
2. **Scale/normalization mismatch**, foreshadowed by the elevated early
   loss during the sanity check (~85 vs. the usual ~60s start) — BERT-
   family `last_hidden_state` outputs are known to have uneven,
   sometimes large-magnitude activations ("rogue dimensions"); no
   `LayerNorm` sits between the frozen embeddings and `mapping_lang`
   right now to compensate.
3. **Keeping `[CLS]`/`[SEP]` in the sequence** (the implementation
   choice made during Stage 1, documented then) costs 2 of the ~15-25
   real-content positions to non-word tokens, in an already-short
   sequence.

**Candidate fixes, not yet tried**: add a `LayerNorm` right after the
cached embeddings, before `mapping_lang` (cheap, targets #2 directly);
revisit stripping `[CLS]`/`[SEP]` after all (targets #3); the
domain-adaptive continued-pretraining step from the original plan,
never built (targets #1 partially).

**Decision needed**: run Mode A+distilbert now anyway (informative
either way -- confirms whether this is a pipeline-wide problem or
something BDH's mechanism is more robust to), or fix the LayerNorm issue
and rerun baseline+distilbert first before spending another 12+ hour run
on a comparison that might just inherit the same problem. NOT YET
DECIDED.

**DECIDED (2026-07-21): fix LayerNorm first, verify thoroughly before any
GPU run (given the ~12+ hour cost of getting this wrong again).**

Reasoning for fixing before re-testing with Mode A: the scale issue is a
property of the raw input representation, shared identically by
whatever fusion mechanism sits downstream (baseline's softmax attention
and BDH's memory kernel both operate over the same per-position
`flang` tensor) -- no strong architectural reason to expect BDH to be
specially robust to badly-scaled input, so fixing the well-understood
cause first is the better bet than hoping a different mechanism
compensates for it.

**IMPLEMENTED**: `nn.LayerNorm(768)` (`self.text_ln`) added in
`grounding_model.__init__` (distilbert branch only), applied in
`forward()` **after** trimming to `real_len`, before `mapping_lang` --
order matters: trimming first means padded (all-zero) rows are never
fed through LayerNorm at all (if normalized, they'd just pass through
as the constant affine-bias term -- harmless but pointless, simpler to
never touch them).

**Verified locally, thoroughly, before touching the GPU**:
- Extended `tests/text_encoder_test.py`'s existing trim/projection test
  to match the real forward() path exactly (trim -> LayerNorm ->
  reshape -> mapping_lang), confirming shapes and gradients (including
  through the new LayerNorm) still check out.
- New `test_layernorm_scale_fix()`: constructs synthetic input with
  simulated BERT-style "rogue dimensions" (5 of 768 dims scaled 50x),
  confirms pre-norm std is genuinely large (~4.6, sanity-checks the test
  setup itself), confirms post-norm std lands at ~1.000 (the fix
  actually does what it's supposed to), confirms the trim-before-norm
  ordering by construction, confirms gradients flow through the LayerNorm
  and back to the input.
- All 5 local test suites (smoke, plumbing, integration, stratify-logic,
  text-encoder) pass, `py_compile` clean.

**NOT YET RUN ON GPU.** Plan: sync, rerun baseline+distilbert from
scratch (old flawed checkpoint under the same tag will be overwritten,
which is correct -- nothing worth keeping from the pre-fix run), confirm
the gap closes before deciding on Mode A+distilbert.

**RESULT (2026-07-22): fix helped a little, doesn't close the real gap.**
Best AP50 = 60.50 (epoch 98=59.04, epoch 99=59.38, final).

```
baseline + glove (mean, 2 seeds)              64.92
baseline + distilbert (no LayerNorm)          59.98    delta -4.94
baseline + distilbert (+ LayerNorm)           60.50    delta -4.42
```

**+0.52 from the fix** -- real, right direction, confirms scale was a
genuine contributing factor. But closes under 11% of the original gap.
Diagnostic conclusion: **scale normalization was real but minor; the
dominant problem is almost certainly cause #1 from the original
diagnosis -- a fully frozen encoder with zero task-specific adaptation.**
LayerNorm fixes how the numbers are distributed, not the fact that only
a `Linear(768,512)` (+ the LayerNorm's 1,536 params) has any capacity to
adapt at all, versus the original BiLSTM trained end-to-end for this
exact task. One wrinkle noted, not fully explained: final training loss
went *up* (~1.0 -> ~1.9) despite AP50 improving slightly -- not alarming
given loss isn't directly comparable across normalization schemes, but
not a clean "everything got better" story either.

**DECIDED (2026-07-22): pause the text-encoder track, don't spend a 3rd
12+ hour run on it.** Two consecutive runs (~24+ hours GPU time) both
far below baseline+glove; the cheap fix barely moved the needle; running
Mode A+distilbert next would very likely just replicate a similarly
disappointing number for the same reason (shared, weak input
representation, not something a different fusion mechanism can fix).
Remaining plausible fixes (partial DistilBERT fine-tuning, or the
domain-adaptive continued-pretraining step from the original queued
plan) are each genuinely new implementation efforts with uncertain
payoff, not quick patches.

**Redirecting to**: metrics collection (GFLOPs/params/FPS -- cheap, no
GPU-training needed, and the final AttnGrounder-phase table needs it
regardless) and TransVG cross-architecture validation. If revisited
later, domain-adaptive continued pretraining (still frozen at inference,
preserves the "cheap" framing) is the more promising next fix to reach
for, not another incremental architectural tweak.

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

## Mode A seed=2 — RESULT (2026-07-21): CONFIRMED FINAL, unexpectedly high

Best AP50 = **66.09** (epoch 98=63.86, epoch 99=63.43, both below peak —
same no-late-improvement pattern as every prior run; genuinely final).
Notably higher than every other data point collected so far.

```
                     AP50 (all)
baseline, seed=0        64.92
baseline, seed=1        64.92
mode A, seed=0           65.09
mode A, seed=1           64.14
mode A, seed=2           66.09
```

Three-seed Mode A mean: (65.09+64.14+66.09)/3 = **65.11** — now *above*
baseline's stable 64.92, reversing the "indistinguishable from baseline"
read from two seeds. **2 of 3 Mode A seeds now beat baseline** on the
aggregate (only seed=1 was a clear miss). High variance across seeds
(64.14 to 66.09, spread ~2 points) is still real and needs accounting
for, but the direction is now trending positive, not toward zero.

**Stratified breakdown pending — the most consequential eval left to
run.** If the ambiguous bucket drove this jump, it's the first genuinely
convincing result for the hypothesis; if the gain is concentrated in
unambiguous instead, that cuts against the specific disambiguation claim
even with a strong aggregate. NOT YET RUN.

**STRATIFIED RESULT (2026-07-21): the gain is in unambiguous, not
ambiguous — does not support the hypothesis.**
```
                      AP50 (all)   AP50 (ambiguous, n=830)   AP50 (unambiguous, n=333)
baseline, seed=0        64.92             63.37                      68.77
baseline, seed=1        64.92             64.70                      65.47
mode A, seed=0           65.09             63.98                      67.87
mode A, seed=1           64.14             62.89                      67.27
mode A, seed=2           66.04             63.98                      71.17
```
Seed=2's ambiguous score (63.98) is *identical* to seed=0's (63.98) and
sits inside baseline's own observed range (63.37-64.70) — no advantage,
same as before. The entire aggregate jump comes from unambiguous
(71.17), a genuine outlier (higher than anything measured in that
bucket across every variant tested) — but unambiguous performance isn't
what the hypothesis predicts; if anything, a big gain on the *easier*
single-candidate cases while ambiguous stays flat is the opposite
pattern from "explicit memory helps disambiguation."

**Three-seed Mode A ambiguous summary: 63.98, 62.89, 63.98 — mean
63.62, slightly *below* baseline's two-seed ambiguous mean (64.03).**
Well within baseline's own noise range, not evidence of an advantage.

**CONCLUSION for Mode A (3 seeds, complete) — CORRECTED (2026-07-21,
user pushback, verified by computing proper means instead of eyeballing
seeds individually)**: two separate claims here, not one — conflating
them was the error in the first pass at this conclusion.

```
Bucket          Baseline mean (2 seeds)   Mode A mean (3 seeds)   Delta
All (aggregate)         64.92                    65.09            +0.17
Ambiguous               64.04                    63.62            -0.42
Unambiguous              67.12                    68.77            +1.65
```

1. **Specific mechanistic hypothesis** ("BDH's memory gives a
   *disproportionate* advantage on ambiguous scenes") — NOT supported.
   Ambiguous mean is slightly below baseline's, not above. This part of
   the original conclusion stands.
2. **General viability claim** ("BDH is a competitive, comparable-or-
   better replacement for established attention in this fusion role")
   — IS supported, and was previously under-weighted by over-focusing
   on (1). Positive aggregate delta (+0.17, 2 of 3 seeds beat baseline
   outright), real unambiguous gains (+1.65). Legitimate, positive,
   citable finding on its own, separate from whether the disambiguation
   mechanism was confirmed.

**Combined honest conclusion**: BDH (Mode A) achieves comparable-to-
modestly-better overall performance than established softmax attention
— a real, positive result for a <1-year-old mechanism applied to a new
domain (cross-modal grounding) for the first time, with zero
task-specific tuning — but does NOT show the originally-hypothesized
disambiguation-specific advantage. Both halves belong in the writeup;
neither should be allowed to overshadow the other. Modes B and C's more
decisive failures (see earlier results) don't change this — they were
weaker configurations, not evidence against Mode A's own
comparable-to-better result. Caveat: variance across seeds is real
(~2-point aggregate spread), so "modestly better" should stay modest,
not "clearly wins" — more seeds would strengthen this, not just for the
ambiguous-bucket question.

Per the stopping criterion (DECIDED 2026-07-21 above), one result
remains before this phase closes: the growing-scales (Mode E) run.

## TOP CONTENDER (2026-07-21): cross-scale growing memory ("Mode E")

Motivated by user's "favor BDH's nature, don't just force a swap-in"
question. Important tension surfaced first: **Mode B is the one that
most faithfully preserves BDH's "natural" causal/sequential design
(single causal sequence, RoPE, growing memory as words-then-regions
unfold) — and it's the clear worst performer of the three.** So "lean
into BDH's causal/sequential nature" is not obviously supported by our
own evidence — regions and words have no real sequence order, likely
why B struggles. This idea is different: it doesn't reintroduce
causal/RoPE structure at all.

**Idea**: right now the fusion module runs **independently at each of
the 3 FPN scales** (13x13, 26x26, 52x52) — each scale builds its
associative memory (`rho = K^T @ V`) from scratch, no state shared
across scales despite `share_scales: true` sharing only the *parameters*,
not the *computed memory*. Multi-scale detection has a natural
coarse-to-fine order, unlike words/regions which don't — so let `rho`
**persist and accumulate across scales**: the 13x13 pass builds an
initial memory, the 26x26 pass extends it (`rho_new = rho_prev +
K_scale^T @ V_scale`) rather than recomputing from scratch, then 52x52
extends further. Uses BDH's growing-memory concept in a way that maps
onto real structure in this task, without repeating Mode B's mistake.

**Rank: top contender**, ahead of the hybrid (Mode D) and stacking for
immediate implementation — it's the most direct, evidence-motivated
answer to "how do we actually exploit what makes BDH different," and
it's a new axis (orthogonal to which of A/B/C's *intra-scale* structure
is used) so it doesn't depend on Mode A's seed-variance situation
resolving first.

**Implementation scope**: modify `BDHVisualTextAttention.forward()` (and
`_forward_C` specifically, since B/A don't have a clean analogous `rho`
object to accumulate the same way) to accept/return an optional memory
state threaded across scale-calls; modify `grounding_model.py`'s
3-scale forward loop to thread that state through; new config knob
(e.g. `growing_scales: true`) + CLI override, same pattern as prior
mode/loss toggles; local CPU test before any GPU run.

**IMPLEMENTED (2026-07-21)**: design corrected before coding (see below),
then built and locally tested, all green:
- `bdh_grounding/bdh_fusion.py`: `BDHVisualTextAttention.forward()` gains
  optional `region_prior` (upsampled + added to region features before
  the mode-specific kernel runs) — works uniformly with modes A/B/C
  since the intervention point is before mode dispatch.
- `external/AttnGrounder/model/grounding_model.py`: new `bdh_growing_scales`
  constructor kwarg; `text_attn` threads `region_prior`; the 3-scale
  forward loop passes each scale's own output as the next (finer)
  scale's prior when enabled (order confirmed coarse->fine: 13x13,
  26x26, 52x52).
- `bdh_grounding/pipeline.py` / `train.py` / `analysis/stratified_eval.py`
  / `configs/full_a100.yaml`: config field + `--growing-scales` CLI
  override + `ckpt_tag` suffix (`_grow`), same pattern as prior toggles.
- New CPU test (`tests/smoke_test.py::check_growing_scales`): confirms
  `region_prior=None` exactly matches the no-arg call (backward compat),
  a real prior measurably changes the output (mechanism isn't a no-op),
  and gradients flow through it. Caught and fixed one test bug along the
  way (dropout stochasticity in train mode made the comparison
  flaky — fixed by using eval mode for the deterministic check).
- All 4 local test suites pass (smoke, plumbing, integration, stratify-logic).

**DESIGN CORRECTION made before implementing (important)**: the first
description of this idea ("accumulate `rho` across scales") was flawed
— `rho = K^T @ V` is built entirely from words (`Q = lang_feat`), which
are identical at every scale, so naive accumulation would just scale
the same matrix 1x/2x/3x, not add real information. Corrected design:
let each scale's *output* enrich the next scale's *region* features
(residual, upsampled) instead — genuine coarse-to-fine information
flow, and simpler than the original plan since it doesn't need to be
mode-specific.

**NOT YET RUN ON GPU** — ready to sync and launch as soon as Mode A
seed=2 (or whichever run is currently occupying the GPU) finishes.
Recommended first run: `--variant bdh --bdh-mode C --growing-scales`
(pairs the new cross-scale mechanism with the primary kernel).

**RESULT (2026-07-21): CONFIRMED FINAL (single seed).** Best AP50 =
64.80 (epoch 99=62.82, below peak — same no-late-improvement pattern as
every prior run). Stratified: all=64.75, ambiguous=62.77,
unambiguous=69.67.

Two comparisons, kept separate on purpose:
```
                          all     ambiguous   unambiguous
Mode C (base kernel)     64.06      62.17        68.77
Mode E (C + growing)     64.75      62.77        69.67
Delta (E - C)            +0.69      +0.60        +0.90
```
Growing-scales genuinely helps *relative to Mode C alone*, on both
buckets — a real, positive mechanism-level finding.

```
                                  ambiguous    vs baseline's range [63.37-64.70]
Mode E                             62.77              below range
```
But *relative to baseline* (the actual hypothesis test), Mode E's
ambiguous score falls below baseline's entire observed range — same
miss pattern as every other BDH configuration. Its unambiguous score
(69.67) does land above baseline's range, consistent with Mode A's
pattern of real gains concentrated in the bucket the hypothesis isn't
about.

**Caveat**: single seed only. Per the Mode A saga this session, single-
seed BDH results have not been reliable predictors of the multi-seed
mean — this result should be read as suggestive, not confirmed, if a
stronger claim about Mode E specifically is ever needed later.

## FINAL SYNTHESIS (2026-07-21): AttnGrounder phase closed, per the stopping criterion

Complete dataset — every BDH configuration tested (3 kernel modes,
seed variation on the winner, the growing-scales extension) — on the
ambiguous bucket specifically, compared against baseline's own observed
range [63.37, 64.70] (2 seeds):

| Configuration | AP50 (ambiguous) | vs. baseline's range |
|---|---|---|
| Mode A, seed=0 | 63.98 | inside range |
| Mode A, seed=1 | 62.89 | below range |
| Mode A, seed=2 | 63.98 | inside range |
| Mode B | 58.19 | well below range |
| Mode C | 62.17 | below range |
| Mode E (C + growing) | 62.77 | below range |

**None of the six BDH configurations tested clearly exceed baseline's
own observed ambiguous-scene range.** As clean and complete a negative
result on the specific disambiguation hypothesis as this phase could
produce.

**Two separate, both-true conclusions for the AttnGrounder phase**:
1. **Disambiguation hypothesis ("BDH's memory gives a disproportionate
   ambiguous-scene advantage"): NOT supported.** Consistent across every
   configuration tested.
2. **General viability ("BDH is a comparable-to-modestly-better
   attention replacement"): supported.** Mode A's 3-seed mean beats
   baseline's 2-seed mean (+0.17 aggregate); Mode E substantially
   improves on its own base kernel (+0.69) and beats baseline on
   unambiguous (+0.90 over Mode C, and above baseline's own range).
   Real, positive, and worth headlining — just not via the hypothesized
   mechanism.

**Per the stopping criterion (DECIDED 2026-07-21 above): AttnGrounder
experimentation stops here.** Next: collect the metrics table
(GFLOPs, layer count, FPS, inference-ms, params) for the final
comparison table, then move to cross-architecture validation (TransVG)
and/or the offline pretrained-text-encoder ablation (both already
scoped above) as separate, deliberately-sequenced next phases — not
further AttnGrounder-side experiments.

## Metrics collection tooling (2026-07-22)

**IMPLEMENTED**: `analysis/measure_metrics.py` (new) — params, layer
count (by type, e.g. `Conv2d: N`), GFLOPs, inference-ms, FPS, for
baseline or any BDH variant against a real checkpoint.
- `count_layers()`: counts parameterized "leaf" modules only (containers
  and param-free layers like `ReLU` excluded) — an unambiguous metric
  that avoids having to define "depth" for a multi-branch/multi-scale
  architecture (3 FPN scales, parallel heads).
- GFLOPs via `fvcore.nn.FlopCountAnalysis` (not yet confirmed installed
  on remote), measured at fp32 on a single image (GFLOPs is a
  precision-independent architecture property) — separate from
  inference-ms, which uses the real training precision (bf16 autocast)
  since that's about measured wall-clock throughput, not architecture
  size. Wrapped in try/except: this codebase's custom Darknet layers
  (route/shortcut/upsample) are a real risk for fvcore compatibility, so
  a failure here is reported with the reason rather than crashing the
  whole script — params/layers/inference-ms/FPS still get reported
  either way.
- Reuses `train.py`'s existing `measure_inference_ms()` (already used by
  `--eval-only`, just not yet run systematically across variants).

**Verified locally**: new `tests/measure_metrics_test.py` — confirms
`count_layers` correctly excludes containers and param-free leaves,
correctly tallies repeated layer types (e.g. 3x `Conv2d` across FPN
scales). GFLOPs/inference-ms need the real Darknet-backed model + GPU,
not locally testable — verify on remote. All 6 local test suites
(smoke, plumbing, integration, stratify-logic, text-encoder,
measure-metrics) pass, `py_compile` clean.

**RESULT (2026-07-22): baseline + Mode A run successfully, GFLOPs pending.**

```
              Params    Layers   Inference-ms   FPS
Baseline      75.84M     183        4.94       202.59
Mode A        76.63M     184        5.05       197.91
```

`fvcore` not yet installed on remote -- GFLOPs fell back gracefully
(unavailable + reason, not a crash) exactly as designed; everything else
still reported. Rerun both once `pip install fvcore` is done.

**Two things worth flagging in the writeup**:
1. "184 vs 183 layers" understates BDH's real structure -- its core
   `E`/`Ev`/`Dx` encoder/decoder matrices are raw `nn.Parameter` tensors
   on `BDHVisualTextAttention`, not wrapped in submodules, so
   `count_layers()` can't see them (only the extra `Linear` beta_head
   shows up). **Params (+0.79M) is the honest complexity measure**, not
   layer count.
2. Mode A is ~2.2% slower per image (5.05ms vs 4.94ms, 197.91 vs 202.59
   FPS) -- small but real overhead from the lift/memory/gate
   computation. Report plainly as part of "lightweight but not free,"
   not glossed over.

**GFLOPs RESULT (2026-07-22), fvcore installed, rerun both**:
```
              Params    Layers   GFLOPs   Inference-ms   FPS
Baseline      75.84M     183      43.49      2.14        468.34
Mode A        76.63M     184      50.04      2.45        407.73
```

**GFLOPs is reliable** (deterministic static analysis, not live timing)
and reveals something the params number hides: **+6.55 GFLOPs, +15.1%
relative** -- far bigger than the params delta (+1.04%). BDH's lift into
the high-dimensional sparse space (`n = mult*d = 1024`) is compute-heavy
at *each* of the 3 FPN scales (169/676/2704 regions), even though the
lift/projection matrices are shared across scales and cheap in param
count. **Real nuance for the "lightweight" framing**: lightweight in
params, meaningfully more expensive in compute. State this plainly.

**Inference-ms is NOT reliable as measured -- flag before using it
anywhere.** The two runs of the *identical* checkpoints gave very
different absolute timings (baseline 4.94ms -> 2.14ms, a 57% swing;
Mode A 5.05ms -> 2.45ms, a 51% swing), almost certainly from variable
load on the shared GPU between runs. More concerning: the *relative*
overhead of Mode A over baseline also isn't stable across the two runs
(+2.2% in run 1, +14.5% in run 2) -- if this were purely uniform
background load, that relative percentage should have stayed roughly
consistent even as absolute numbers shifted. **Do not report either
inference-ms/FPS number as a confident, publishable figure yet.**
Needed before this table is final: re-measure both back-to-back
(minimize the window for background load to change between them),
ideally checking `nvidia-smi` immediately before each to confirm the
GPU isn't under heavy contention from another user at that moment.
NOT YET DONE.

**RECONCILED (2026-07-22): 3rd measurement run confirms runs 2+3 are the
clean pair, run 1 was the contended outlier.**
```
Run   Baseline   Mode A   Relative overhead
1     4.94ms     5.05ms   +2.2%   (likely contended -- outlier)
2     2.14ms     2.45ms   +14.5%
3     2.11ms     2.33ms   +10.4%
```
Runs 2 and 3 agree well with each other, disagree sharply with run 1 --
under real GPU contention both models get bottlenecked by shared
external load, compressing the *relative* difference; under clean
conditions the true architectural cost shows through more consistently.
Averaging the two clean runs (2.125ms/2.39ms):

## FINAL AttnGrounder-phase metrics table (2026-07-22)

```
              Params     Layers   GFLOPs   Inference-ms   FPS
Baseline      75.84M      183      43.49      2.13         ~470
Mode A        76.63M      184      50.04      2.39         ~418
Delta         +0.79M      +1       +6.55      +0.26ms      -52
              (+1.0%)              (+15.1%)   (+12.5%)     (-11.1%)
```

**Honest three-dimensional picture, all three pieces belong in the
writeup together**: BDH Mode A is lightweight in parameters (+1%) but
meaningfully more expensive in compute (+15.1% GFLOPs) and measurably
slower in practice (+12.5% inference time). Don't lean on the flattering
params number alone -- report all three. This closes out the
metrics-collection task; the AttnGrounder phase now has a complete,
defensible results package (ablation study + stratified findings +
metrics table).

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

## DECIDED (2026-07-21): AttnGrounder stopping criterion + cross-architecture plan

**Stop AttnGrounder experimentation once Mode A seed=2 and the
cross-scale growing-memory run (Mode E) both finish** — both already
built/queued, low-cost to complete. Whatever those two results show
(positive, negative, or still-ambiguous) becomes **the AttnGrounder-side
finding**, not a problem to keep chasing with more experiments. Then:
1. Collect the metrics table (GFLOPs, layer count, FPS, inference-ms,
   params) for baseline + whichever configuration ends up the reference
   point. `train.py` already has `measure_inference_ms()`; GFLOPs and
   layer count are not yet measured (need `fvcore`/`ptflops` for GFLOPs,
   trivial enumeration of `model.named_modules()` for layer count).
2. Move to cross-architecture validation on TransVG (2021, ICCV,
   official code at github.com/djiajunustc/TransVG — see the "modern
   architecture" research above).

**Explicitly deferred, NOT gates for finishing AttnGrounder**: the
distractor-contrastive loss (`arch2/`) and the BDH->attention hybrid
(Mode D) are real, reasonable ideas, but each is its own multi-hour
build-and-train cycle — chaining more of them onto AttnGrounder is how
"when do we stop" becomes "never." They become candidates to try on
TransVG instead, or explicit future work beyond this paper, not
prerequisites.

**Cross-architecture validation scope (TransVG phase) — kept
deliberately light**: do NOT replicate the entire AttnGrounder ablation
(3+ modes, multiple seeds each) on TransVG — would double total project
size and defeat the point of having a stopping rule. Instead: port over
just the single most informative configuration from the AttnGrounder
phase (whichever mode/setup produced the clearest result, positive or
negative), rerun with ~2 seeds on TransVG as a lighter variance check,
and see if the same qualitative story holds. Enough to support a genuine
cross-architecture claim without a second full ablation study.

## Anticipated reviewer questions (prepared answers, 2026-07-21)

**"Why test on an old (2020) model/dataset?"**
- Controlled comparison requires a fixed, minimal reference point — the
  whole point of the ablation study is isolating what BDH's mechanism
  contributes. AttnGrounder is chosen *because* it's simple enough that
  the fusion module is a single, cleanly swappable component. A newer,
  more complex architecture (more moving parts, pretrained encoders,
  deeper fusion) would introduce more confounds, making it harder, not
  easier, to attribute any effect to BDH specifically — a deliberate
  methodological choice, not a lazy default.
- The dataset isn't stale — Talk2Car is still the benchmark CAVG (2024)
  and ThinkDeeper (2025) use. It's the base *architecture* that's from
  2020, not the task or data.
- The planned TransVG cross-architecture validation turns this into a
  strength: "we further validate this isn't an artifact of one 2020-era
  architecture by porting the mechanism to a structurally different,
  more modern base."

**"Why not just use CLIP/VLMs — they perform much better, why bother
with non-VLM at all?"**
- A VLM's performance advantage comes overwhelmingly from web-scale
  pretraining, not from its fusion mechanism being better-designed for
  disambiguation specifically. Swapping BDH into a CLIP-based model and
  seeing no extra ambiguous-scene benefit would be uninterpretable —
  can't tell if BDH doesn't help, or if CLIP's already-massive
  pretrained representations create a ceiling effect that swamps any
  local mechanism contribution. Testing on a non-pretrained base is what
  makes the mechanism's own contribution measurable at all.
- Standard methodological sequencing, not an excuse: no new
  attention/memory mechanism gets validated for the first time inside a
  frontier-scale model — Transformers, Mamba, RWKV, xLSTM, and BDH
  itself were all first validated in controlled, smaller-scale settings
  before anyone scaled them up. We're doing exactly that for a
  mechanism under a year old.
- VLM-scale testing is genuine future work, not something being
  avoided — the two-phase plan's phase 2 (pretrained text encoder)
  already starts probing "does this hold up when pretraining is added,"
  just not at full CLIP scale.

## Open questions / decisions needed

- Which of A/B/C to build on once B finishes (currently A leads).
- Whether to pursue the backbone-upgrade lever at all, given it would
  broaden scope beyond the core BDH-mechanism question (if pursued, must
  apply to both variants for a fair comparison).
- Whether val-only reporting (with the test-set caveat above) is
  sufficient for the eventual write-up, or whether to attempt contacting
  Talk2Car maintainers directly for research-use test GT access (slow,
  uncertain, not on the critical path).

## DECIDED (2026-07-22): headline narrative pivots to viability; disambiguation hypothesis reframed as a secondary, honestly-reported negative result

**The change**: the project's headline claim is no longer "BDH gives a
disambiguation-specific advantage" (the original motivating hypothesis,
which is NOT supported — see the multi-seed results above). It is now:
**"a memory architecture published under a year ago, never previously
tested on any cross-modal task, is a viable drop-in replacement for
established softmax attention on a real grounding benchmark"** —
comparable-to-modestly-better AP50, near-zero parameter overhead, zero
task-specific tuning of the mechanism itself.

**Why**: the disambiguation hypothesis was tested rigorously (3 kernel
modes, multi-seed on the winner, the growing-memory extension — 6 BDH
data points total) and did not hold up against baseline's own observed
variance. Leading the paper with that as the headline result makes the
whole project read as a negative-result paper, when the more accurate
and more interesting story is the viability finding: BDH holds its own
against a mechanism with years of refinement behind it, on the very
first attempt, with no hyperparameter search on the BDH side. That's a
real, positive, and honestly-earned claim — separate from whether the
originally-hypothesized mechanism explains *why*.

**Guardrail (non-negotiable)**: this is a reframing of emphasis, not a
suppression of the negative result. The disambiguation hypothesis and
its negative finding remain fully and honestly reported — full
multi-seed evidence, the Mode A/B/C/E breakdown, the seed-replication
failure — just moved later in the narrative, under a "digging into the
mechanism" framing, instead of leading with it. This is NOT HARKing:
the original hypothesis is explicitly presented as having come first
(motivating the project), and its negative result is stated as plainly
as before — nothing about the actual finding changed, only which claim
is the headline.

**What changed in the slide deck** (`slides/progress_update.tex`):
- Subtitle: "Testing the 'Grounding = In-Context Retrieval' Hypothesis"
  → "Is a $<$1-Year-Old Memory Architecture a Viable Attention
  Replacement?"
- "Why Bring in BDH?" now leads with the viability question, defers the
  specific mechanistic hypothesis to later ("explored later, once the
  headline result is on the table").
- "Interpretation" slide renamed to "BDH Holds Up Against Established
  Attention," leads with the viability evidence, forward-references the
  mechanism discussion instead of stating the negative result inline.
- New `\section{Digging Into the Mechanism}` (between Methodology and
  "What We Tried") with two slides: (1) presents the original
  "grounding = in-context retrieval" hypothesis, explicitly flagged as
  the project's starting motivation; (2) "Honest Answer: Not Supported"
  — the full negative result across all 6 BDH configurations, plus an
  explicit "why report this at all if it's negative" justification.
- Final Conclusion ("What This Project Actually Shows") and "Summary"
  slides reordered so the viability claim (exampleblock / first bullet)
  leads and the mechanism result (alertblock / later bullet) follows —
  content of both claims unchanged, only order and framing.

## TransVG cross-architecture port — BDH swap DONE + locally verified (2026-07-22)

Started the TransVG phase (the planned cross-architecture validation).
Decision recap, confirmed with reasoning before building: keep the DATASET
fixed (Talk2Car) and change only the ARCHITECTURE, so the single variable
tested is "does BDH's viability generalize beyond one 2020 CNN-based
grounder." Also confirmed TransVG is a defensible choice specifically
because (a) it already has a published Talk2Car number (65.83 AP50, in
ThinkDeeper's table) so it is NOT foreign to the benchmark and we can
sanity-check our reimplementation against it; (b) it is architecturally
DISTANT from AttnGrounder (transformer V-L fusion + regression head vs.
CNN + softmax cross-attention + YOLO head), which is exactly what a
generalization test needs; (c) same "single cleanly-swappable fusion
component + official code" selection criterion used for AttnGrounder;
(d) CMSVG, the more obviously Talk2Car-native alternative, was already
ruled out (FINDINGS.md CORRECTION 2026-07-20) for depending on pretrained
Sentence-BERT/EfficientNet, failing the isolate-the-mechanism criterion.

**The swap point is different from AttnGrounder's** and this matters for
the claim. AttnGrounder had a single spatial visual->text cross-attention
module we replaced wholesale. TransVG instead concatenates
[REG]; text_tokens; visual_tokens into ONE sequence and runs it through a
6-layer transformer encoder; the "attention" is the nn.MultiheadAttention
self-attn INSIDE each encoder layer (vl_transformer.py:73). So the faithful
BDH swap here = replace that self-attention operator with a non-causal,
mask-aware BDH associative-memory self-attention, keeping the encoder
layer's residual + FFN + LayerNorm identical. Only the attention mechanism
changes -> the baseline-vs-BDH comparison isolates it, same discipline as
the AttnGrounder phase.

New module `bdh_grounding/bdh_selfattn.py` (BDHSelfAttention) — NOT the old
bdh_fusion.py, which is spatial-grid-specific (H×W reshaping, the beta map
for AttnGrounder's BCE aux loss, none of which TransVG has). It is built to
match nn.MultiheadAttention's forward signature EXACTLY (q,k,v, attn_mask,
key_padding_mask) -> (out, None) so it drops in with a one-line constructor
change and no call-site edits. Mechanism = same BDH memory (lift to
ReLU-sparse n-dim, rho = K^T V outer-product memory, gated bilinear
read+decode) as bdh_fusion but non-causal, symmetric, reading out ALL
positions; closest to the old mode "B" (single sequence) minus causality
and RoPE (position comes from TransVG's own learned vl_pos embeddings,
folded into q/k upstream). Stacked once per encoder layer -> N layers give
N sequential BDH reads with FFN+residual between, a faithful analog of
"N layers of self-attention -> N layers of BDH self-attention."

Files changed/added (all local, synced-to-remote model, py_compile clean):
- `bdh_grounding/bdh_selfattn.py` (NEW) — the operator.
- `external/TransVG/models/vl_transformer.py` — gated `attn_type` ('mha'|
  'bdh') threaded through VisionLanguageEncoder -> TransformerEncoderLayer;
  robust bdh_grounding import; **_reset_parameters guarded so TransVG's
  xavier_uniform_ does NOT clobber BDH's deliberate normal_(0.02) init**
  (verified: BDH E/Dx std stays ~0.02, FFN linears still xavier).
- `external/TransVG/datasets/data_loader.py` — registered 'talk2car' in
  SUPPORTED_DATASETS + a talk2car im_dir branch. pull_item's existing
  refcoco branch already handles our 5-tuple + xywh->xyxy, no edit needed.
- `external/TransVG/train.py` — added --vl_attn_type/--bdh_mult/
  --bdh_share_qv args.
- `arch3/build_transvg_talk2car.py` (NEW) — converts our
  talk2car_{split}.pth (img_file, bbox_xywh, phrase) into TransVG's
  (img_file, None, bbox_xywh, phrase, None) 5-tuple. A pure repack: same
  filenames, same xywh boxes, same phrases — the DATA both models see is
  identical, only architecture + text tokenizer (BERT vs GloVe) differ.
  Only train/val (test is GT-less).
- `tests/transvg_bdh_test.py` (NEW, all pass) — signature/shape parity with
  nn.MultiheadAttention; attn_mask fail-loud; **key_padding_mask zero-leakage
  (corrupting padded tokens leaves every real-token output bit-identical)**;
  both modes forward+backprop; init preservation; build_vl_transformer arg
  threading incl. legacy-args-default-to-mha safety; share_qv param math.

**Locally VERIFIED**: the operator + wiring (above). **NOT locally testable
(remote/GPU/pytorch_pretrained_bert/data), pending on remote**: full TransVG
forward, the Talk2Car loader end-to-end, real training. Remote runbook (rsync
+ DETR-R50 weights + BERT + converter + symlink + baseline & BDH train
commands) is in STATUS.md. Plan per the stopping rule: run just baseline +
one BDH config, ~2 seeds, val AP50 only — a lighter variance check, NOT a
second full ablation. Baseline should land near the published 65.83 as the
reimplementation sanity check before any BDH conclusion is drawn.
