# Project status — BDH-swapped AttnGrounder on Talk2Car

Last updated: 2026-07-18 — baseline eval-only pipeline confirmed working
end-to-end on remote (see "BLOCKER RESOLVED" below). **Read this file first in
the next session to resume with full context.**

## Goal (one line)
Swap AttnGrounder's visual-text attention module for a BDH-based module, train
both on Talk2Car, and test whether BDH's growing outer-product associative
memory (unlike Mamba) shows an AP50 advantage that concentrates on scenes with
multiple same-class candidate objects ("ambiguous" scenes) — testing the
"grounding = in-context retrieval" hypothesis from the Shaking-Up-VLMs paper.

## Environment split
- **Local**: this Mac (Apple M1 Pro, no CUDA). All code is authored/edited HERE,
  then synced to the remote. Local is also used for CPU-only smoke tests.
- **Remote**: `ssh cs24d0010@172.16.1.199`, conda env `brats`
  (torch 2.4.1+cu118, Python 3.10), **A100-PCIE-40GB on `cuda:0`** (ignore the
  Quadro P2000 on cuda:1 — always set `CUDA_VISIBLE_DEVICES=0`). All real
  data/training/eval happens here.
- Sync direction: **local → remote only**, via `rsync -avz <paths> cs24d0010@172.16.1.199:~/BDH/Talk2Car/AttnGrounder/`
  (or `scp -r` for a first-time copy). Never edit code directly on the remote.
- Remote has an open **tmux session named `attngroun`** — reattach with
  `tmux attach -t attngroun` (or `tmux ls` to check, `tmux new -s attngroun` if gone).

## Repo layout
- **Local** (`~/Desktop/Talk2Car BDH/`):
  - `external/AttnGrounder/`, `external/bdh/` — cloned reference repos (read-only reference, not run locally)
  - `bdh_grounding/` — the actual BDH module + pipeline helpers (see below)
  - `configs/`, `train.py`, `tests/`, `analysis/` — all authored here
  - `papers/` — BDH paper (`2509.26507v1.md`), AttnGrounder paper (`attngrounder.md`), IRRA paper (`2303.md`, not directly relevant)
- **Remote** (`~/BDH/Talk2Car/`):
  - `AttnGrounder/` — the repo we run everything from (our `train.py` + `bdh_grounding/` + `configs/` + `tests/` + `analysis/` copied into its root so imports resolve)
  - `Talk2Car/` — Talk2Car repo (commands JSON + slim images), used for Step 6 data
  - `bdh/` — pathwaycom/bdh reference (not run directly; `bdh_fusion.py` is our own reimplementation)

## Design decisions (LOCKED, user-approved)
- **BDH module = Option C**: symmetric NON-causal kernel. Regions are queries,
  words are keys/values. ReLU-sparse high-`n` addressing, explicit outer-product
  associative memory `ρ = Kᵀ@V`, multiplicative gate for bilinear fusion. Drops
  causal mask + RoPE (grounding has no sequence order). True drop-in: outputs
  `T'ᵏ (B,C,H,W)` + `βᵏ (B,H,W)` matching AttnGrounder's original `text_attn`.
- **Fusion trim**: fuse only `[fvisu, flang_attn]` (2 streams, drop `β⊙visual`),
  `fcn_emb` input width `emb_size*2` not `*3`.
- **Single-head** for the first pass.
- **`share_qv_encoder` lever**: ties query/value encoders to reclaim `n·d` params
  (config knob, default `false`).
- **A and B modes** (causal-context ablation, single-causal-sequence ablation)
  are implemented in `bdh_fusion.py` and selectable via `model.bdh.mode`, but not
  yet validated end-to-end — reserved for a future ablation study.
- **Param honesty**: AttnGrounder's REAL `emb_size` is 512 (not the paper's
  illustrative 256). Only `mult=1` (no expansion) is strictly ≤ baseline 75.84M;
  `mult=2` (our default) costs **+0.79M (~+1%)**, or **+0.26M** with
  `share_qv_encoder:true`. Backbone dominates either way — framing should be
  "comparable params, richer fusion, higher AP50 especially on ambiguous
  scenes," not a headline param win.
- **Step 6 data source = nuScenes metadata** (user's explicit choice over a
  commands-only proxy or hunting for a 2D-detection file). Needs
  `v1.0-trainval_meta.tgz` (~445MB, metadata only, NOT the 300GB sensor data)
  + a free nuscenes.org account + `nuscenes-devkit`. **NOT YET DOWNLOADED.**

## Files built (all local, all CPU-tested green as of last run)
```
bdh_grounding/
  bdh_fusion.py       BDHVisualTextAttention: modes C (primary), A, B (ablations)
  pipeline.py         config->args, bf16 autocast ctx, CSV logger, checkpoint/resume, subset helper
  INTEGRATION.md       exact 3-edit patch for grounding_model.py + the .cuda()->device fix
                        (NOT YET APPLIED on remote — only needed for variant=bdh)
  __init__.py
configs/
  smoke_local.yaml     CPU, fp32, emb_size 256, 20 steps
  full_a100.yaml       bf16, emb_size 512 (real AttnGrounder value), batch 14, 100 epochs,
                        mode C, mult 2, lr 1e-4 / backbone lr/10, wd 5e-4, poly power 1
train.py               config-driven trainer/evaluator, mirrors train_yolo.py's loss/target/
                        eval logic exactly, device+precision agnostic, --eval-only mode reports
                        AP50 + inference-ms + params, --variant CLI override
tests/
  smoke_test.py            BDH module contract (all 3 FPN scales, all 3 modes, grad, params)
  integration_smoke.py     BDH swap + 2-stream fusion -> mock YOLO head, shape-verified
  plumbing_test.py         config parsing, CSV logger, checkpoint roundtrip, subset/anchors
  stratify_logic_test.py   Step 6 bucketing math + category matching (synthetic data)
analysis/
  build_scene_index.py     nuScenes + Talk2Car commands -> scene_index_{split}.json
                            (per-image same-class-other-object counts). NOT YET RUN (needs nuScenes dl)
  stratified_eval.py       source-agnostic: AP50 split into ambiguous/unambiguous/unknown
  fix_corpus.py            one-time spaCy2->3 corpus.pth converter (see below)
```
All local tests pass: `python3 tests/smoke_test.py`, `integration_smoke.py`,
`plumbing_test.py`, `stratify_logic_test.py` — run these again after any local
edit before syncing.

## Remote environment fixes already applied (don't redo)
1. `utils/transforms.py`: `from collections import Iterable` →
   `from collections.abc import Iterable` (Python 3.10 removed the old path).
   Applied via `sed` directly on remote.
2. `corpus.pth` (487MB) embedded a **spaCy 2.x** pipeline → broke under the
   installed spaCy 3.x (`ModuleNotFoundError: spacy.lemmatizer`, and further
   pickle-reconstruction edge cases). Fixed by `analysis/fix_corpus.py`: loads
   with a spaCy-agnostic stub-unpickler (a `_Dummy` class + metaclass that
   absorbs any pickle reconstruction protocol — init, setstate, setitem, even
   direct `getattr()` on the class object itself), keeps the real
   `dictionary`/`glove` (plain Python objects, untouched by stubbing), swaps
   `nlp` for `spacy.blank("en")` (same tokenizer surface tokens), re-saves.
   **Already run successfully on remote** — original backed up to
   `ln_data/corpus.pth.spacy2.bak`. Result verified: vocab=1897,
   glove_present=True, tokenize output correct.
3. `pip install spacy && python -m spacy download en_core_web_sm` — done
   (avoid `Talk2Car/requirements.txt` directly, it pins `torch==1.8.1` which
   doesn't exist for Python 3.10 and aborts the whole install).
4. `train.py build_model()`: baseline variant calls the **stock**
   `grounding_model` (no BDH kwargs) so baseline/author-checkpoint eval works
   against the unmodified repo — `INTEGRATION.md` is only needed once we run
   `variant=bdh`.

## Data status on remote (confirmed real, not assumed)
- `AttnGrounder/ln_data/`: `corpus.pth` (fixed), `talk2car_{train,val,test}.pth`,
  `images/` (Talk2CarSlim, from Drive), `version_n2.0_continued_model_best_continued.pth.tar`
  (857MB — **authors' own pretrained checkpoint**, bonus find in the Drive folder).
- `AttnGrounder/saved_models/yolov3.weights` (237MB, COCO-pretrained Darknet-53).
- Splits: **train 8349**, **val 1163** (both `(img_file, [x,y,w,h], phrase)`,
  has GT) — **test 3610** (`(img_file, phrase)`, **no GT**, held out for
  leaderboard). **Evaluate on val, not test.**
- `Talk2Car/data/commands/{train,val,test}_commands.json`: dict
  `{"commands": [...]}`, each entry has `t2c_img` (== AttnGrounder `img_file`,
  clean 1:1 join key), `obj_name` (referred object class, e.g. `vehicle.car`),
  `2d_box`, `sample_token`, 3D translation/size/rotation. **Only the referred
  object** — no all-objects-in-scene list, hence the nuScenes-metadata plan for
  Step 6. Multiple commands can share a `sample_token` (same image, several commands).

## BLOCKER RESOLVED (2026-07-18)
The bf16 `.numpy()` fix (`decode_pred_boxes()` now `.float()`s `pred_anchor`
before any numpy conversion) worked. Confirmed on remote:
```
[eval] AP50=65.32  inference=3.2ms  params=75.84M
```
75.84M params matches the paper exactly; AP50=65.32 is close to the
checkpoint's own tracked best (65.75%, epoch=39). **This validates the entire
remote pipeline end-to-end**: corpus (spaCy-fixed), Darknet weights, data
loader, model build, our eval path, checkpoint loading. `train.py` on remote
is confirmed in sync with local (fix present, no redo needed).

## Roadmap after the blocker clears
- **Step 2 (real)** — DONE (2026-07-18): applied `bdh_grounding/INTEGRATION.md`'s
  3 edits + the `generate_coord` `.cuda()`→`device` fix to local
  `external/AttnGrounder/model/grounding_model.py` (reference copy, edited
  locally per the local→remote sync model), verified `py_compile` clean and
  kwargs match `train.py build_model()` exactly, synced to remote
  `~/BDH/Talk2Car/AttnGrounder/model/grounding_model.py`. `variant=bdh` is now
  runnable on remote.
- **Step 5.1 — actual baseline reproduction** — DONE (2026-07-19): trained
  AttnGrounder from scratch (`--variant baseline`, `full_a100.yaml`, 100
  epochs) on remote A100. **Best AP50 = 64.89** (epoch 98; epoch 99 dipped to
  63.77, checkpointer correctly kept the epoch-98 weights). Compares well to
  paper's 63.30% and the author checkpoint's 65.75% — solid reproduction,
  validates the training loop end-to-end (not just eval-only).
- **Step 5.2 — BDH variant training** — DONE (2026-07-19): **best AP50 = 64.11**
  (epoch < 99; both epoch 98 and 99 logged 62.31, so best was set on an
  earlier epoch not shown in the tail). Vs baseline's 64.89, BDH trails by
  -0.78 AP50 in aggregate. **This is not yet informative on its own** — the
  hypothesis is about ambiguous-scene concentration, not aggregate AP50; a
  small aggregate deficit is consistent with BDH still winning on the
  ambiguous subset while losing on unambiguous (or just being noise from a
  single seed). Step 6 stratified eval is what actually tests this.
  Original crash/fix history below, for reference:
  `mode: C` (config default already `variant: bdh`, `mode: C`, `mult: 2`).
  First attempt (2026-07-19) crashed on batch 1 with
  `RuntimeError: Found dtype Float but expected BFloat16` in the BCE aux mask
  loss (`train_epoch`, `train.py` map_loss line). **Root cause**: baseline's
  `beta` happens to land in fp32 because `torch.sum` is on autocast's
  fp32-promotion list, but BDH's `beta = sigmoid(Linear(...))` stays in the
  autocast dtype (bf16) — `obmap` is explicitly `.float()`'d, so `BCELoss`
  saw a dtype mismatch only for `variant=bdh`. **Fixed** (train.py, in
  `train_epoch`): `attn_map[k].float()` before the loss call — same class of
  bug/fix as the earlier `decode_pred_boxes` bf16→numpy issue. Synced to
  remote; crashed before any checkpoint was written so this is a clean
  restart, not a resume:
  ```bash
  cd ~/BDH/Talk2Car/AttnGrounder
  tmux new -s bdh_train   # or reuse attngroun/a fresh window
  CUDA_VISIBLE_DEVICES=0 python train.py --config configs/full_a100.yaml --variant bdh
  ```
  Expect ~similar wall-clock to the baseline run just completed (same epoch
  count/batch size; BDH module adds ~1% params, shouldn't meaningfully change
  step time). Detach with `Ctrl+b d`, reattach anytime to check progress —
  no need to babysit it. **NOT YET RE-RUN / CONFIRMED PAST BATCH 1.**
- **Step 6 data prep** — DONE (2026-07-19): downloaded `v1.0-trainval_meta.tgz`
  (0.43GB, Trainval > Metadata only — NOT the sensor-blob parts) from
  nuscenes.org, extracted to `~/BDH/Talk2Car/nuscenes/v1.0-trainval/` on
  remote, `pip install nuscenes-devkit` (v1.2.0) in the `brats` env.
  **Bug found + fixed**: `build_scene_index.py` originally did
  `nusc.get("sample", c["sample_token"])` — but Talk2Car's `sample_token`
  field is actually the **CAM_FRONT `sample_data` token**, not nuScenes'
  keyframe `sample` token (confirmed via direct lookup:
  `channel='CAM_FRONT'`, `is_key_frame=True`). All 1163 val commands failed
  with `KeyError` until fixed to call `nusc.get_sample_data(c["sample_token"],
  ...)` directly, skipping the `sample` table lookup entirely. Fixed, synced,
  rerun — confirmed working: `analysis/scene_index_val.json` built,
  **830 ambiguous / 333 unambiguous / 0 errors** (out of 1163 val commands).
- **Step 6 eval** (NEXT): `analysis/stratified_eval.py` on both trained
  models (`checkpoints/full/baseline_best.pth.tar`,
  `checkpoints/full/bdh_best.pth.tar`) against `scene_index_val.json` → AP50
  split into ambiguous/unambiguous — the actual hypothesis test. Baseline
  aggregate AP50=64.89, BDH aggregate AP50=64.11 (BDH trails by -0.78 overall,
  but that alone doesn't confirm/deny the hypothesis — what matters is
  whether BDH's relative AP50 is better specifically on the 830 ambiguous
  scenes vs. the 333 unambiguous ones).
- **Final table**: AP50 / inference-ms / params for baseline vs BDH (matching
  AttnGrounder's paper format), plus the stratified breakdown.

## RESULT (2026-07-19) — Step 6 stratified eval, hypothesis NOT supported
```
              AP50 (all)   AP50 (ambiguous, n=830)   AP50 (unambiguous, n=333)
baseline        64.92             63.37                      68.77
bdh             64.06             62.17                      68.77
Δ (bdh-base)    -0.86             -1.20                       0.00
```
(Params: baseline 75.84M, bdh 76.63M, from `[model]` build lines.)

Both variants tie exactly on unambiguous scenes (229/333 hits either way).
BDH's entire aggregate deficit is concentrated in the ambiguous bucket — but
in the OPPOSITE direction from the hypothesis: if BDH's growing outer-product
associative memory gave a retrieval edge on multi-same-class-candidate
scenes, the ambiguous-bucket gap should favor BDH (or at least be smaller
than the unambiguous gap). Instead BDH loses more ground exactly where it
was predicted to win. **The "grounding = in-context retrieval" hypothesis
from Shaking-Up-VLMs is not supported by this experiment**, at least for
mode C / mult=2 / single-head / single seed.

Caveats before over-interpreting: single training run per variant (no seed
averaging), ~1 AP50 gap on 830 examples has non-trivial noise given the
scale.

**DECIDED (2026-07-19): proceed with modes A and B**, then look at
improvements beyond the attention swap. Not yet decided: seed-averaging /
multi-seed reruns — revisit after A/B land.

## Slides
`slides/progress_update.tex` (+ compiled `progress_update.pdf`, 12 slides,
Beamer/Madrid theme) — progress update for advisor covering hypothesis,
architecture, method, Step 6 stratification methodology, the mode-C result
table + bar chart, honest interpretation, and the A/B ablation plan. Built
2026-07-19, compiles clean (`pdflatex`, 2 passes, no errors/overfull boxes).
Regenerate after A/B results land — the results slide and interpretation
slide will need updating to a 3-mode table.

## IN PROGRESS (2026-07-19): ablation modes A and B
**Checkpoint-tag bug found + fixed before launching**: `train.py` tagged
checkpoints by `args.variant` alone (`"bdh"`), so training mode A would have
silently overwritten mode C's `bdh_best.pth.tar`/`bdh_last.pth.tar`. Fixed:
added `--bdh-mode {A,B,C}` CLI override (mirrors the existing `--variant`
override) to both `train.py` and `analysis/stratified_eval.py`, and
introduced `args.ckpt_tag` (`f"bdh_{bdh_mode}"` for the bdh variant,
otherwise just `args.variant`) used at both `save_checkpoint` call sites.
Going forward checkpoints are `bdh_A_*`, `bdh_B_*`, `bdh_C_*`,
`baseline_*` — no more collisions. Synced to remote.

**Manual step required before mode A's first checkpoint**: rename the
existing mode-C checkpoint on remote to the new naming scheme so it isn't
orphaned/confusing:
```bash
cd ~/BDH/Talk2Car/AttnGrounder/checkpoints/full
mv bdh_best.pth.tar bdh_C_best.pth.tar
mv bdh_last.pth.tar bdh_C_last.pth.tar
```

**Launch commands** (sequential — same A100, don't run A and B in parallel):
```bash
cd ~/BDH/Talk2Car/AttnGrounder
tmux new -s bdh_A
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/full_a100.yaml --variant bdh --bdh-mode A
# after it finishes:
tmux new -s bdh_B
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/full_a100.yaml --variant bdh --bdh-mode B
```

After both finish, eval each against the Step 6 scene index:
```bash
CUDA_VISIBLE_DEVICES=0 python analysis/stratified_eval.py \
    --config configs/full_a100.yaml --variant bdh --bdh-mode A \
    --resume checkpoints/full/bdh_A_best.pth.tar \
    --scene-index analysis/scene_index_val.json

CUDA_VISIBLE_DEVICES=0 python analysis/stratified_eval.py \
    --config configs/full_a100.yaml --variant bdh --bdh-mode B \
    --resume checkpoints/full/bdh_B_best.pth.tar \
    --scene-index analysis/scene_index_val.json
```
**Mode A — DONE (2026-07-19): best AP50 = 65.15** (set at epoch 60, held
through epoch 99 — epoch 98 dipped to 63.86, epoch 99 to 64.54, checkpointer
correctly kept the epoch-60 weights). Stratified eval on
`bdh_A_best.pth.tar`:
```
              AP50 (all)   AP50 (ambiguous, n=830)   AP50 (unambiguous, n=333)
baseline        64.92             63.37                      68.77
bdh mode C      64.06             62.17                      68.77
bdh mode A      65.09             63.98                      67.87
Δ A - baseline  +0.17             +0.61                      -0.90
```
**This is the first result pointing in the hypothesis's predicted
direction**: mode A beats baseline specifically on ambiguous scenes
(+0.61), while ceding ground on unambiguous scenes (-0.90) — a genuine
trade-off toward disambiguation, not the uniform shift mode C showed. The
modest aggregate gain (+0.17) undersells it; the split is the interesting
part. Still one run/one seed — not conclusive, but a much more encouraging
data point than mode C.

**Mode B — DONE (2026-07-20): best AP50 = 60.50** (train), 60.53 (val
eval), stratified: ambiguous 58.19, unambiguous 66.37 — meaningfully worse
than baseline/C/A across the board. See FINDINGS.md for the full 3-mode
table and the interpretive note on why mode A specifically wins (causal
text-self-contextualization, not just "having associative memory" — modes
B and C both have that too and both lose on ambiguous scenes).

**Decision: Mode A is the ablation-study winner.** Proceed with Mode A as
the base for the next round of improvements (Tversky+Focal loss, BDH
stacking, BDH->attention hybrid — see FINDINGS.md).

## LATER (after A/B land): improvements beyond the attention swap
User wants to explore this next, scope not yet defined. Candidate
directions to evaluate/discuss when we get there (brainstormed, not
committed):
- Multi-head BDH (currently locked to single-head for the first pass, see
  design decisions above) — revisit now that a single-head baseline result
  exists to compare against.
- `share_qv_encoder: true` (already a config knob, default false, saves
  ~0.53M params) — cheap to try, currently unused.
- Different `mult` values (memory width n=mult*emb_size) — mult=1 (param-
  neutral, weakest) vs mult=4 (+2.36M, strongest) vs current mult=2.
  Currently no sweep has been run at all, only mult=2.
- Possibly reconsidering the fusion trim (2-stream vs 3-stream) as an
  independent variable from the attention-mechanism swap itself, to
  separate "does BDH help" from "does dropping the beta*visual stream
  help/hurt".

## CURRENT (2026-07-20): running mode A seed confirmation
Tversky+Focal loss on Mode A: confirmed final negative, best AP50=64.03
(see FINDINGS.md for full detail) — not pursued further. Ablation winner
remains Mode A (plain BCE), best AP50=65.09.

**Now running**: multi-seed confirmation of Mode A's base result (does
the +0.61 ambiguous-scene advantage replicate, or was it noise from one
seed?). Added `--seed` CLI override to `train.py` (same pattern as
`--variant`/`--bdh-mode`/`--map-loss`), synced, launched in tmux
`bdh_A_seed1`:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/full_a100.yaml --variant bdh --bdh-mode A --seed 1
```
Checkpoint tag: `bdh_A_seed1` (won't collide with the original
`bdh_A_best.pth.tar`, seed=0). NOT YET DONE.

**`arch2/` folder created** (local + needs sync) — placeholder home for
the Distractor-Contrastive Loss idea (see FINDINGS.md "Idea queued:
Distractor-Contrastive Loss"). Explicitly scoped but NOT YET BUILT —
user wants this logged now, implemented later.

## Useful commands reference
```bash
# reattach remote tmux
ssh cs24d0010@172.16.1.199
tmux attach -t attngroun

# sync any locally-changed file/dir up
rsync -avz <path> cs24d0010@172.16.1.199:~/BDH/Talk2Car/AttnGrounder/

# always pin the A100, never let anything land on the Quadro P2000
CUDA_VISIBLE_DEVICES=0 python ...
```

## TransVG cross-architecture phase — REMOTE RUNBOOK (2026-07-22)

BDH swap is code-complete + locally verified (see FINDINGS.md "TransVG
cross-architecture port"). Everything below runs ON REMOTE — the data,
DETR/BERT weights, and GPU are all there; none of it is locally testable.
Proposed remote home: `~/BDH/Talk2Car/TransVG/`.

**Step 0 — sync the ported code up** (from local):
```bash
rsync -avz external/TransVG/ cs24d0010@172.16.1.199:~/BDH/Talk2Car/TransVG/
rsync -avz bdh_grounding/  cs24d0010@172.16.1.199:~/BDH/Talk2Car/TransVG/bdh_grounding/
rsync -avz arch3/          cs24d0010@172.16.1.199:~/BDH/Talk2Car/TransVG/arch3/
# bdh_grounding must sit at TransVG root so vl_transformer.py's
# `from bdh_grounding.bdh_selfattn import ...` resolves (run from that root).
```

**Step 1 — env + pretrained weights** (remote, `brats` env):
```bash
conda activate brats
pip install pytorch_pretrained_bert            # TransVG's BERT dep (if missing)
cd ~/BDH/Talk2Car/TransVG
bash checkpoints/download_detr_model.sh        # -> checkpoints/detr-r50.pth
# BERT (bert-base-uncased) auto-downloads on first run; needs internet once.
```

**Step 2 — build the Talk2Car splits in TransVG format + images symlink**:
```bash
python arch3/build_transvg_talk2car.py \
    --src-root ~/BDH/Talk2Car/AttnGrounder/ln_data \
    --out-root ~/BDH/Talk2Car/TransVG/data \
    --splits train val
ln -s ~/BDH/Talk2Car/AttnGrounder/ln_data/images \
      ~/BDH/Talk2Car/TransVG/data/talk2car/images
```

**Step 3 — SANITY FIRST: reproduce the TransVG baseline on Talk2Car**
(must land near the published 65.83 AP50 before trusting ANY BDH number):
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset talk2car --data_root ./data --split_root ./data \
    --detr_model ./checkpoints/detr-r50.pth --max_query_len 20 \
    --vl_attn_type mha \
    --batch_size 8 --epochs 90 --output_dir ./outputs/talk2car_mha
```

**Step 4 — the BDH swap run** (identical except `--vl_attn_type bdh`):
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset talk2car --data_root ./data --split_root ./data \
    --detr_model ./checkpoints/detr-r50.pth --max_query_len 20 \
    --vl_attn_type bdh --bdh_mult 4 \
    --batch_size 8 --epochs 90 --output_dir ./outputs/talk2car_bdh
```

Scope (stopping rule, FINDINGS.md 2026-07-21): baseline + ONE BDH config,
~2 seeds (`--seed`), val AP50 only. A lighter variance check, NOT a second
full ablation. First confirm the mha baseline reproduces ~65.83, then draw
the BDH-vs-baseline comparison the same way as the AttnGrounder phase.

Watch-outs (untested remotely yet): first BDH forward pass shape/mask
sanity on real batches; TransVG's `utils.collate_fn` NestedTensor path with
our data; single-GPU (non-distributed) init — `init_distributed_mode`
should no-op without env vars. Run a 1-epoch smoke of the mha baseline
first to shake out data/weights/env before committing to full runs.
