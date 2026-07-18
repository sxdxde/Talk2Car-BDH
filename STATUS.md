# Project status — BDH-swapped AttnGrounder on Talk2Car

Last updated: 2026-07-18, right before a local Mac restart (terminal hit
`fork failed: resource temporarily unavailable` — system resource exhaustion,
unrelated to the project; a restart should clear it). **Read this file first in
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

## CURRENT BLOCKER (fix written, NOT YET CONFIRMED on remote)
Ran the first real pipeline test — baseline eval using the **author checkpoint**
(sanity anchor, not a substitute for training our own baseline — see Roadmap):
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/full_a100.yaml --eval-only --variant baseline \
    --resume ln_data/version_n2.0_continued_model_best_continued.pth.tar
```
Got most of the way through — model built correctly (**75.84M params**, exactly
matching the paper), checkpoint loaded (their own tracked best AP50 =
**65.75%**, i.e. `epoch=39, best=0.6575` — close to the paper's 63.30%, great
sanity signal) — then crashed in `validate()`:
```
TypeError: Got unsupported ScalarType BFloat16
  at decode_pred_boxes(): conf_s = pred_conf_list[best_scale].view(...).data.cpu().numpy()
```
**Root cause**: forward pass runs under bf16 autocast, but `decode_pred_boxes`
called `.numpy()` directly on bf16 tensors (NumPy has no bfloat16 type).

**Fix applied** (in local `train.py`, `decode_pred_boxes()`, line ~150):
```python
def decode_pred_boxes(pred_anchor, anchors_full, args, device):
    pred_anchor = [p.float() for p in pred_anchor]   # <-- added: local rebind only,
    ...                                                #     doesn't affect bf16 loss/backward elsewhere
```
Verified by careful code review (my local Bash tool was down this session, same
fork-exhaustion issue as the user's terminal — could not execute a live test,
but traced every `.numpy()` call site in `train.py` and confirmed this is the
only one touching bf16-derived tensors; `yolo_loss`/`build_target` never call
`.numpy()` on model outputs, and `bbox_iou`/`accm` downstream operate on
already-float32 `pred_box`).

## IMMEDIATE NEXT STEPS (after restart)
1. Confirm Mac is healthy: open a fresh terminal, `echo ok` should work normally.
2. Sync the fixed `train.py`:
   ```bash
   cd "~/Desktop/Talk2Car BDH"
   rsync -avz train.py cs24d0010@172.16.1.199:~/BDH/Talk2Car/AttnGrounder/
   ```
3. On remote (reattach `tmux attach -t attngroun` or start fresh):
   ```bash
   cd ~/BDH/Talk2Car/AttnGrounder
   CUDA_VISIBLE_DEVICES=0 python train.py --config configs/full_a100.yaml --eval-only --variant baseline \
       --resume ln_data/version_n2.0_continued_model_best_continued.pth.tar
   ```
4. Report back the `[eval] AP50=... inference=...ms params=...M` line (or any
   new traceback). Expect ~1-3 min total runtime (eval-only, forward-pass only).
5. If it errors again: paste the traceback, will fix and re-sync.
6. If it succeeds: this validates the ENTIRE remote pipeline end-to-end
   (corpus, Darknet weights, loader, model, our eval path, checkpoint loading).
   Then proceed to the roadmap below.

## Roadmap after the blocker clears
- **Step 2 (real)**: Apply `bdh_grounding/INTEGRATION.md`'s 3 edits (+ the
  `generate_coord` `.cuda()`→`device` fix) to `model/grounding_model.py` on
  remote, so `variant=bdh` becomes runnable. (Can write an auto-patch script
  instead of hand-editing if preferred — offered earlier, not yet built.)
- **Step 5.1 — actual baseline reproduction**: train AttnGrounder from scratch
  (`variant: baseline` in `full_a100.yaml`) on our data/environment — the
  author-checkpoint eval above is a sanity anchor, NOT a substitute for this.
  Use tmux (`tmux new -s baseline_train`), expect on the order of 1-3
  hours/run on the A100 (rough estimate, not measured — extrapolate from a few
  real epochs once running).
- **Step 5.2 — BDH variant training**: same recipe, `variant: bdh`, `mode: C`.
- **Step 6 data prep**: get a free nuscenes.org account, download
  `v1.0-trainval_meta.tgz` (~445MB metadata only), `pip install nuscenes-devkit`,
  run `analysis/build_scene_index.py` against `val_commands.json`.
- **Step 6 eval**: `analysis/stratified_eval.py` on both trained models against
  the scene index → AP50 split into ambiguous/unambiguous — the actual
  hypothesis test.
- **Final table**: AP50 / inference-ms / params for baseline vs BDH (matching
  AttnGrounder's paper format), plus the stratified breakdown.

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
