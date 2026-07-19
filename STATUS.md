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
- **Step 5.2 — BDH variant training** (NEXT): same recipe, `--variant bdh`,
  `mode: C` (config default already `variant: bdh`, `mode: C`, `mult: 2`).
  ```bash
  cd ~/BDH/Talk2Car/AttnGrounder
  tmux new -s bdh_train   # or reuse attngroun/a fresh window
  CUDA_VISIBLE_DEVICES=0 python train.py --config configs/full_a100.yaml --variant bdh
  ```
  Expect ~similar wall-clock to the baseline run just completed (same epoch
  count/batch size; BDH module adds ~1% params, shouldn't meaningfully change
  step time). Detach with `Ctrl+b d`, reattach anytime to check progress —
  no need to babysit it.
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
