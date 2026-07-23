"""Overfit-a-tiny-batch sanity test — the definitive bug-vs-recipe localizer
for the flat-loss TransVG-on-Talk2Car baseline. RUNS ON REMOTE (GPU).

Grabs ONE fixed batch of 8 training samples and trains on just that batch for
a few hundred steps. Interpretation:

  * loss -> ~0 and accu -> ~1.0  => the forward/loss/target plumbing is CORRECT.
    The full-run flat loss is then a training-dynamics/recipe/scale problem
    (effective batch size, augmentation, lr, dataset size) -- NOT a code bug.
  * loss stuck / accu stuck ~0    => there IS a real bug in the model forward,
    the loss, or the target construction, and no amount of recipe tuning helps.

A correct model can always memorize 8 examples. If it can't, nothing else
matters until that's fixed.

Usage (from the TransVG root on remote):
    CUDA_VISIBLE_DEVICES=0 python arch3/overfit_test.py --steps 300 --vl_attn_type mha
    # also worth running with --vl_attn_type bdh to confirm the BDH path too
"""
import os
import sys
import argparse

_TRANSVG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRANSVG_ROOT)

import torch
from torch.utils.data import DataLoader

from train import get_args_parser
import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils
from models import build_model
from datasets import build_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--detr_model", default="./checkpoints/detr-r50.pth")
    p.add_argument("--vl_attn_type", default="mha", choices=["mha", "bdh"])
    extra = p.parse_args()

    # full TransVG defaults, then override for the Talk2Car overfit run
    args = get_args_parser().parse_args([])
    args.dataset = "talk2car"
    args.data_root = "./data"
    args.split_root = "./data"
    args.max_query_len = 20
    args.batch_size = 8
    args.device = "cuda"
    args.distributed = False
    args.vl_attn_type = extra.vl_attn_type
    args.detr_model = extra.detr_model

    device = torch.device("cuda")
    model = build_model(args).to(device)

    # load the DETR visual weights exactly as train.py does
    if os.path.isfile(args.detr_model):
        ckpt = torch.load(args.detr_model, map_location="cpu")
        missing, unexpected = model.visumodel.load_state_dict(ckpt["model"], strict=False)
        print(f"[detr] loaded {args.detr_model}: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[detr] WARNING: {args.detr_model} not found, visual branch is random-init")

    # one fixed batch of 8 training samples (augmentation applied once, then frozen)
    ds = build_dataset("train", args)
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=utils.collate_fn, num_workers=2)
    img_data, text_data, target = next(iter(loader))
    img_data = img_data.to(device)
    text_data = text_data.to(device)
    target = target.to(device)
    print(f"[batch] fixed batch of {target.shape[0]} samples; target boxes (norm cxcywh):")
    for i in range(target.shape[0]):
        print("   ", [round(float(v), 3) for v in target[i]])

    opt = torch.optim.AdamW(model.parameters(), lr=extra.lr)
    model.train()
    print(f"\n[overfit] {extra.steps} steps on the SAME 8 samples, lr={extra.lr}, "
          f"attn={extra.vl_attn_type}")
    for step in range(extra.steps):
        out = model(img_data, text_data)
        ld = loss_utils.trans_vg_loss(out, target)
        loss = sum(ld.values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == extra.steps - 1:
            with torch.no_grad():
                miou, accu = eval_utils.trans_vg_eval_val(out.detach(), target)
            print(f"  step {step:4d}  loss {loss.item():.4f}  "
                  f"bbox {float(ld['loss_bbox']):.4f}  giou {float(ld['loss_giou']):.4f}  "
                  f"miou {float(miou.mean()):.3f}  accu {float(accu):.3f}")

    print("\nVERDICT: if accu climbed toward ~1.0, the plumbing is correct and the "
          "full-run flat loss is a recipe/scale issue (batch/aug/lr/data size). If "
          "accu stayed ~0, there is a real bug in forward/loss/target.")


if __name__ == "__main__":
    main()
