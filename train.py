"""Config-driven trainer/evaluator for AttnGrounder + BDH swap on Talk2Car.

RUNS ON THE REMOTE (GPU). Same code path for both profiles; only the YAML differs:
    python train.py --config configs/full_a100.yaml                 # full A100 training
    python train.py --config configs/smoke_local.yaml               # tiny CPU smoke
    python train.py --config configs/full_a100.yaml --eval-only \
        --resume checkpoints/full/bdh_best.pth.tar                  # AP50 + ms + params

Placement on the remote: copy this file + the `bdh_grounding/` package into the
AttnGrounder repo root (so `dataset/`, `model/`, `utils/` import cleanly), after
applying bdh_grounding/INTEGRATION.md to model/grounding_model.py.

Reuses AttnGrounder's dataset/model/utils verbatim. The loss and target-building
below mirror train_yolo.py exactly, with `.cuda()` -> `.to(device)` so the same
code runs on CPU (smoke) and A100 (full).
"""
import os
import sys
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

# make sibling bdh_grounding/ importable when this file sits in AttnGrounder root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.talk2car_loader import Talk2CarDataset
from model.grounding_model import grounding_model
from utils.utils import AverageMeter, bbox_iou, xywh2xyxy

from bdh_grounding import pipeline as P


# --------------------------------------------------------------------------- #
# loss + target (device-agnostic mirrors of train_yolo.py)
# --------------------------------------------------------------------------- #
def yolo_loss(pred, target, gi, gj, best_n_list, device, w_coord=5.0):
    mseloss = nn.MSELoss()
    celoss = nn.CrossEntropyLoss()
    batch = pred[0].size(0)
    pred_bbox = torch.zeros(batch, 4, device=device)
    gt_bbox = torch.zeros(batch, 4, device=device)
    for ii in range(batch):
        s = best_n_list[ii] // 3
        a = best_n_list[ii] % 3
        pred_bbox[ii, 0:2] = torch.sigmoid(pred[s][ii, a, 0:2, gj[ii], gi[ii]])
        pred_bbox[ii, 2:4] = pred[s][ii, a, 2:4, gj[ii], gi[ii]]
        gt_bbox[ii, :] = target[s][ii, a, :4, gj[ii], gi[ii]]
    loss_xywh = (mseloss(pred_bbox[:, 0], gt_bbox[:, 0]) + mseloss(pred_bbox[:, 1], gt_bbox[:, 1])
                 + mseloss(pred_bbox[:, 2], gt_bbox[:, 2]) + mseloss(pred_bbox[:, 3], gt_bbox[:, 3]))
    pred_conf = torch.cat([pred[s][:, :, 4, :, :].contiguous().view(batch, -1) for s in range(len(pred))], dim=1)
    gt_conf = torch.cat([target[s][:, :, 4, :, :].contiguous().view(batch, -1) for s in range(len(target))], dim=1)
    loss_conf = celoss(pred_conf, gt_conf.max(1)[1])
    return loss_xywh * w_coord + loss_conf


def build_target(raw_coord, pred, args, anchors_full, device):
    coord_list, bbox_list = [], []
    for s in range(len(pred)):
        grid = args.size // (32 // (2 ** s))
        coord = torch.zeros(raw_coord.size(0), raw_coord.size(1), device=device)
        coord[:, 0] = (raw_coord[:, 0] + raw_coord[:, 2]) / (2 * args.size)
        coord[:, 1] = (raw_coord[:, 1] + raw_coord[:, 3]) / (2 * args.size)
        coord[:, 2] = (raw_coord[:, 2] - raw_coord[:, 0]) / args.size
        coord[:, 3] = (raw_coord[:, 3] - raw_coord[:, 1]) / args.size
        coord = coord * grid
        coord_list.append(coord)
        bbox_list.append(torch.zeros(coord.size(0), 3, 5, grid, grid))

    batch = raw_coord.size(0)
    best_n_list, best_gi, best_gj = [], [], []
    for ii in range(batch):
        anch_ious = []
        for s in range(len(pred)):
            grid = args.size // (32 // (2 ** s))
            gw, gh = coord_list[s][ii, 2], coord_list[s][ii, 3]
            anchor_idxs = [x + 3 * s for x in (0, 1, 2)]
            scaled = [(anchors_full[i][0] / (args.anchor_imsize / grid),
                       anchors_full[i][1] / (args.anchor_imsize / grid)) for i in anchor_idxs]
            gt_box = torch.FloatTensor(np.array([0, 0, float(gw), float(gh)])).unsqueeze(0)
            anchor_shapes = torch.FloatTensor(
                np.concatenate((np.zeros((len(scaled), 2)), np.array(scaled)), 1))
            anch_ious += list(bbox_iou(gt_box, anchor_shapes))
        best_n = int(np.argmax(np.array(anch_ious)))
        best_scale = best_n // 3
        grid = args.size // (32 // (2 ** best_scale))
        anchor_idxs = [x + 3 * best_scale for x in (0, 1, 2)]
        scaled = [(anchors_full[i][0] / (args.anchor_imsize / grid),
                   anchors_full[i][1] / (args.anchor_imsize / grid)) for i in anchor_idxs]
        gi = int(coord_list[best_scale][ii, 0].long())
        gj = int(coord_list[best_scale][ii, 1].long())
        tx = coord_list[best_scale][ii, 0] - gi
        ty = coord_list[best_scale][ii, 1] - gj
        gw, gh = coord_list[best_scale][ii, 2], coord_list[best_scale][ii, 3]
        tw = torch.log(gw / scaled[best_n % 3][0] + 1e-16)
        th = torch.log(gh / scaled[best_n % 3][1] + 1e-16)
        bbox_list[best_scale][ii, best_n % 3, :, gj, gi] = torch.stack(
            [tx.cpu(), ty.cpu(), tw.cpu(), th.cpu(), torch.ones(1).squeeze()])
        best_n_list.append(best_n); best_gi.append(gi); best_gj.append(gj)
    bbox_list = [b.to(device) for b in bbox_list]
    return bbox_list, best_gi, best_gj, best_n_list


# --------------------------------------------------------------------------- #
# model builder (variant-aware; assumes INTEGRATION.md applied)
# --------------------------------------------------------------------------- #
def build_model(args, corpus):
    if args.variant == "bdh":
        # requires bdh_grounding/INTEGRATION.md applied to grounding_model.py
        return grounding_model(corpus=corpus, emb_size=args.emb_size, variant="bdh",
                               bdh_mode=args.bdh_mode, bdh_mult=args.bdh_mult,
                               bdh_n_head=args.bdh_n_head, bdh_dropout=args.bdh_dropout,
                               bdh_share_qv_encoder=args.bdh_share_qv_encoder)
    # baseline: call the STOCK grounding_model — no INTEGRATION.md edits needed,
    # so the reproduction / author-checkpoint eval runs against the unmodified repo.
    return grounding_model(corpus=corpus, emb_size=args.emb_size)


def make_optimizer(model, args):
    visu = list(model.visumodel.parameters())
    visu_ids = {id(p) for p in visu}
    rest = [p for p in model.parameters() if id(p) not in visu_ids]
    groups = [{"params": rest}, {"params": visu, "lr": args.lr / args.backbone_lr_divisor}]
    if args.optimizer == "adam":
        return torch.optim.Adam(groups, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=args.lr, momentum=0.99)
    return torch.optim.RMSprop(groups, lr=args.lr, weight_decay=args.weight_decay)


def adjust_lr(optimizer, epoch, args):
    lr = args.lr * ((1 - float(epoch) / max(1, args.nb_epoch)) ** args.power)
    optimizer.param_groups[0]["lr"] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]["lr"] = lr / args.backbone_lr_divisor


# --------------------------------------------------------------------------- #
# decode predicted boxes (shared by train-eval and validate)
# --------------------------------------------------------------------------- #
def decode_pred_boxes(pred_anchor, anchors_full, args, device):
    # local rebind only (doesn't mutate the caller's list/tensors): numpy() has no
    # bfloat16 support, and under bf16 autocast pred_anchor arrives as bf16.
    pred_anchor = [p.float() for p in pred_anchor]
    batch = pred_anchor[0].size(0)
    pred_conf_list = [pred_anchor[s][:, :, 4, :, :].contiguous().view(batch, -1) for s in range(len(pred_anchor))]
    pred_conf = torch.cat(pred_conf_list, dim=1)
    max_conf, max_loc = torch.max(pred_conf, dim=1)
    pred_bbox = torch.zeros(batch, 4)
    pred_gi, pred_gj = [], []
    for ii in range(batch):
        if max_loc[ii] < 3 * (args.size // 32) ** 2:
            best_scale = 0
        elif max_loc[ii] < 3 * (args.size // 32) ** 2 + 3 * (args.size // 16) ** 2:
            best_scale = 1
        else:
            best_scale = 2
        grid, grid_size = args.size // (32 // (2 ** best_scale)), 32 // (2 ** best_scale)
        anchor_idxs = [x + 3 * best_scale for x in (0, 1, 2)]
        scaled = [(anchors_full[i][0] / (args.anchor_imsize / grid),
                   anchors_full[i][1] / (args.anchor_imsize / grid)) for i in anchor_idxs]
        conf_s = pred_conf_list[best_scale].view(batch, 3, grid, grid).data.cpu().numpy()
        (bn, gj, gi) = np.where(conf_s[ii] == max_conf.data.cpu().numpy()[ii])
        bn, gi, gj = int(bn[0]), int(gi[0]), int(gj[0])
        pred_gi.append(gi); pred_gj.append(gj)
        pred_bbox[ii, 0] = torch.sigmoid(pred_anchor[best_scale][ii, bn, 0, gj, gi]) + gi
        pred_bbox[ii, 1] = torch.sigmoid(pred_anchor[best_scale][ii, bn, 1, gj, gi]) + gj
        pred_bbox[ii, 2] = torch.exp(pred_anchor[best_scale][ii, bn, 2, gj, gi]) * scaled[bn][0]
        pred_bbox[ii, 3] = torch.exp(pred_anchor[best_scale][ii, bn, 3, gj, gi]) * scaled[bn][1]
        pred_bbox[ii, :] = pred_bbox[ii, :] * grid_size
    return xywh2xyxy(pred_bbox)


# --------------------------------------------------------------------------- #
# train / validate
# --------------------------------------------------------------------------- #
def reshape_anchor(pred_anchor):
    for ii in range(len(pred_anchor)):
        p = pred_anchor[ii]
        pred_anchor[ii] = p.view(p.size(0), 3, 5, p.size(2), p.size(3))
    return pred_anchor


def build_map_loss_fxn(args):
    if args.map_loss == "tversky_focal":
        return P.TverskyFocalLoss(
            tversky_alpha=args.tversky_alpha, tversky_beta=args.tversky_beta,
            tversky_eps=args.tversky_eps, focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha, lambda_tve=args.lambda_tve, lambda_foc=args.lambda_foc)
    return nn.BCELoss()


def train_epoch(loader, model, optimizer, epoch, args, anchors_full, device, logger, global_step):
    model.train()
    map_loss_fxn = build_map_loss_fxn(args)
    losses, accm = AverageMeter(), AverageMeter()
    for batch_idx, (imgs, ob8, ob16, ob32, word_id, bbox) in enumerate(loader):
        # NOTE: loader yields object maps coarse->fine already; ob8/16/32 naming is
        # AttnGrounder's, order is [stride32(13), stride16(26), stride8(52)].
        imgs, word_id = imgs.to(device), word_id.to(device)
        obmap = [ob8.to(device).float(), ob16.to(device).float(), ob32.to(device).float()]
        bbox = torch.clamp(bbox.to(device), min=0, max=args.size - 1)

        with P.autocast_ctx(device, args.precision):
            pred_anchor, attn_map = model(imgs, word_id)
        gt_param, gi, gj, best_n_list = build_target(bbox, pred_anchor, args, anchors_full, device)
        pred_anchor = reshape_anchor(pred_anchor)

        bs = imgs.size(0)
        # BCELoss requires matching dtypes; the baseline path's beta lands in fp32
        # (torch.sum is on autocast's fp32-promotion list) but the BDH module's
        # beta (sigmoid(Linear(...))) stays in the autocast dtype (bf16) — cast
        # explicitly so both variants match obmap's explicit .float() above.
        map_loss = sum(map_loss_fxn(attn_map[k].float().view(bs, -1), obmap[k].view(bs, -1))
                       for k in range(len(attn_map)))
        loss = yolo_loss(pred_anchor, gt_param, gi, gj, best_n_list, device) + args.lambda_map * map_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), bs)

        with torch.no_grad():
            pred_box = decode_pred_boxes(pred_anchor, anchors_full, args, device)
            iou = bbox_iou(pred_box, bbox.data.cpu(), x1y1x2y2=True)
            accm.update(float(np.mean((iou.data.cpu().numpy() > 0.5))), bs)

        global_step += 1
        if global_step % args.log_every == 0:
            print(f"epoch {epoch} step {global_step} [{batch_idx}/{len(loader)}] "
                  f"loss {losses.avg:.4f} acc {accm.avg:.4f}")
            logger.log(phase="train", epoch=epoch, step=global_step,
                       loss=f"{losses.avg:.6f}", accu=f"{accm.avg:.6f}", ap50="")
        if args.ckpt_every and global_step % args.ckpt_every == 0:
            P.save_checkpoint(args.ckpt_dir, args.ckpt_tag, epoch, global_step, model, optimizer, -1)
        if args.max_steps and global_step >= args.max_steps:
            break
    return global_step


@torch.no_grad()
def validate(loader, model, args, anchors_full, device):
    model.eval()
    accm = AverageMeter()
    for (imgs, ob8, ob16, ob32, word_id, bbox) in loader:
        imgs, word_id = imgs.to(device), word_id.to(device)
        bbox = torch.clamp(bbox.to(device), min=0, max=args.size - 1)
        with P.autocast_ctx(device, args.precision):
            pred_anchor, _ = model(imgs, word_id)
        pred_anchor = reshape_anchor(pred_anchor)
        pred_box = decode_pred_boxes(pred_anchor, anchors_full, args, device)
        iou = bbox_iou(pred_box, bbox.data.cpu(), x1y1x2y2=True)
        accm.update(float(np.mean((iou.data.cpu().numpy() > 0.5))), imgs.size(0))
    return accm.avg  # AP50


@torch.no_grad()
def measure_inference_ms(model, loader, args, device, n=50):
    model.eval()
    it = iter(loader)
    times = []
    for _ in range(min(n, len(loader))):
        imgs, _, _, _, word_id, _ = next(it)
        imgs, word_id = imgs.to(device), word_id.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with P.autocast_ctx(device, args.precision):
            model(imgs, word_id)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000.0 / imgs.size(0))
    return float(np.mean(times)) if times else float("nan")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--variant", default=None, choices=["bdh", "baseline"],
                    help="override model.variant from the config")
    ap.add_argument("--bdh-mode", default=None, choices=["A", "B", "C"],
                    help="override model.bdh.mode from the config (bdh variant only)")
    ap.add_argument("--map-loss", default=None, choices=["bce", "tversky_focal"],
                    help="override model.map_loss.type from the config")
    cli = ap.parse_args()

    cfg = P.load_config(cli.config)
    args = P.config_to_args(cfg)
    if cli.resume:
        args.resume = cli.resume
    if cli.variant:
        args.variant = cli.variant
    if cli.bdh_mode:
        args.bdh_mode = cli.bdh_mode
    if cli.map_loss:
        args.map_loss = cli.map_loss
    # checkpoint tag must encode bdh_mode and map_loss too, otherwise runs
    # that differ only in these overwrite each other's checkpoints
    base_tag = args.variant if args.variant != "bdh" else f"bdh_{args.bdh_mode}"
    args.ckpt_tag = base_tag + ("_tve" if args.map_loss == "tversky_focal" else "")
    device = P.resolve_device(args.device)

    random.seed(args.seed); np.random.seed(args.seed + 1); torch.manual_seed(args.seed + 2)
    anchors_full = P.anchors_full_from_list(args.anchors)

    tf = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_ds = Talk2CarDataset(data_root=args.data_root, split="train", imsize=args.size,
                               transform=tf, max_query_len=args.time, augment=True)
    val_ds = Talk2CarDataset(data_root=args.data_root, split=args.eval_split, imsize=args.size,
                             transform=tf, max_query_len=args.time)
    train_ds = P.maybe_subset(train_ds, args.subset_size)
    val_ds = P.maybe_subset(val_ds, args.subset_size)
    corpus = getattr(train_ds, "corpus", getattr(getattr(train_ds, "dataset", None), "corpus", None))

    dl_kw = dict(pin_memory=(device.type == "cuda"), num_workers=0 if args.profile == "smoke" else 8)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True, **dl_kw)

    model = build_model(args, corpus).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] variant={args.variant} bdh_mode={args.bdh_mode} map_loss={args.map_loss} "
          f"ckpt_tag={args.ckpt_tag} emb_size={args.emb_size} params={n_params/1e6:.2f}M")

    optimizer = make_optimizer(model, args)
    start_epoch, global_step, best = 0, 0, -float("inf")
    if args.resume and os.path.isfile(args.resume):
        start_epoch, global_step, best = P.load_checkpoint(args.resume, model, optimizer, map_location=device)
        print(f"[resume] from {args.resume} epoch={start_epoch} step={global_step} best={best}")

    if cli.eval_only:
        ap50 = validate(val_loader, model, args, anchors_full, device)
        ms = measure_inference_ms(model, val_loader, args, device)
        print(f"[eval] AP50={ap50*100:.2f}  inference={ms:.1f}ms  params={n_params/1e6:.2f}M")
        return

    logger = P.CSVLogger(args.log_csv, fieldnames=["phase", "epoch", "step", "loss", "accu", "ap50"])
    for epoch in range(start_epoch, args.nb_epoch):
        adjust_lr(optimizer, epoch, args)
        global_step = train_epoch(train_loader, model, optimizer, epoch, args,
                                  anchors_full, device, logger, global_step)
        ap50 = validate(val_loader, model, args, anchors_full, device)
        is_best = ap50 > best
        best = max(best, ap50)
        print(f"[val] epoch {epoch} AP50={ap50*100:.2f} (best {best*100:.2f})")
        logger.log(phase="val", epoch=epoch, step=global_step, loss="", accu="", ap50=f"{ap50:.6f}")
        P.save_checkpoint(args.ckpt_dir, args.ckpt_tag, epoch + 1, global_step, model, optimizer, best, is_best)
        if args.max_steps and global_step >= args.max_steps:
            print("[smoke] reached max_steps; stopping"); break
    logger.close()
    print(f"[done] best AP50 = {best*100:.2f}")


if __name__ == "__main__":
    main()
