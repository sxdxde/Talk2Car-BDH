"""Step 6 evaluation: AP50 split into ambiguous vs unambiguous scenes.

Runs on the REMOTE (needs GPU + trained checkpoint + scene_index JSON from
build_scene_index.py). Source-agnostic: any scene_index with the schema
{img: {"n_same_class_others": int}} works.

    ambiguous   = images where >=1 OTHER object shares the referred object's class
    unambiguous = images where the referred object is the only one of its class

Usage (remote, from AttnGrounder root with train.py + bdh_grounding on path):
    CUDA_VISIBLE_DEVICES=0 python analysis/stratified_eval.py \
        --config configs/full_a100.yaml \
        --resume checkpoints/full/bdh_best.pth.tar \
        --scene-index analysis/scene_index_val.json
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE: heavy imports (torch, AttnGrounder dataset/model, train.py) are done lazily
# inside run() so this module's pure logic (stratify) is importable/testable without them.


def stratify(hits_by_img, scene_index):
    """hits_by_img: {img_file: 1/0 (IoU>0.5)}. Returns AP50 for all/ambiguous/unambiguous."""
    buckets = {"all": [], "ambiguous": [], "unambiguous": [], "unknown": []}
    for img, hit in hits_by_img.items():
        buckets["all"].append(hit)
        info = scene_index.get(img)
        n = info.get("n_same_class_others") if info else None
        if n is None:
            buckets["unknown"].append(hit)
        elif n >= 1:
            buckets["ambiguous"].append(hit)
        else:
            buckets["unambiguous"].append(hit)
    def ap50(xs):
        return (100.0 * np.mean(xs)) if xs else float("nan")
    return {k: (ap50(v), len(v)) for k, v in buckets.items()}


def run(config, resume, scene_index_path):
    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor, Normalize
    from dataset.talk2car_loader import Talk2CarDataset
    from utils.utils import bbox_iou
    from bdh_grounding import pipeline as P
    import train as T  # reuse build_model / decode_pred_boxes / reshape_anchor

    torch.set_grad_enabled(False)
    cfg = P.load_config(config)
    args = P.config_to_args(cfg)
    device = P.resolve_device(args.device)
    anchors_full = P.anchors_full_from_list(args.anchors)
    scene_index = json.load(open(scene_index_path))

    tf = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # FULL val set (no subset), deterministic order so global index -> img_file
    val_ds = Talk2CarDataset(data_root=args.data_root, split=args.eval_split, imsize=args.size,
                             transform=tf, max_query_len=args.time)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=8, pin_memory=(device.type == "cuda"))

    corpus = val_ds.corpus
    model = T.build_model(args, corpus).to(device)
    if resume and os.path.isfile(resume):
        P.load_checkpoint(resume, model, map_location=device)
        print(f"[loaded] {resume}")
    model.eval()

    hits_by_img = {}
    for batch_idx, (imgs, ob8, ob16, ob32, word_id, bbox) in enumerate(loader):
        imgs, word_id = imgs.to(device), word_id.to(device)
        bbox = torch.clamp(bbox.to(device), min=0, max=args.size - 1)
        with P.autocast_ctx(device, args.precision):
            pred_anchor, _ = model(imgs, word_id)
        pred_anchor = T.reshape_anchor(pred_anchor)
        pred_box = T.decode_pred_boxes(pred_anchor, anchors_full, args, device)
        iou = bbox_iou(pred_box, bbox.data.cpu(), x1y1x2y2=True).data.cpu().numpy()
        for i in range(imgs.size(0)):
            gidx = batch_idx * args.batch_size + i
            img_file = val_ds.images[gidx][0]
            hits_by_img[img_file] = int(iou[i] > 0.5)

    res = stratify(hits_by_img, scene_index)
    print("\n==================== Step 6: stratified AP50 ====================")
    print(f"  variant={args.variant} mode={args.bdh_mode} eval_split={args.eval_split}")
    for k in ("all", "ambiguous", "unambiguous", "unknown"):
        ap, n = res[k]
        print(f"  {k:12s}: AP50 = {ap:6.2f}   (n={n})")
    print("================================================================")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", required=True)
    ap.add_argument("--scene-index", required=True)
    a = ap.parse_args()
    run(a.config, a.resume, a.scene_index)
