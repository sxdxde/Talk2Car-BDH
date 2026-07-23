"""Diagnose why the TransVG baseline on Talk2Car collapses to the mean box
(flat loss, ~1% accu). RUNS ON REMOTE (needs the converted .pth + images).

The mha baseline should reach ~65% -- getting ~1% with dead-flat training loss
means the visual pathway sees no signal correlating with the target box. This
script checks the two things that can cause that in a data port:

  1. box<->image alignment: is each stored xywh box actually inside its image,
     with a sane area? (a misaligned/out-of-bounds box -> garbage target)
  2. the END-TO-END target the model is trained against: run the real val
     transform and confirm the normalized cxcywh box lands in [0,1] and is
     non-degenerate. A target that clamps to a corner / collapses to ~0 area
     for most samples explains a mean-box collapse.

It also re-reads the ORIGINAL AttnGrounder split to confirm our converter
preserved values exactly.

Usage (from the TransVG root on remote):
    python arch3/diagnose_talk2car_transvg.py \
        --transvg-val  ./data/talk2car/talk2car_val.pth \
        --attn-val     ~/BDH/Talk2Car/AttnGrounder/ln_data/talk2car_val.pth \
        --imdir        ./data/talk2car/images \
        --imsize 640 --n 8
"""
import os
import sys
import argparse

# This script lives in arch3/, so the TransVG root isn't on sys.path by default
# and `import datasets.transforms` would grab the HuggingFace `datasets` package
# from site-packages instead of TransVG's local datasets/ dir. Put the TransVG
# root (arch3/..) first so the local package wins.
_TRANSVG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRANSVG_ROOT)

import numpy as np
import torch
from PIL import Image

import datasets.transforms as T   # local TransVG transforms (no BERT dep)


def summarize_boxes(entries, imdir, imsize, tag):
    """Whole-split stats on box validity + the end-to-end normalized target."""
    val_tf = T.Compose([
        T.RandomResize([imsize]),
        T.ToTensor(),
        T.NormalizeAndPad(size=imsize),
    ])
    n = len(entries)
    n_oob = 0            # box out of image bounds
    n_tiny = 0           # box area < 0.1% of image
    n_bad_norm = 0       # normalized target outside [0,1] or degenerate
    areas = []
    norm_c = []          # normalized center coords, to see if targets are diverse
    missing_img = 0
    for e in entries:
        # TransVG-format 5-tuple (img_file, _, bbox_xywh, phrase, _)
        img_file, _, bbox, phrase, _ = e
        x, y, w, h = [float(v) for v in bbox]
        path = os.path.join(imdir, img_file)
        if not os.path.isfile(path):
            missing_img += 1
            continue
        im = Image.open(path).convert("RGB")
        W, H = im.width, im.height
        # bounds / area check on RAW stored xywh
        if x < 0 or y < 0 or x + w > W + 1 or y + h > H + 1:
            n_oob += 1
        frac = (w * h) / float(W * H)
        areas.append(frac)
        if frac < 0.001:
            n_tiny += 1
        # end-to-end normalized target the model actually trains on
        box_xyxy = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)
        out = val_tf({"img": im, "box": box_xyxy, "text": phrase})
        nb = out["box"]  # normalized cxcywh
        norm_c.append((float(nb[0]), float(nb[1])))
        if (nb < -1e-3).any() or (nb > 1 + 1e-3).any() or float(nb[2]) < 1e-4 or float(nb[3]) < 1e-4:
            n_bad_norm += 1
    areas = np.array(areas) if areas else np.array([0.0])
    norm_c = np.array(norm_c) if norm_c else np.zeros((1, 2))
    print(f"\n==== {tag}: {n} entries ====")
    print(f"  missing images         : {missing_img}")
    print(f"  box out of bounds      : {n_oob}  ({100*n_oob/max(n,1):.1f}%)")
    print(f"  tiny boxes (<0.1% area): {n_tiny}  ({100*n_tiny/max(n,1):.1f}%)")
    print(f"  bad normalized target  : {n_bad_norm}  ({100*n_bad_norm/max(n,1):.1f}%)  <-- should be ~0")
    print(f"  box area frac: min={areas.min():.5f} mean={areas.mean():.5f} max={areas.max():.5f}")
    print(f"  norm center cx: mean={norm_c[:,0].mean():.3f} std={norm_c[:,0].std():.3f}  "
          f"cy: mean={norm_c[:,1].mean():.3f} std={norm_c[:,1].std():.3f}")
    print(f"    (low std here => targets barely vary => nothing for the model to learn)")


def show_samples(tv_entries, attn_entries, imdir, imsize, k):
    val_tf = T.Compose([T.RandomResize([imsize]), T.ToTensor(), T.NormalizeAndPad(size=imsize)])
    # index AttnGrounder entries by (img_file, phrase) for a fidelity cross-check
    attn_idx = {}
    for a in attn_entries:
        if len(a) == 3:
            af, ab, ap = a
            attn_idx[(af, ap)] = ab
    print("\n==== sample entries (raw xywh -> image dims -> normalized cxcywh) ====")
    for e in tv_entries[:k]:
        img_file, _, bbox, phrase, _ = e
        x, y, w, h = [float(v) for v in bbox]
        im = Image.open(os.path.join(imdir, img_file)).convert("RGB")
        W, H = im.width, im.height
        box_xyxy = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)
        nb = val_tf({"img": im, "box": box_xyxy.clone(), "text": phrase})["box"]
        src = attn_idx.get((img_file, phrase))
        match = "OK" if (src is not None and [float(v) for v in src] == [x, y, w, h]) else "MISMATCH/absent"
        print(f"  {img_file}  imgWH=({W},{H})")
        print(f"    xywh={[round(x,1),round(y,1),round(w,1),round(h,1)]}  "
              f"norm_cxcywh={[round(float(v),3) for v in nb]}  converter={match}")
        print(f"    phrase={phrase!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transvg-val", required=True)
    ap.add_argument("--attn-val", required=True)
    ap.add_argument("--imdir", required=True)
    ap.add_argument("--imsize", type=int, default=640)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    tv = torch.load(a.transvg_val, weights_only=False)
    attn = torch.load(a.attn_val, weights_only=False)
    print(f"loaded {len(tv)} TransVG-format val entries, {len(attn)} AttnGrounder val entries")

    show_samples(tv, attn, a.imdir, a.imsize, a.n)
    summarize_boxes(tv, a.imdir, a.imsize, "TransVG val split")


if __name__ == "__main__":
    main()
