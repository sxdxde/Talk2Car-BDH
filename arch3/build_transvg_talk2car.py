"""Convert our Talk2Car split files into TransVG's expected format.

RUNS ON THE REMOTE (that's where the Talk2Car .pth files + images live).

Our AttnGrounder-format split file `talk2car_{split}.pth` is a list of
    (img_file, bbox_xywh, phrase)
tuples (see external/AttnGrounder/dataset/talk2car_loader.py :: pull_item,
which reads exactly this and does bbox[2],bbox[3] = x+w, y+h).

TransVG's refcoco-family loader expects a list of
    (img_file, _, bbox_xywh, phrase, attri)
5-tuples (see external/TransVG/datasets/data_loader.py :: pull_item, which
for non-referit/flickr datasets does the SAME xywh -> x1y1x2y2 conversion).

So the conversion is a straight repack: same image filenames, same bbox
convention (xywh, untouched), same phrase strings. The two unused slots (`_`
and `attri`) are filled with None. This is intentionally a no-op on the
actual values -- it only changes the tuple arity so TransVG's unpacker is
happy -- which is what makes this a low-risk, faithful cross-architecture
port (the DATA the two models see is identical; only the architecture and
the text tokenizer differ).

Only train/val are converted by default: Talk2Car's public test set has no
GT (AIcrowd eval server closed 2020), so val is the evaluation split, exactly
as in the AttnGrounder phase.

Usage (on remote):
    python arch3/build_transvg_talk2car.py \
        --src-root  ~/BDH/Talk2Car/AttnGrounder/ln_data \
        --out-root  ~/BDH/Talk2Car/TransVG/data \
        --splits train val
Then symlink the images so TransVG's loader finds them:
    ln -s ~/BDH/Talk2Car/AttnGrounder/ln_data/images \
          ~/BDH/Talk2Car/TransVG/data/talk2car/images
"""
import os
import argparse

import torch


def _to_xywh_list(bbox):
    """Coerce a stored bbox (list / np.ndarray / tensor) to a plain python
    list of 4 numbers, WITHOUT changing the xywh convention."""
    if hasattr(bbox, "tolist"):
        bbox = bbox.tolist()
    bbox = list(bbox)
    if len(bbox) != 4:
        raise ValueError(f"expected a 4-element bbox, got {bbox!r}")
    return [float(v) for v in bbox]


def convert_split(src_path, out_path):
    entries = torch.load(src_path, weights_only=False)
    out = []
    n_skipped = 0
    for e in entries:
        if len(e) == 3:
            img_file, bbox, phrase = e
        elif len(e) == 2:
            # test-style (img_file, phrase) -- no GT box, cannot train/eval AP
            n_skipped += 1
            continue
        else:
            raise ValueError(
                f"unexpected entry arity {len(e)} in {src_path}: {e!r}")
        out.append((img_file, None, _to_xywh_list(bbox), phrase, None))
    torch.save(out, out_path)
    return len(out), n_skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True,
                    help="dir containing our talk2car_{split}.pth files")
    ap.add_argument("--out-root", required=True,
                    help="TransVG split_root; files go to <out-root>/talk2car/")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    src_root = os.path.expanduser(args.src_root)
    out_dir = os.path.join(os.path.expanduser(args.out_root), "talk2car")
    os.makedirs(out_dir, exist_ok=True)

    for split in args.splits:
        src_path = os.path.join(src_root, f"talk2car_{split}.pth")
        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"source split not found: {src_path}")
        out_path = os.path.join(out_dir, f"talk2car_{split}.pth")
        n, n_skipped = convert_split(src_path, out_path)
        msg = f"[{split}] wrote {n} entries -> {out_path}"
        if n_skipped:
            msg += f"  (skipped {n_skipped} GT-less test-style entries)"
        print(msg)

    print("\nDone. Remember to make the images visible to TransVG's loader, e.g.:")
    print(f"    ln -s <talk2car images dir> {os.path.join(out_dir, 'images')}")


if __name__ == "__main__":
    main()
