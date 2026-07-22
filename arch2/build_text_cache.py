"""Offline text-encoder cache builder: extracts frozen DistilBERT-base
per-token embeddings for every Talk2Car phrase (+ its left/right-swapped
augmentation variant), caches to disk. Run ONCE on remote (needs the real
ln_data/ .pth files + `pip install transformers`) before any
--text-encoder distilbert training/eval run.

Why cache both the phrase and its swapped variant: talk2car_loader.py's
augmentation path does a horizontal-flip-triggered left/right swap
(phrase.replace('right',...).replace('left','right')...) on ~50% of
augmented training samples -- the cache must contain both forms, in
lockstep with the loader's own replace-chain, or training hits
cache-misses on swapped phrases.

DistilBERT's own [CLS]/[SEP] special tokens are kept in the cached
sequence (not stripped) -- the fusion module (BDH or baseline attention)
treats every position uniformly, so this needs no special-casing
downstream, at the cost of 2 of the 40 query_len slots.

Usage (remote, from AttnGrounder root):
    pip install transformers
    python arch2/build_text_cache.py \
        --data-root ln_data \
        --out arch2/text_cache_distilbert.pth
"""
import os
import argparse

import torch


def lr_swap(phrase):
    """Exact same replace-chain as talk2car_loader.py's augmentation path,
    kept in lockstep so cache keys match what training will look up."""
    return (phrase.replace('right', '*&^special^&*')
                  .replace('left', 'right')
                  .replace('*&^special^&*', 'left'))


def load_phrases(data_root, split):
    path = os.path.join(data_root, f"talk2car_{split}.pth")
    images = torch.load(path, weights_only=False)
    phrases = set()
    for entry in images:
        if len(entry) == 3:        # (img_file, bbox, phrase) -- train/val
            _, _, phrase = entry
        else:                       # (img_file, phrase) -- test (testmode)
            _, phrase = entry
        phrase = phrase.lower()
        phrases.add(phrase)
        phrases.add(lr_swap(phrase))
    return phrases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="dir containing talk2car_{train,val,test}.pth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--query-len", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModel

    all_phrases = set()
    for split in ("train", "val", "test"):
        p = os.path.join(args.data_root, f"talk2car_{split}.pth")
        if os.path.isfile(p):
            before = len(all_phrases)
            all_phrases |= load_phrases(args.data_root, split)
            print(f"  {split}: +{len(all_phrases) - before} -> cumulative {len(all_phrases)} unique phrases")
        else:
            print(f"  {split}: {p} not found, skipping")

    print(f"[build_text_cache] {len(all_phrases)} unique phrases total (incl. left/right-swapped variants)")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    cache = {}
    phrases = sorted(all_phrases)
    with torch.no_grad():
        for i in range(0, len(phrases), args.batch_size):
            batch = phrases[i:i + args.batch_size]
            enc = tok(batch, padding="max_length", truncation=True,
                      max_length=args.query_len, return_tensors="pt")
            out = model(**enc).last_hidden_state            # (B, query_len, 768)
            mask = enc["attention_mask"].unsqueeze(-1)       # (B, query_len, 1)
            out = (out * mask).half()                        # zero padded rows, halve storage
            for j, phrase in enumerate(batch):
                cache[phrase] = out[j].clone()
            done = min(i + args.batch_size, len(phrases))
            if (i // args.batch_size + 1) % 20 == 0 or done == len(phrases):
                print(f"  processed {done}/{len(phrases)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(cache, args.out)
    print(f"[done] {len(cache)} embeddings -> {args.out}")


if __name__ == "__main__":
    main()
