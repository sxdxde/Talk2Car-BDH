"""Full metrics table for the AttnGrounder-phase comparison: params, layer
count, GFLOPs, inference-ms, FPS -- for baseline and any BDH variant.

Runs on the REMOTE (needs GPU + a trained checkpoint + Darknet weights).

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis/measure_metrics.py \
        --config configs/full_a100.yaml --variant baseline \
        --resume checkpoints/full/baseline_best.pth.tar

    CUDA_VISIBLE_DEVICES=0 python analysis/measure_metrics.py \
        --config configs/full_a100.yaml --variant bdh --bdh-mode A \
        --resume checkpoints/full/bdh_A_best.pth.tar

GFLOPs needs `pip install fvcore` (not yet confirmed installed on remote).
Measured against a single image at fp32 (GFLOPs is an architecture
property, precision-independent) -- separate from inference-ms, which
DOES use the real training precision (bf16 autocast) since that's about
actual measured wall-clock throughput, not architecture size. If fvcore
is missing, or fails on this codebase's custom Darknet layers (route/
shortcut/upsample), GFLOPs is reported as unavailable with the reason --
everything else (params, layers, inference-ms, FPS) is still reported
either way, this is intentionally the piece most likely to need a retry.
"""
import os
import sys
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE: count_layers is pure (no torch-heavy deps beyond what's already
# imported by the caller), kept import-light so it's locally unit-testable.


def count_layers(model):
    """Count parameterized 'leaf' modules by type -- unambiguous, avoids
    having to define 'depth' for a multi-branch/multi-scale architecture
    (3 FPN scales, multiple parallel heads)."""
    counts = Counter()
    total = 0
    for m in model.modules():
        if len(list(m.children())) == 0 and len(list(m.parameters(recurse=False))) > 0:
            counts[type(m).__name__] += 1
            total += 1
    return total, dict(counts)


def measure_gflops(model, imgs1, word_id1):
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None, "fvcore not installed (pip install fvcore)"
    try:
        model.eval()
        flops = FlopCountAnalysis(model, (imgs1, word_id1))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total = flops.total()
        # fvcore counts multiply-accumulates as 1 "flop" each (standard
        # convention in most reported papers, e.g. "GFLOPs" == GMACs here)
        return total / 1e9, None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:200]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", required=True)
    ap.add_argument("--variant", default=None, choices=["bdh", "baseline"])
    ap.add_argument("--bdh-mode", default=None, choices=["A", "B", "C"])
    ap.add_argument("--growing-scales", action="store_true")
    ap.add_argument("--text-encoder", default=None, choices=["glove", "distilbert"])
    ap.add_argument("--text-cache", default=None)
    ap.add_argument("--n-infer", type=int, default=50, help="batches for inference-ms timing")
    cli = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor, Normalize
    from dataset.talk2car_loader import Talk2CarDataset
    from bdh_grounding import pipeline as P
    import train as T

    torch.set_grad_enabled(False)
    cfg = P.load_config(cli.config)
    args = P.config_to_args(cfg)
    if cli.variant:
        args.variant = cli.variant
    if cli.bdh_mode:
        args.bdh_mode = cli.bdh_mode
    if cli.growing_scales:
        args.bdh_growing_scales = True
    if cli.text_encoder:
        args.text_encoder = cli.text_encoder
    if cli.text_cache:
        args.text_cache_path = cli.text_cache
    device = P.resolve_device(args.device)

    tf = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_ds = Talk2CarDataset(data_root=args.data_root, split=args.eval_split, imsize=args.size,
                             transform=tf, max_query_len=args.time,
                             text_encoder=args.text_encoder, text_cache_path=args.text_cache_path)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=4, pin_memory=(device.type == "cuda"))

    corpus = val_ds.corpus
    model = T.build_model(args, corpus).to(device)
    if not os.path.isfile(cli.resume):
        raise FileNotFoundError(f"--resume checkpoint not found: {cli.resume!r}")
    P.load_checkpoint(cli.resume, model, map_location=device)
    model.eval()
    print(f"[loaded] {cli.resume}")

    n_params = sum(p.numel() for p in model.parameters())
    n_layers, layer_counts = count_layers(model)

    imgs, _, _, _, word_id, _ = next(iter(loader))
    imgs1 = imgs[:1].to(device)
    word_id1 = word_id[:1].to(device)
    gflops, gflops_err = measure_gflops(model, imgs1, word_id1)

    ms = T.measure_inference_ms(model, loader, args, device, n=cli.n_infer)
    fps = 1000.0 / ms if ms == ms else float("nan")

    print("\n==================== Metrics ====================")
    print(f"  variant={args.variant} mode={args.bdh_mode} growing_scales={args.bdh_growing_scales} "
          f"text_encoder={args.text_encoder}")
    print(f"  params        : {n_params/1e6:.2f} M")
    print(f"  layers (total): {n_layers}")
    for k, v in sorted(layer_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {k:20s}: {v}")
    if gflops is not None:
        print(f"  GFLOPs (1 img): {gflops:.3f}")
    else:
        print(f"  GFLOPs        : unavailable ({gflops_err})")
    print(f"  inference-ms  : {ms:.2f}  (batch={args.batch_size}, n={cli.n_infer} batches, "
          f"precision={args.precision})")
    print(f"  FPS           : {fps:.2f}")
    print("===================================================")


if __name__ == "__main__":
    main()
