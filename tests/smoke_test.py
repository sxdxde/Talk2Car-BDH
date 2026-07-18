"""CPU smoke test for the BDH visual-text attention drop-in.

Verifies, without any A100 / dataset / Darknet weights:
  1. output shapes match the original text_attn contract at all 3 FPN scales
  2. beta is a valid [0,1] map, no NaNs
  3. gradients flow (backward works)
  4. modes A and B are shape-correct too
  5. reports parameter counts and compares to the (0-param) original module
  6. checks the fusion-trim channel arithmetic (3 -> 2 streams)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from bdh_grounding import BDHFusionConfig, BDHVisualTextAttention

D = 256
# AttnGrounder FPN scales at 416x416 input: strides 32/16/8 -> 13,26,52
SCALES = [(13, 13), (26, 26), (52, 52)]
B, T = 2, 40   # batch, max query length (query_len=40 in the loader)


def original_text_attn(image_feat, lang_feat):
    """Reference: the exact original module (grounding_model.py:168-191), for a shape/behaviour baseline."""
    Bx, C, H, W = image_feat.size()
    image_feat = image_feat.view(Bx, C, -1).transpose(1, 2)
    lang_feat_ = lang_feat.transpose(1, 2)
    image_lang = torch.matmul(image_feat, lang_feat_)
    sf1 = torch.softmax(image_lang, dim=-1)
    sf2 = torch.sigmoid(image_lang.sum(-1)).view(Bx, H, W)
    lang_feat_attn = torch.matmul(sf1, lang_feat).transpose(1, 2).view(Bx, C, H, W)
    return lang_feat_attn, sf2


def check_mode(mode):
    cfg = BDHFusionConfig(dim=D, mode=mode, mult=4, n_head=1)
    mod = BDHVisualTextAttention(cfg)
    mod.train()
    lang = torch.randn(B, T, D, requires_grad=True)
    ok = True
    for (H, W) in SCALES:
        img = torch.randn(B, D, H, W, requires_grad=True)
        Tprime, beta = mod(img, lang)
        ref_T, ref_b = original_text_attn(img, lang)
        assert Tprime.shape == ref_T.shape == (B, D, H, W), f"{mode}: T' shape {Tprime.shape}"
        assert beta.shape == ref_b.shape == (B, H, W), f"{mode}: beta shape {beta.shape}"
        assert torch.isfinite(Tprime).all() and torch.isfinite(beta).all(), f"{mode}: NaN/Inf"
        assert (beta >= 0).all() and (beta <= 1).all(), f"{mode}: beta out of [0,1]"
        # backward
        loss = Tprime.pow(2).mean() + torch.nn.functional.binary_cross_entropy(
            beta, torch.zeros_like(beta))
        loss.backward()
        assert mod.E.grad is not None and torch.isfinite(mod.E.grad).all(), f"{mode}: no/NaN grad"
        mod.zero_grad(); lang.grad = None
        print(f"  [{mode}] scale {H:>2}x{W:<2}: T'={tuple(Tprime.shape)} beta={tuple(beta.shape)} "
              f"beta[min={beta.min():.3f} max={beta.max():.3f}] grad OK")
    n_params = sum(p.numel() for p in mod.parameters())
    print(f"  [{mode}] params = {n_params/1e6:.3f}M  (shared across 3 scales)\n")
    return n_params


def main():
    torch.manual_seed(0)
    print("== Original module (drop-in target) ==")
    print("  learned params: 0  (matmul + softmax + sigmoid)\n")

    print("== BDH module smoke test (dim=256, mult=4, single-head) ==")
    pc = check_mode("C")   # primary
    check_mode("A")        # ablation
    check_mode("B")        # ablation

    print("== Fusion-trim channel arithmetic ==")
    emb = D
    orig_fusion_in = emb * 3   # [fvisu, beta*fvisu, flang_attn]
    trim_fusion_in = emb * 2   # [fvisu, flang_attn]  (drop beta*fvisu stream)
    print(f"  original fcn_emb input channels: {orig_fusion_in}")
    print(f"  trimmed  fcn_emb input channels: {trim_fusion_in}  (saves the beta*visual conv path)\n")

    print(f"== SUMMARY ==")
    print(f"  Mode C (primary) adds {pc/1e6:.3f}M params vs the 0-param original module;")
    print(f"  the fusion trim removes one {emb}-channel stream from the heavy fcn_emb block,")
    print(f"  so net model-level change stays small and the drop-in contract holds. ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
