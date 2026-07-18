"""End-to-end wiring smoke test for the BDH swap + 2-stream fusion trim.

Runs the fusion -> YOLO-head path on CPU with a MOCK backbone (no Darknet weights,
no dataset), proving the trimmed 2-stream fusion produces the correct YOLO output
shape (B, 3*5, H, W) at all 3 FPN scales with the BDH module in the loop.
Mirrors grounding_model.forward's fusion/fcn blocks (fcn_emb: 2*emb->emb, fcn_out: emb->15).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from bdh_grounding import BDHFusionConfig, BDHVisualTextAttention

D = 256
SCALES = [(13, 13), (26, 26), (52, 52)]
B, T = 2, 40


def cbr(cin, cout, k, p):
    return nn.Sequential(nn.Conv2d(cin, cout, k, 1, p), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class MiniHead(nn.Module):
    """Mimics grounding_model's fcn_emb (2*emb->emb) + fcn_out (emb->3*5) for the trimmed fusion."""
    def __init__(self, emb):
        super().__init__()
        self.fcn_emb = nn.Sequential(cbr(emb * 2, emb, 1, 0), cbr(emb, emb, 3, 1), cbr(emb, emb, 1, 0))
        self.fcn_out = nn.Sequential(cbr(emb, emb // 2, 1, 0), nn.Conv2d(emb // 2, 3 * 5, 1))

    def forward(self, fused):
        return self.fcn_out(self.fcn_emb(fused))


def main():
    torch.manual_seed(0)
    bdh = BDHVisualTextAttention(BDHFusionConfig(dim=D, mode="C", mult=4, n_head=1))
    head = MiniHead(D)

    lang = torch.randn(B, T, D)                       # mock BiLSTM+proj output (post mapping_lang)
    print("== Integration smoke: BDH swap + 2-stream fusion -> YOLO head ==")
    total_ok = True
    for (H, W) in SCALES:
        fvisu = torch.randn(B, D, H, W)               # mock backbone+mapping_visu output
        flang_attn, beta = bdh(fvisu, lang)           # BDH module (drop-in)

        # trimmed 2-stream fusion (Edit 3): [fvisu, flang_attn], l2-normalized
        f1 = F.normalize(fvisu, p=2, dim=1)
        f2 = F.normalize(flang_attn, p=2, dim=1)
        fused = torch.cat([f1, f2], dim=1)            # (B, 2*emb, H, W)
        assert fused.shape == (B, 2 * D, H, W)

        out = head(fused)                             # (B, 15, H, W) YOLO: 3 anchors * (x,y,w,h,conf)
        assert out.shape == (B, 3 * 5, H, W), out.shape
        assert beta.shape == (B, H, W)
        assert torch.isfinite(out).all() and torch.isfinite(beta).all()
        print(f"  scale {H:>2}x{W:<2}: fused={tuple(fused.shape)} -> yolo_out={tuple(out.shape)} "
              f"beta={tuple(beta.shape)} OK")

    print("\n  Trimmed fusion wiring verified end-to-end (mock backbone). "
          "Ready to apply INTEGRATION.md edits on the remote where Darknet weights exist.")


if __name__ == "__main__":
    main()
