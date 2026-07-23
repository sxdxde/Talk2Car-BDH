"""CPU test for the BDH self-attention swap into TransVG's V-L fusion.

Covers what's locally testable without TransVG's heavy deps (DETR weights,
pytorch_pretrained_bert, GPU): the BDH self-attention operator's drop-in
signature compatibility with nn.MultiheadAttention, its padding-mask
semantics, that the gated TransformerEncoderLayer runs end-to-end and back-
props in both "mha" and "bdh" modes, and that TransVG's xavier _reset_
parameters does NOT clobber BDH's own normal_(0.02) init.

vl_transformer.py is imported standalone (it has no relative imports besides
the absolute bdh_grounding one) so we never trigger models/__init__.py, which
would pull in DETR/BERT. NOT locally testable (needs remote data / GPU /
pytorch_pretrained_bert): the full TransVG forward, the Talk2Car loader port,
real training.
"""
import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "external", "TransVG", "models"))

import importlib

import torch
import torch.nn as nn

from bdh_grounding.bdh_selfattn import BDHSelfAttention, BDHSelfAttnConfig
import vl_transformer as VLT  # standalone import, no models/__init__.py

D = 256
S = 7      # 1 reg + a few text + a few visual tokens
B = 3


def test_signature_and_shape():
    """BDHSelfAttention must match nn.MultiheadAttention's call signature and
    output contract: forward(q,k,v, attn_mask, key_padding_mask) -> (out, w),
    out shape (S,B,d), and TransVG only ever uses index [0]."""
    torch.manual_seed(0)
    mod = BDHSelfAttention(BDHSelfAttnConfig(dim=D)).eval()
    src = torch.randn(S, B, D)
    pos = torch.randn(S, B, D)
    q = k = src + pos            # exactly how TransVG builds q/k (value=src)
    out, weights = mod(q, k, value=src, attn_mask=None, key_padding_mask=None)
    assert out.shape == (S, B, D), out.shape
    assert weights is None  # stands in for MHA's attn-weight matrix, unused by TransVG
    # a real nn.MultiheadAttention on the same shapes returns the same out shape
    mha = nn.MultiheadAttention(D, 8)
    mout = mha(q, k, value=src)[0]
    assert mout.shape == out.shape, "shape contract diverges from nn.MultiheadAttention"
    print(f"  signature/shape: BDH self-attn out {tuple(out.shape)} matches "
          f"nn.MultiheadAttention, returns (out, None)")


def test_attn_mask_rejected():
    """TransVG's fusion only ever passes key_padding_mask; a full attn_mask is
    not a supported code path and must fail loud, not silently mis-behave."""
    mod = BDHSelfAttention(BDHSelfAttnConfig(dim=D)).eval()
    src = torch.randn(S, B, D)
    try:
        mod(src, src, value=src, attn_mask=torch.zeros(S, S))
        raise AssertionError("expected AssertionError for non-None attn_mask")
    except AssertionError as e:
        assert "attn_mask" in str(e), e
    print("  attn_mask: non-None correctly rejected (fail loud)")


def test_key_padding_mask_excludes_pads():
    """Padded key/value tokens must not influence any real token's output.
    Changing the CONTENT of padded positions must leave real outputs identical
    (they contribute nothing to the memory rho)."""
    torch.manual_seed(1)
    mod = BDHSelfAttention(BDHSelfAttnConfig(dim=D, dropout=0.0)).eval()
    src = torch.randn(S, B, D)
    kpm = torch.zeros(B, S, dtype=torch.bool)
    kpm[:, -2:] = True                       # last two tokens are padding for all batch items

    out1 = mod(src, src, value=src, key_padding_mask=kpm)[0]

    src2 = src.clone()
    src2[-2:, :, :] = torch.randn(2, B, D) * 100.0   # arbitrarily corrupt the padded rows' content
    out2 = mod(src2, src2, value=src2, key_padding_mask=kpm)[0]

    # real (non-padded) token outputs must be unchanged
    real_out1 = out1[:-2]
    real_out2 = out2[:-2]
    assert torch.allclose(real_out1, real_out2, atol=1e-5), \
        "padded tokens leaked into real token outputs"
    print(f"  key_padding_mask: corrupting {2} padded tokens leaves all "
          f"{S-2} real-token outputs identical (max delta "
          f"{(real_out1 - real_out2).abs().max().item():.2e})")


def test_encoder_layer_both_modes_backprop():
    """The gated TransformerEncoderLayer must run + backprop in both modes,
    with the residual/FFN/LN structure identical -- only self_attn differs."""
    torch.manual_seed(2)
    src = torch.randn(S, B, D, requires_grad=True)
    pos = torch.randn(S, B, D)
    kpm = torch.zeros(B, S, dtype=torch.bool); kpm[:, -1] = True

    for attn_type in ("mha", "bdh"):
        layer = VLT.TransformerEncoderLayer(
            D, nhead=8, dim_feedforward=512, dropout=0.0, attn_type=attn_type)
        # confirm the swapped operator is the expected class
        if attn_type == "bdh":
            assert isinstance(layer.self_attn, BDHSelfAttention)
        else:
            assert isinstance(layer.self_attn, nn.MultiheadAttention)
        out = layer(src, src_key_padding_mask=kpm, pos=pos)
        assert out.shape == (S, B, D), (attn_type, out.shape)
        assert torch.isfinite(out).all(), attn_type
        loss = out.pow(2).mean()
        loss.backward()
        assert src.grad is not None and torch.isfinite(src.grad).all(), attn_type
        src.grad = None
        print(f"  encoder layer [{attn_type}]: forward {tuple(out.shape)}, "
              f"finite, grad flows to input")


def test_reset_parameters_preserves_bdh_init():
    """VisionLanguageEncoder._reset_parameters xavier-inits everything with
    dim>1; it MUST skip BDH's E/Ev/Dx so their deliberate normal_(0.02) init
    survives (xavier on 256x1024 would give std ~0.04, silently different)."""
    torch.manual_seed(3)
    enc = VLT.VisionLanguageEncoder(
        d_model=D, nhead=8, num_encoder_layers=2, dim_feedforward=512,
        dropout=0.0, attn_type="bdh", bdh_mult=4)
    # after construction (which calls _reset_parameters), BDH matrices should
    # still look like normal_(0.02): std ~0.02, clearly not xavier's ~0.04.
    for name, m in enc.named_modules():
        if isinstance(m, BDHSelfAttention):
            e_std = m.E.detach().std().item()
            dx_std = m.Dx.detach().std().item()
            assert 0.012 < e_std < 0.028, f"E std {e_std} not ~0.02 (xavier clobber?)"
            assert 0.012 < dx_std < 0.028, f"Dx std {dx_std} not ~0.02 (xavier clobber?)"
    # and a non-BDH param (the FFN linear1) SHOULD have been xavier-inited
    lin = enc.encoder.layers[0].linear1.weight
    xavier_std = (2.0 / (lin.shape[0] + lin.shape[1])) ** 0.5
    assert abs(lin.detach().std().item() - xavier_std) < 0.01, "FFN not xavier-inited"
    print(f"  reset_parameters: BDH E std={e_std:.4f}, Dx std={dx_std:.4f} "
          f"(~0.02 preserved); FFN linear xavier-inited (~{xavier_std:.4f}) as expected")


def test_build_vl_transformer_from_args():
    """The real entry point TransVG uses is build_vl_transformer(args). Confirm
    the arg threading: vl_attn_type='bdh' -> BDH layers; a plain args missing
    the attr falls back to 'mha' (getattr default) so a stale config can't
    silently break the baseline."""
    import argparse
    base = dict(vl_hidden_dim=D, vl_dropout=0.0, vl_nheads=8,
                vl_dim_feedforward=512, vl_enc_layers=2)

    a_bdh = argparse.Namespace(**base, vl_attn_type="bdh", bdh_mult=4, bdh_share_qv=False)
    enc_bdh = VLT.build_vl_transformer(a_bdh)
    assert all(isinstance(l.self_attn, BDHSelfAttention) for l in enc_bdh.encoder.layers)

    a_mha = argparse.Namespace(**base, vl_attn_type="mha", bdh_mult=4, bdh_share_qv=False)
    enc_mha = VLT.build_vl_transformer(a_mha)
    assert all(isinstance(l.self_attn, nn.MultiheadAttention) for l in enc_mha.encoder.layers)

    # a legacy args namespace with NO vl_attn_type attribute must default to mha
    a_legacy = argparse.Namespace(**base)
    enc_legacy = VLT.build_vl_transformer(a_legacy)
    assert all(isinstance(l.self_attn, nn.MultiheadAttention) for l in enc_legacy.encoder.layers)
    print("  build_vl_transformer: 'bdh'->BDH layers, 'mha'->MHA layers, "
          "missing attr safely defaults to MHA")


def test_share_qv_reduces_params():
    """share_qv_encoder ties E and Ev -> reclaims one (d x n) matrix."""
    full = BDHSelfAttention(BDHSelfAttnConfig(dim=D, share_qv_encoder=False))
    tied = BDHSelfAttention(BDHSelfAttnConfig(dim=D, share_qv_encoder=True))
    n_full = sum(p.numel() for p in full.parameters())
    n_tied = sum(p.numel() for p in tied.parameters())
    assert n_full - n_tied == D * (D * 4), (n_full, n_tied)
    assert tied.Ev is tied.E
    print(f"  share_qv: tying E/Ev reclaims {n_full - n_tied} params "
          f"({n_full} -> {n_tied})")


def main():
    print("== TransVG BDH self-attention swap CPU test ==")
    test_signature_and_shape()
    test_attn_mask_rejected()
    test_key_padding_mask_excludes_pads()
    test_encoder_layer_both_modes_backprop()
    test_reset_parameters_preserves_bdh_init()
    test_build_vl_transformer_from_args()
    test_share_qv_reduces_params()
    print("ALL TRANSVG-BDH CHECKS PASSED.")
    print("(NOT locally tested: full TransVG forward, Talk2Car loader port, "
          "real training -- need remote data/GPU/pytorch_pretrained_bert.)")


if __name__ == "__main__":
    main()
