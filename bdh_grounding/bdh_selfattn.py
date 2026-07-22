"""
BDH-based drop-in replacement for the self-attention inside TransVG's
Vision-Language fusion transformer (external/TransVG/models/vl_transformer.py).

Where the AttnGrounder swap (bdh_fusion.py) replaced a single spatial
visual->text CROSS-attention (region grid queries a word sequence, plus a
beta attention-map for the BCE aux loss), TransVG has no such module. Its
fusion is a stack of standard transformer ENCODER layers running over one
concatenated multimodal token sequence:

    [REG] ; text_tokens ; visual_tokens          (S = 1 + L + N tokens)

and the "attention" being swapped is the `nn.MultiheadAttention` self-attn
operator inside each encoder layer. So this module is a self-attention
operator, not a cross-attention one, and it is deliberately built to match
`nn.MultiheadAttention`'s forward signature EXACTLY so it drops straight
into TransformerEncoderLayer with a one-line constructor change and no call-
site edits:

    forward(query, key, value, attn_mask=None, key_padding_mask=None,
            need_weights=..., ...) -> (attn_output, None)

Shapes are sequence-first (S, B, C) -- TransVG's convention (text/visual are
permuted to LenxBxC, the reg token is (1,B,C)). In TransVG's encoder layer
q = k = src + pos and value = src (positional embeddings folded into the
query/key addresses only, DETR-style), which this module respects since it
just consumes whatever q/k/v it is handed.

Mechanism (the same BDH memory as bdh_fusion.py, but non-causal, symmetric,
and reading out ALL sequence positions rather than a region sub-grid):

  * lift q and k into a ReLU-sparse, positive, high-dimensional "neuron"
    space (n = mult * d, n >> d)   -> many similar tokens stay separable
  * build an explicit growing OUTER-PRODUCT associative memory
        rho = K_sparse^T @ V       (n, d)   -> content-addressable retrieval
    with padded key/value tokens masked out so they never contribute
  * each token's sparse query address reads from rho, then a multiplicative
    gate x_sparse * read_sparse -> bilinear cross-token feature -> decode.

This is closest in spirit to bdh_fusion.py's mode "B" (a single token
sequence), but non-causal (TransVG's fusion is bidirectional -- no causal
mask) and with no RoPE (position comes from TransVG's own learned vl_pos
embeddings, already folded into q/k upstream), and it is stacked once per
encoder layer -- so N stacked encoder layers give N sequential BDH reads
with the layer's own residual + FFN + LayerNorm in between, a faithful
analog of "N layers of self-attention -> N layers of BDH self-attention."
"""

import math
import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class BDHSelfAttnConfig:
    dim: int = 256                  # d : == TransVG's vl_hidden_dim
    mult: int = 4                   # n = mult * dim  (neuron / high dimension)
    dropout: float = 0.1
    share_qv_encoder: bool = False  # tie query-address & value-path encoders -> reclaims n*d params


class BDHSelfAttention(nn.Module):
    """Non-causal BDH associative-memory self-attention, signature-compatible
    with nn.MultiheadAttention so it drops into TransVG's encoder layer."""

    def __init__(self, cfg: BDHSelfAttnConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        n = cfg.mult * d
        self.d, self.n = d, n

        # BDH-GPU parameter matrices (paper: E in R^{d x n}, Dx in R^{n x d}).
        self.E = nn.Parameter(torch.zeros(d, n).normal_(std=0.02))    # shared address-space encoder
        if cfg.share_qv_encoder:
            self.Ev = self.E                                          # tie -> reclaim n*d params
        else:
            self.Ev = nn.Parameter(torch.zeros(d, n).normal_(std=0.02))  # value-path encoder

        self.Dx = nn.Parameter(torch.zeros(n, d).normal_(std=0.02))   # decoder

        # Parameter-free LayerNorm, as in BDH.
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.scale = 1.0 / math.sqrt(n)  # keep linear-attention scores well-scaled for bf16

    def _lift(self, feat):
        """d-dim features -> ReLU-sparse positive n-dim addresses."""
        return F.relu(feat @ self.E)

    def forward(self, query, key, value, attn_mask=None,
                key_padding_mask=None, need_weights=True, **kwargs):
        """query/key/value: (S, B, d) sequence-first (TransVG convention).

        key_padding_mask: (B, S) bool, True == padded token to be excluded as
        a key/value (matches nn.MultiheadAttention semantics). attn_mask is
        accepted for signature compatibility but must be None here (TransVG's
        fusion never passes a full attention mask -- only key_padding_mask).

        Returns (attn_output (S, B, d), None) -- the None stands in for the
        attention-weight matrix nn.MultiheadAttention returns; TransVG only
        ever uses index [0], so weights are never needed.
        """
        assert attn_mask is None, (
            "BDHSelfAttention got a non-None attn_mask; TransVG's V-L fusion "
            "only uses key_padding_mask. A full attn_mask is not supported.")

        # (S, B, d) -> (B, S, d) so the batched matmuls below are natural.
        q = query.transpose(0, 1)
        k = key.transpose(0, 1)
        v = value.transpose(0, 1)

        q = self.ln(q)
        k = self.ln(k)
        v = self.ln(v)

        Qs = self._lift(q)                               # (B, S, n) query addresses
        Ks = self._lift(k)                               # (B, S, n) key addresses

        if key_padding_mask is not None:
            # (B, S) True==pad -> zero those rows out of BOTH the key addresses
            # and the values, so padded tokens contribute nothing to the memory
            # rho. (Their own output rows are computed but never read: only the
            # [REG] token at position 0, which is never padded, is used at the
            # end -- and padded tokens, excluded as keys here on every layer,
            # never leak into real tokens.)
            keep = (~key_padding_mask).to(v.dtype).unsqueeze(-1)   # (B, S, 1)
            Ks = Ks * keep
            v = v * keep

        # Explicit outer-product associative memory rho = K^T @ V  (B, n, d)
        rho = Ks.transpose(1, 2) @ v                     # (B, n, d)   the BDH memory state
        read = (Qs @ rho) * self.scale                   # (B, S, d)   each token retrieves from memory
        read = self.ln(read)
        read_sparse = F.relu(read @ self.Ev)             # (B, S, n)
        xy = Qs * read_sparse                            # (B, S, n) multiplicative bilinear gate
        xy = self.drop(xy)
        out = xy @ self.Dx                               # (B, S, d)
        out = self.ln(out)

        out = out.transpose(0, 1)                        # (B, S, d) -> (S, B, d)
        return out, None


def build_bdh_selfattn(cfg: BDHSelfAttnConfig) -> BDHSelfAttention:
    return BDHSelfAttention(cfg)
