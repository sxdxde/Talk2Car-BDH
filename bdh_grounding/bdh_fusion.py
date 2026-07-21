"""
BDH-based drop-in replacement for AttnGrounder's visual-text attention module.

Original target (parameter-free):
    external/AttnGrounder/model/grounding_model.py :: grounding_model.text_attn
        in : image_feat (B, C, H, W), lang_feat (B, T, C)
        out: lang_feat_attn (B, C, H, W)   # T'^k : per-region text representation
             beta           (B, H, W)      # attention map in [0,1] for the BCE aux loss

This module keeps that exact I/O contract so it is a true drop-in. It injects
BDH-GPU's memory mechanism (see external/bdh/bdh.py, which is the ground-truth
reference for the paper 2509.26507v1 whose Appendix E / equations are corrupted
in extraction):

  * lift features into a ReLU-sparse, positive, high-dimensional "neuron" space
    (n = mult * d, n >> d)   -> many similar candidates stay separable
  * linear attention -> an explicit growing OUTER-PRODUCT associative memory
    rho = K^T @ V            -> content-addressable retrieval (the anti-Mamba bit)
  * multiplicative gate x_sparse * y_sparse -> a bilinear cross-modal feature
    (richer than the original's linear softmax-weighted sum)

Three modes (see bdh-module-design-decision memory):
  "C" (PRIMARY, validated): symmetric NON-causal kernel. Regions query words.
  "A" (ablation): text runs as a causal BDH stream first (contextualized memory),
       then regions query it. Sharpens the "in-context retrieval" framing.
  "B" (ablation): single causal sequence [words ; regions], vanilla-BDH style
       with RoPE + causal mask; outputs read from the region positions.
"""

import math
import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class BDHFusionConfig:
    dim: int = 256                # d : common semantic dim (== AttnGrounder emb_size)
    mode: str = "C"               # "C" (primary) | "A" | "B" (ablations)
    mult: int = 4                 # n = mult * dim  (neuron / high dimension)
    n_head: int = 1               # single-head locked for first pass
    dropout: float = 0.1
    share_scales: bool = True     # one module instance reused across the 3 FPN scales
    share_qv_encoder: bool = False  # tie query-address & value-path encoders -> reclaims n*d params
    rope_theta: float = 1e4       # only used by mode "B"


def _get_freqs(n, theta, dtype):
    # RoPE frequencies, following external/bdh/bdh.py:get_freqs (mode "B" only)
    def quantize(t, q=2):
        return (t / q).floor() * q
    return 1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n)) / (2 * math.pi)


class BDHVisualTextAttention(nn.Module):
    """Drop-in BDH replacement for grounding_model.text_attn (single-head)."""

    def __init__(self, cfg: BDHFusionConfig):
        super().__init__()
        assert cfg.n_head == 1, "single-head locked for the first pass (multi-head is a future knob)"
        assert cfg.mode in ("C", "A", "B"), f"unknown mode {cfg.mode!r}"
        self.cfg = cfg
        d = cfg.dim
        n = cfg.mult * d
        self.d, self.n = d, n

        # BDH-GPU parameter matrices (paper: E in R^{d x n}, Dx/Dy in R^{n x d}).
        # Shapes chosen so (feat @ E) lifts d -> n.
        self.E = nn.Parameter(torch.zeros(d, n).normal_(std=0.02))    # encoder (shared address space)
        if cfg.share_qv_encoder:
            self.Ev = self.E                                          # tie -> reclaim n*d params
        else:
            self.Ev = nn.Parameter(torch.zeros(d, n).normal_(std=0.02))  # value-path encoder (encoder_v)
        self.Dx = nn.Parameter(torch.zeros(n, d).normal_(std=0.02))   # decoder

        # Learned 1-d scoring head on the associative read so beta can dip below 0.5
        # (raw ReLU-sparse dot products are >= 0). Tiny: d+1 params.
        self.beta_head = nn.Linear(d, 1)

        # Parameter-free LayerNorm, as in BDH.
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.scale = 1.0 / math.sqrt(n)  # keep linear-attention scores well-scaled for bf16

        if cfg.mode == "B":
            self.register_buffer(
                "freqs", _get_freqs(n, cfg.rope_theta, torch.float32).view(1, 1, n), persistent=False
            )

    # ---- helpers -----------------------------------------------------------
    def _lift(self, feat):
        """d-dim features -> ReLU-sparse positive n-dim addresses."""
        return F.relu(feat @ self.E)

    def _read_to_output(self, queries_sparse, read):
        """Shared BDH tail: gate + decode a read into a d-dim region/text feature.

        queries_sparse : (B, S, n)   sparse positive query addresses
        read           : (B, S, d)   associative-memory read (word features retrieved)
        returns        : (B, S, d)
        """
        read = self.ln(read)
        read_sparse = F.relu(read @ self.Ev)             # (B, S, n)
        xy = queries_sparse * read_sparse                # (B, S, n) multiplicative bilinear gate
        xy = self.drop(xy)
        out = xy @ self.Dx                               # (B, S, d)
        return self.ln(out)

    @staticmethod
    def _rope(freqs, positions, v):
        # RoPE on the last dim, following bdh.py rope() (mode "B" only)
        phases = (positions.view(1, -1, 1) * freqs)      # (1, S, n)
        phases = (phases % 1) * (2 * math.pi)
        cos, sin = torch.cos(phases), torch.sin(phases)
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view_as(v)
        return v * cos + v_rot * sin

    # ---- forward -----------------------------------------------------------
    def forward(self, image_feat, lang_feat, region_prior=None):
        """region_prior: optional (B, d, H_prev, W_prev) — the previous (coarser)
        scale's own output, for cross-scale growing memory ("Mode E"). Upsampled
        and folded into this scale's region features as a residual prior, so
        finer scales build on what coarser scales found, instead of each scale
        starting from scratch. None (default) reproduces the original per-scale-
        independent behavior exactly. Callers reuse the previous call's own
        `lang_feat_attn` output directly as the next call's `region_prior` --
        already the right shape, no extra state needed."""
        B, C, H, W = image_feat.shape
        assert C == self.d, f"channel {C} != configured dim {self.d}"
        R = H * W
        G = image_feat.flatten(2).transpose(1, 2)        # (B, R, d)  region features
        if region_prior is not None:
            prior_up = F.interpolate(region_prior, size=(H, W), mode="nearest")
            G = G + prior_up.flatten(2).transpose(1, 2)
        Q = lang_feat                                    # (B, T, d)  word features
        G = self.ln(G)
        Q = self.ln(Q)

        if self.cfg.mode == "C":
            lang_feat_attn, beta_map = self._forward_C(G, Q, B, H, W)
        elif self.cfg.mode == "A":
            lang_feat_attn, beta_map = self._forward_A(G, Q, B, H, W)
        else:
            lang_feat_attn, beta_map = self._forward_B(G, Q, B, H, W, R)
        return lang_feat_attn, beta_map

    def _forward_C(self, G, Q, B, H, W):
        """Symmetric non-causal: regions query words. PRIMARY."""
        Gq = self._lift(G)                               # (B, R, n) region query addresses
        Kk = self._lift(Q)                               # (B, T, n) word key addresses
        # Explicit outer-product associative memory rho = K^T @ V  (V = word features Q)
        rho = Kk.transpose(1, 2) @ Q                     # (B, n, d)   the BDH memory state
        read = (Gq @ rho) * self.scale                   # (B, R, d)   region retrieves from memory
        Tprime = self._read_to_output(Gq, read)          # (B, R, d)
        lang_feat_attn = Tprime.transpose(1, 2).view(B, self.d, H, W)
        beta = torch.sigmoid(self.beta_head(read).squeeze(-1)).view(B, H, W)
        return lang_feat_attn, beta

    def _forward_A(self, G, Q, B, H, W):
        """Ablation: contextualize text with a causal BDH self-attention pass first,
        then regions query the contextualized text memory. (Validation deferred.)"""
        T = Q.size(1)
        Qs = self._lift(Q)                               # (B, T, n)
        causal = torch.ones(T, T, device=Q.device).tril(-1)
        self_scores = (Qs @ Qs.transpose(1, 2)) * self.scale * causal   # (B, T, T) causal
        Q_ctx = self.ln(self_scores @ Q)                 # (B, T, d) contextualized words
        # regions query the contextualized text memory (non-causal read)
        Gq = self._lift(G)
        rho = self._lift(Q_ctx).transpose(1, 2) @ Q_ctx  # (B, n, d)
        read = (Gq @ rho) * self.scale
        Tprime = self._read_to_output(Gq, read)
        lang_feat_attn = Tprime.transpose(1, 2).view(B, self.d, H, W)
        beta = torch.sigmoid(self.beta_head(read).squeeze(-1)).view(B, H, W)
        return lang_feat_attn, beta

    def _forward_B(self, G, Q, B, H, W, R):
        """Ablation: single causal sequence [words ; regions] with RoPE. (Validation deferred.)"""
        T = Q.size(1)
        seq = torch.cat([Q, G], dim=1)                   # (B, T+R, d)
        S = T + R
        s_sparse = self._lift(seq)                       # (B, S, n)
        pos = torch.arange(S, device=seq.device, dtype=self.freqs.dtype)
        s_rope = self._rope(self.freqs, pos, s_sparse)
        scores = (s_rope @ s_rope.transpose(1, 2)).tril(-1) * self.scale   # (B, S, S) causal
        read = scores @ seq                              # (B, S, d)
        out = self._read_to_output(s_sparse, read)       # (B, S, d)
        region_out = out[:, T:, :]                       # (B, R, d) region positions
        region_read = read[:, T:, :]
        lang_feat_attn = region_out.transpose(1, 2).view(B, self.d, H, W)
        beta = torch.sigmoid(self.beta_head(region_read).squeeze(-1)).view(B, H, W)
        return lang_feat_attn, beta


def build_bdh_fusion(cfg: BDHFusionConfig) -> BDHVisualTextAttention:
    return BDHVisualTextAttention(cfg)
