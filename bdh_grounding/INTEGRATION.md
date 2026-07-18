# Integrating the BDH module into AttnGrounder

Three precise edits to `external/AttnGrounder/model/grounding_model.py`. Everything
else (Darknet-53, RNNEncoder/BiLSTM+GloVe, YOLO head, the BCE aux mask loss in
`train_yolo.py`, hyperparameters) stays byte-for-byte unchanged. This isolates the
single-component swap the study is about.

Copy `bdh_grounding/bdh_fusion.py` next to `grounding_model.py` (or keep the package
importable) and select the variant from config (`model.variant: bdh|baseline`,
`model.bdh.mode: C|A|B`).

---

## Edit 1 — `grounding_model.__init__`: build the BDH module + trim fusion width

```python
# --- add import at top of file ---
from bdh_grounding.bdh_fusion import BDHFusionConfig, BDHVisualTextAttention

# --- extend grounding_model.__init__ signature with the variant/bdh kwargs
#     (defaults keep the original baseline behaviour if omitted): ---
def __init__(self, corpus=None, emb_size=256, jemb_drop_out=0.1,
             coordmap=True, leaky=False, dataset=None, light=False,
             variant="baseline", bdh_mode="C", bdh_mult=2, bdh_n_head=1,
             bdh_dropout=0.1, bdh_share_qv_encoder=False):

# --- inside __init__, REPLACE the line:
#     self.attn_sfmax = nn.Softmax(-1)
# with: ---
self.variant = variant                    # "bdh" | "baseline", threaded from config
if self.variant == "bdh":
    self.bdh_attn = BDHVisualTextAttention(
        BDHFusionConfig(dim=emb_size, mode=bdh_mode, mult=bdh_mult,
                        n_head=bdh_n_head, dropout=bdh_dropout,
                        share_qv_encoder=bdh_share_qv_encoder))
else:
    self.attn_sfmax = nn.Softmax(-1)      # original parameter-free path

# --- change fusion input width for the 2-stream trim ---
#   was:  embin_size = emb_size*3
embin_size = emb_size * (3 if self.variant == "baseline" else 2)
```

`fcn_emb` then maps `emb_size*2 -> emb_size` in the bdh variant (vs `*3` original),
removing one 256-channel conv path — this is the fusion trim that offsets the BDH
module's added params.

---

## Edit 2 — `text_attn`: delegate to the BDH module in the bdh variant

```python
def text_attn(self, image_feat, lang_feat):
    if self.variant == "bdh":
        # returns (lang_feat_attn (B,C,H,W), beta (B,H,W)) — same contract
        return self.bdh_attn(image_feat, lang_feat)
    # ---- original path (unchanged) ----
    B, C, H, W = image_feat.size()
    image_feat = image_feat.view(B, C, -1)
    image_feat_ = image_feat.transpose(1, 2)
    lang_feat_ = lang_feat.transpose(1, 2)
    image_lang = torch.matmul(image_feat_, lang_feat_)
    image_lang_sf1 = self.attn_sfmax(image_lang)
    image_lang_sum = torch.sum(image_lang, dim=-1)
    image_lang_sf2 = torch.sigmoid(image_lang_sum).view(B, H, W)
    lang_feat_attn = torch.matmul(image_lang_sf1, lang_feat).transpose(1, 2).view(B, C, H, W)
    return lang_feat_attn, image_lang_sf2
```

---

## Edit 3 — `forward`: fuse 2 streams (drop the beta*visual stream) in the bdh variant

```python
# in the fusion loop, REPLACE the 3-stream concat:
#     flangvisu.append(torch.cat([fvisu[ii], fvisu_ii_attn, flang_attn[ii]], dim=1))
if self.variant == "bdh":
    flang_attn[ii] = F.normalize(flang_attn[ii], p=2, dim=1)
    fvisu[ii] = F.normalize(fvisu[ii], p=2, dim=1)
    flangvisu.append(torch.cat([fvisu[ii], flang_attn[ii]], dim=1))   # 2 streams
else:
    fvisu_ii_attn = attn_map[ii].unsqueeze(1) * fvisu[ii]
    fvisu_ii_attn = F.normalize(fvisu_ii_attn, p=2, dim=1)
    fvisu[ii] = F.normalize(fvisu[ii], p=2, dim=1)
    flang_attn[ii] = F.normalize(flang_attn[ii], p=2, dim=1)
    flangvisu.append(torch.cat([fvisu[ii], fvisu_ii_attn, flang_attn[ii]], dim=1))
```

Note: `attn_map` (= beta) is still returned by `text_attn`/`forward` and still drives
the BCE aux mask loss in `train_yolo.py` — only the *fused* beta*visual stream is dropped.

---

## Also: device-agnostic fix (needed for local CPU smoke)

`generate_coord` (grounding_model.py:37) hard-codes `.cuda()`. Replace with the
tensor's device (`device=image.device`) so the same code runs on CPU and A100.
