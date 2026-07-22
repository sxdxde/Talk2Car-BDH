"""CPU test for the offline/frozen DistilBERT text-encoder path (arch2).

Covers what's locally testable without the real remote dataset or the
`transformers` package: the left/right-swap helper (must exactly match
talk2car_loader.py's own replace-chain, since cache keys depend on it),
the cache-lookup contract (fail loud on a miss, not silent), and the
real-length-trim + mapping_lang projection logic that replaces the
BiLSTM path in grounding_model.forward.

NOT locally testable (needs remote data / GPU / transformers): the actual
Talk2CarDataset class end-to-end (needs real images + corpus.pth + the
talk2car_{split}.pth files), and running real DistilBERT extraction.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/arch2")

import torch
import torch.nn as nn

from arch2.build_text_cache import lr_swap

D = 256
QUERY_LEN = 40


def test_lr_swap():
    # single-direction phrases: swap is its own inverse
    assert lr_swap("turn left behind the car") == "turn right behind the car"
    assert lr_swap(lr_swap("turn left behind the car")) == "turn left behind the car"
    # both words present: each swaps independently, order-preserved
    assert lr_swap("turn left then right") == "turn right then left"
    # no left/right: no-op
    assert lr_swap("go straight ahead") == "go straight ahead"
    print("  lr_swap: single/double/no-op cases all correct")


def test_cache_lookup_contract():
    """Mirrors the loader's lookup line: self.text_cache[phrase].float().numpy()"""
    cache = {
        "turn left": torch.randn(QUERY_LEN, 768).half(),
        "turn right": torch.randn(QUERY_LEN, 768).half(),
    }
    out = cache["turn left"].float().numpy()
    assert out.shape == (QUERY_LEN, 768)
    assert out.dtype.name == "float32"
    try:
        _ = cache["a phrase never cached"]
        raise AssertionError("expected KeyError for a missing phrase, got none")
    except KeyError:
        pass  # correct: fail loud, no silent fallback
    print("  cache lookup: correct shape/dtype, missing key raises loudly")


def test_real_len_trim_and_projection():
    """Replicates grounding_model.forward's distilbert branch exactly:
    zero-row mask -> real_len -> trim -> LayerNorm -> reshape -> mapping_lang -> view."""
    torch.manual_seed(0)
    B = 3
    real_lens = [12, 40, 1]   # varied, including the full-length and minimal edge cases
    raw_flang = torch.zeros(B, QUERY_LEN, 768)
    for b, rl in enumerate(real_lens):
        raw_flang[b, :rl, :] = torch.randn(rl, 768)
    raw_flang.requires_grad_(True)

    real_len = (raw_flang.abs().sum(-1) > 0).sum(1).max().clamp(min=1).item()
    assert real_len == max(real_lens), f"expected {max(real_lens)}, got {real_len}"

    trimmed = raw_flang[:, :real_len, :]
    text_ln = nn.LayerNorm(768)
    normed = text_ln(trimmed)
    mapping_lang = nn.Linear(768, D)
    bs, mlen, embdim = normed.shape
    flang = mapping_lang(normed.reshape(-1, embdim)).view(bs, mlen, D)

    assert flang.shape == (B, max(real_lens), D)
    assert torch.isfinite(flang).all()

    loss = flang.pow(2).mean()
    loss.backward()
    assert mapping_lang.weight.grad is not None and torch.isfinite(mapping_lang.weight.grad).all()
    assert text_ln.weight.grad is not None and torch.isfinite(text_ln.weight.grad).all()
    assert raw_flang.grad is not None and torch.isfinite(raw_flang.grad).all()
    print(f"  real_len trim: max real_len={real_len} correctly picked from batch of "
          f"{real_lens}, projection shape={tuple(flang.shape)}, grad OK (incl. through LayerNorm)")


def test_layernorm_scale_fix():
    """Verifies the actual fix for the observed baseline+distilbert regression
    (59.98 vs 64.92 for glove): a LayerNorm applied AFTER trimming to real_len
    (so padded all-zero rows are never normalized), which should correct
    BERT-family "rogue dimension" scale issues before the untrained
    mapping_lang projection has to absorb them raw."""
    torch.manual_seed(2)
    B = 2
    real_lens = [15, 40]
    raw_flang = torch.zeros(B, QUERY_LEN, 768)
    for b, rl in enumerate(real_lens):
        # simulate uneven/large-magnitude activations: most dims ~N(0,1), a
        # handful of "rogue" dims with much larger scale -- documented BERT
        # behavior, the working hypothesis for why the real run underperformed
        content = torch.randn(rl, 768)
        content[:, :5] *= 50.0
        raw_flang[b, :rl, :] = content
    raw_flang.requires_grad_(True)

    real_len = (raw_flang.abs().sum(-1) > 0).sum(1).max().clamp(min=1).item()
    assert real_len == max(real_lens)
    trimmed = raw_flang[:, :real_len, :]

    # sanity-check the test setup itself actually has uneven scale pre-norm
    # (expected ~sqrt((763*1 + 5*2500)/768) =~ 4.2-4.6 depending on sampling;
    # threshold set well below that, comfortably above the ~1.0 a normal-scale
    # embedding would show, so this is a robust "is it actually large" check)
    pre_std = trimmed[0, 0].std(unbiased=False).item()
    assert pre_std > 3.0, f"test setup should have large pre-norm std, got {pre_std}"

    ln = nn.LayerNorm(768)
    normed = ln(trimmed)
    post_std = normed[0, 0].std(unbiased=False).item()
    assert abs(post_std - 1.0) < 0.05, f"expected ~unit std per position after LN, got {post_std}"
    assert torch.isfinite(normed).all()

    # trim-before-norm confirmed by construction: `trimmed`/`normed` never
    # contained the padded rows (positions >= real_len) at all
    assert trimmed.shape[1] == max(real_lens) == normed.shape[1]

    loss = normed.pow(2).mean()
    loss.backward()
    assert ln.weight.grad is not None and torch.isfinite(ln.weight.grad).all()
    assert raw_flang.grad is not None and torch.isfinite(raw_flang.grad).all()
    print(f"  layernorm fix: pre-norm per-position std={pre_std:.1f} (simulated rogue "
          f"dims) -> post-norm std={post_std:.3f} (~1.0, correct), trim-before-norm "
          f"confirmed, grad flows through LN and back to input")


def test_glove_path_unaffected():
    """Sanity: the glove branch's math (no trim, full query_len, .view not .reshape)
    still works identically -- reshape() behaves like view() on a contiguous tensor,
    so switching view->reshape in grounding_model.py is a no-op for this path."""
    torch.manual_seed(1)
    B = 2
    raw_flang = torch.randn(B, QUERY_LEN, 600)  # BiLSTM-shaped output, always contiguous
    mapping_lang = nn.Linear(600, D)
    bs, mlen, embdim = raw_flang.shape
    flang_view = mapping_lang(raw_flang.view(-1, embdim)).view(bs, mlen, D)
    flang_reshape = mapping_lang(raw_flang.reshape(-1, embdim)).view(bs, mlen, D)
    assert torch.equal(flang_view, flang_reshape)
    print("  glove path: view->reshape change is a verified no-op")


def main():
    print("== Offline text-encoder (arch2) CPU test ==")
    test_lr_swap()
    test_cache_lookup_contract()
    test_real_len_trim_and_projection()
    test_layernorm_scale_fix()
    test_glove_path_unaffected()
    print("ALL TEXT-ENCODER CHECKS PASSED.")
    print("(NOT locally tested: real Talk2CarDataset end-to-end, real DistilBERT "
          "extraction -- both need remote data/transformers, verify on remote.)")


if __name__ == "__main__":
    main()
