"""CPU plumbing test for the config-driven pipeline (no AttnGrounder / CUDA needed).

Verifies: both YAMLs parse into a complete args namespace; CSV logger writes header
+ rows; checkpoint save/load roundtrips model+optimizer+bookkeeping; subset + anchors.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from bdh_grounding import pipeline as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_configs():
    for name in ("smoke_local.yaml", "full_a100.yaml"):
        cfg = P.load_config(os.path.join(ROOT, "configs", name))
        a = P.config_to_args(cfg)
        for field in ("device", "precision", "emb_size", "variant", "lr",
                      "backbone_lr_divisor", "lambda_map", "batch_size", "nb_epoch",
                      "bdh_mode", "bdh_mult", "ckpt_dir", "log_csv"):
            assert hasattr(a, field), f"{name}: missing {field}"
        assert a.variant in ("bdh", "baseline")
        assert a.bdh_mode in ("C", "A", "B")
        print(f"  {name:18s} -> variant={a.variant} emb={a.emb_size} mode={a.bdh_mode} "
              f"mult={a.bdh_mult} lr={a.lr} bb/÷{a.backbone_lr_divisor} steps={a.max_steps}")


def test_csv():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub", "loss.csv")
        lg = P.CSVLogger(path, ["phase", "epoch", "step", "loss", "ap50"])
        lg.log(phase="train", epoch=0, step=1, loss=1.23)
        lg.log(phase="val", epoch=0, step=1, ap50=0.5)
        lg.close()
        lines = open(path).read().strip().splitlines()
        assert lines[0] == "phase,epoch,step,loss,ap50"
        assert len(lines) == 3
        print(f"  csv: header + {len(lines)-1} rows OK")


def test_checkpoint():
    with tempfile.TemporaryDirectory() as d:
        m = torch.nn.Linear(4, 3)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        opt.step()  # populate optimizer state
        P.save_checkpoint(d, "bdh", epoch=2, step=40, model=m, optimizer=opt, best_accu=0.42, is_best=True)
        m2 = torch.nn.Linear(4, 3)
        opt2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
        ep, st, best = P.load_checkpoint(os.path.join(d, "bdh_last.pth.tar"), m2, opt2)
        assert (ep, st, best) == (2, 40, 0.42)
        assert torch.allclose(m.weight, m2.weight)
        assert os.path.exists(os.path.join(d, "bdh_best.pth.tar"))
        print(f"  checkpoint: roundtrip epoch={ep} step={st} best={best} + best-copy OK")


def test_subset_and_anchors():
    ds = list(range(100))
    assert len(P.maybe_subset(ds, 10)) == 10
    assert len(P.maybe_subset(ds, None)) == 100
    anchors = [[10, 13], [16, 30], [33, 23]]
    full = P.anchors_full_from_list(anchors)
    assert full[0] == (33.0, 23.0)  # reversed (coarse first), matches AttnGrounder
    print(f"  subset + anchors (reversed) OK")


def main():
    print("== Pipeline plumbing test (CPU, no AttnGrounder deps) ==")
    test_configs()
    test_csv()
    test_checkpoint()
    test_subset_and_anchors()
    print("ALL PLUMBING CHECKS PASSED.")


if __name__ == "__main__":
    main()
