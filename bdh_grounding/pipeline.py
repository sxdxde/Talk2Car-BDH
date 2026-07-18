"""Config-driven pipeline helpers, device-agnostic and CPU-testable.

These are the parts of the trainer that DON'T need CUDA / Darknet weights, so they
can be unit-tested locally. `train.py` (the remote entrypoint) wires them around
AttnGrounder's dataset/model/loss.
"""
import os
import csv
import time
import types

import yaml
import torch


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def config_to_args(cfg):
    """Flatten the nested YAML into a flat namespace with the fields train.py uses.
    Mirrors AttnGrounder's argparse names where they overlap, so the reused loss /
    target-building code needs no changes beyond device-agnosticism."""
    a = types.SimpleNamespace()
    # run
    a.profile = cfg["run"]["profile"]
    a.device = cfg["run"]["device"]
    a.precision = cfg["run"]["precision"]           # "fp32" | "bf16"
    a.seed = cfg["run"]["seed"]
    a.max_steps = cfg["run"].get("max_steps")       # None = full
    a.eval_split = cfg["run"].get("eval_split", "val")
    # data
    a.data_root = cfg["data"]["root"]
    a.subset_size = cfg["data"].get("subset_size")
    a.time = cfg["data"]["query_len"]
    a.size = cfg["data"]["imsize"]
    a.anchor_imsize = cfg["data"]["imsize"]
    # model
    a.emb_size = cfg["model"]["emb_size"]
    a.variant = cfg["model"]["variant"]             # "bdh" | "baseline"
    bdh = cfg["model"].get("bdh", {})
    a.bdh_mode = bdh.get("mode", "C")
    a.bdh_mult = bdh.get("mult", 4)
    a.bdh_n_head = bdh.get("n_head", 1)
    a.bdh_dropout = bdh.get("dropout", 0.1)
    a.bdh_share_qv_encoder = bdh.get("share_qv_encoder", False)
    a.fusion_streams = cfg["model"].get("fusion", {}).get("streams", ["fvisu", "flang_attn"])
    a.lambda_map = cfg["model"].get("lambda_map", 0.1)
    # train
    t = cfg["train"]
    a.batch_size = t["batch_size"]
    a.lr = float(t["lr"])
    a.backbone_lr_divisor = float(t.get("backbone_lr_divisor", 10.0))
    a.weight_decay = float(t.get("weight_decay", 5.0e-4))
    a.nb_epoch = t["epochs"]
    a.optimizer = t["optimizer"]
    a.power = float(t.get("scheduler_power", 1.0))
    a.anchors = t["anchors"]
    # checkpoint / log
    a.ckpt_dir = cfg["checkpoint"]["dir"]
    a.ckpt_every = cfg["checkpoint"].get("every_n_steps", 500)
    a.resume = cfg["checkpoint"].get("resume")
    a.log_csv = cfg["log"]["csv"]
    a.log_every = cfg["log"].get("every_n_steps", 20)
    return a


def resolve_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        print("[pipeline] cuda requested but unavailable -> falling back to cpu")
        return torch.device("cpu")
    return torch.device(name)


def autocast_ctx(device, precision):
    """bf16 autocast on cuda, no-op otherwise."""
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    class _Null:
        def __enter__(self): return None
        def __exit__(self, *a): return False
    return _Null()


# --------------------------------------------------------------------------- #
# csv logging
# --------------------------------------------------------------------------- #
class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._fh = open(path, "a", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=fieldnames)
        if new:
            self._w.writeheader()
            self._fh.flush()

    def log(self, **row):
        self._w.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._fh.flush()

    def close(self):
        self._fh.close()


# --------------------------------------------------------------------------- #
# checkpointing (with resume)
# --------------------------------------------------------------------------- #
def save_checkpoint(ckpt_dir, tag, epoch, step, model, optimizer, best_accu, is_best=False):
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {
        "epoch": epoch, "step": step, "best_accu": best_accu,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    last = os.path.join(ckpt_dir, f"{tag}_last.pth.tar")
    torch.save(state, last)
    if is_best:
        torch.save(state, os.path.join(ckpt_dir, f"{tag}_best.pth.tar"))
    return last


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("step", 0), ckpt.get("best_accu", -float("inf"))


def maybe_subset(dataset, n):
    """Take the first n samples (deterministic) for local smoke runs."""
    if not n or n >= len(dataset):
        return dataset
    return torch.utils.data.Subset(dataset, list(range(n)))


def anchors_full_from_list(anchor_list):
    """AttnGrounder orders anchors reversed (coarse scale first)."""
    return [(float(a[0]), float(a[1])) for a in anchor_list][::-1]
