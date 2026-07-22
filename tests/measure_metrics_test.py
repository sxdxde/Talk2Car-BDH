"""CPU test for analysis/measure_metrics.py's count_layers helper -- the
one piece of that script that's testable without Darknet weights / a real
checkpoint / GPU. GFLOPs and inference-ms measurement (needs fvcore + the
real model) are verified on remote instead.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from analysis.measure_metrics import count_layers


class Tiny(nn.Module):
    """Small nested hierarchy: containers (Sequential) should NOT be
    counted, only their parameterized leaves; a leaf with no params
    (e.g. ReLU) should NOT be counted either."""
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(inplace=True))
        self.fc = nn.Linear(8, 4)
        self.ln = nn.LayerNorm(4)

    def forward(self, x):
        return x


def test_count_layers_basic():
    m = Tiny()
    total, counts = count_layers(m)
    # expected leaves with params: Conv2d(1), BatchNorm2d(1), Linear(1), LayerNorm(1)
    # NOT counted: the Sequential container itself, ReLU (no params), Tiny itself
    assert total == 4, f"expected 4 parameterized leaves, got {total}: {counts}"
    assert counts == {"Conv2d": 1, "BatchNorm2d": 1, "Linear": 1, "LayerNorm": 1}, counts
    print(f"  count_layers: total={total}, breakdown={counts} -- containers and "
          f"param-free leaves correctly excluded")


def test_count_layers_repeated_types():
    """Multiple instances of the same layer type should tally correctly,
    e.g. AttnGrounder's 3 FPN-scale branches each with their own Conv2d."""
    m = nn.ModuleList([nn.Conv2d(3, 3, 1) for _ in range(3)] + [nn.Linear(3, 3)])
    total, counts = count_layers(m)
    assert total == 4
    assert counts == {"Conv2d": 3, "Linear": 1}, counts
    print(f"  count_layers: repeated types tally correctly -> {counts}")


def main():
    print("== measure_metrics.count_layers CPU test ==")
    test_count_layers_basic()
    test_count_layers_repeated_types()
    print("ALL MEASURE_METRICS CHECKS PASSED.")
    print("(NOT locally tested: GFLOPs via fvcore, inference-ms -- both need "
          "the real Darknet-backed model + GPU/checkpoint, verify on remote.)")


if __name__ == "__main__":
    main()
