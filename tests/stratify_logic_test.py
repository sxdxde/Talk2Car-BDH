"""CPU test for the Step 6 bucketing + category matching (no nuScenes / GPU)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.stratified_eval import stratify
from analysis.build_scene_index import categories_match


def test_categories_match():
    assert categories_match("vehicle.car", "vehicle.car")
    assert categories_match("vehicle", "vehicle.car")        # coarse ref matches fine
    assert categories_match("vehicle.car", "vehicle")        # fine ref matches coarse
    assert not categories_match("vehicle.car", "vehicle.truck")
    assert not categories_match("vehicle.car", "human.pedestrian.adult")
    print("  category matching OK")


def test_stratify():
    scene = {
        "a.jpg": {"n_same_class_others": 3},   # ambiguous, hit
        "b.jpg": {"n_same_class_others": 1},   # ambiguous, miss
        "c.jpg": {"n_same_class_others": 0},   # unambiguous, hit
        "d.jpg": {"n_same_class_others": 0},   # unambiguous, hit
        "e.jpg": {"n_same_class_others": None},# unknown, hit
    }
    hits = {"a.jpg": 1, "b.jpg": 0, "c.jpg": 1, "d.jpg": 1, "e.jpg": 1}
    res = stratify(hits, scene)
    assert res["all"] == (80.0, 5), res["all"]              # 4/5
    assert res["ambiguous"] == (50.0, 2), res["ambiguous"]  # 1/2
    assert res["unambiguous"] == (100.0, 2), res["unambiguous"]  # 2/2
    assert res["unknown"] == (100.0, 1), res["unknown"]
    print("  stratify buckets OK: "
          f"all={res['all']} amb={res['ambiguous']} unamb={res['unambiguous']}")


def main():
    print("== Step 6 stratification logic test (CPU) ==")
    test_categories_match()
    test_stratify()
    print("ALL STEP-6 LOGIC CHECKS PASSED.")


if __name__ == "__main__":
    main()
