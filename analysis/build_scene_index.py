"""Step 6 data prep: build a per-image same-class candidate index from nuScenes
metadata + Talk2Car commands. Runs on the REMOTE (needs nuscenes-devkit + the
~445MB v1.0-trainval metadata; NO sensor data, NO GPU).

For each Talk2Car command it looks up the nuScenes sample, lists every object
visible in CAM_FRONT (the Talk2Car image) with its category, and counts how many
OTHER objects share the referred object's class. Output: a JSON mapping

    { "img_val_0.jpg": {"ref_class": "vehicle.car",
                         "n_same_class_others": 2,
                         "n_front_objects": 9}, ... }

which stratified_eval.py consumes (source-agnostic: any producer of this schema works).

Usage (remote):
    pip install nuscenes-devkit
    # download + extract v1.0-trainval_meta.tgz to $NUSC_META (creates v1.0-trainval/)
    python analysis/build_scene_index.py \
        --nuscenes-root $NUSC_META \
        --commands Talk2Car/data/commands/val_commands.json \
        --out analysis/scene_index_val.json
"""
import os
import json
import argparse


def categories_match(ref, other):
    """Match nuScenes category strings across granularity, e.g.
    'vehicle.car' vs 'vehicle.car', or coarse 'vehicle' vs 'vehicle.car'."""
    if ref == other:
        return True
    return other.startswith(ref + ".") or ref.startswith(other + ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuscenes-root", required=True, help="dir containing v1.0-trainval/ metadata")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--commands", required=True, help="Talk2Car {val,test,train}_commands.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vis-level", default="ANY", choices=["ANY", "NONE", "MIN", "ALL"],
                    help="CAM_FRONT visibility filter (BoxVisibility)")
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import BoxVisibility
    vis = getattr(BoxVisibility, args.vis_level)

    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_root, verbose=False)

    data = json.load(open(args.commands))
    commands = data["commands"] if isinstance(data, dict) and "commands" in data else data

    index, n_missing = {}, 0
    for i, c in enumerate(commands):
        img = c["t2c_img"]
        ref_class = c["obj_name"]
        ref_box_token = c.get("box_token")
        # Talk2Car's "sample_token" field is actually the CAM_FRONT sample_data
        # token, not nuScenes' keyframe "sample" token (confirmed via
        # nusc.get('sample_data', tok): channel='CAM_FRONT', is_key_frame=True) -
        # so it can be passed to get_sample_data directly, no sample lookup needed.
        cam_token = c["sample_token"]
        try:
            _, boxes, _ = nusc.get_sample_data(cam_token, box_vis_level=vis)
        except Exception as e:
            n_missing += 1
            index[img] = {"ref_class": ref_class, "n_same_class_others": None,
                          "n_front_objects": None, "error": str(e)[:120]}
            continue
        same = [b for b in boxes if categories_match(ref_class, b.name)]
        # exclude the referred object itself (by annotation token) when present
        n_same_others = sum(1 for b in same if b.token != ref_box_token)
        if all(b.token != ref_box_token for b in same):
            # referred box not among visible same-class boxes -> don't subtract
            n_same_others = len(same)
        index[img] = {"ref_class": ref_class,
                      "n_same_class_others": n_same_others,
                      "n_front_objects": len(boxes)}
        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{len(commands)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(index, open(args.out, "w"))
    n = len(index)
    amb = sum(1 for v in index.values() if v.get("n_same_class_others"))
    print(f"[done] {n} images -> {args.out}")
    print(f"  ambiguous (>=1 same-class other): {amb}  unambiguous: {n-amb-n_missing}  errors: {n_missing}")


if __name__ == "__main__":
    main()
