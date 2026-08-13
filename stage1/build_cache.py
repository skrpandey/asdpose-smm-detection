"""Stage 1 cache builder.

Converts dataset.pkl into a float32 training cache: one memory-mapped
array of per-segment keypoints (T, 17, 3) with x, y, score channels, plus
a metadata table (identifier, child_id, split, binary_label, action_name,
n_frames, quality features for the hard-negative filter).

Dedup: dataset.pkl contains 2 duplicate identifiers; the first occurrence
wins and both are logged. Temporal quantities are stored in frames; the
fps fields are copied into metadata but flagged unreliable.

Also runs the subject-level split integrity check: no child id may appear
in both train and test.

Output layout under --cache-dir (default cache/segments/):
    keypoints.f32   flat float32 file, segments concatenated
    index.csv       one row per segment with byte offsets and metadata

Usage:
    python stage1/build_cache.py
"""

import argparse
import csv
import os
import pickle

import numpy as np

ARM_JOINTS = [5, 6, 7, 8, 9, 10]
LOW_CONF = 0.3


def quality_features(kp, sc):
    """The four Gate 0.2 quality features used by naive baselines."""
    lowconf = float((sc < LOW_CONF).mean())
    missing = float((sc == 0).mean())
    frozen = float((kp[1:] == kp[:-1]).all(axis=(1, 2)).mean()) if kp.shape[0] > 1 else 0.0
    energy = float(np.linalg.norm(np.diff(kp, axis=0), axis=2).mean()) if kp.shape[0] > 1 else 0.0
    arm_lowconf = float((sc[:, ARM_JOINTS] < LOW_CONF).mean())
    return lowconf, missing, frozen, energy, arm_lowconf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="/hd/data/StimmingADSMotion")
    ap.add_argument("--cache-dir", default="cache/segments")
    args = ap.parse_args()

    with open(f"{args.data_root}/dataset.pkl", "rb") as f:
        d = pickle.load(f)
    ann, split = d["annotations"], d["split"]
    train_set, test_set = set(split["train"]), set(split["test"])

    os.makedirs(args.cache_dir, exist_ok=True)
    seen = set()
    dupes = []
    offset = 0
    rows = []
    kp_path = os.path.join(args.cache_dir, "keypoints.f32")
    with open(kp_path, "wb") as out:
        for a in ann:
            ident = a["identifier"]
            if ident in seen:
                dupes.append(ident)
                continue
            seen.add(ident)
            kp = a["keypoint"].astype(np.float32)          # (T,17,2)
            sc = a["keypoint_score"].astype(np.float32)    # (T,17)
            arr = np.concatenate([kp, sc[..., None]], axis=2)  # (T,17,3)
            out.write(arr.tobytes())
            in_train = ident in train_set
            in_test = ident in test_set
            lowconf, missing, frozen, energy, arm_lowconf = quality_features(kp, sc)
            rows.append({
                "identifier": ident,
                "child_id": int(ident.split("_")[0]),
                "split": "train" if in_train else ("test" if in_test else "none"),
                "binary_label": a["binary_label"],
                "action_name": a["action_name"],
                "n_frames": arr.shape[0],
                "byte_offset": offset,
                "fps_unreliable": a["fps"],
                "img_h": a["img_shape"][0],
                "img_w": a["img_shape"][1],
                "q_lowconf": f"{lowconf:.6f}",
                "q_missing": f"{missing:.6f}",
                "q_frozen": f"{frozen:.6f}",
                "q_energy": f"{energy:.6f}",
                "q_arm_lowconf": f"{arm_lowconf:.6f}",
            })
            offset += arr.nbytes

    with open(os.path.join(args.cache_dir, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"cached {len(rows)} segments, {offset/2**30:.2f} GiB")
    print(f"duplicate identifiers dropped ({len(dupes)}): {dupes}")

    tr_children = {r["child_id"] for r in rows if r["split"] == "train"}
    te_children = {r["child_id"] for r in rows if r["split"] == "test"}
    overlap = tr_children & te_children
    print(f"split integrity: {len(tr_children)} train children, "
          f"{len(te_children)} test children, overlap {sorted(overlap)}")
    if overlap:
        raise SystemExit("FAIL: subject overlap between train and test")
    none = sum(1 for r in rows if r["split"] == "none")
    print(f"segments in neither split: {none}")


if __name__ == "__main__":
    main()
