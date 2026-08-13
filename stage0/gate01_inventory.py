"""Gate 0.1: data inventory for the ASDPose release.

Answers: what files ship, what schema the annotations have, whether SMM type
labels exist, whether a child identifier is recoverable, whether the published
220/21 child split can be reconstructed, and how counts compare to the paper
(7352 SMM segments, 241 children, 319 assessments, 6597/755 train/test SMM).

Usage:
    python stage0/gate01_inventory.py --data-root /hd/data/StimmingADSMotion
"""

import argparse
import csv
import io
import pickle
import statistics
import zipfile
from collections import Counter


def parse_identifier(ident):
    """identifier format: {child}_{assessment_idx}_{camera}_{action}_{start_s}_{end_s}"""
    parts = ident.split("_")
    return int(parts[0]), int(parts[1]), int(parts[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="/hd/data/StimmingADSMotion")
    args = ap.parse_args()
    root = args.data_root

    print("## Gate 0.1 data inventory\n")

    # --- zip: untrimmed recordings + db.csv ---
    zf = zipfile.ZipFile(f"{root}/asdpose.zip")
    names = zf.namelist()
    pkls = [n for n in names if n.endswith(".pkl")]
    total_bytes = sum(i.file_size for i in zf.infolist())
    print(f"asdpose.zip: {len(names)} entries, {len(pkls)} recording pickles, "
          f"{total_bytes/2**30:.1f} GiB uncompressed")

    rows = list(csv.DictReader(io.TextIOWrapper(zf.open("db.csv"))))
    print(f"db.csv rows: {len(rows)}, columns: {list(rows[0].keys())}")
    db_children = {int(r["child_id"]) for r in rows}
    db_assessments = {(r["child_id"], r["assessment"], r["assessment_index"]) for r in rows}
    print(f"db.csv children: {len(db_children)} (ids {min(db_children)}..{max(db_children)})")
    print(f"db.csv assessments: {len(db_assessments)} "
          f"(paper says 319; types: {Counter(r['assessment'] for r in rows)})")

    # --- dataset.pkl: annotated segments + split ---
    with open(f"{root}/dataset.pkl", "rb") as f:
        d = pickle.load(f)
    ann, split = d["annotations"], d["split"]
    train_set, test_set = set(split["train"]), set(split["test"])
    print(f"\ndataset.pkl: {len(ann)} segments, split sizes train={len(train_set)} test={len(test_set)}")

    seg_keys = sorted(ann[0].keys())
    print(f"segment keys: {seg_keys}")
    a0 = ann[0]
    print(f"keypoint shape example: {a0['keypoint'].shape} (T, V=17, C=2), "
          f"score shape: {a0['keypoint_score'].shape}")

    labels = Counter(a["binary_label"] for a in ann)
    print(f"\nbinary labels: SMM={labels[1]} non-SMM={labels[0]} (paper: 7352 SMM)")

    base_types = Counter()
    for a in ann:
        if a["binary_label"] == 1:
            for t in a["action_name"].split(","):
                base_types[t.strip()] += 1
    print(f"SMM type labels PRESENT. Base types ({len(base_types)}): {base_types.most_common()}")

    smm_tr = sum(1 for a in ann if a["binary_label"] == 1 and a["identifier"] in train_set)
    smm_te = sum(1 for a in ann if a["binary_label"] == 1 and a["identifier"] in test_set)
    print(f"SMM per split: train={smm_tr} test={smm_te} (paper: 6597/755)")

    tr_children = {parse_identifier(i)[0] for i in train_set}
    te_children = {parse_identifier(i)[0] for i in test_set}
    print(f"\nchild id from identifier prefix: train children={len(tr_children)} "
          f"test children={len(te_children)} overlap={sorted(tr_children & te_children)}")
    print(f"test children: {sorted(te_children)}")
    pkl_children = tr_children | te_children
    print(f"dataset.pkl children total: {len(pkl_children)} vs db.csv {len(db_children)}; "
          f"pkl-only ids: {min(pkl_children - db_children)}..{max(pkl_children - db_children)} "
          f"({len(pkl_children - db_children)} children with no untrimmed recording in the zip)")
    te_in_zip = te_children & db_children
    print(f"test children with untrimmed recordings available: {len(te_in_zip)} of {len(te_children)}")

    lens = [a["total_frames"] / a["fps"] for a in ann if a["binary_label"] == 1]
    print(f"\nSMM segment length seconds (incl ~2s context padding): "
          f"mean {statistics.mean(lens):.2f} median {statistics.median(lens):.2f} "
          f"min {min(lens):.2f} max {max(lens):.2f} (paper mean 9.9s on the tight interval)")
    print(f"fps values: {Counter(round(a['fps'], 2) for a in ann)}")

    dup = len(ann) - len({a["identifier"] for a in ann})
    print(f"duplicate identifiers among segments: {dup}")


if __name__ == "__main__":
    main()
