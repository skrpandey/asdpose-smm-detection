"""Feasibility diagnostic for SMM type classification.

Answers three questions before committing to a type-classification
section: (1) what does the type label distribution look like per split
(single- versus multi-label, long tail); (2) how separable are the base
types in the frozen v3 detection features, via a multinomial probe
(train children fit, test children evaluate, subject-level by
construction); (3) does binary detection quality vary by type (per-type
AUC of the existing v3 scores against all non-SMM negatives).

Usage:
    python revision/type_feasibility.py
"""

import csv
import sys
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "stage1")
from dataset import SegmentDataset
from model import CompactCTRGCN
from protocols import load_index, auc


def main():
    rows = load_index()
    smm = [r for r in rows if r["binary_label"] == 1]
    for split in ("train", "test"):
        rs = [r for r in smm if r["split"] == split]
        single = [r for r in rs if "," not in r["action_name"]]
        print(f"{split}: {len(rs)} SMM segments, {len(single)} single-label "
              f"({len(single)/len(rs):.0%})")
    types_tr = Counter(r["action_name"] for r in smm
                       if r["split"] == "train" and "," not in r["action_name"])
    types_te = Counter(r["action_name"] for r in smm
                       if r["split"] == "test" and "," not in r["action_name"])
    print(f"\n{'type':22s} {'train':>6s} {'test':>5s}")
    for t, n in types_tr.most_common():
        print(f"{t:22s} {n:>6d} {types_te.get(t, 0):>5d}")

    # types with enough support on both sides
    keep = [t for t in types_tr if types_tr[t] >= 100 and types_te.get(t, 0) >= 20]
    print(f"\ntypes with >=100 train and >=20 test single-label segments: "
          f"{len(keep)}: {sorted(keep)}")

    # frozen-feature probe
    ckpt = torch.load("cache/stage1_detector_v3.pt", map_location="cpu",
                      weights_only=False)
    cfg = ckpt["config"]
    model = CompactCTRGCN(c_in=cfg["model"]["c_in"],
                          widths=tuple(cfg["model"]["widths"]),
                          blocks_per_stage=cfg["model"]["blocks_per_stage"])
    model.load_state_dict(ckpt["state_dict"])
    model = model.cuda().eval()

    probe_rows = [r for r in smm if "," not in r["action_name"]
                  and r["action_name"] in keep]
    ds = SegmentDataset(probe_rows, cfg["data"]["cache_dir"],
                        cfg["data"]["window"], train=False,
                        velocity=cfg["data"]["velocity"],
                        normalize=cfg["data"]["normalize_mode"])
    dl = DataLoader(ds, batch_size=128, num_workers=6)
    feats = []
    with torch.no_grad():
        for x, _ in dl:
            feats.append(model.forward_features(x.cuda()).cpu().numpy())
    X = np.concatenate(feats)
    y = np.array([keep.index(r["action_name"]) for r in probe_rows])
    tr = np.array([r["split"] == "train" for r in probe_rows])

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, random_state=0)
    clf.fit(scaler.transform(X[tr]), y[tr])
    pred = clf.predict(scaler.transform(X[~tr]))
    yt = y[~tr]
    acc = (pred == yt).mean()
    maj = Counter(yt).most_common(1)[0][1] / len(yt)
    print(f"\nprobe on frozen v3 features, {len(keep)} classes, "
          f"subject-level split:")
    print(f"top-1 accuracy {acc:.3f} against majority baseline {maj:.3f}")
    print(f"{'type':22s} {'n test':>6s} {'recall':>7s}")
    for i, t in enumerate(keep):
        m = yt == i
        if m.sum() == 0:
            continue
        print(f"{t:22s} {int(m.sum()):>6d} {(pred[m] == i).mean():>7.3f}")

    # per-type binary detection AUC of the deployed v3 scores
    sc = {r["identifier"]: float(r["smm_score"]) for r in
          csv.DictReader(open("results/segment_scores/stage2_sapl_lam0_seed0_test_scores.csv"))}
    te = [r for r in rows if r["split"] == "test"]
    neg = [sc[r["identifier"]] for r in te if r["binary_label"] == 0]
    print(f"\nper-type binary detection AUC (v3, vs all test non-SMM):")
    for t in sorted(keep):
        pos = [sc[r["identifier"]] for r in te
               if r["binary_label"] == 1 and r["action_name"] == t]
        if len(pos) < 10:
            continue
        s = np.array(pos + neg)
        lab = np.array([1] * len(pos) + [0] * len(neg))
        print(f"{t:22s} n={len(pos):>4d}  AUC {auc(s, lab):.3f}")


if __name__ == "__main__":
    main()
