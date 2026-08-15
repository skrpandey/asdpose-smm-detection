"""Stage 2: per-child performance spread on the released test split.

Answers: how uneven is detection quality across the 40 test children, for
any model's per-segment scores? Reports per-child AUC (children with at
least --min-pos SMM and --min-neg non-SMM segments), the best-to-worst
child gap, and the same statistics stratified by per-child tracking
quality (Addition 1 of the Stage 2 spec: invariance claims must not be
confounded by per-child pose quality).

Usage:
    python stage2/per_child_analysis.py --scores results/segment_scores/stage1_detector_v3_test_scores.csv
"""

import argparse
import csv
import sys

import numpy as np

sys.path.insert(0, "stage1")
from protocols import load_index, auc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--min-pos", type=int, default=10)
    ap.add_argument("--min-neg", type=int, default=10)
    args = ap.parse_args()

    rows = load_index()
    sc = {r["identifier"]: float(r["smm_score"])
          for r in csv.DictReader(open(args.scores))}
    test = [r for r in rows if r["split"] == "test" and r["identifier"] in sc]

    by_child = {}
    for r in test:
        by_child.setdefault(r["child_id"], []).append(r)

    print(f"## per-child analysis: {args.scores}")
    print(f"test children: {len(by_child)}; inclusion rule: >= {args.min_pos} "
          f"SMM and >= {args.min_neg} non-SMM segments\n")

    included = []
    for child, rs in sorted(by_child.items()):
        labels = np.array([r["binary_label"] for r in rs])
        n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
        if n_pos < args.min_pos or n_neg < args.min_neg:
            continue
        scores = np.array([sc[r["identifier"]] for r in rs])
        child_auc = auc(scores, labels)
        quality = float(np.mean([r["q_lowconf"] for r in rs]))
        included.append((child, child_auc, n_pos, n_neg, quality))

    aucs = np.array([x[1] for x in included])
    print(f"included children: {len(included)}")
    print(f"per-child AUC: mean {aucs.mean():.4f}, median {np.median(aucs):.4f}, "
          f"std {aucs.std():.4f}")
    print(f"best {aucs.max():.4f}, worst {aucs.min():.4f}, "
          f"GAP {aucs.max()-aucs.min():.4f}")
    print(f"children below 0.8: {(aucs < 0.8).sum()}, below 0.7: {(aucs < 0.7).sum()}\n")

    for child, a, npos, nneg, q in sorted(included, key=lambda x: x[1]):
        print(f"  child {child:4d}: AUC {a:.4f} ({npos:4d} SMM / {nneg:4d} non-SMM, "
              f"lowconf {q:.2f})")

    # quality stratification: median split on per-child mean low-conf fraction
    qs = np.array([x[4] for x in included])
    med = np.median(qs)
    good = aucs[qs <= med]
    poor = aucs[qs > med]
    print(f"\nquality-stratified (median split on child mean low-conf {med:.2f}):")
    print(f"  well-tracked half : mean AUC {good.mean():.4f}, worst {good.min():.4f}")
    print(f"  poorly-tracked half: mean AUC {poor.mean():.4f}, worst {poor.min():.4f}")
    from scipy.stats import spearmanr
    r = spearmanr(qs, aucs)
    print(f"  Spearman(child quality, child AUC): {r.statistic:+.3f} (p {r.pvalue:.3f})")


if __name__ == "__main__":
    main()
