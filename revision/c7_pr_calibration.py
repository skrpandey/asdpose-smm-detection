"""W2 analysis C7: precision-recall and calibration metrics.

ROC AUC is prevalence-insensitive; deployment is not. This script adds,
for the checkpoint and the v3 family under the standard protocol:
precision-recall AUC (average precision), Brier score, expected
calibration error (10 equal-width bins), and the positive predictive
value the test-set operating point at threshold 0.85 implies at
plausible deployment prevalences (1%, 3%, 7%) via
PPV = sens*pi / (sens*pi + (1-spec)*(1-pi)).

Usage:
    python revision/c7_pr_calibration.py
"""

import csv
import sys

import numpy as np

sys.path.insert(0, "stage1")
from protocols import load_index, prf_at

MODELS = [
    ("checkpoint", ["results/segment_scores/gate03_segment_scores.csv"]),
    ("v3 family", ["results/segment_scores/stage2_sapl_lam0_seed0_test_scores.csv",
                   "results/segment_scores/stage2_sapl_lam0_seed1_test_scores.csv",
                   "results/segment_scores/stage2_sapl_lam0_seed2_test_scores.csv"]),
]


def average_precision(scores, labels):
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def brier(scores, labels):
    return float(np.mean((scores - labels) ** 2))


def ece(scores, labels, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (scores >= lo) & (scores < hi if hi < 1 else scores <= hi)
        if m.sum() == 0:
            continue
        total += m.mean() * abs(scores[m].mean() - labels[m].mean())
    return float(total)


def main():
    rows = load_index()
    te = [r for r in rows if r["split"] == "test"]
    labels = np.array([r["binary_label"] for r in te])

    print("## C7 Precision-recall and calibration (standard protocol)\n")
    print("| model | PR-AUC | Brier | ECE | sens@0.85 | spec@0.85 | "
          "PPV at 1% / 3% / 7% prevalence |")
    print("|---|---|---|---|---|---|---|")
    for name, paths in MODELS:
        ap_, br_, ec_, se_, sp_, ppvs_ = [], [], [], [], [], []
        for p in paths:
            sc = {r["identifier"]: float(r["smm_score"])
                  for r in csv.DictReader(open(p))}
            s = np.array([sc[r["identifier"]] for r in te])
            ap_.append(average_precision(s, labels))
            br_.append(brier(s, labels))
            ec_.append(ece(s, labels))
            pr, rc, sp = prf_at(s, labels, 0.85)
            se_.append(rc)
            sp_.append(sp)
            ppvs_.append([rc * pi / (rc * pi + (1 - sp) * (1 - pi))
                          for pi in (0.01, 0.03, 0.07)])
        ppv = np.mean(ppvs_, axis=0)
        print(f"| {name} | {np.mean(ap_):.3f} | {np.mean(br_):.3f} | "
              f"{np.mean(ec_):.3f} | {np.mean(se_):.3f} | {np.mean(sp_):.3f} | "
              f"{ppv[0]:.2f} / {ppv[1]:.2f} / {ppv[2]:.2f} |")


if __name__ == "__main__":
    main()
