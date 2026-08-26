"""W2 analysis C6: a quality-matched secondary protocol.

The QC filter is one-sided (it drops poorly tracked negatives and keeps
all positives), so within-positive quality variation stays available.
The two-sided construction pairs every test positive with a distinct
test negative that is nearest in the four z-scored quality statistics,
then evaluates on the matched set. Reports covariate balance
(standardized mean differences before and after matching) and matched
AUC per model.

Usage:
    python revision/c6_quality_matched.py
"""

import csv
import sys

import numpy as np

sys.path.insert(0, "stage1")
from protocols import load_index, auc

FEATS = ["q_lowconf", "q_missing", "q_frozen", "q_energy"]
MODELS = [
    ("checkpoint", ["results/segment_scores/gate03_segment_scores.csv"]),
    ("v1", ["results/segment_scores/stage1_detector_test_scores.csv"]),
    ("v3 family", ["results/segment_scores/stage2_sapl_lam0_seed0_test_scores.csv",
                   "results/segment_scores/stage2_sapl_lam0_seed1_test_scores.csv",
                   "results/segment_scores/stage2_sapl_lam0_seed2_test_scores.csv"]),
]


def smd(a, b):
    """Standardized mean difference."""
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def main():
    rows = load_index()
    te = [r for r in rows if r["split"] == "test"]
    labels = np.array([r["binary_label"] for r in te])
    X = np.array([[r[k] for k in FEATS] for r in te])
    Z = (X - X.mean(0)) / X.std(0)

    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]

    # greedy nearest-neighbor matching without replacement, hardest
    # positives (largest distance to their nearest negative) matched first
    d = np.linalg.norm(Z[pos][:, None, :] - Z[neg][None, :, :], axis=2)
    order = np.argsort(-d.min(axis=1))
    taken = np.zeros(len(neg), bool)
    pairs = []
    for pi in order:
        cand = np.where(~taken)[0]
        j = cand[np.argmin(d[pi, cand])]
        taken[j] = True
        pairs.append((pos[pi], neg[j]))
    matched = np.array([i for pr in pairs for i in pr])

    print("## C6 Quality-matched secondary protocol "
          f"({len(pairs)} matched pairs)\n")
    print("| feature | SMD before | SMD after matching |")
    print("|---|---|---|")
    for k, f in enumerate(FEATS):
        before = smd(X[pos, k], X[neg, k])
        mp = np.array([p for p, _ in pairs])
        mn = np.array([n for _, n in pairs])
        after = smd(X[mp, k], X[mn, k])
        print(f"| {f} | {before:+.2f} | {after:+.2f} |")

    print("\n| model | standard AUC | quality-matched AUC |")
    print("|---|---|---|")
    y_m = labels[matched]
    for name, paths in MODELS:
        st, ma = [], []
        for p in paths:
            sc = {r["identifier"]: float(r["smm_score"])
                  for r in csv.DictReader(open(p))}
            s = np.array([sc[r["identifier"]] for r in te])
            st.append(auc(s, labels))
            ma.append(auc(s[matched], y_m))
        print(f"| {name} | {np.mean(st):.3f} | {np.mean(ma):.3f} |")


if __name__ == "__main__":
    main()
