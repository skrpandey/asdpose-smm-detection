"""W2 analysis C2: within-length-decile AUC for every evaluated arm.

The annotated-length confound (revision B5: 0.903 AUC from length alone)
is controlled by stratification: segments are binned into deciles of
annotated length over the test split, AUC is computed inside each decile,
and the deciles are averaged weighted by their number of
positive-negative pairs (the pairs that contribute to AUC). A model
exploiting length would drop substantially inside deciles; the earlier
two-model spot check becomes a column for every arm here.

Usage:
    python revision/c2_length_decile.py
"""

import csv
import sys

import numpy as np

sys.path.insert(0, "stage1")
from protocols import load_index, auc

MODELS = [
    ("checkpoint", ["results/segment_scores/gate03_segment_scores.csv"]),
    ("v1", ["results/segment_scores/stage1_detector_test_scores.csv"]),
    ("velocity only (v1 norm)", ["results/segment_scores/stage1_abl_velocity_test_scores.csv"]),
    ("dropout only (v1 norm)", ["results/segment_scores/stage1_abl_dropout_test_scores.csv"]),
    ("v2", ["results/segment_scores/stage1_detector_v2_test_scores.csv"]),
    ("v3 family", ["results/segment_scores/stage2_sapl_lam0_seed0_test_scores.csv",
                   "results/segment_scores/stage2_sapl_lam0_seed1_test_scores.csv",
                   "results/segment_scores/stage2_sapl_lam0_seed2_test_scores.csv"]),
    ("stgcn base", ["results/segment_scores/wacv_stgcn_base_test_scores.csv"]),
    ("stgcn remedies", ["results/segment_scores/wacv_stgcn_remedies_test_scores.csv"]),
    ("agcn base", ["results/segment_scores/wacv_agcn_base_test_scores.csv"]),
    ("agcn remedies", ["results/segment_scores/wacv_agcn_remedies_test_scores.csv"]),
    ("ours base", ["results/segment_scores/wacv_ours_base_test_scores.csv"]),
]


def decile_auc(scores, labels, deciles):
    num = den = 0.0
    for d in np.unique(deciles):
        m = deciles == d
        y = labels[m]
        n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
        if n_pos == 0 or n_neg == 0:
            continue
        w = n_pos * n_neg
        num += w * auc(scores[m], y)
        den += w
    return num / den


def main():
    rows = load_index()
    te = [r for r in rows if r["split"] == "test"]
    labels = np.array([r["binary_label"] for r in te])
    frames = np.array([r["n_frames"] for r in te])
    edges = np.percentile(frames, np.arange(10, 100, 10))
    deciles = np.digitize(frames, edges)

    print("## C2 Within-length-decile AUC (standard protocol, pair-weighted)\n")
    print("| model | pooled AUC | within-decile AUC |")
    print("|---|---|---|")
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"model & pooled AUC & within-decile AUC \\", r"\midrule"]
    for name, paths in MODELS:
        pooled, strat = [], []
        for p in paths:
            sc = {r["identifier"]: float(r["smm_score"])
                  for r in csv.DictReader(open(p))}
            s = np.array([sc[r["identifier"]] for r in te])
            pooled.append(auc(s, labels))
            strat.append(decile_auc(s, labels, deciles))
        print(f"| {name} | {np.mean(pooled):.3f} | {np.mean(strat):.3f} |")
        lines.append(f"{name} & {np.mean(pooled):.3f} & {np.mean(strat):.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open("results/tables/table_length_decile.tex", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
