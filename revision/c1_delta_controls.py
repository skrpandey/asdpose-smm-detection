"""W2 analysis C1: uncertainty and controls for the reliance statistic.

Three questions about Delta = AUC_standard - AUC_quality_controlled:

1. Uncertainty: paired child-clustered bootstrap CIs for Delta, for every
   model in the main and breadth tables. The two protocols share the same
   positives, so the paired construction over resampled children is the
   right one.
2. Identification: does Delta measure reliance on the tracking cue, or
   just a harder negative set? Control: evaluate each model on repeated
   size-matched RANDOM subsamples of the negatives (1,648 of 3,694,
   matching the filter's retention). If the random-subsample Delta
   centers on zero while the quality-filtered Delta does not, the drop is
   attributable to the quality-based selection, not to negative-set size.
3. Sensitivity: sweep both filter thresholds +/-0.05 around the frozen
   values and report retention, Delta, and the model ranking per cell.
   The frozen thresholds stay the headline; this is post-hoc robustness.

Usage:
    python revision/c1_delta_controls.py [--iters 10000] [--draws 1000]
"""

import argparse
import csv
import sys

import numpy as np

sys.path.insert(0, "stage1")
from protocols import load_index, auc

MODELS = [
    ("checkpoint", ["results/segment_scores/gate03_segment_scores.csv"]),
    ("v1", ["results/segment_scores/stage1_detector_test_scores.csv"]),
    ("v3 family", ["results/segment_scores/stage2_sapl_lam0_seed0_test_scores.csv",
                   "results/segment_scores/stage2_sapl_lam0_seed1_test_scores.csv",
                   "results/segment_scores/stage2_sapl_lam0_seed2_test_scores.csv"]),
    ("ST-GCN base", ["results/segment_scores/wacv_stgcn_base_test_scores.csv"]),
    ("ST-GCN remedies", ["results/segment_scores/wacv_stgcn_remedies_test_scores.csv"]),
    ("AGCN base", ["results/segment_scores/wacv_agcn_base_test_scores.csv"]),
    ("AGCN remedies", ["results/segment_scores/wacv_agcn_remedies_test_scores.csv"]),
    ("ours base", ["results/segment_scores/wacv_ours_base_test_scores.csv"]),
    ("v3 no-confidence family",
     ["results/segment_scores/stage1_v3_noconf_seed0_test_scores.csv",
      "results/segment_scores/stage1_v3_noconf_seed1_test_scores.csv",
      "results/segment_scores/stage1_v3_noconf_seed2_test_scores.csv"]),
]


def quality_lr_scores(rows):
    """Refit the four-feature quality regression exactly as in
    stage1/eval_baselines.py and results/make_paper_tables.py."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.array([[r[k] for k in ("q_lowconf", "q_missing", "q_frozen", "q_energy")]
                  for r in rows])
    y = np.array([r["binary_label"] for r in rows])
    tr = np.array([r["split"] == "train" for r in rows])
    scaler = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=1000, random_state=0).fit(
        scaler.transform(X[tr]), y[tr])
    return clf.predict_proba(scaler.transform(X))[:, 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep-only", action="store_true",
                    help="skip the bootstrap and subsample blocks; recompute "
                         "only the threshold sweep and its figure")
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)

    rows = load_index()
    te = [r for r in rows if r["split"] == "test"]
    labels = np.array([r["binary_label"] for r in te])
    child = np.array([r["child_id"] for r in te])
    q_frozen = np.array([r["q_frozen"] for r in te])
    q_arm = np.array([r["q_arm_lowconf"] for r in te])
    good = (q_frozen < 0.10) & (q_arm < 0.30)
    hn_mask = (labels == 1) | good
    n_hn_neg = int((hn_mask & (labels == 0)).sum())

    qlr_all = quality_lr_scores(rows)
    te_ident_idx = {r["identifier"]: i for i, r in enumerate(te)}
    qlr = np.full(len(te), np.nan)
    for r, s in zip(rows, qlr_all):
        if r["split"] == "test":
            qlr[te_ident_idx[r["identifier"]]] = s

    model_scores = {"quality LR": [qlr]}
    for name, paths in MODELS:
        arrs = []
        for p in paths:
            sc = {r["identifier"]: float(r["smm_score"])
                  for r in csv.DictReader(open(p))}
            arrs.append(np.array([sc[r["identifier"]] for r in te]))
        model_scores[name] = arrs

    def family_delta(arrs, seg_mask):
        st, hn = [], []
        for s in arrs:
            st.append(auc(s[seg_mask], labels[seg_mask]))
            hn.append(auc(s[seg_mask & hn_mask], labels[seg_mask & hn_mask]))
        return float(np.mean(st) - np.mean(hn))

    children = np.unique(child)
    by_child = {c: child == c for c in children}

    full = np.ones(len(te), bool)
    print("## C1.1 Paired child-clustered bootstrap CIs for Delta "
          f"({args.iters} resamples of {len(children)} children)\n")
    print("| model | Delta | 95% CI |")
    print("|---|---|---|")
    ci_rows = []
    for name, arrs in (model_scores.items() if not args.sweep_only else []):
        point = family_delta(arrs, full)
        boots = []
        for _ in range(args.iters):
            draw = rng.choice(children, len(children), replace=True)
            # child multiplicity is kept by concatenating index lists
            idx = np.concatenate([np.where(by_child[c])[0] for c in draw])
            seg_labels = labels[idx]
            seg_hn = hn_mask[idx]
            st, hn = [], []
            for s in arrs:
                ss = s[idx]
                st.append(auc(ss, seg_labels))
                hn.append(auc(ss[seg_hn], seg_labels[seg_hn]))
            boots.append(np.mean(st) - np.mean(hn))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"| {name} | {point:.3f} | [{lo:.3f}, {hi:.3f}] |")
        ci_rows.append((name, point, lo, hi))

    if ci_rows:
        lines = [r"\begin{tabular}{lcc}", r"\toprule",
                 r"model & $\Delta$ & 95\% CI \\", r"\midrule"]
        for name, point, lo, hi in ci_rows:
            lines.append(f"{name} & {point:.3f} & [{lo:.3f}, {hi:.3f}] \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open("results/tables/table_delta_ci.tex", "w").write("\n".join(lines) + "\n")

    print(f"\n## C1.2 Random-subsample negative control "
          f"({args.draws} draws of {n_hn_neg} of {(labels == 0).sum()} negatives)\n")
    print("| model | quality-filtered Delta | random-subsample Delta, "
          "mean [2.5%, 97.5%] |")
    print("|---|---|---|")
    neg_idx = np.where(labels == 0)[0]
    for name, arrs in (model_scores.items() if not args.sweep_only else []):
        point = family_delta(arrs, full)
        rand_deltas = []
        for _ in range(args.draws):
            keep = rng.choice(neg_idx, n_hn_neg, replace=False)
            m = labels == 1
            m[keep] = True
            st, sub = [], []
            for s in arrs:
                st.append(auc(s, labels))
                sub.append(auc(s[m], labels[m]))
            rand_deltas.append(np.mean(st) - np.mean(sub))
        rd = np.array(rand_deltas)
        lo, hi = np.percentile(rd, [2.5, 97.5])
        print(f"| {name} | {point:.3f} | {rd.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] |")

    print("\n## C1.3 Threshold sensitivity sweep (frozen values 0.10 / 0.30)\n")
    frozen_grid = [0.05, 0.075, 0.10, 0.125, 0.15]
    arm_grid = [0.25, 0.275, 0.30, 0.325, 0.35]
    sweep_models = ["checkpoint", "v1", "v3 family", "ST-GCN remedies", "quality LR"]
    header = "| f_max | a_max | neg kept | " + " | ".join(sweep_models) + " | ranking by QC AUC |"
    print(header)
    print("|" + "---|" * (3 + len(sweep_models) + 1))
    sweep_records = []
    for fm in frozen_grid:
        for am in arm_grid:
            g = (q_frozen < fm) & (q_arm < am)
            m = (labels == 1) | g
            kept = int((m & (labels == 0)).sum())
            deltas, qc_aucs = [], {}
            for name in sweep_models:
                arrs = model_scores[name]
                st = np.mean([auc(s, labels) for s in arrs])
                qc = np.mean([auc(s[m], labels[m]) for s in arrs])
                deltas.append(st - qc)
                qc_aucs[name] = qc
            rank = " > ".join(sorted(qc_aucs, key=qc_aucs.get, reverse=True))
            cells = " | ".join(f"{d:.3f}" for d in deltas)
            print(f"| {fm} | {am} | {kept} | {cells} | {rank} |")
            sweep_records.append((fm, am, kept, list(deltas)))

    # main-paper figure: Delta per model across the 25 sweep cells,
    # sized and set for a two-column-layout column
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 2.2))
    xs = np.arange(len(sweep_records))
    frozen_cell = [i for i, (fm, am, _, _) in enumerate(sweep_records)
                   if fm == 0.10 and am == 0.30][0]
    colors = ["#555555", "#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    for j, name in enumerate(sweep_models):
        ax.plot(xs, [rec[3][j] for rec in sweep_records], marker="o",
                markersize=3.5, linewidth=1.4, label=name, color=colors[j])
    ax.axvline(frozen_cell, color="0.6", linestyle=(0, (4, 3)), linewidth=1.1)
    ax.annotate("frozen thresholds", (frozen_cell, ax.get_ylim()[1]),
                xytext=(5, -11), textcoords="offset points", fontsize=9,
                color="0.4")
    ax.set_xlabel("sweep cell (25 combinations of the two cutoffs)",
                  fontsize=10)
    ax.set_ylabel(r"$\Delta$ (standard $-$ QC AUC)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("results/figures/fig_threshold_sweep.pdf")
    fig.savefig("results/figures/fig_threshold_sweep.png", dpi=200)
    print("\nwrote results/figures/fig_threshold_sweep.{pdf,png}")


if __name__ == "__main__":
    main()
