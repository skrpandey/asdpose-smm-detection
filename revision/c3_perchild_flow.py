"""W2 analysis C3: who enters the per-child analysis, and what changes if
the bar is lowered.

The per-child analysis requires at least 10 segments of each class and
keeps 25 of 40 test children. Questions answered here:
1. Flow: per-child segment counts (both classes), QC-filter retention,
   and tracking quality for all 40 children.
2. Selection: do the excluded 15 differ from the included 25 in tracking
   quality or segment counts (rank-sum tests)?
3. Sensitivity: per-child AUC statistics at the 10/10 bar and at a 5/5
   bar, for the checkpoint and the v3 family, with the 10th-percentile
   child reported beside the minimum (the minimum of 25 is unstable).

Usage:
    python revision/c3_perchild_flow.py
"""

import csv
import sys

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, "stage1")
from protocols import load_index, auc

V3_PATHS = ["results/segment_scores/stage2_sapl_lam0_seed0_test_scores.csv",
            "results/segment_scores/stage2_sapl_lam0_seed1_test_scores.csv",
            "results/segment_scores/stage2_sapl_lam0_seed2_test_scores.csv"]
CKPT_PATH = "results/segment_scores/gate03_segment_scores.csv"


def perchild_aucs(scores, te, min_per_class):
    by_child = {}
    for i, r in enumerate(te):
        by_child.setdefault(r["child_id"], []).append(i)
    out = {}
    for c, idx in by_child.items():
        y = np.array([te[i]["binary_label"] for i in idx])
        if (y == 1).sum() < min_per_class or (y == 0).sum() < min_per_class:
            continue
        out[c] = auc(scores[idx], y)
    return out


def main():
    rows = load_index()
    te = [r for r in rows if r["split"] == "test"]
    good = [(r["q_frozen"] < 0.10 and r["q_arm_lowconf"] < 0.30) for r in te]

    by_child = {}
    for i, r in enumerate(te):
        by_child.setdefault(r["child_id"], []).append(i)

    print("## C3.1 Participant flow, all 40 test children\n")
    print("| child | SMM | non-SMM | non-SMM kept by QC filter | "
          "mean low-conf | included (10/10) |")
    print("|---|---|---|---|---|---|")
    included, excluded = [], []
    tex = [r"\begin{tabular}{rrrrcc}", r"\toprule",
           r"child & SMM & non-SMM & kept by filter & mean low-conf "
           r"& included \\", r"\midrule"]
    for c in sorted(by_child):
        idx = by_child[c]
        y = np.array([te[i]["binary_label"] for i in idx])
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        kept = sum(1 for i in idx if te[i]["binary_label"] == 0 and good[i])
        q = np.mean([te[i]["q_lowconf"] for i in idx])
        inc = n_pos >= 10 and n_neg >= 10
        (included if inc else excluded).append(
            {"child": c, "n_pos": n_pos, "n_neg": n_neg, "kept": kept,
             "q": q, "n": len(idx)})
        print(f"| {c} | {n_pos} | {n_neg} | {kept} | {q:.2f} | "
              f"{'yes' if inc else 'no'} |")
        tex.append(f"{c} & {n_pos} & {n_neg} & {kept} & {q:.2f} & "
                   f"{'yes' if inc else 'no'} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open("results/tables/table_perchild_flow.tex", "w").write(
        "\n".join(tex) + "\n")

    print(f"\nincluded {len(included)}, excluded {len(excluded)}\n")

    print("## C3.2 Excluded 15 vs included 25\n")
    print("| statistic | included median | excluded median | "
          "Mann-Whitney p |")
    print("|---|---|---|---|")
    for key, label in [("q", "mean low-confidence"), ("n", "total segments"),
                       ("n_pos", "SMM segments")]:
        a = [d[key] for d in included]
        b = [d[key] for d in excluded]
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        print(f"| {label} | {np.median(a):.2f} | {np.median(b):.2f} | "
              f"{p:.3f} |")

    ckpt = {r["identifier"]: float(r["smm_score"])
            for r in csv.DictReader(open(CKPT_PATH))}
    ckpt_s = np.array([ckpt[r["identifier"]] for r in te])
    v3_s = []
    for p in V3_PATHS:
        sc = {r["identifier"]: float(r["smm_score"])
              for r in csv.DictReader(open(p))}
        v3_s.append(np.array([sc[r["identifier"]] for r in te]))

    print("\n## C3.3 Per-child AUC at two inclusion bars\n")
    print("| model | bar | n children | mean | p10 | min |")
    print("|---|---|---|---|---|---|")
    for bar in (10, 5):
        a = perchild_aucs(ckpt_s, te, bar)
        v = np.array(sorted(a.values()))
        print(f"| checkpoint | {bar}/{bar} | {len(v)} | {v.mean():.3f} | "
              f"{np.percentile(v, 10):.3f} | {v.min():.3f} |")
        fam_mean, fam_p10, fam_min, ns = [], [], [], []
        for s in v3_s:
            a = perchild_aucs(s, te, bar)
            v = np.array(sorted(a.values()))
            ns.append(len(v))
            fam_mean.append(v.mean())
            fam_p10.append(np.percentile(v, 10))
            fam_min.append(v.min())
        print(f"| v3 family | {bar}/{bar} | {ns[0]} | "
              f"{np.mean(fam_mean):.3f} $\\pm$ {np.std(fam_mean):.3f} | "
              f"{np.mean(fam_p10):.3f} $\\pm$ {np.std(fam_p10):.3f} | "
              f"{np.mean(fam_min):.3f} $\\pm$ {np.std(fam_min):.3f} |")


if __name__ == "__main__":
    main()
