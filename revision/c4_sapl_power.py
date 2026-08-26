"""W2 analysis C4: how small a worst-child effect could the adversarial
comparison have detected?

The matched families (three seeds at lambda 0 and 0.3) showed 0.718 vs
0.696 worst-child AUC, a difference of the same order as the seed
spread. This script quantifies the claim boundary: a child-clustered
bootstrap CI on the paired per-child difference between the family-mean
score vectors, and a minimum detectable effect for the worst-child
statistic from the observed seed variances (the difference two matched
three-seed families would need before a two-sample comparison at
roughly two standard errors could see it).

Usage:
    python revision/c4_sapl_power.py
"""

import csv
import sys

import numpy as np

sys.path.insert(0, "stage1")
from protocols import load_index, auc

FAMS = {
    "lam0": ["results/segment_scores/stage2_sapl_lam0_seed0_test_scores.csv",
             "results/segment_scores/stage2_sapl_lam0_seed1_test_scores.csv",
             "results/segment_scores/stage2_sapl_lam0_seed2_test_scores.csv"],
    "lam0.3": ["results/segment_scores/stage2_sapl_lam0.3_test_scores.csv",
               "results/segment_scores/stage2_sapl_lam0.3_seed1_test_scores.csv",
               "results/segment_scores/stage2_sapl_lam0.3_seed2_test_scores.csv"],
}


def main():
    rng = np.random.RandomState(0)
    rows = load_index()
    te = [r for r in rows if r["split"] == "test"]

    by_child = {}
    for i, r in enumerate(te):
        by_child.setdefault(r["child_id"], []).append(i)
    eligible = {c: idx for c, idx in by_child.items()
                if sum(te[i]["binary_label"] for i in idx) >= 10
                and sum(1 - te[i]["binary_label"] for i in idx) >= 10}

    scores = {}
    for fam, paths in FAMS.items():
        arrs = []
        for p in paths:
            sc = {r["identifier"]: float(r["smm_score"])
                  for r in csv.DictReader(open(p))}
            arrs.append(np.array([sc[r["identifier"]] for r in te]))
        scores[fam] = arrs

    # per-child AUC per seed per family
    pc = {fam: {c: [auc(s[idx], [te[i]["binary_label"] for i in idx])
                    for s in arrs]
                for c, idx in eligible.items()}
          for fam, arrs in scores.items()}

    children = sorted(eligible)
    d_child = np.array([np.mean(pc["lam0"][c]) - np.mean(pc["lam0.3"][c])
                        for c in children])
    boots = []
    for _ in range(10000):
        draw = rng.choice(len(children), len(children), replace=True)
        boots.append(d_child[draw].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print("## C4 Adversarial worst-child comparison: resolution limit\n")
    print(f"paired per-child AUC difference (lam0 - lam0.3), family means, "
          f"n={len(children)} children:")
    print(f"  mean {d_child.mean():+.3f}, child-clustered bootstrap 95% CI "
          f"[{lo:+.3f}, {hi:+.3f}]\n")

    worst = {fam: [min(pc[fam][c][k] for c in children) for k in range(3)]
             for fam in FAMS}
    for fam, w in worst.items():
        print(f"worst-child AUC, {fam}: {np.mean(w):.3f} sd {np.std(w, ddof=1):.3f} "
              f"(seeds: {', '.join(f'{x:.3f}' for x in w)})")
    se = np.sqrt(np.var(worst["lam0"], ddof=1) / 3
                 + np.var(worst["lam0.3"], ddof=1) / 3)
    print(f"\nobserved worst-child difference: "
          f"{np.mean(worst['lam0']) - np.mean(worst['lam0.3']):+.3f}")
    print(f"standard error of the family difference: {se:.3f}")
    print(f"minimum detectable effect (~2 SE): {2 * se:.3f}")


if __name__ == "__main__":
    main()
