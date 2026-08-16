"""Revision analysis B2: child-clustered bootstrap confidence intervals.

The test children, not the segments, are the sampling unit: a different
draw of 40 children gives different AUCs. Resamples test children with
replacement (default 10,000 iterations) and reports 95% percentile CIs
for: each model's AUC under both protocols, the shortcut delta, the
v3-minus-checkpoint margin, and the two per-child Spearman correlations.

Usage:
    python revision/b2_bootstrap_cis.py
"""

import argparse
import csv
import sys

import numpy as np

sys.path.insert(0, "stage1")
from protocols import load_index, protocol_masks, auc

MODELS = {
    "checkpoint": "results/segment_scores/gate03_segment_scores.csv",
    "v3": "results/segment_scores/stage2_sapl_lam0_seed0_test_scores.csv",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)

    rows = load_index()
    masks = protocol_masks(rows)
    te_idx = np.array([i for i, r in enumerate(rows) if r["split"] == "test"])
    child_of = np.array([rows[i]["child_id"] for i in te_idx])
    labels = np.array([rows[i]["binary_label"] for i in te_idx])
    hn = masks["hard_negative"][te_idx]
    quality = np.array([rows[i]["q_lowconf"] for i in te_idx])

    scores = {}
    for name, path in MODELS.items():
        sc = {r["identifier"]: float(r["smm_score"]) for r in csv.DictReader(open(path))}
        scores[name] = np.array([sc[rows[i]["identifier"]] for i in te_idx])

    children = np.unique(child_of)
    by_child = {c: np.where(child_of == c)[0] for c in children}

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        if ra.std() == 0 or rb.std() == 0:
            return np.nan
        return np.corrcoef(ra, rb)[0, 1]

    def stats_for(idx):
        y, h = labels[idx], hn[idx]
        out = {}
        for name, s in scores.items():
            sv = s[idx]
            out[f"{name}_std"] = auc(sv, y)
            out[f"{name}_hn"] = auc(sv[h], y[h])
            out[f"{name}_delta"] = out[f"{name}_hn"] - out[f"{name}_std"]
        out["margin_std"] = out["v3_std"] - out["checkpoint_std"]
        out["margin_hn"] = out["v3_hn"] - out["checkpoint_hn"]
        # per-child spearman(quality, auc) for each model
        cids = child_of[idx]
        for name, s in scores.items():
            sv = s[idx]
            pcs, qs = [], []
            for c in np.unique(cids):
                m = cids == c
                yy = y[m]
                if (yy == 1).sum() < 10 or (yy == 0).sum() < 10:
                    continue
                pcs.append(auc(sv[m], yy))
                qs.append(quality[idx][m].mean())
            out[f"{name}_qual_rho"] = spearman(np.array(qs), np.array(pcs)) if len(pcs) > 5 else np.nan
        return out

    point = stats_for(np.arange(len(te_idx)))
    boots = {k: [] for k in point}
    for _ in range(args.iters):
        draw = rng.choice(children, size=len(children), replace=True)
        idx = np.concatenate([by_child[c] for c in draw])
        st = stats_for(idx)
        for k, v in st.items():
            boots[k].append(v)

    print(f"child-clustered bootstrap, {args.iters} iterations, "
          f"{len(children)} test children\n")
    for k in point:
        b = np.array(boots[k], dtype=float)
        b = b[np.isfinite(b)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        print(f"{k:22s} point {point[k]:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
