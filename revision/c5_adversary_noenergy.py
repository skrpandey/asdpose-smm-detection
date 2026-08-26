"""W2 analysis C5: quality adversary without motion energy.

Motion energy (mean inter-frame keypoint displacement) is computed from
the trajectory and can carry genuine behavioral signal, so a
"quality-only" adversary that includes it is not purely non-behavioral.
This script refits the logistic adversary on the three pure tracking
statistics (low-confidence fraction, missing fraction, frozen fraction)
and reports each feature alone, under both protocols.

Feature definitions (stage1/build_cache.py):
  lowconf  = fraction of joint-frames with confidence < 0.3
  missing  = fraction of joint-frames with confidence == 0
  frozen   = fraction of frame pairs with exactly identical coordinates
             across all 17 joints (exact float equality, no tolerance)
  energy   = mean over frames and joints of the L2 inter-frame
             keypoint displacement in pixels

Usage:
    python revision/c5_adversary_noenergy.py
"""

import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "stage1")
from protocols import load_index, evaluate

FEATS = ["q_lowconf", "q_missing", "q_frozen", "q_energy"]


def fit_eval(rows, feats):
    X = np.array([[r[k] for k in feats] for r in rows])
    y = np.array([r["binary_label"] for r in rows])
    tr = np.array([r["split"] == "train" for r in rows])
    scaler = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=1000, random_state=0).fit(
        scaler.transform(X[tr]), y[tr])
    s = clf.predict_proba(scaler.transform(X))[:, 1]
    res = evaluate(s, rows)
    return res["standard"]["auc"], res["hard_negative"]["auc"]


def main():
    rows = load_index()
    print("## C5 Quality adversary with and without motion energy\n")
    print("| features | standard AUC | QC AUC |")
    print("|---|---|---|")
    for feats, label in [
            (FEATS, "all four (paper)"),
            (["q_lowconf", "q_missing", "q_frozen"], "tracking only, no energy"),
            (["q_lowconf"], "lowconf alone"),
            (["q_missing"], "missing alone"),
            (["q_frozen"], "frozen alone"),
            (["q_energy"], "energy alone")]:
        st, hn = fit_eval(rows, feats)
        print(f"| {label} | {st:.3f} | {hn:.3f} |")


if __name__ == "__main__":
    main()
