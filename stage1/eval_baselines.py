"""Stage 1 baselines under both protocols.

Evaluates, on the released test split (STANDARD) and the filtered
HARD-NEGATIVE protocol:
  - the four naive quality scores from Gate 0.2,
  - a quality-only logistic regression trained on the released TRAIN split
    (the shortcut ceiling any real model must clear),
  - the released PoseConv3D checkpoint (scores from stage0/gate03_segment_eval.py).

Usage:
    python stage1/eval_baselines.py
"""

import csv
import sys

import numpy as np

sys.path.insert(0, "stage1")
from protocols import load_index, evaluate

CKPT_SCORES = "cache/gate03_segment_scores.csv"
QUALITY_FEATURES = ["q_lowconf", "q_missing", "q_frozen", "q_energy"]


def fmt(name, block):
    return (f"{name:28s} n={block['n']:5d} AUC {block['auc']:.4f}  "
            f"p@.85 {block['p@0.85']:.3f} r@.85 {block['r@0.85']:.3f}  "
            f"p@matched-recall {block['p@matched_recall']:.3f}")


def main():
    rows = load_index()

    print("## naive quality scores (sign chosen so higher = SMM)")
    for feat, sign in [("q_lowconf", -1), ("q_missing", -1),
                       ("q_frozen", -1), ("q_energy", 1)]:
        scores = np.array([sign * r[feat] for r in rows])
        res = evaluate(scores, rows)
        for proto, block in res.items():
            print(fmt(f"{feat} ({proto})", block))
        print()

    print("## quality-only logistic regression (trained on train split)")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.array([[r[f] for f in QUALITY_FEATURES] for r in rows])
    y = np.array([r["binary_label"] for r in rows])
    tr = np.array([r["split"] == "train" for r in rows])
    scaler = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(scaler.transform(X[tr]), y[tr])
    scores = clf.predict_proba(scaler.transform(X))[:, 1]
    print("coefficients:", dict(zip(QUALITY_FEATURES, clf.coef_[0].round(3))))
    for proto, block in evaluate(scores, rows).items():
        print(fmt(f"quality-LR ({proto})", block))
    print()

    print("## released PoseConv3D checkpoint")
    ckpt = {r["identifier"]: float(r["smm_score"])
            for r in csv.DictReader(open(CKPT_SCORES))}
    scores = np.array([ckpt.get(r["identifier"], np.nan) for r in rows])
    n_missing = int(np.isnan(scores[[r["split"] == "test" for r in rows]]).sum())
    if n_missing:
        print(f"warning: {n_missing} test segments without checkpoint scores")
    for proto, block in evaluate(scores, rows).items():
        print(fmt(f"checkpoint ({proto})", block))


if __name__ == "__main__":
    main()
