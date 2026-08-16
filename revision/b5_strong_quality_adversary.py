"""Revision analysis B5: a stronger quality-only adversary under both
protocols.

The original displacement check used a logistic regression on the same
four features the hard-negative filter was built from, which is partly
circular. This adversary gets an expanded quality feature set (per-joint
confidence means/stds, joint-group missingness, frozen-run statistics,
bounding-box scale and motion of the raw coordinates, segment length)
and gradient boosting. If it also sits near 0.5 under the hard-negative
protocol, the shortcut has been removed rather than displaced into
unmeasured quality correlates.

Usage:
    python revision/b5_strong_quality_adversary.py
"""

import sys

import numpy as np

sys.path.insert(0, "stage1")
from dataset import SegmentDataset
from protocols import load_index, evaluate

HEAD = [0, 1, 2, 3, 4]
ARMS = [5, 6, 7, 8, 9, 10]
LEGS = [11, 12, 13, 14, 15, 16]


def features_of(arr):
    """Quality-only features from one (T,17,3) segment array. No
    normalized pose shape information: only confidence structure,
    missingness, frozen runs, scale, and gross magnitude statistics."""
    kp, sc = arr[..., :2], arr[..., 2]
    t = arr.shape[0]
    f = []
    f.append(t)
    # per-joint-group confidence structure
    for grp in (HEAD, ARMS, LEGS):
        g = sc[:, grp]
        f += [g.mean(), g.std(), (g == 0).mean(), (g < 0.3).mean()]
    # per-joint confidence means (17 features)
    f += list(sc.mean(axis=0))
    # frozen-run statistics
    if t > 1:
        same = (kp[1:] == kp[:-1]).all(axis=(1, 2))
        f.append(same.mean())
        runs, cur = [], 0
        for s in same:
            cur = cur + 1 if s else 0
            runs.append(cur)
        f.append(max(runs) / t)
    else:
        f += [0.0, 0.0]
    # bbox scale and motion of raw coordinates
    valid = sc > 0
    if valid.any():
        xs = np.where(valid, kp[..., 0], np.nan)
        ys = np.where(valid, kp[..., 1], np.nan)
        with np.errstate(all="ignore"):
            w = np.nanmax(xs, axis=1) - np.nanmin(xs, axis=1)
            h = np.nanmax(ys, axis=1) - np.nanmin(ys, axis=1)
            f += [np.nanmean(w), np.nanmean(h), np.nanstd(w), np.nanstd(h)]
            cx = np.nanmean(xs, axis=1)
            f.append(np.nanstd(cx[np.isfinite(cx)]) if np.isfinite(cx).any() else 0.0)
    else:
        f += [0.0] * 5
    return np.nan_to_num(np.array(f, dtype=np.float64))


def main():
    rows = load_index()
    ds = SegmentDataset(rows, train=False)
    X = np.stack([features_of(ds.load_segment(r)) for r in rows])
    y = np.array([r["binary_label"] for r in rows])
    tr = np.array([r["split"] == "train" for r in rows])
    print(f"features: {X.shape[1]}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    clf.fit(X[tr], y[tr])
    scores = clf.predict_proba(X)[:, 1]
    res = evaluate(scores, rows)
    print(f"strong quality adversary: standard {res['standard']['auc']:.4f}  "
          f"hard-negative {res['hard_negative']['auc']:.4f}")


if __name__ == "__main__":
    main()
