"""Revision analysis B7: does truncation alone explain the per-child
quantification null?

The paper attributes the null correlation between predicted and
annotation-derived SMM totals to the annotation/recording mismatch. The
alternative reading is that segment-trained models simply fail on
untrimmed video. This control simulates the mismatch's simplest form:
if the released recordings are truncated versions of the annotation
source videos, totals computed only from annotations that fit inside
each recording's duration should be what a perfect detector could
recover. If those truncated totals still rank children and recordings
the same way as the full totals, truncation alone cannot produce the
observed null, and the deployment-failure alternative gains weight.

Usage:
    python revision/b7_truncation_control.py
"""

import csv
import io
import pickle
import zipfile
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

DATA_ROOT = "/hd/data/StimmingADSMotion"
TEST_CHILDREN = [3, 21, 30, 59, 61, 66, 68, 108, 113, 116, 132, 139, 148,
                 156, 180, 183, 184, 185, 186, 190, 198, 204, 210, 215,
                 233, 237, 241]


def main():
    zf = zipfile.ZipFile(f"{DATA_ROOT}/asdpose.zip")
    db = list(csv.DictReader(io.TextIOWrapper(zf.open("db.csv"))))
    rec_len = {}
    for r in db:
        if int(r["child_id"]) not in TEST_CHILDREN:
            continue
        with zf.open(f"data/{r['filename']}.pkl") as f:
            rec = pickle.load(f)
        rec_len[(int(r["child_id"]), int(r["assessment_index"]),
                 int(r["camera_index"]))] = rec["length_seconds"]
    print(f"recordings with durations: {len(rec_len)}")

    with open(f"{DATA_ROOT}/dataset.pkl", "rb") as f:
        d = pickle.load(f)
    full = defaultdict(float)
    trunc = defaultdict(float)
    for a in d["annotations"]:
        if a["binary_label"] != 1:
            continue
        p = a["identifier"].split("_")
        key = (int(p[0]), int(p[1]), int(p[2]))
        if key not in rec_len:
            continue
        s, e = float(p[4]), float(p[5])
        full[key] += e - s
        # clip the annotated interval to the released recording's duration
        e_c = min(e, rec_len[key])
        if e_c > s:
            trunc[key] += e_c - s

    keys = sorted(full)
    f_ = np.array([full[k] for k in keys])
    t_ = np.array([trunc.get(k, 0.0) for k in keys])
    r_rec = spearmanr(f_, t_)
    print(f"\nper-recording (n={len(keys)}): full vs truncated annotated "
          f"SMM seconds Spearman {r_rec.statistic:+.4f} (p {r_rec.pvalue:.1e})")
    lost = 1 - t_.sum() / f_.sum()
    print(f"fraction of annotated SMM seconds lost to truncation: {lost:.1%}")

    fc, tc = defaultdict(float), defaultdict(float)
    for k in keys:
        fc[k[0]] += full[k]
        tc[k[0]] += trunc.get(k, 0.0)
    kids = sorted(fc)
    r_ch = spearmanr([fc[c] for c in kids], [tc[c] for c in kids])
    print(f"per-child (n={len(kids)}): Spearman {r_ch.statistic:+.4f} "
          f"(p {r_ch.pvalue:.1e})")

    print("\nreading: if these correlations are high, truncation of the")
    print("annotation timeline alone preserves the ranking, so the observed")
    print("null between model predictions and annotation totals cannot be")
    print("explained by truncation-style misalignment by itself.")


if __name__ == "__main__":
    main()
