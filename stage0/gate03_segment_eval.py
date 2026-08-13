"""Gate 0.3a: segment-level evaluation of the released checkpoint on the
released test split.

Scores every test segment in dataset.pkl with the reimplemented PoseConv3D
test pipeline (10 clips x flip double, softmax-prob average) and reports
precision / recall / specificity at threshold 0.85 plus ROC AUC. Segments
longer than 200 frames get the same random 200-frame crop their test-time
Splitter applies (seeded here for repeatability).

This is the sanity half of the reproduction. The published numbers are
frame-wise over untrimmed videos (gate03_framewise.py); segment-level numbers
should be in the same neighborhood, not identical.

Usage:
    python stage0/gate03_segment_eval.py --out cache/gate03_segment_scores.csv
"""

import argparse
import csv
import os
import pickle
import random
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from posec3d import PoseConv3D, preprocess_window

SEQ_LEN = 200


def crop_and_preprocess(args):
    idx, kp, sc, img_shape, seed = args
    total = kp.shape[1]
    if total > SEQ_LEN:
        rng = random.Random(seed)
        s = rng.randint(0, total - SEQ_LEN)
        kp, sc = kp[:, s:s + SEQ_LEN], sc[:, s:s + SEQ_LEN]
    return idx, preprocess_window(kp, sc, img_shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="/hd/data/StimmingADSMotion")
    ap.add_argument("--out", default="cache/gate03_segment_scores.csv")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    with open(f"{args.data_root}/dataset.pkl", "rb") as f:
        d = pickle.load(f)
    test_ids = set(d["split"]["test"])
    segs = [a for a in d["annotations"] if a["identifier"] in test_ids]
    print(f"test segments: {len(segs)}", flush=True)

    model = PoseConv3D()
    model.load_mmaction_checkpoint(f"{args.data_root}/model.pth")
    model = model.cuda().eval()

    jobs = [(i, a["keypoint"][None], a["keypoint_score"][None],
             a["img_shape"], args.seed + i) for i, a in enumerate(segs)]

    scores = np.zeros(len(segs))
    t0 = time.time()
    with Pool(args.workers) as pool, torch.no_grad():
        for n_done, (idx, imgs) in enumerate(
                pool.imap_unordered(crop_and_preprocess, jobs, chunksize=4), 1):
            x = torch.from_numpy(imgs).cuda()
            prob = F.softmax(model(x), dim=1).mean(dim=0)
            scores[idx] = prob[1].item()
            if n_done % 250 == 0:
                rate = n_done / (time.time() - t0)
                print(f"{n_done}/{len(segs)} ({rate:.1f}/s, "
                      f"eta {(len(segs)-n_done)/rate/60:.0f} min)", flush=True)

    labels = np.array([a["binary_label"] for a in segs])
    pred = scores > args.threshold
    tp = int(((labels == 1) & pred).sum())
    fp = int(((labels == 0) & pred).sum())
    fn = int(((labels == 1) & ~pred).sum())
    tn = int(((labels == 0) & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)

    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = (labels == 1).sum(), (labels == 0).sum()
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    print(f"\nthreshold {args.threshold}")
    print(f"TP {tp} FP {fp} FN {fn} TN {tn}")
    print(f"precision {precision:.4f} recall {recall:.4f} "
          f"specificity {specificity:.4f} AUC {auc:.4f}")
    print(f"(published frame-wise: precision 0.6682 recall 0.9253 "
          f"specificity 0.9545 on the 21-child reannotated test set)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["identifier", "binary_label", "action_name", "smm_score"])
        for a, s in zip(segs, scores):
            w.writerow([a["identifier"], a["binary_label"], a["action_name"], f"{s:.6f}"])
    print(f"per-segment scores written to {args.out}")


if __name__ == "__main__":
    main()
