"""Gate 0.4: does a training-free periodicity score separate SMM frames from
non-SMM frames?

Score: sliding-window autocorrelation strength of arm-joint speed. Per
segment, build a mean speed signal over well-tracked arm joints (shoulders,
elbows, wrists; Gate 0.2 found these the most reliable), then for each frame
take a centered window and compute the max normalized autocorrelation over
lags 0.25 to 2.0 seconds. Frames inherit their segment's binary label.

Untrimmed recordings cannot be used because the released annotations do not
align with the released recordings (annotated end times exceed recording
durations, and no segment matches its stated timestamp by cross-correlation;
main paper Section 3.1), so this runs on the annotated segments instead.

Controls (both must lose to the periodicity score, otherwise the signal is a
confound):
  1. valid-data fraction per window (pose quality differs by class, Gate 0.2)
  2. raw motion energy (mean speed) per window

Secondary: envelope-structure metric per segment (correlation of the speed
envelope with its heavily smoothed version).

PASS: median per-recording periodicity AUC >= 0.65. MARGINAL: 0.55-0.65.
FAIL: ~0.5.

Usage:
    python stage0/gate04_periodicity.py --split test
"""

import argparse
import pickle
from collections import defaultdict

import numpy as np

ARM_JOINTS = [5, 6, 7, 8, 9, 10]
LOW_CONF = 0.3


def speed_signal(kp, score):
    """(T,) mean speed over arm joints, nan where untracked."""
    xy = kp[:, ARM_JOINTS, :]
    conf = score[:, ARM_JOINTS]
    v = np.linalg.norm(np.diff(xy, axis=0), axis=2)          # (T-1, J)
    valid = (conf[1:] >= LOW_CONF) & (conf[:-1] >= LOW_CONF) \
        & (xy[1:] != 0).any(axis=2) & (xy[:-1] != 0).any(axis=2)
    v[~valid] = np.nan
    with np.errstate(invalid="ignore"):
        sig = np.nanmean(v, axis=1)
    return sig, np.isfinite(sig)


def interp_nan(sig):
    out = sig.copy()
    bad = ~np.isfinite(out)
    if bad.all():
        return None
    idx = np.arange(len(out))
    out[bad] = np.interp(idx[bad], idx[~bad], out[~bad])
    return out


def window_autocorr_score(sig, fps, win_s=3.0, lag_lo_s=0.25, lag_hi_s=2.0):
    """(T,) max normalized autocorrelation per centered window."""
    W = int(win_s * fps)
    lags = np.arange(max(2, int(lag_lo_s * fps)), int(lag_hi_s * fps) + 1)
    T = len(sig)
    if T < W or W <= lags[-1] + 4:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(sig, W)   # (T-W+1, W)
    w = windows - windows.mean(axis=1, keepdims=True)
    denom = (w ** 2).sum(axis=1)
    best = np.full(w.shape[0], -1.0)
    for lag in lags:
        num = (w[:, :-lag] * w[:, lag:]).sum(axis=1)
        d = np.sqrt((w[:, :-lag] ** 2).sum(axis=1) * (w[:, lag:] ** 2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            ac = np.where(d > 0, num / d, -1.0)
        np.maximum(best, ac, out=best)
    # windows -> frames (centered)
    out = np.full(T, np.nan)
    out[W // 2: W // 2 + len(best)] = best
    # edge frames take the nearest computed value
    out[:W // 2] = best[0]
    out[W // 2 + len(best):] = best[-1]
    return out


def window_mean(x, fps, win_s=3.0):
    W = int(win_s * fps)
    if len(x) < W:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(x, W)
    m = windows.mean(axis=1)
    out = np.full(len(x), np.nan)
    out[W // 2: W // 2 + len(m)] = m
    out[:W // 2] = m[0]
    out[W // 2 + len(m):] = m[-1]
    return out


def envelope_structure(sig, fps):
    """Correlation of |signal| with its heavily smoothed version."""
    env = np.abs(sig - np.nanmean(sig))
    k = max(3, int(0.5 * fps))
    kernel = np.ones(k) / k
    smooth = np.convolve(env, kernel, mode="same")
    if env.std() == 0 or smooth.std() == 0:
        return np.nan
    return float(np.corrcoef(env, smooth)[0, 1])


def auc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    m = np.isfinite(scores)
    scores, labels = scores[m], labels[m]
    n_pos, n_neg = (labels == 1).sum(), (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="data/StimmingADSMotion")
    ap.add_argument("--split", choices=["test", "train", "all"], default="test")
    ap.add_argument("--min-valid", type=float, default=0.5,
                    help="skip segments with less than this fraction of usable speed samples")
    args = ap.parse_args()

    with open(f"{args.data_root}/dataset.pkl", "rb") as f:
        d = pickle.load(f)
    ann = d["annotations"]
    if args.split != "all":
        wanted = set(d["split"][args.split])
        ann = [a for a in ann if a["identifier"] in wanted]

    frame_scores = {"periodicity": [], "valid_frac": [], "energy": []}
    frame_labels = []
    per_recording = defaultdict(lambda: {"p": [], "l": []})
    seg_env, seg_lab = [], []
    n_used = n_skipped = 0

    for a in ann:
        fps = a["fps"]
        sig_raw, valid = speed_signal(a["keypoint"], a["keypoint_score"])
        if valid.mean() < args.min_valid:
            n_skipped += 1
            continue
        sig = interp_nan(sig_raw)
        per = window_autocorr_score(sig, fps)
        if per is None:
            n_skipped += 1
            continue
        vf = window_mean(valid.astype(float), fps)
        en = window_mean(sig, fps)
        n_used += 1
        lab = a["binary_label"]
        labs = np.full(len(per), lab)
        frame_scores["periodicity"].append(per)
        frame_scores["valid_frac"].append(vf)
        frame_scores["energy"].append(en)
        frame_labels.append(labs)
        rec = "_".join(a["identifier"].split("_")[:3])
        per_recording[rec]["p"].append(per)
        per_recording[rec]["l"].append(labs)
        seg_env.append(envelope_structure(sig, fps))
        seg_lab.append(lab)

    labels = np.concatenate(frame_labels)
    print(f"## Gate 0.4 periodicity check ({args.split} split)")
    print(f"segments used: {n_used}, skipped (poor tracking/too short): {n_skipped}")
    print(f"frames: {len(labels)} ({(labels == 1).mean():.1%} SMM)\n")
    for name, chunks in frame_scores.items():
        a_ = auc(np.concatenate(chunks), labels)
        print(f"frame-level AUC, {name}: {a_:.4f}")

    rec_aucs = []
    for rec, v in per_recording.items():
        a_ = auc(np.concatenate(v["p"]), np.concatenate(v["l"]))
        if np.isfinite(a_):
            rec_aucs.append(a_)
    rec_aucs = np.array(rec_aucs)
    print(f"\nper-recording periodicity AUC ({len(rec_aucs)} recordings with both classes):")
    print(f"median {np.median(rec_aucs):.4f}, mean {rec_aucs.mean():.4f}, "
          f"q25 {np.percentile(rec_aucs, 25):.4f}, q75 {np.percentile(rec_aucs, 75):.4f}, "
          f"min {rec_aucs.min():.4f}, max {rec_aucs.max():.4f}")
    print(f"recordings with AUC > 0.65: {(rec_aucs > 0.65).mean():.1%}, "
          f"> 0.55: {(rec_aucs > 0.55).mean():.1%}")

    env_auc = auc(seg_env, seg_lab)
    print(f"\nsecondary, segment-level envelope-structure AUC: {env_auc:.4f}")


if __name__ == "__main__":
    main()
