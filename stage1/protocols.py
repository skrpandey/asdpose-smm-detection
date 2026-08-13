"""Shared evaluation protocols for Stage 1.

Answers, for any per-segment score vector: how does it do under the
STANDARD protocol (full released test split) and the HARD-NEGATIVE
protocol (same positives, negatives restricted to child-present,
well-tracked segments per configs/protocols.yaml)?

Import surface: load_index(), protocol_masks(), evaluate(), auc().
Run directly to print the protocol composition.
"""

import csv

import numpy as np
import yaml

CONFIG_PATH = "configs/protocols.yaml"
INDEX_PATH = "cache/segments/index.csv"


def load_index(path=INDEX_PATH):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["binary_label"] = int(r["binary_label"])
        r["child_id"] = int(r["child_id"])
        r["n_frames"] = int(r["n_frames"])
        r["byte_offset"] = int(r["byte_offset"])
        for k in ("q_lowconf", "q_missing", "q_frozen", "q_energy", "q_arm_lowconf"):
            r[k] = float(r[k])
    return rows


def protocol_masks(rows, config_path=CONFIG_PATH):
    """Returns dict protocol name -> boolean mask over rows."""
    cfg = yaml.safe_load(open(config_path))
    test = np.array([r["split"] == "test" for r in rows])
    labels = np.array([r["binary_label"] for r in rows])

    masks = {"standard": test}

    nf = cfg["hard_negative"]["negative_filter"]
    good = np.array([r["q_frozen"] < nf["q_frozen_max"]
                     and r["q_arm_lowconf"] < nf["q_arm_lowconf_max"]
                     for r in rows])
    masks["hard_negative"] = test & ((labels == 1) | good)
    return masks


def auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    m = np.isfinite(scores)
    scores, labels = scores[m], labels[m]
    n_pos, n_neg = (labels == 1).sum(), (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def prf_at(scores, labels, threshold):
    p = np.asarray(scores) > threshold
    labels = np.asarray(labels)
    tp = int(((labels == 1) & p).sum())
    fp = int(((labels == 0) & p).sum())
    fn = int(((labels == 1) & ~p).sum())
    tn = int(((labels == 0) & ~p).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return precision, recall, specificity


def precision_at_matched_recall(scores, labels, target_recall):
    """Highest threshold whose recall still reaches the target."""
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    pos_scores = np.sort(scores[labels == 1])[::-1]
    k = int(np.ceil(target_recall * len(pos_scores)))
    if k == 0 or k > len(pos_scores):
        return float("nan"), float("nan")
    threshold = pos_scores[k - 1]
    pr, rc, _ = prf_at(scores, labels, threshold - 1e-12)
    return pr, rc


def evaluate(scores, rows, config_path=CONFIG_PATH):
    """Full metric block per protocol. scores aligned with rows."""
    cfg = yaml.safe_load(open(config_path))
    scores = np.asarray(scores, float)
    labels = np.array([r["binary_label"] for r in rows])
    out = {}
    for name, mask in protocol_masks(rows, config_path).items():
        s, y = scores[mask], labels[mask]
        block = {"n": int(mask.sum()), "n_pos": int((y == 1).sum()),
                 "n_neg": int((y == 0).sum()), "auc": auc(s, y)}
        for th in cfg["report_thresholds"]:
            pr, rc, sp = prf_at(s, y, th)
            block[f"p@{th}"], block[f"r@{th}"], block[f"s@{th}"] = pr, rc, sp
        pr, rc = precision_at_matched_recall(s, y, cfg["matched_recall"])
        block["p@matched_recall"] = pr
        out[name] = block
    return out


if __name__ == "__main__":
    rows = load_index()
    masks = protocol_masks(rows)
    labels = np.array([r["binary_label"] for r in rows])
    for name, mask in masks.items():
        y = labels[mask]
        print(f"{name}: {mask.sum()} segments ({(y == 1).sum()} SMM, "
              f"{(y == 0).sum()} non-SMM)")
    hn, st = masks["hard_negative"], masks["standard"]
    dropped = int((st & ~hn).sum())
    print(f"hard-negative filter drops {dropped} negatives "
          f"({dropped / (labels[st] == 0).sum():.1%} of standard negatives)")
