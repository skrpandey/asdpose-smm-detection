"""Stage 1 detector training.

Trains CompactCTRGCN on the released train split (subject-level val carved
out for epoch selection), then writes per-segment test scores for the
protocol evaluation. With --permute, labels are shuffled once at the start
(subject-blind permutation control); everything else is identical.

Usage:
    python stage1/train.py --config configs/stage1_detector.yaml
    python stage1/train.py --config configs/stage1_detector.yaml --permute \
        --out-prefix cache/stage1_permuted
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, "stage1")
from dataset import SegmentDataset, subject_level_val_split
from model import CompactCTRGCN
from protocols import load_index, auc


@torch.no_grad()
def score_rows(model, rows, cfg, device, batch=128):
    ds = SegmentDataset(rows, cfg["data"]["cache_dir"],
                        cfg["data"]["window"], train=False,
                        velocity=cfg["data"].get("velocity", False))
    dl = DataLoader(ds, batch_size=batch, num_workers=6)
    model.eval()
    out = []
    for x, _ in dl:
        out.append(F.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/stage1_detector.yaml")
    ap.add_argument("--permute", action="store_true")
    ap.add_argument("--out-prefix", default="cache/stage1_detector")
    ap.add_argument("--seed", type=int, default=None,
                    help="override config seed (for seed-spread runs)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed = args.seed if args.seed is not None else cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    rows = load_index()
    tr_rows, va_rows, val_children = subject_level_val_split(
        rows, cfg["data"]["val_child_fraction"], seed=cfg["seed"])
    te_rows = [r for r in rows if r["split"] == "test"]
    print(f"train {len(tr_rows)} / val {len(va_rows)} "
          f"({len(val_children)} children) / test {len(te_rows)}", flush=True)

    if args.permute:
        rng = np.random.RandomState(seed)
        labels = np.array([r["binary_label"] for r in tr_rows])
        rng.shuffle(labels)
        for r, l in zip(tr_rows, labels):
            r["binary_label"] = int(l)
        print("permutation control: train labels shuffled", flush=True)

    device = "cuda"
    model = CompactCTRGCN(c_in=cfg["model"]["c_in"],
                          widths=tuple(cfg["model"]["widths"]),
                          blocks_per_stage=cfg["model"]["blocks_per_stage"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e3:.1f}K", flush=True)

    tcfg = cfg["train"]
    ds = SegmentDataset(tr_rows, cfg["data"]["cache_dir"], cfg["data"]["window"],
                        train=True, flip_prob=tcfg["flip_prob"], seed=seed,
                        joint_dropout=tcfg.get("joint_dropout", 0.0),
                        velocity=cfg["data"].get("velocity", False))
    labels = np.array([r["binary_label"] for r in tr_rows])
    if tcfg["balanced_sampling"]:
        w = np.where(labels == 1, 1.0 / (labels == 1).sum(),
                     1.0 / (labels == 0).sum())
        sampler = WeightedRandomSampler(torch.from_numpy(w), len(w))
        dl = DataLoader(ds, batch_size=tcfg["batch_size"], sampler=sampler,
                        num_workers=6, drop_last=True)
    else:
        dl = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True,
                        num_workers=6, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tcfg["epochs"])
    crit = nn.CrossEntropyLoss()

    best = (-1.0, None)
    t0 = time.time()
    for epoch in range(tcfg["epochs"]):
        model.train()
        tot = seen = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            seen += len(y)
        sched.step()
        va_scores = score_rows(model, va_rows, cfg, device)
        va_auc = auc(va_scores, [r["binary_label"] for r in va_rows])
        marker = ""
        if va_auc > best[0]:
            best = (va_auc, {k: v.cpu().clone() for k, v in model.state_dict().items()})
            marker = " *"
        print(f"epoch {epoch+1:3d} loss {tot/seen:.4f} val AUC {va_auc:.4f}"
              f" ({(time.time()-t0)/60:.1f} min){marker}", flush=True)

    model.load_state_dict(best[1])
    te_scores = score_rows(model, te_rows, cfg, device)
    te_auc = auc(te_scores, [r["binary_label"] for r in te_rows])
    print(f"\nbest val AUC {best[0]:.4f}; test AUC {te_auc:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    torch.save({"state_dict": best[1], "config": cfg, "seed": seed,
                "val_children": val_children, "permuted": args.permute,
                "val_auc": best[0]},
               f"{args.out_prefix}.pt")
    with open(f"{args.out_prefix}_test_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["identifier", "binary_label", "smm_score"])
        for r, s in zip(te_rows, te_scores):
            w.writerow([r["identifier"], r["binary_label"], f"{s:.6f}"])
    print(f"saved {args.out_prefix}.pt and test scores", flush=True)


if __name__ == "__main__":
    main()
