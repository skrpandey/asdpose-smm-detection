"""Stage 2: subject-adversarial training (SAPL) on the v3 recipe.

A gradient-reversal adversary head predicts which of the training
children a pooled feature vector came from; the backbone is trained to
defeat it while keeping the SMM task loss. lambda scales the reversed
gradient (lambda 0 is exactly the v3 recipe, so cache/stage1_detector_v3
serves as that arm of the tradeoff curve).

Identity classes are the train-split children after the val carve-out
(val children are unseen by both heads). Outputs match stage1/train.py:
a checkpoint plus per-segment test scores for the protocol and per-child
analyses.

Usage:
    python stage2/train_sapl.py --config configs/stage2_sapl.yaml \
        --lam 0.3 --out-prefix cache/stage2_sapl_lam0.3
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
from torch.utils.data import DataLoader

sys.path.insert(0, "stage1")
sys.path.insert(0, "stage2")
from dataset import SegmentDataset, subject_level_val_split
from model import CompactCTRGCN
from protocols import load_index, auc


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class SAPLModel(nn.Module):
    def __init__(self, backbone, n_identities, adv_hidden=128):
        super().__init__()
        self.backbone = backbone
        feat = backbone.fc.in_features
        self.adversary = nn.Sequential(
            nn.Linear(feat, adv_hidden), nn.ReLU(inplace=True),
            nn.Linear(adv_hidden, n_identities))

    def forward(self, x, lam=0.0):
        feats = self.backbone.forward_features(x)
        task_logits = self.backbone.fc(feats)
        adv_logits = self.adversary(GradReverse.apply(feats, lam))
        return task_logits, adv_logits


class IdentityDataset(SegmentDataset):
    """SegmentDataset that also yields the child-identity class index."""

    def __init__(self, rows, child_to_class, **kw):
        super().__init__(rows, **kw)
        self.child_to_class = child_to_class

    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        return x, y, self.child_to_class[self.rows[i]["child_id"]]


@torch.no_grad()
def score_rows(model, rows, cfg, device, batch=128):
    ds = SegmentDataset(rows, cfg["data"]["cache_dir"], cfg["data"]["window"],
                        train=False, velocity=cfg["data"]["velocity"],
                        normalize=cfg["data"]["normalize_mode"])
    dl = DataLoader(ds, batch_size=batch, num_workers=6)
    model.eval()
    out = []
    for x, _ in dl:
        logits, _ = model(x.to(device))
        out.append(F.softmax(logits, 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/stage2_sapl.yaml")
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="override config seed (val split stays fixed)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed = args.seed if args.seed is not None else cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    rows = load_index()
    tr_rows, va_rows, _ = subject_level_val_split(
        rows, cfg["data"]["val_child_fraction"], seed=cfg["seed"])
    te_rows = [r for r in rows if r["split"] == "test"]
    train_children = sorted({r["child_id"] for r in tr_rows})
    child_to_class = {c: i for i, c in enumerate(train_children)}
    print(f"train {len(tr_rows)} segments / {len(train_children)} identities; "
          f"val {len(va_rows)}; test {len(te_rows)}; lambda {args.lam}", flush=True)

    device = "cuda"
    backbone = CompactCTRGCN(c_in=cfg["model"]["c_in"],
                             widths=tuple(cfg["model"]["widths"]),
                             blocks_per_stage=cfg["model"]["blocks_per_stage"])
    model = SAPLModel(backbone, len(train_children)).to(device)

    tcfg = cfg["train"]
    ds = IdentityDataset(tr_rows, child_to_class,
                         cache_dir=cfg["data"]["cache_dir"],
                         window=cfg["data"]["window"], train=True,
                         flip_prob=tcfg["flip_prob"], seed=seed,
                         joint_dropout=tcfg["joint_dropout"],
                         velocity=cfg["data"]["velocity"],
                         normalize=cfg["data"]["normalize_mode"])
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
        task_sum = adv_sum = adv_hits = seen = 0.0
        for x, y, cid in dl:
            x, y, cid = x.to(device), y.to(device), cid.to(device)
            opt.zero_grad()
            task_logits, adv_logits = model(x, lam=args.lam)
            task_loss = crit(task_logits, y)
            adv_loss = crit(adv_logits, cid)
            (task_loss + adv_loss).backward()
            opt.step()
            task_sum += task_loss.item() * len(y)
            adv_sum += adv_loss.item() * len(y)
            adv_hits += (adv_logits.argmax(1) == cid).float().sum().item()
            seen += len(y)
        sched.step()
        va_scores = score_rows(model, va_rows, cfg, device)
        va_auc = auc(va_scores, [r["binary_label"] for r in va_rows])
        marker = ""
        if va_auc > best[0]:
            best = (va_auc, {k: v.cpu().clone() for k, v in model.state_dict().items()})
            marker = " *"
        print(f"epoch {epoch+1:3d} task {task_sum/seen:.4f} adv {adv_sum/seen:.4f} "
              f"adv-acc {adv_hits/seen:.3f} val AUC {va_auc:.4f} "
              f"({(time.time()-t0)/60:.1f} min){marker}", flush=True)

    model.load_state_dict(best[1])
    te_scores = score_rows(model, te_rows, cfg, device)
    te_auc = auc(te_scores, [r["binary_label"] for r in te_rows])
    print(f"\nlambda {args.lam}: best val AUC {best[0]:.4f}; test AUC {te_auc:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    torch.save({"state_dict": best[1], "config": cfg, "lam": args.lam,
                "seed": seed, "n_identities": len(train_children),
                "val_auc": best[0]}, f"{args.out_prefix}.pt")
    with open(f"{args.out_prefix}_test_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["identifier", "binary_label", "smm_score"])
        for r, s in zip(te_rows, te_scores):
            w.writerow([r["identifier"], r["binary_label"], f"{s:.6f}"])
    print(f"saved {args.out_prefix}.pt and test scores", flush=True)


if __name__ == "__main__":
    main()
