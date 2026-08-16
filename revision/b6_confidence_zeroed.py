"""Revision analysis B6: score v3 with the confidence channel zeroed at
test time.

Answers: how much of v3's performance flows through the OpenPose
confidence input? If the hard-negative AUC barely moves with the channel
zeroed, the model is not using confidence as a quality side-channel on
well-tracked segments.

Usage:
    python revision/b6_confidence_zeroed.py
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "stage1")
from dataset import SegmentDataset
from model import CompactCTRGCN
from protocols import load_index, evaluate


class ConfZeroDataset(SegmentDataset):
    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        x[-1] = 0.0   # confidence is the last channel
        return x, y


def main():
    ckpt = torch.load("cache/stage1_detector_v3.pt", map_location="cpu",
                      weights_only=False)
    cfg = ckpt["config"]
    model = CompactCTRGCN(c_in=cfg["model"]["c_in"],
                          widths=tuple(cfg["model"]["widths"]),
                          blocks_per_stage=cfg["model"]["blocks_per_stage"])
    model.load_state_dict(ckpt["state_dict"])
    model = model.cuda().eval()

    rows = load_index()
    te = [r for r in rows if r["split"] == "test"]
    for name, ds_cls in [("intact", SegmentDataset), ("confidence zeroed", ConfZeroDataset)]:
        ds = ds_cls(te, cfg["data"]["cache_dir"], cfg["data"]["window"],
                    train=False, velocity=cfg["data"]["velocity"],
                    normalize=cfg["data"]["normalize_mode"])
        dl = DataLoader(ds, batch_size=128, num_workers=6)
        out = []
        with torch.no_grad():
            for x, _ in dl:
                out.append(F.softmax(model(x.cuda()), 1)[:, 1].cpu().numpy())
        sc = np.concatenate(out)
        scores = np.full(len(rows), np.nan)
        te_idx = [i for i, r in enumerate(rows) if r["split"] == "test"]
        scores[te_idx] = sc
        res = evaluate(scores, rows)
        print(f"v3, {name:18s}: standard {res['standard']['auc']:.4f}  "
              f"hard-negative {res['hard_negative']['auc']:.4f}")


if __name__ == "__main__":
    main()
