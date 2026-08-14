"""Cache-backed segment dataset for Stage 1.

Reads windows from cache/segments/keypoints.f32 (built by build_cache.py).
Each item is (3, window, 17): window-normalized x, y plus raw confidence.
Missing joints (score == 0) keep zero coordinates after normalization.

Train mode: random temporal crop, optional horizontal flip (x -> -x plus
left/right joint swap, valid in centered coordinates). Eval mode: center
crop. Short segments are looped to window length.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

LEFT_KP = [1, 3, 5, 7, 9, 11, 13, 15]
RIGHT_KP = [2, 4, 6, 8, 10, 12, 14, 16]
FLIP_ORDER = list(range(17))
for l, r in zip(LEFT_KP, RIGHT_KP):
    FLIP_ORDER[l], FLIP_ORDER[r] = r, l


class SegmentDataset(Dataset):
    def __init__(self, rows, cache_dir="cache/segments", window=200,
                 train=False, flip_prob=0.5, seed=0, joint_dropout=0.0,
                 velocity=False, normalize="window-std"):
        self.rows = rows
        self.mm = np.memmap(f"{cache_dir}/keypoints.f32", dtype=np.float32,
                            mode="r")
        self.window = window
        self.train = train
        self.flip_prob = flip_prob
        self.joint_dropout = joint_dropout
        self.velocity = velocity
        assert normalize in ("window-std", "image")
        self.normalize = normalize
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.rows)

    def load_segment(self, r):
        start = r["byte_offset"] // 4
        n = r["n_frames"] * 17 * 3
        return self.mm[start:start + n].reshape(r["n_frames"], 17, 3).copy()

    def __getitem__(self, i):
        r = self.rows[i]
        arr = self.load_segment(r)                      # (T,17,3)
        t = arr.shape[0]
        w = self.window
        if t < w:
            reps = int(np.ceil(w / t))
            arr = np.tile(arr, (reps, 1, 1))[:w]
        elif t > w:
            s = self.rng.randint(0, t - w) if self.train else (t - w) // 2
            arr = arr[s:s + w]

        xy, score = arr[..., :2], arr[..., 2]
        valid = score > 0
        if valid.any():
            center = xy[valid].mean(axis=0)
            if self.normalize == "window-std":
                scale = max(xy[valid].std(), 1e-3)
            else:
                # image scale: a nearly still child stays nearly still instead
                # of having its tracking jitter amplified to unit variance
                scale = float(r["img_h"]) / 2.0
            xy = np.where(valid[..., None], (xy - center) / scale, 0.0)
        if self.train and self.rng.rand() < self.flip_prob:
            xy = xy[:, FLIP_ORDER]
            score = score[:, FLIP_ORDER]
            xy[..., 0] = -xy[..., 0]
        if self.train and self.joint_dropout > 0:
            # zero whole joints for the whole window so missing data stops
            # being informative (shortcut control, see stage1_report.md)
            drop = self.rng.rand(17) < self.joint_dropout
            xy[:, drop] = 0.0
            score = score.copy()
            score[:, drop] = 0.0

        chans = [xy, score[..., None]]
        if self.velocity:
            v = np.diff(xy, axis=0, prepend=xy[:1])       # (w,17,2)
            chans.insert(1, v)
        out = np.concatenate(chans, axis=2)               # (w,17,3 or 5)
        x = torch.from_numpy(np.ascontiguousarray(out.transpose(2, 0, 1))).float()
        return x, int(r["binary_label"])


def subject_level_val_split(rows, val_fraction=0.2, seed=0):
    """Deterministically carve val children out of the train split."""
    train_rows = [r for r in rows if r["split"] == "train"]
    children = sorted({r["child_id"] for r in train_rows})
    rng = np.random.RandomState(seed)
    val_children = set(rng.choice(children, int(len(children) * val_fraction),
                                  replace=False).tolist())
    tr = [r for r in train_rows if r["child_id"] not in val_children]
    va = [r for r in train_rows if r["child_id"] in val_children]
    tr_c = {r["child_id"] for r in tr}
    assert not (tr_c & {r["child_id"] for r in va}), "subject overlap in val split"
    return tr, va, sorted(val_children)
