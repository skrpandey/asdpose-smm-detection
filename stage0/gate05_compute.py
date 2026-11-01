"""Gate 0.5: compute reality check.

Times forward and backward passes for (a) a ~351K parameter CTR-GCN-sized
graph model on 17-joint input and (b) the 3M baseline PoseConv3D, on this
machine's GPU. Estimates epoch and full-training wall clock against the
released training split (32,177 segments, 10x repeat per their config) and
reports the disk budget.

The GCN stand-in below is not the Stage 1 architecture, it only has to match
the parameter count and rough op mix (temporal convs + per-frame graph mixing)
well enough for a wall-clock estimate.

Usage:
    python stage0/gate05_compute.py
"""

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn


class GCNBlock(nn.Module):
    def __init__(self, c_in, c_out, v=17, stride=1):
        super().__init__()
        self.adj = nn.Parameter(torch.randn(v, v) * 0.01)
        self.theta = nn.Conv2d(c_in, c_out, 1)
        self.tcn = nn.Conv2d(c_out, c_out, (9, 1), stride=(stride, 1), padding=(4, 0))
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):            # (N, C, T, V)
        x = torch.einsum("nctv,vw->nctw", self.theta(x), self.adj.softmax(-1))
        return self.relu(self.bn(self.tcn(x)))


class SmallGCN(nn.Module):
    """~351K params at widths (64, 96, 128)."""

    def __init__(self, c_in=3, num_classes=2):
        super().__init__()
        self.blocks = nn.Sequential(
            GCNBlock(c_in, 64),
            GCNBlock(64, 64),
            GCNBlock(64, 96, stride=2),
            GCNBlock(96, 96),
            GCNBlock(96, 128, stride=2),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.blocks(x)
        return self.fc(x.mean(dim=(2, 3)))


def time_model(model, x, label, steps=20, train=False):
    opt = torch.optim.SGD(model.parameters(), lr=0.01) if train else None
    crit = nn.CrossEntropyLoss()
    y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    for _ in range(3):
        if train:
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        else:
            with torch.no_grad():
                model(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        if train:
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        else:
            with torch.no_grad():
                model(x)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    print(f"{label}: {dt*1000:.1f} ms/iter, batch {x.shape[0]} "
          f"-> {x.shape[0]/dt:.0f} samples/s")
    return dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-segments", type=int, default=32177)
    ap.add_argument("--repeat", type=int, default=10,
                    help="their RepeatDataset factor per epoch")
    args = ap.parse_args()

    dev = "cuda"
    print("## Gate 0.5 compute check")
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB\n")

    gcn = SmallGCN().to(dev)
    n = sum(p.numel() for p in gcn.parameters())
    print(f"GCN stand-in params: {n/1e3:.0f}K")
    x = torch.randn(64, 3, 200, 17, device=dev)
    time_model(gcn, x, "GCN fwd (batch 64, T=200, V=17)", train=False)
    dt = time_model(gcn, x, "GCN fwd+bwd", train=True)
    iters = args.train_segments / 64
    print(f"epoch (1 pass over {args.train_segments} train segments): "
          f"{iters*dt/60:.1f} min")
    print(f"50 epochs: {50*iters*dt/3600:.2f} h; "
          f"with their 10x repeat: {50*args.repeat*iters*dt/3600:.1f} h\n")

    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from posec3d import PoseConv3D
    pc = PoseConv3D().to(dev)
    n = sum(p.numel() for p in pc.parameters())
    print(f"baseline PoseConv3D params: {n/1e6:.2f}M")
    bs = 8  # batch 16 does not fit next to the model on 8 GiB
    xb = torch.randn(bs, 17, 48, 56, 56, device=dev)
    dtb = time_model(pc, xb, f"PoseConv3D fwd+bwd (batch {bs}, their train shape 48x56x56)", train=True)
    iters = args.train_segments / bs
    print(f"PoseConv3D 50 epochs, 10x repeat: {50*args.repeat*iters*dtb/3600:.1f} h "
          f"(what retraining THEIR model here would cost)\n")

    total, used, free = shutil.disk_usage(".")
    print(f"disk: {free/2**30:.0f} GiB free of {total/2**30:.0f} GiB; "
          f"dataset 12 GiB compressed + 26 GiB if the zip is extracted; "
          f"a float32 cache of all segment arrays adds roughly 6 GiB")


if __name__ == "__main__":
    main()
