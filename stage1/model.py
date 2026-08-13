"""Compact CTR-GCN-lineage detector for 17-joint ASDPose skeletons.

Input (N, 3, T, 17): x, y in window-normalized coordinates, plus the raw
OpenPose confidence as the third channel (kept deliberately; the
hard-negative protocol is what prevents quality from being a winning
shortcut, see configs/protocols.yaml).

Blocks are CTR-GC spatial units (channel-wise topology refinement on a
COCO-17 base adjacency) followed by multi-scale temporal convolutions.
Widths are chosen to land near the 351K parameter budget from the study plan.

Run directly to print the parameter count and a forward-pass shape check.
"""

import torch
import torch.nn as nn

COCO_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7), (7, 9),
              (0, 6), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15),
              (6, 12), (12, 14), (14, 16), (11, 12)]
NUM_JOINTS = 17


def base_adjacency():
    a = torch.eye(NUM_JOINTS)
    for i, j in COCO_EDGES:
        a[i, j] = a[j, i] = 1.0
    d = a.sum(1, keepdim=True)
    return a / d


class CTRGC(nn.Module):
    """Channel-wise topology refinement graph convolution (one branch)."""

    def __init__(self, c_in, c_out, rel_reduction=8):
        super().__init__()
        c_rel = max(c_in // rel_reduction, 8)
        self.theta = nn.Conv2d(c_in, c_rel, 1)
        self.phi = nn.Conv2d(c_in, c_rel, 1)
        self.value = nn.Conv2d(c_in, c_out, 1)
        self.refine = nn.Conv2d(c_rel, c_out, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, a_base):                      # x: (N,C,T,V)
        q = self.theta(x).mean(2)                      # (N,R,V)
        k = self.phi(x).mean(2)                        # (N,R,V)
        rel = self.refine(torch.tanh(q.unsqueeze(-1) - k.unsqueeze(-2))
                          .unsqueeze(2).squeeze(2))    # (N,C_out,V,V)
        adj = rel * self.alpha + a_base                # broadcast over channels
        v = self.value(x)                              # (N,C_out,T,V)
        return torch.einsum("nctv,ncvw->nctw", v, adj)


class SpatialUnit(nn.Module):
    def __init__(self, c_in, c_out, branches=3):
        super().__init__()
        self.a_base = nn.Parameter(base_adjacency().repeat(branches, 1, 1))
        self.branches = nn.ModuleList(CTRGC(c_in, c_out) for _ in range(branches))
        self.bn = nn.BatchNorm2d(c_out)
        self.residual = (nn.Identity() if c_in == c_out
                         else nn.Conv2d(c_in, c_out, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = None
        for i, branch in enumerate(self.branches):
            y = branch(x, self.a_base[i].unsqueeze(0).unsqueeze(0))
            out = y if out is None else out + y
        return self.relu(self.bn(out) + self.residual(x))


class TemporalUnit(nn.Module):
    """Two dilated 5x1 branches + max-pool branch + pointwise branch."""

    def __init__(self, channels, stride=1):
        super().__init__()
        c = channels // 4
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                          nn.Conv2d(c, c, (5, 1), stride=(stride, 1),
                                    padding=(2, 0)), nn.BatchNorm2d(c)),
            nn.Sequential(nn.Conv2d(channels, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                          nn.Conv2d(c, c, (5, 1), stride=(stride, 1),
                                    padding=(4, 0), dilation=(2, 1)), nn.BatchNorm2d(c)),
            nn.Sequential(nn.Conv2d(channels, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                          nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0))),
            nn.Sequential(nn.Conv2d(channels, c, (1, 1), stride=(stride, 1)),
                          nn.BatchNorm2d(c)),
        ])
        self.residual = (nn.Identity() if stride == 1 else
                         nn.Sequential(nn.Conv2d(channels, channels, 1,
                                                 stride=(stride, 1)),
                                       nn.BatchNorm2d(channels)))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = torch.cat([b(x) for b in self.branches], dim=1)
        return self.relu(out + self.residual(x))


class Block(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.spatial = SpatialUnit(c_in, c_out)
        self.temporal = TemporalUnit(c_out, stride)

    def forward(self, x):
        return self.temporal(self.spatial(x))


class CompactCTRGCN(nn.Module):
    def __init__(self, c_in=3, num_classes=2, widths=(48, 96, 136),
                 blocks_per_stage=2):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(c_in * NUM_JOINTS)
        layers = []
        prev = c_in
        for i, w in enumerate(widths):
            for b in range(blocks_per_stage):
                stride = 2 if (i > 0 and b == 0) else 1
                layers.append(Block(prev, w, stride))
                prev = w
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(prev, num_classes)

    def forward(self, x):                              # (N, C, T, V)
        n, c, t, v = x.shape
        y = self.input_bn(x.permute(0, 1, 3, 2).reshape(n, c * v, t))
        x = y.reshape(n, c, v, t).permute(0, 1, 3, 2)
        x = self.blocks(x)
        return self.fc(x.mean(dim=(2, 3)))


if __name__ == "__main__":
    m = CompactCTRGCN()
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n/1e3:.1f}K")
    x = torch.randn(2, 3, 200, 17)
    print("forward:", m(x).shape)
