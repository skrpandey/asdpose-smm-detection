"""External skeleton baselines for the architecture-breadth experiment:
ST-GCN (Yan et al. 2018) and single-stream AGCN (Shi et al. 2019),
adapted only in input layout (17 COCO joints, 3 or 5 channels). Widths
and depths follow the originals (10 blocks, 64/128/256). Both expose
forward_features like CompactCTRGCN so stage1/train.py can swap them in.

Run directly for a parameter-count and shape smoke test.
"""

import numpy as np
import torch
import torch.nn as nn

COCO_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7), (7, 9),
              (0, 6), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15),
              (6, 12), (12, 14), (14, 16), (11, 12)]
V = 17
CENTER = 0  # nose; partition distances are measured to this joint


def _hop_distance():
    inf = 999
    d = np.full((V, V), inf)
    np.fill_diagonal(d, 0)
    for i, j in COCO_EDGES:
        d[i, j] = d[j, i] = 1
    for k in range(V):
        for i in range(V):
            for j in range(V):
                d[i, j] = min(d[i, j], d[i, k] + d[k, j])
    return d


def spatial_partitions():
    """Three-subset spatial partition (identity, centripetal, centrifugal),
    each column-normalized as in ST-GCN."""
    hop = _hop_distance()
    adj = np.zeros((V, V))
    np.fill_diagonal(adj, 1)
    for i, j in COCO_EDGES:
        adj[i, j] = adj[j, i] = 1
    deg = adj.sum(0)
    norm = adj / deg[None, :]
    parts = np.zeros((3, V, V))
    for i in range(V):
        for j in range(V):
            if norm[i, j] == 0:
                continue
            if hop[j, CENTER] == hop[i, CENTER]:
                parts[0, i, j] = norm[i, j]
            elif hop[j, CENTER] > hop[i, CENTER]:
                parts[1, i, j] = norm[i, j]
            else:
                parts[2, i, j] = norm[i, j]
    return torch.tensor(parts, dtype=torch.float32)


class STGCNBlock(nn.Module):
    def __init__(self, c_in, c_out, a, stride=1, residual=True):
        super().__init__()
        self.register_buffer("a", a)
        k = a.shape[0]
        self.edge = nn.Parameter(torch.ones_like(a))
        self.gcn = nn.Conv2d(c_in, c_out * k, 1)
        self.gbn = nn.BatchNorm2d(c_out)
        self.tcn = nn.Sequential(
            nn.Conv2d(c_out, c_out, (9, 1), stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(c_out))
        if not residual:
            self.residual = None
        elif c_in == c_out and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=(stride, 1)),
                nn.BatchNorm2d(c_out))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):                       # (N,C,T,V)
        n, _, t, v = x.shape
        k = self.a.shape[0]
        y = self.gcn(x).view(n, k, -1, t, v)
        y = torch.einsum("nkctv,kvw->nctw", y, self.a * self.edge)
        y = self.relu(self.gbn(y))
        y = self.tcn(y)
        res = 0 if self.residual is None else self.residual(x)
        return self.relu(y + res)


class STGCN(nn.Module):
    def __init__(self, c_in=3, num_classes=2):
        super().__init__()
        a = spatial_partitions()
        self.input_bn = nn.BatchNorm1d(c_in * V)
        widths = [(c_in, 64, 1, False), (64, 64, 1, True), (64, 64, 1, True),
                  (64, 64, 1, True), (64, 128, 2, True), (128, 128, 1, True),
                  (128, 128, 1, True), (128, 256, 2, True),
                  (256, 256, 1, True), (256, 256, 1, True)]
        self.blocks = nn.ModuleList(
            STGCNBlock(i, o, a.clone(), s, r) for i, o, s, r in widths)
        self.fc = nn.Linear(256, num_classes)

    def forward_features(self, x):              # (N,C,T,V) -> (N,256)
        n, c, t, v = x.shape
        y = self.input_bn(x.permute(0, 1, 3, 2).reshape(n, c * v, t))
        x = y.reshape(n, c, v, t).permute(0, 1, 3, 2)
        for b in self.blocks:
            x = b(x)
        return x.mean(dim=(2, 3))

    def forward(self, x):
        return self.fc(self.forward_features(x))


class AGCNBlock(nn.Module):
    """Single-stream 2s-AGCN unit: fixed A + learned B + attention C."""

    def __init__(self, c_in, c_out, a, stride=1, residual=True, embed=None):
        super().__init__()
        self.register_buffer("a", a)
        k = a.shape[0]
        self.b = nn.Parameter(torch.zeros_like(a))
        embed = embed or max(c_out // 4, 16)
        self.theta = nn.ModuleList(nn.Conv2d(c_in, embed, 1) for _ in range(k))
        self.phi = nn.ModuleList(nn.Conv2d(c_in, embed, 1) for _ in range(k))
        self.value = nn.ModuleList(nn.Conv2d(c_in, c_out, 1) for _ in range(k))
        self.gbn = nn.BatchNorm2d(c_out)
        self.tcn = nn.Sequential(
            nn.Conv2d(c_out, c_out, (9, 1), stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(c_out))
        if not residual:
            self.residual = None
        elif c_in == c_out and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=(stride, 1)),
                nn.BatchNorm2d(c_out))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):                       # (N,C,T,V)
        n, _, t, v = x.shape
        out = None
        for k in range(self.a.shape[0]):
            q = self.theta[k](x).mean(2)        # (N,E,V)
            kk = self.phi[k](x).mean(2)
            c_att = torch.softmax(torch.einsum("nev,new->nvw", q, kk), dim=-1)
            adj = self.a[k][None] + self.b[k][None] + c_att
            y = torch.einsum("nctv,nvw->nctw", self.value[k](x), adj)
            out = y if out is None else out + y
        out = self.relu(self.gbn(out))
        out = self.tcn(out)
        res = 0 if self.residual is None else self.residual(x)
        return self.relu(out + res)


class AGCN(nn.Module):
    def __init__(self, c_in=3, num_classes=2):
        super().__init__()
        a = spatial_partitions()
        self.input_bn = nn.BatchNorm1d(c_in * V)
        widths = [(c_in, 64, 1, False), (64, 64, 1, True), (64, 64, 1, True),
                  (64, 64, 1, True), (64, 128, 2, True), (128, 128, 1, True),
                  (128, 128, 1, True), (128, 256, 2, True),
                  (256, 256, 1, True), (256, 256, 1, True)]
        self.blocks = nn.ModuleList(
            AGCNBlock(i, o, a.clone(), s, r) for i, o, s, r in widths)
        self.fc = nn.Linear(256, num_classes)

    def forward_features(self, x):
        n, c, t, v = x.shape
        y = self.input_bn(x.permute(0, 1, 3, 2).reshape(n, c * v, t))
        x = y.reshape(n, c, v, t).permute(0, 1, 3, 2)
        for b in self.blocks:
            x = b(x)
        return x.mean(dim=(2, 3))

    def forward(self, x):
        return self.fc(self.forward_features(x))


if __name__ == "__main__":
    for cls in (STGCN, AGCN):
        for c in (3, 5):
            m = cls(c_in=c)
            n = sum(p.numel() for p in m.parameters())
            x = torch.randn(2, c, 200, 17)
            print(f"{cls.__name__} c_in={c}: {n/1e6:.2f}M params, "
                  f"forward {tuple(m(x).shape)}")
