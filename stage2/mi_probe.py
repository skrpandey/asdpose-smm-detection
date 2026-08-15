"""Stage 2: identity-probe mutual information estimate (ITEA protocol).

How much child-identity information do the pooled features carry? A
logistic-regression probe is trained on frozen features to predict which
training child a segment belongs to; the cross-entropy of the probe on
held-out segments gives a lower bound on I(features; identity):

    I >= log(N) - CE_heldout        (nats; uniform prior over N children)

The bound uses the EMPIRICAL entropy of the identity marginal, not
log N: children contribute unequal segment counts, so H(Y) < log N, and
substituting log N would inflate the bound (it would no longer be a
lower bound). The chance floor reported is the majority-class rate, not
1/N, for the same reason. This corrects an error in the first version
of this script, found in review.

The probe split is by segment within child (identity must appear on both
sides, so a subject-level split is impossible here by construction).
Reported for the v3 backbone (lambda 0) and each SAPL checkpoint.

Usage:
    python stage2/mi_probe.py --weights cache/stage1_detector_v3.pt
    python stage2/mi_probe.py --weights cache/stage2_sapl_lam0.3.pt --sapl
"""

import argparse
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, "stage1")
sys.path.insert(0, "stage2")
from dataset import SegmentDataset, subject_level_val_split
from model import CompactCTRGCN
from protocols import load_index


@torch.no_grad()
def extract_features(backbone, rows, cfg, device):
    ds = SegmentDataset(rows, cfg["data"]["cache_dir"], cfg["data"]["window"],
                        train=False, velocity=cfg["data"]["velocity"],
                        normalize=cfg["data"].get("normalize_mode", "window-std"))
    dl = DataLoader(ds, batch_size=128, num_workers=6)
    backbone.eval()
    feats = []
    for x, _ in dl:
        feats.append(backbone.forward_features(x.to(device)).cpu().numpy())
    return np.concatenate(feats)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--sapl", action="store_true",
                    help="checkpoint is a SAPLModel (backbone.* prefixed keys)")
    ap.add_argument("--probe-frac", type=float, default=0.75,
                    help="fraction of each child's segments used to fit the probe")
    args = ap.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    backbone = CompactCTRGCN(c_in=cfg["model"]["c_in"],
                             widths=tuple(cfg["model"]["widths"]),
                             blocks_per_stage=cfg["model"]["blocks_per_stage"])
    sd = ckpt["state_dict"]
    if args.sapl:
        sd = {k[len("backbone."):]: v for k, v in sd.items()
              if k.startswith("backbone.")}
    backbone.load_state_dict(sd, strict=True)
    device = "cuda"
    backbone = backbone.to(device)

    rows = load_index()
    tr_rows, _, _ = subject_level_val_split(rows, cfg["data"]["val_child_fraction"],
                                            seed=cfg["seed"])
    children = sorted({r["child_id"] for r in tr_rows})
    cls = {c: i for i, c in enumerate(children)}
    print(f"{len(tr_rows)} segments, {len(children)} identities")

    feats = extract_features(backbone, tr_rows, cfg, device)
    labels = np.array([cls[r["child_id"]] for r in tr_rows])

    rng = np.random.RandomState(0)
    fit_mask = np.zeros(len(labels), bool)
    for c in range(len(children)):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        fit_mask[idx[:max(1, int(len(idx) * args.probe_frac))]] = True

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(feats[fit_mask])
    probe = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial",
                               random_state=0)
    probe.fit(scaler.transform(feats[fit_mask]), labels[fit_mask])
    proba = probe.predict_proba(scaler.transform(feats[~fit_mask]))
    held = labels[~fit_mask]
    # probe classes may be a subset ordering; align columns
    col = {c: i for i, c in enumerate(probe.classes_)}
    p_true = np.array([proba[i, col[c]] if c in col else 1e-12
                       for i, c in enumerate(held)])
    ce = -np.mean(np.log(np.clip(p_true, 1e-12, 1)))
    acc = float((probe.predict(scaler.transform(feats[~fit_mask])) == held).mean())
    n = len(children)
    # empirical entropy of the held-out identity marginal (valid bound)
    counts = np.bincount(held, minlength=n).astype(float)
    p = counts / counts.sum()
    h_emp = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    majority = float(p.max())
    mi = max(0.0, h_emp - ce)
    mi_naive = max(0.0, np.log(n) - ce)
    print(f"probe accuracy {acc:.4f} (majority-class floor {majority:.4f}, "
          f"uniform 1/N {1/n:.4f})")
    print(f"held-out CE {ce:.4f} nats; H_emp(Y) {h_emp:.4f} nats "
          f"({h_emp/np.log(2):.2f} bits); log(N) {np.log(n):.4f} "
          f"(inflation if used: {(np.log(n)-h_emp)/np.log(2):.3f} bits)")
    print(f"MI lower bound (corrected): {mi:.4f} nats = {mi/np.log(2):.4f} bits "
          f"(max H_emp {h_emp/np.log(2):.2f} bits)")
    print(f"[previous, invalid log-N bound would have read: "
          f"{mi_naive/np.log(2):.4f} bits]")


if __name__ == "__main__":
    main()
