"""Per-child SMM quantification on the released untrimmed recordings.

For each recording of the zip-covered test children, scores sliding
windows (200 frames, step 30, mirroring the deployment protocol) with a
chosen model and converts window scores to per-frame scores by max over
covering windows. Predicted SMM frames are those above threshold;
per-child totals (frames, nominal seconds, rate per minute) are then
correlated against totals derived from the annotation segments.

Known noise source, reported alongside: the released recordings are
shorter than the videos the annotations were made on (Gate 0.3), so
annotation-derived totals overestimate what is visible in the released
recordings.

Two model backends:
  --model ours: CompactCTRGCN checkpoint from stage1/train.py (fast)
  --model baseline: the released PoseConv3D via the stage0 port (slow,
    plan for an overnight run)

Usage:
    python stage1/quantify_per_child.py --model ours \
        --weights cache/stage1_detector.pt --out cache/quant_ours.csv
    python stage1/quantify_per_child.py --model baseline \
        --out cache/quant_baseline.csv
"""

import argparse
import csv
import io
import os
import pickle
import sys
import time
import zipfile
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "stage1")
sys.path.insert(0, "stage0")

WINDOW = 200
STEP = 30
TEST_CHILDREN_IN_ZIP = [3, 21, 30, 59, 61, 66, 68, 108, 113, 116, 132, 139,
                        148, 156, 180, 183, 184, 185, 186, 190, 198, 204,
                        210, 215, 233, 237, 241]


def windows_of(total):
    starts = list(range(0, max(1, total - WINDOW + STEP), STEP))
    return [(s, min(s + WINDOW, total)) for s in starts if min(s + WINDOW, total) - s >= 2 * STEP]


def normalize_window(kp, sc, mode="window-std", img_h=None):
    """Same per-window normalization as SegmentDataset."""
    xy = kp.astype(np.float32).copy()
    valid = sc > 0
    if valid.any():
        center = xy[valid].mean(axis=0)
        if mode == "window-std":
            scale = max(xy[valid].std(), 1e-3)
        else:
            scale = img_h / 2.0
        xy = np.where(valid[..., None], (xy - center) / scale, 0.0)
    return xy


class OursBackend:
    def __init__(self, weights, device):
        from model import CompactCTRGCN
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        mcfg = ckpt["config"]["model"]
        self.model = CompactCTRGCN(c_in=mcfg["c_in"], widths=tuple(mcfg["widths"]),
                                   blocks_per_stage=mcfg["blocks_per_stage"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model = self.model.to(device).eval()
        self.device = device
        self.velocity = ckpt["config"]["data"].get("velocity", False)
        self.normalize = ckpt["config"]["data"].get("normalize_mode", "window-std")

    @torch.no_grad()
    def score_windows(self, kp, sc, spans, img_shape):
        xs = []
        for (s, t) in spans:
            w_kp, w_sc = kp[s:t], sc[s:t]
            if t - s < WINDOW:
                reps = int(np.ceil(WINDOW / (t - s)))
                w_kp = np.tile(w_kp, (reps, 1, 1))[:WINDOW]
                w_sc = np.tile(w_sc, (reps, 1))[:WINDOW]
            xy = normalize_window(w_kp, w_sc, self.normalize, img_shape[0])
            chans = [xy, w_sc[..., None].astype(np.float32)]
            if self.velocity:
                v = np.diff(xy, axis=0, prepend=xy[:1])
                chans.insert(1, v)
            arr = np.concatenate(chans, axis=2)
            xs.append(arr.transpose(2, 0, 1))
        out = []
        for i in range(0, len(xs), 256):
            x = torch.from_numpy(np.stack(xs[i:i + 256])).float().to(self.device)
            out.append(F.softmax(self.model(x), 1)[:, 1].cpu().numpy())
        return np.concatenate(out)


def _baseline_preprocess(args):
    from posec3d import preprocess_window
    i, kp, sc, img_shape = args
    return i, preprocess_window(kp, sc, img_shape)


class BaselineBackend:
    """Heatmap preprocessing is the bottleneck (0.22 s/window single core),
    so it runs in a worker pool overlapped with the GPU forward."""

    def __init__(self, weights, device, workers=8):
        from multiprocessing import Pool
        from posec3d import PoseConv3D
        self.model = PoseConv3D()
        self.model.load_mmaction_checkpoint(weights)
        self.model = self.model.to(device).eval()
        self.device = device
        self.pool = Pool(workers)

    @torch.no_grad()
    def score_windows(self, kp, sc, spans, img_shape, batch_windows=2):
        jobs = [(i, kp[None, s:t], sc[None, s:t], img_shape)
                for i, (s, t) in enumerate(spans)]
        out = np.zeros(len(spans))
        pending = []

        def flush():
            idxs = [i for i, _ in pending]
            x = torch.from_numpy(np.concatenate([m for _, m in pending])).to(self.device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = self.model(x)
            probs = F.softmax(logits.float(), 1).view(len(pending), -1, 2).mean(1)
            for j, i in enumerate(idxs):
                out[i] = probs[j, 1].item()
            pending.clear()

        n_done = 0
        for i, imgs in self.pool.imap_unordered(_baseline_preprocess, jobs,
                                                chunksize=4):
            pending.append((i, imgs))
            if len(pending) >= batch_windows:
                flush()
            n_done += 1
            if n_done % 200 == 0:
                print(f"    {n_done}/{len(jobs)} windows", flush=True)
        if pending:
            flush()
        return out


def annotation_totals(data_root):
    """Per-child annotated SMM frames from dataset.pkl segments, restricted
    to (assessment_index, camera) pairs present in the zip for that child so
    both sides count roughly the same streams. Exact per-recording matching
    is impossible (Gate 0.3); this pair-level filter is the best available
    approximation and the residual mismatch is reported as a noise source."""
    zf = zipfile.ZipFile(f"{data_root}/asdpose.zip")
    rows = list(csv.DictReader(io.TextIOWrapper(zf.open("db.csv"))))
    zip_pairs = defaultdict(set)
    for r in rows:
        zip_pairs[int(r["child_id"])].add(
            (int(r["assessment_index"]), int(r["camera_index"])))
    with open(f"{data_root}/dataset.pkl", "rb") as f:
        d = pickle.load(f)
    totals = defaultdict(int)
    for a in d["annotations"]:
        p = a["identifier"].split("_")
        child, pair = int(p[0]), (int(p[1]), int(p[2]))
        if (child in TEST_CHILDREN_IN_ZIP and a["binary_label"] == 1
                and pair in zip_pairs[child]):
            totals[child] += a["total_frames"]
    return totals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="/hd/data/StimmingADSMotion")
    ap.add_argument("--model", choices=["ours", "baseline"], required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-correlation", action="store_true",
                    help="skip the correlation step (no dataset.pkl needed); "
                    "compute correlations separately from the CSV")
    ap.add_argument("--shard", default=None,
                    help="k/n: score only recordings with sorted-index %% n == k")
    args = ap.parse_args()

    device = "cuda"
    if args.model == "ours":
        backend = OursBackend(args.weights or "cache/stage1_detector.pt", device)
    else:
        backend = BaselineBackend(args.weights or f"{args.data_root}/model.pth", device)

    zf = zipfile.ZipFile(f"{args.data_root}/asdpose.zip")
    rows = list(csv.DictReader(io.TextIOWrapper(zf.open("db.csv"))))
    by_child = defaultdict(list)
    all_files = sorted(r["filename"] for r in rows
                       if int(r["child_id"]) in TEST_CHILDREN_IN_ZIP)
    if args.shard:
        k, n = map(int, args.shard.split("/"))
        keep = {f for i, f in enumerate(all_files) if i % n == k}
        print(f"shard {k}/{n}: {len(keep)} of {len(all_files)} recordings")
    else:
        keep = set(all_files)
    for r in rows:
        c = int(r["child_id"])
        if c in TEST_CHILDREN_IN_ZIP and r["filename"] in keep:
            by_child[c].append(r["filename"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = ["child_id", "filename", "n_frames", "fps_recording",
                  "n_windows", "smm_frames", "smm_seconds_nominal",
                  "score_mass", "minutes"]
    # resume: skip recordings already scored in a previous partial run
    done_files = set()
    if os.path.exists(args.out):
        done_rows = list(csv.DictReader(open(args.out)))
        done_files = {r["filename"] for r in done_rows}
        print(f"resuming: {len(done_files)} recordings already in {args.out}",
              flush=True)
    else:
        with open(args.out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    t0 = time.time()
    n_rec = sum(len(v) for v in by_child.values())
    done = len(done_files)
    for child in sorted(by_child):
        for fname in by_child[child]:
            if fname in done_files:
                continue
            with zf.open(f"data/{fname}.pkl") as f:
                rec = pickle.load(f)
            kp = rec["keypoint"][0]
            sc = rec["keypoint_score"][0]
            total = kp.shape[0]
            spans = windows_of(total)
            h, w = rec["original_shape"]
            scores = backend.score_windows(kp, sc, spans, (int(h), int(w)))
            frame_score = np.zeros(total, dtype=np.float32)
            for (s, t), v in zip(spans, scores):
                np.maximum(frame_score[s:t], v, out=frame_score[s:t])
            smm_frames = int((frame_score > args.threshold).sum())
            row = {
                "child_id": child, "filename": fname, "n_frames": total,
                "fps_recording": rec["fps"], "n_windows": len(spans),
                "smm_frames": smm_frames,
                "smm_seconds_nominal": smm_frames / rec["fps"],
                "score_mass": f"{float(frame_score.sum()):.2f}",
                "minutes": total / rec["fps"] / 60,
            }
            with open(args.out, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
            done += 1
            print(f"[{done}/{n_rec}] {fname}: {len(spans)} windows, "
                  f"{smm_frames} SMM frames "
                  f"({(time.time()-t0)/60:.1f} min elapsed)", flush=True)

    if args.no_correlation:
        print(f"written {args.out} (correlation step skipped)")
        return

    out_rows = list(csv.DictReader(open(args.out)))
    for r in out_rows:
        r["child_id"] = int(r["child_id"])
        r["smm_frames"] = int(r["smm_frames"])
        r["minutes"] = float(r["minutes"])

    # per-child correlation against annotation-derived totals
    pred = defaultdict(int)
    minutes = defaultdict(float)
    mass = defaultdict(float)
    for r in out_rows:
        pred[r["child_id"]] += r["smm_frames"]
        minutes[r["child_id"]] += r["minutes"]
        mass[r["child_id"]] += float(r["score_mass"])
    ann = annotation_totals(args.data_root)
    children = sorted(set(pred) & set(ann))
    x = np.array([pred[c] for c in children], float)
    y = np.array([ann[c] for c in children], float)
    from scipy.stats import pearsonr, spearmanr
    pr, sp = pearsonr(x, y), spearmanr(x, y)
    mx, my, vx, vy = x.mean(), y.mean(), x.var(), y.var()
    ccc = 2 * np.cov(x, y, bias=True)[0, 1] / (vx + vy + (mx - my) ** 2)
    print(f"\nper-child totals over {len(children)} children (frames):")
    print(f"Pearson r {pr.statistic:.4f} (p {pr.pvalue:.2e}), "
          f"Spearman rho {sp.statistic:.4f} (p {sp.pvalue:.2e}), "
          f"concordance {ccc:.4f}")
    rate_x = np.array([pred[c] / minutes[c] for c in children])
    sp2 = spearmanr(rate_x, y)
    print(f"rate/min vs annotated frames: Spearman {sp2.statistic:.4f}")
    mass_x = np.array([mass[c] for c in children])
    sp3 = spearmanr(mass_x, y)
    print(f"score mass (threshold-free) vs annotated frames: "
          f"Spearman {sp3.statistic:.4f}")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
