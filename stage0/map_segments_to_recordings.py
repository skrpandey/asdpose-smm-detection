"""Map dataset.pkl segments to their source recordings in asdpose.zip by
exact keypoint matching.

The segment metadata (fps, start/end fields) is not a reliable pointer into
the untrimmed recordings, but segment keypoint arrays are verbatim slices of
recording keypoint arrays. So for each child we index every recording frame
by its keypoint bytes and locate each segment by hashing an anchor frame,
then verify the whole span with a nan-aware comparison.

Output: a CSV with identifier, recording filename, start/end frame offsets,
and the verified fraction. Needed for the frame-wise Gate 0.3 protocol and
the Gate 0.4 periodicity check on untrimmed streams.

Usage:
    python stage0/map_segments_to_recordings.py --split test \
        --out cache/segment_map_test.csv
"""

import argparse
import csv
import io
import pickle
import zipfile
from collections import defaultdict

import numpy as np


def build_frame_index(rkp):
    """dict: float32 frame bytes -> first frame index. Skips empty/nan frames.

    Segment arrays went through a float32 roundtrip at some point, so both
    sides are cast to float32 before hashing.
    """
    index = {}
    rkp32 = rkp.astype(np.float32)
    for t in range(rkp32.shape[0]):
        frame = rkp32[t]
        if not np.any(frame):
            continue
        if np.isnan(frame).any():
            continue
        key = frame.tobytes()
        if key not in index:
            index[key] = t
    return index


def verify_span(rkp, skp, offset):
    """Fraction of finite, nonzero keypoint values equal (as float32)."""
    end = offset + skp.shape[0]
    if offset < 0 or end > rkp.shape[0]:
        return 0.0
    rec = rkp[offset:end].astype(np.float32)
    seg = skp.astype(np.float32)
    mask = np.isfinite(rec) & np.isfinite(seg) & (seg != 0)
    if mask.sum() == 0:
        return 0.0
    return float((rec[mask] == seg[mask]).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="data/StimmingADSMotion")
    ap.add_argument("--split", choices=["test", "train", "all"], default="test")
    ap.add_argument("--out", default="cache/segment_map_test.csv")
    ap.add_argument("--min-verified", type=float, default=0.95)
    args = ap.parse_args()

    with open(f"{args.data_root}/dataset.pkl", "rb") as f:
        d = pickle.load(f)
    ann = d["annotations"]
    if args.split != "all":
        wanted = set(d["split"][args.split])
        ann = [a for a in ann if a["identifier"] in wanted]

    by_child = defaultdict(list)
    for a in ann:
        by_child[int(a["identifier"].split("_")[0])].append(a)

    zf = zipfile.ZipFile(f"{args.data_root}/asdpose.zip")
    rows = list(csv.DictReader(io.TextIOWrapper(zf.open("db.csv"))))
    files_by_child = defaultdict(list)
    for r in rows:
        files_by_child[int(r["child_id"])].append(r["filename"])

    results = []
    n_total = n_mapped = 0
    for child in sorted(by_child):
        segs = by_child[child]
        n_total += len(segs)
        files = files_by_child.get(child, [])
        if not files:
            results += [(a["identifier"], "", -1, -1, 0.0) for a in segs]
            continue

        indices = {}
        recordings = {}
        for fname in files:
            with zf.open(f"data/{fname}.pkl") as f:
                rec = pickle.load(f)
            rkp = rec["keypoint"][0]
            recordings[fname] = rkp
            indices[fname] = build_frame_index(rkp)

        mapped_here = 0
        for a in segs:
            skp = a["keypoint"]
            skp32 = skp.astype(np.float32)
            # anchor frames: a few frames with data, spread over the segment
            cand = [t for t in range(skp.shape[0])
                    if np.any(skp32[t]) and not np.isnan(skp32[t]).any()]
            anchors = cand[:1] + cand[len(cand) // 2:len(cand) // 2 + 1] + cand[-1:]
            hit = None
            for fname in files:
                for t in anchors:
                    off = indices[fname].get(skp32[t].tobytes())
                    if off is None:
                        continue
                    start = off - t
                    frac = verify_span(recordings[fname], skp, start)
                    if frac >= args.min_verified:
                        hit = (fname, start, start + skp.shape[0], frac)
                        break
                if hit:
                    break
            if hit:
                results.append((a["identifier"], *hit))
                mapped_here += 1
            else:
                results.append((a["identifier"], "", -1, -1, 0.0))
        n_mapped += mapped_here
        print(f"child {child}: {mapped_here}/{len(segs)} segments mapped "
              f"({len(files)} recordings)", flush=True)

    print(f"\ntotal mapped: {n_mapped}/{n_total} ({n_mapped/n_total:.1%})")
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["identifier", "filename", "start_frame", "end_frame", "verified"])
        for row in results:
            w.writerow(row)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
