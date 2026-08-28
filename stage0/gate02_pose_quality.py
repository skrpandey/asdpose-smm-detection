"""Gate 0.2: loadability and pose quality audit for the ASDPose release.

Answers: can segments be loaded end to end into arrays with a documented shape
convention, which joint groups are trustworthy (per-joint confidence audit),
and how common frozen keypoints (tracker failure) are. Also spot-checks one
untrimmed recording from asdpose.zip.

Usage:
    python stage0/gate02_pose_quality.py --data-root data/StimmingADSMotion \
        --sample 2000 --seed 0
"""

import argparse
import pickle
import random
import zipfile

import numpy as np

# COCO-17 order (their pipeline converts OpenPose BODY_25 to this layout)
JOINT_GROUPS = {
    "head (nose,eyes,ears)": [0, 1, 2, 3, 4],
    "arms (shoulders,elbows,wrists)": [5, 6, 7, 8, 9, 10],
    "legs (hips,knees,ankles)": [11, 12, 13, 14, 15, 16],
}
LOW_CONF = 0.3


def audit_segments(segs, title):
    print(f"\n### {title} (n={len(segs)})")
    ok = 0
    for a in segs:
        kp, sc = a["keypoint"], a["keypoint_score"]
        assert kp.ndim == 3 and kp.shape[1:] == (17, 2), kp.shape
        assert sc.shape == kp.shape[:2], (sc.shape, kp.shape)
        assert kp.shape[0] == a["total_frames"] or abs(kp.shape[0] - a["total_frames"]) <= 1
        if np.isfinite(kp).all() and np.isfinite(sc).all():
            ok += 1
    print(f"loadable with finite values: {ok}/{len(segs)}")

    scores = np.concatenate([a["keypoint_score"] for a in segs])  # (sumT, 17)
    coords = np.concatenate([a["keypoint"] for a in segs])
    print(f"score range: {scores.min():.3f}..{scores.max():.3f}, mean {scores.mean():.3f}")
    print(f"fraction of joint-frames with score == 0 (joint missing): {(scores == 0).mean():.3f}")
    for name, idx in JOINT_GROUPS.items():
        frac = (scores[:, idx] < LOW_CONF).mean()
        zero = (scores[:, idx] == 0).mean()
        print(f"  {name}: frac score<{LOW_CONF}: {frac:.3f}, frac score==0: {zero:.3f}")

    # frozen keypoints: consecutive frames with identical coordinates on all joints
    frozen_frac = []
    for a in segs:
        kp = a["keypoint"]
        if kp.shape[0] < 2:
            continue
        same = (kp[1:] == kp[:-1]).all(axis=(1, 2))
        frozen_frac.append(same.mean())
    frozen_frac = np.array(frozen_frac)
    print(f"frozen-frame fraction per segment: mean {frozen_frac.mean():.4f}, "
          f"median {np.median(frozen_frac):.4f}, max {frozen_frac.max():.4f}, "
          f"segments with >20% frozen: {(frozen_frac > 0.2).mean():.4f}")

    # unusable segment definition: >50% of arm joint-frames below LOW_CONF
    arm = JOINT_GROUPS["arms (shoulders,elbows,wrists)"]
    unusable = 0
    for a in segs:
        if (a["keypoint_score"][:, arm] < LOW_CONF).mean() > 0.5:
            unusable += 1
    print(f"segments with >50% low-confidence arm joint-frames: "
          f"{unusable}/{len(segs)} ({unusable/len(segs):.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="data/StimmingADSMotion")
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print(f"## Gate 0.2 pose quality audit (seed={args.seed})")

    with open(f"{args.data_root}/dataset.pkl", "rb") as f:
        d = pickle.load(f)
    ann = d["annotations"]

    smm = [a for a in ann if a["binary_label"] == 1]
    non = [a for a in ann if a["binary_label"] == 0]

    # the required end-to-end check: 20 + 20 random segments
    audit_segments(rng.sample(smm, 20), "20 random SMM segments (end-to-end check)")
    audit_segments(rng.sample(non, 20), "20 random non-SMM segments (end-to-end check)")

    # larger samples for stable statistics
    audit_segments(rng.sample(smm, min(args.sample, len(smm))), f"SMM sample")
    audit_segments(rng.sample(non, min(args.sample, len(non))), f"non-SMM sample")

    # one untrimmed recording from the zip
    print("\n### untrimmed recording spot check")
    zf = zipfile.ZipFile(f"{args.data_root}/asdpose.zip")
    name = rng.choice([n for n in zf.namelist() if n.endswith(".pkl")])
    with zf.open(name) as f:
        rec = pickle.load(f)
    kp, sc = rec["keypoint"], rec["keypoint_score"]
    print(f"{name}: keypoint {kp.shape} (M, T, 17, 2), score {sc.shape}, "
          f"fps {rec['fps']:.2f}, {rec['length_seconds']/60:.1f} min")
    det = rec["child_detection_score"]
    print(f"child_detection_score: mean {det.mean():.3f}, frac == 0: {(det == 0).mean():.3f}")
    for gname, idx in JOINT_GROUPS.items():
        print(f"  {gname}: frac score<{LOW_CONF}: {(sc[0][:, idx] < LOW_CONF).mean():.3f}")


if __name__ == "__main__":
    main()
