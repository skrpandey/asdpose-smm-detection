"""Revision analysis B4: numeric parity between our ported inference
pipeline and the original fork's transform code.

The released checkpoint shipped with TalBarami/mmaction2 (mmaction 0.24
lineage), whose environment (mmcv 1.x, Python 3.9) cannot be installed
here. Instead of trusting the port, this script executes the fork's OWN
transform classes, extracted by AST from the cloned fork source with a
minimal mmcv stub (only rescale_size, is_tuple_of, and _pair are touched
on the keypoint path), runs test segments through both pipelines, and
compares the generated heatmap tensors and the final checkpoint scores.

Exact tensor equality is the pass criterion for the heatmaps; score
agreement is reported as max absolute probability difference.

Usage:
    python revision/b4_parity_check.py --n 50
"""

import argparse
import ast
import copy as cp
import pickle
import random
import sys
import types

import numpy as np

sys.path.insert(0, "stage0")

FORK = "mmaction2-fork/mmaction/datasets/pipelines"


def build_fork_namespace():
    """Extract the needed classes/functions from the fork's source files
    and exec them with stubs for mmcv and the registry."""
    mmcv_stub = types.ModuleType("mmcv")

    def rescale_size(old_size, scale, return_scale=False):
        w, h = old_size
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
        new_size = (int(w * scale_factor + 0.5), int(h * scale_factor + 0.5))
        return (new_size, scale_factor) if return_scale else new_size

    mmcv_stub.rescale_size = rescale_size
    mmcv_stub.is_tuple_of = lambda x, t: isinstance(x, tuple) and all(isinstance(i, t) for i in x)
    mmcv_stub.imresize = None  # never called on the keypoint path

    ns = {
        "np": np, "cp": cp, "random": random, "mmcv": mmcv_stub,
        "_pair": lambda x: (x, x) if not isinstance(x, tuple) else x,
        "warnings": __import__("warnings"),
    }

    wanted = {
        f"{FORK}/augmentations.py": ["_combine_quadruple", "_init_lazy_if_proper",
                                      "RandomCrop", "PoseCompact", "Resize", "CenterCrop", "Flip"],
        f"{FORK}/pose_loading.py": ["UniformSampleFrames", "PoseDecode",
                                     "GeneratePoseTarget"],
    }
    for path, names in wanted.items():
        tree = ast.parse(open(path).read())
        keep = [n for n in tree.body
                if (isinstance(n, (ast.ClassDef, ast.FunctionDef))
                    and n.name in names)]
        mod = ast.Module(body=keep, type_ignores=[])
        # strip decorators (the PIPELINES registry)
        for n in keep:
            n.decorator_list = []
        exec(compile(mod, path, "exec"), ns)
    missing = [n for names in wanted.values() for n in names if n not in ns]
    assert not missing, f"failed to extract: {missing}"
    return ns


def fork_preprocess(ns, kp, score, img_shape):
    """The fork's test pipeline, driven by its own class objects."""
    results = {
        "keypoint": kp.astype(np.float32),
        "keypoint_score": score.astype(np.float32),
        "img_shape": img_shape,
        "total_frames": kp.shape[1],
        "start_index": 0,
        "modality": "Pose",
        "label": -1,
    }
    steps = [
        ns["UniformSampleFrames"](clip_len=48, num_clips=10, test_mode=True),
        ns["PoseDecode"](),
        ns["PoseCompact"](hw_ratio=1.0, allow_imgpad=True),
        ns["Resize"](scale=(-1, 56)),
        ns["CenterCrop"](crop_size=56),
        ns["GeneratePoseTarget"](sigma=0.6, use_score=True, with_kp=True,
                                 with_limb=False, double=True,
                                 left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
                                 right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
    ]
    for t in steps:
        results = t(results)
    imgs = results["imgs"]
    imgs = imgs.reshape((-1, 10, 48) + imgs.shape[1:])
    imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
    return imgs.reshape((-1,) + imgs.shape[2:])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from posec3d import PoseConv3D, preprocess_window

    # numpy 2 removed aliases the 2022-era fork uses
    if not hasattr(np, "int"):
        np.int = int
    np.Inf = np.inf

    ns = build_fork_namespace()

    with open("data/StimmingADSMotion/dataset.pkl", "rb") as f:
        d = pickle.load(f)
    test_ids = set(d["split"]["test"])
    segs = [a for a in d["annotations"] if a["identifier"] in test_ids]
    rng = random.Random(args.seed)
    sample = rng.sample(segs, args.n)

    import torch
    import torch.nn.functional as F
    model = PoseConv3D()
    model.load_mmaction_checkpoint("data/StimmingADSMotion/model.pth")
    model = model.cuda().eval()

    max_hm_diff = 0.0
    max_score_diff = 0.0
    exact = 0
    with torch.no_grad():
        for a in sample:
            kp = a["keypoint"][None]
            sc = a["keypoint_score"][None]
            # both pipelines share the >200-frame crop upstream; use full
            # segment when <= 200, center window otherwise (as in eval)
            if kp.shape[1] > 200:
                s = (kp.shape[1] - 200) // 2
                kp, sc = kp[:, s:s + 200], sc[:, s:s + 200]
            ours = preprocess_window(kp, sc, a["img_shape"])
            theirs = fork_preprocess(ns, kp, sc, a["img_shape"])
            diff = float(np.abs(ours - theirs).max())
            max_hm_diff = max(max_hm_diff, diff)
            if diff == 0.0:
                exact += 1
            p_ours = F.softmax(model(torch.from_numpy(ours).cuda()), 1).mean(0)[1].item()
            p_theirs = F.softmax(model(torch.from_numpy(theirs).cuda()), 1).mean(0)[1].item()
            max_score_diff = max(max_score_diff, abs(p_ours - p_theirs))

    print(f"segments compared: {args.n}")
    print(f"heatmap tensors exactly equal: {exact}/{args.n}")
    print(f"max abs heatmap difference: {max_hm_diff:.2e}")
    print(f"max abs SMM-probability difference: {max_score_diff:.2e}")


if __name__ == "__main__":
    main()
