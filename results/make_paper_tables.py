"""Generate the paper's LaTeX tables from the per-segment score CSVs.

Writes results/tables/*.tex for the paper repo to copy. Numbers are
computed from the score files, never typed in, so regenerating after an
upstream fix keeps the manuscript consistent (paper repo rule).

Usage:
    python results/make_paper_tables.py
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, "stage1")
from protocols import load_index, evaluate, auc

OUT = "results/tables"
rows = load_index()
test_rows = [r for r in rows if r["split"] == "test"]
by_child = {}
for r in test_rows:
    by_child.setdefault(r["child_id"], []).append(r)


def scores_of(path):
    sc = {r["identifier"]: float(r["smm_score"]) for r in csv.DictReader(open(path))}
    return sc, np.array([sc.get(r["identifier"], np.nan) for r in rows])


def per_child(sc):
    aucs = []
    for child, rs in by_child.items():
        labels = np.array([r["binary_label"] for r in rs])
        if (labels == 1).sum() < 10 or (labels == 0).sum() < 10:
            continue
        aucs.append(auc(np.array([sc[r["identifier"]] for r in rs]), labels))
    return np.array(aucs)


def f(x, digits=3):
    return f"{x:.{digits}f}"


os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- table 1
# main comparison: both protocols, all models
MODELS = [
    ("quality-only logistic regression", None, "5"),
    ("released PoseConv3D checkpoint", "results/segment_scores/gate03_segment_scores.csv", "3.0M"),
    ("ours, coordinates only (v1)", "results/segment_scores/stage1_detector_test_scores.csv", "353K"),
    ("ours, full recipe (v3, 3 seeds)",
     ["results/segment_scores/stage2_sapl_lam0_seed0_test_scores.csv",
      "results/segment_scores/stage2_sapl_lam0_seed1_test_scores.csv",
      "results/segment_scores/stage2_sapl_lam0_seed2_test_scores.csv"], "353K"),
]
# quality-LR is refit here exactly as in stage1/eval_baselines.py
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
X = np.array([[r[k] for k in ("q_lowconf", "q_missing", "q_frozen", "q_energy")] for r in rows])
y = np.array([r["binary_label"] for r in rows])
tr = np.array([r["split"] == "train" for r in rows])
scaler = StandardScaler().fit(X[tr])
clf = LogisticRegression(max_iter=1000, random_state=0).fit(scaler.transform(X[tr]), y[tr])
qlr_scores = clf.predict_proba(scaler.transform(X))[:, 1]

lines = [
    r"\begin{tabular}{lrccc}", r"\toprule",
    r"model & params & standard AUC & QC AUC & $\Delta$ \\",
    r"\midrule",
]
for name, path, params in MODELS:
    if isinstance(path, list):
        sts, hns = [], []
        for p in path:
            res = evaluate(scores_of(p)[1], rows)
            sts.append(res["standard"]["auc"])
            hns.append(res["hard_negative"]["auc"])
        lines.append(f"{name} & {params} & {f(np.mean(sts))} $\\pm$ {f(np.std(sts))} & "
                     f"{f(np.mean(hns))} $\\pm$ {f(np.std(hns))} & "
                     f"{f(np.mean(sts) - np.mean(hns))} \\\\")
        continue
    s = qlr_scores if path is None else scores_of(path)[1]
    res = evaluate(s, rows)
    st, hn = res["standard"]["auc"], res["hard_negative"]["auc"]
    lines.append(f"{name} & {params} & {f(st)} & {f(hn)} & {f(st - hn)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_main.tex", "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- table 2
# ablation 2x2 plus v2 seed spread
ABL = [
    ("neither (v1)", "results/segment_scores/stage1_detector_test_scores.csv"),
    ("velocity only", "results/segment_scores/stage1_abl_velocity_test_scores.csv"),
    ("joint dropout only", "results/segment_scores/stage1_abl_dropout_test_scores.csv"),
    ("both (v2)", "results/segment_scores/stage1_detector_v2_test_scores.csv"),
]
lines = [r"\begin{tabular}{lcc}", r"\toprule",
         r"variant & standard AUC & QC AUC \\", r"\midrule"]
for name, path in ABL:
    res = evaluate(scores_of(path)[1], rows)
    lines.append(f"{name} & {f(res['standard']['auc'])} & {f(res['hard_negative']['auc'])} \\\\")
v2_st, v2_hn = [], []
for p in ["results/segment_scores/stage1_detector_v2_test_scores.csv",
          "results/segment_scores/stage1_v2_seed1_test_scores.csv",
          "results/segment_scores/stage1_v2_seed2_test_scores.csv"]:
    res = evaluate(scores_of(p)[1], rows)
    v2_st.append(res["standard"]["auc"])
    v2_hn.append(res["hard_negative"]["auc"])
lines.append(r"\midrule")
lines.append(f"both (v2), 3 seeds & {f(np.mean(v2_st))} $\\pm$ {f(np.std(v2_st))} "
             f"& {f(np.mean(v2_hn))} $\\pm$ {f(np.std(v2_hn))} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_ablation.tex", "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- table 3
# SAPL tradeoff with per-child stats and MI (MI values from stage2 logs,
# reproduced by stage2/mi_probe.py; kept as a documented constant map so
# this script has no GPU dependency)
MI_BITS = {"0": 1.3965, "0.1": 1.3444, "0.3": 1.1854, "1.0": 0.5902}  # corrected H_emp(Y)-CE bound
SAPL = [
    ("0", "results/segment_scores/stage1_detector_v3_test_scores.csv"),
    ("0.1", "results/segment_scores/stage2_sapl_lam0.1_test_scores.csv"),
    ("0.3", "results/segment_scores/stage2_sapl_lam0.3_test_scores.csv"),
    ("1.0", "results/segment_scores/stage2_sapl_lam1.0_test_scores.csv"),
]
lines = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"$\lambda$ & standard AUC & per-child mean & worst child & gap & identity MI (bits) \\",
         r"\midrule"]
for lam, path in SAPL:
    sc, s = scores_of(path)
    res = evaluate(s, rows)
    a = per_child(sc)
    lines.append(f"{lam} & {f(res['standard']['auc'])} & {f(a.mean())} & "
                 f"{f(a.min())} & {f(a.max() - a.min())} & {f(MI_BITS[lam], 2)} \\\\")
# matched three-seed families at lambda 0 and 0.3
lines.append(r"\midrule")
for fam, paths in [
    ("0, 3 seeds", ["results/segment_scores/stage2_sapl_lam0_seed0_test_scores.csv",
                    "results/segment_scores/stage2_sapl_lam0_seed1_test_scores.csv",
                    "results/segment_scores/stage2_sapl_lam0_seed2_test_scores.csv"]),
    ("0.3, 3 seeds", ["results/segment_scores/stage2_sapl_lam0.3_test_scores.csv",
                      "results/segment_scores/stage2_sapl_lam0.3_seed1_test_scores.csv",
                      "results/segment_scores/stage2_sapl_lam0.3_seed2_test_scores.csv"])]:
    st3, pm3, w3, g3 = [], [], [], []
    for p in paths:
        sc, s = scores_of(p)
        st3.append(evaluate(s, rows)["standard"]["auc"])
        a = per_child(sc)
        pm3.append(a.mean()); w3.append(a.min()); g3.append(a.max() - a.min())
    lines.append(f"{fam} & {f(np.mean(st3))} $\\pm$ {f(np.std(st3))} & "
                 f"{f(np.mean(pm3))} $\\pm$ {f(np.std(pm3))} & "
                 f"{f(np.mean(w3))} $\\pm$ {f(np.std(w3))} & "
                 f"{f(np.mean(g3))} $\\pm$ {f(np.std(g3))} & \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_sapl.tex", "w").write("\n".join(lines) + "\n")

print("wrote", os.listdir(OUT))

# ---------------------------------------------------------------- table 4
# per-camera AUC and tracking quality (camera = third identifier token)
def camera_of(r):
    return int(r["identifier"].split("_")[2])

CAM_MODELS = [("checkpoint", "results/segment_scores/gate03_segment_scores.csv"),
              ("ours (v3)", "results/segment_scores/stage2_sapl_lam0_seed0_test_scores.csv")]
cams = sorted({camera_of(r) for r in test_rows})
lines = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"camera & segments (SMM/non) & mean low-conf & checkpoint AUC & ours AUC \\",
         r"\midrule"]
scs = {n: scores_of(p)[0] for n, p in CAM_MODELS}
for cam in cams:
    rs = [r for r in test_rows if camera_of(r) == cam]
    labels = np.array([r["binary_label"] for r in rs])
    q = np.mean([r["q_lowconf"] for r in rs])
    aucs = []
    for n, _ in CAM_MODELS:
        s = np.array([scs[n][r["identifier"]] for r in rs])
        aucs.append(auc(s, labels))
    lines.append(f"{cam} & {int((labels==1).sum())}/{int((labels==0).sum())} & "
                 f"{q:.2f} & {f(aucs[0])} & {f(aucs[1])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_camera.tex", "w").write("\n".join(lines) + "\n")
print("wrote table_camera.tex")

# ---------------------------------------------------------------- table 5
# architecture breadth: 3 architectures x 2 recipe arms
BREADTH = [
    ("ST-GCN", "3.1M", "base", "results/segment_scores/wacv_stgcn_base_test_scores.csv"),
    ("ST-GCN", "3.1M", "remedies", "results/segment_scores/wacv_stgcn_remedies_test_scores.csv"),
    ("AGCN", "3.4M", "base", "results/segment_scores/wacv_agcn_base_test_scores.csv"),
    ("AGCN", "3.4M", "remedies", "results/segment_scores/wacv_agcn_remedies_test_scores.csv"),
    ("CompactCTRGCN (ours)", "353K", "base", "results/segment_scores/wacv_ours_base_test_scores.csv"),
    ("CompactCTRGCN (ours)", "353K", "remedies", "results/segment_scores/stage2_sapl_lam0_seed0_test_scores.csv"),
]
lines = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"architecture & params & arm & standard AUC & QC AUC & $\Delta$ \\",
         r"\midrule"]
prev = None
for arch, par, arm, path in BREADTH:
    if prev and prev != arch:
        lines.append(r"\midrule")
    prev = arch
    res = evaluate(scores_of(path)[1], rows)
    st, hn = res["standard"]["auc"], res["hard_negative"]["auc"]
    name = arch if arm == "base" else ""
    p = par if arm == "base" else ""
    lines.append(f"{name} & {p} & {arm} & {f(st)} & {f(hn)} & {f(st-hn)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_breadth.tex", "w").write("\n".join(lines) + "\n")
print("wrote table_breadth.tex")
