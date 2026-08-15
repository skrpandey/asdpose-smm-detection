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
    ("ours, full recipe (v3)", "results/segment_scores/stage1_detector_v3_test_scores.csv", "353K"),
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
    r"model & params & standard AUC & hard-negative AUC & $\Delta$ \\",
    r"\midrule",
]
for name, path, params in MODELS:
    s = qlr_scores if path is None else scores_of(path)[1]
    res = evaluate(s, rows)
    st, hn = res["standard"]["auc"], res["hard_negative"]["auc"]
    lines.append(f"{name} & {params} & {f(st)} & {f(hn)} & {f(hn - st)} \\\\")
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
         r"variant & standard AUC & hard-negative AUC \\", r"\midrule"]
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
lines.append(f"both, 3 seeds & {f(np.mean(v2_st))} $\\pm$ {f(np.std(v2_st))} "
             f"& {f(np.mean(v2_hn))} $\\pm$ {f(np.std(v2_hn))} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_ablation.tex", "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- table 3
# SAPL tradeoff with per-child stats and MI (MI values from stage2 logs,
# reproduced by stage2/mi_probe.py; kept as a documented constant map so
# this script has no GPU dependency)
MI_BITS = {"0": 1.6697, "0.1": 1.6175, "0.3": 1.4586, "1.0": 0.8633}
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
# lambda 0.3 seed spread
st3, w3, g3 = [], [], []
for p in ["results/segment_scores/stage2_sapl_lam0.3_test_scores.csv",
          "results/segment_scores/stage2_sapl_lam0.3_seed1_test_scores.csv",
          "results/segment_scores/stage2_sapl_lam0.3_seed2_test_scores.csv"]:
    sc, s = scores_of(p)
    st3.append(evaluate(s, rows)["standard"]["auc"])
    a = per_child(sc)
    w3.append(a.min())
    g3.append(a.max() - a.min())
lines.append(r"\midrule")
lines.append(f"0.3, 3 seeds & {f(np.mean(st3))} $\\pm$ {f(np.std(st3))} & & "
             f"{f(np.mean(w3))} $\\pm$ {f(np.std(w3))} & {f(np.mean(g3))} $\\pm$ {f(np.std(g3))} & \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_sapl.tex", "w").write("\n".join(lines) + "\n")

print("wrote", os.listdir(OUT))
