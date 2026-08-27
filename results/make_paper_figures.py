"""Generate the paper's figures from the per-segment score CSVs.

Writes results/figures/*.pdf (vector, for LaTeX) and *.png (for visual
inspection). Numbers are computed from the score files; the identity-MI
values are the probe outputs recorded in logs (reproducible via
stage2/mi_probe.py, which needs a GPU).

Figure 1: subject-adversarial tradeoff. Two stacked panels over the same
lambda axis: detection AUC (standard and worst-child) on top, identity
MI below. Error bars at lambda 0.3 are the spread over three seeds.

Figure 2: per-child AUC dumbbell, released checkpoint vs our v3, sorted
by v3, chance line at 0.5.

Usage:
    python results/make_paper_figures.py
"""

import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "stage1")
from protocols import load_index, evaluate, auc

BLUE, ORANGE = "#2a78d6", "#eb6834"   # validated categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})

rows = load_index()
test_rows = [r for r in rows if r["split"] == "test"]
by_child = {}
for r in test_rows:
    by_child.setdefault(r["child_id"], []).append(r)


def scores_of(path):
    return {r["identifier"]: float(r["smm_score"]) for r in csv.DictReader(open(path))}


def standard_auc(sc):
    s = np.array([sc.get(r["identifier"], np.nan) for r in rows])
    return evaluate(s, rows)["standard"]["auc"]


def per_child_aucs(sc):
    out = {}
    for child, rs in by_child.items():
        labels = np.array([r["binary_label"] for r in rs])
        if (labels == 1).sum() < 10 or (labels == 0).sum() < 10:
            continue
        out[child] = auc(np.array([sc[r["identifier"]] for r in rs]), labels)
    return out


# ------------------------------------------------------------- figure 1
LAM_FILES = {
    "0": ["results/segment_scores/stage2_sapl_lam0_seed0_test_scores.csv",
          "results/segment_scores/stage2_sapl_lam0_seed1_test_scores.csv",
          "results/segment_scores/stage2_sapl_lam0_seed2_test_scores.csv"],
    "0.1": ["results/segment_scores/stage2_sapl_lam0.1_test_scores.csv"],
    "0.3": ["results/segment_scores/stage2_sapl_lam0.3_test_scores.csv",
            "results/segment_scores/stage2_sapl_lam0.3_seed1_test_scores.csv",
            "results/segment_scores/stage2_sapl_lam0.3_seed2_test_scores.csv"],
    "0.5": ["results/segment_scores/stage2_sapl_lam0.5_test_scores.csv"],
    "0.7": ["results/segment_scores/stage2_sapl_lam0.7_test_scores.csv"],
    "1.0": ["results/segment_scores/stage2_sapl_lam1.0_test_scores.csv"],
}
MI_BITS = {"0": 1.40, "0.1": 1.34, "0.3": 1.19, "0.5": 1.05,
           "0.7": 0.65, "1.0": 0.59}  # corrected H_emp(Y)-CE bound

lams = list(LAM_FILES)
x = np.arange(len(lams))
std_mean, std_err, worst_mean, worst_err = [], [], [], []
for lam in lams:
    stds = [standard_auc(scores_of(p)) for p in LAM_FILES[lam]]
    worsts = [min(per_child_aucs(scores_of(p)).values()) for p in LAM_FILES[lam]]
    std_mean.append(np.mean(stds)); std_err.append(np.std(stds))
    worst_mean.append(np.mean(worsts)); worst_err.append(np.std(worsts))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.6, 4.2), sharex=True,
                               height_ratios=[2, 1])
ax1.grid(axis="y", color=GRID, linewidth=0.6)
ax1.set_axisbelow(True)
ax1.axhline(0.8347, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
ax1.annotate("released baseline, standard (0.835)", (-0.06, 0.8347),
             textcoords="offset points", xytext=(0, -11), fontsize=7.5, color=MUTED)
ax1.errorbar(x, std_mean, yerr=std_err, color=BLUE, linewidth=1.8,
             marker="o", markersize=6, capsize=3, label="standard AUC")
ax1.errorbar(x, worst_mean, yerr=worst_err, color=ORANGE, linewidth=1.8,
             marker="s", markersize=6, capsize=3, label="worst-child AUC")
ax1.set_ylabel("test AUC")
ax1.set_ylim(0.52, 1.0)
ax1.legend(frameon=False, loc="lower left", fontsize=8)

ax2.grid(axis="y", color=GRID, linewidth=0.6)
ax2.set_axisbelow(True)
mi = [MI_BITS[l] for l in lams]
ax2.axhline(7.58, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
ax2.annotate("empirical identity entropy (7.58 bits)", (-0.06, 7.58),
             textcoords="offset points", xytext=(0, -11), fontsize=7.5,
             color=MUTED)
ax2.plot(x, mi, color=BLUE, linewidth=1.8, marker="o", markersize=6)
for xi, yi in zip(x, mi):
    ax2.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                 xytext=(0, 6), ha="center", fontsize=7.5, color=MUTED)
ax2.set_ylabel("identity MI (bits)")
ax2.set_ylim(0, 8.4)
ax2.set_xlabel(r"adversary strength $\lambda$")
ax2.set_xticks(x, lams)

fig.tight_layout()
fig.savefig(f"{OUT}/fig_sapl_tradeoff.pdf")
fig.savefig(f"{OUT}/fig_sapl_tradeoff.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------- figure 2
ckpt = per_child_aucs(scores_of("results/segment_scores/gate03_segment_scores.csv"))
# The headline detector is the matched lambda=0 three-seed family, so the
# per-child curve averages the three seeds. Plotting one run would show a
# worst-child value that the seed lottery moves by about 0.035, which is the
# reading error this paper documents in Section 4.6.
_ours_seeds = [per_child_aucs(scores_of(p)) for p in LAM_FILES["0"]]
ours = {c: float(np.mean([s[c] for s in _ours_seeds])) for c in _ours_seeds[0]}
children = sorted(ours, key=lambda c: ours[c])
y = np.arange(len(children))

fig, ax = plt.subplots(figsize=(4.6, 5.2))
ax.grid(axis="x", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.axvline(0.5, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
ax.annotate("chance", (0.5, len(children) - 0.2), fontsize=7.5, color=MUTED,
            ha="center", textcoords="offset points", xytext=(0, 6))
for yi, c in zip(y, children):
    ax.plot([ckpt[c], ours[c]], [yi, yi], color=GRID, linewidth=1.4, zorder=1)
ax.scatter([ckpt[c] for c in children], y, s=34, marker="s", color=ORANGE,
           zorder=2, label="released checkpoint")
ax.scatter([ours[c] for c in children], y, s=38, marker="o", color=BLUE,
           zorder=3, label="ours (v3)")
ax.set_yticks(y, [str(c) for c in children], fontsize=7)
ax.set_ylabel("test child")
ax.set_xlabel("per-child AUC")
ax.set_xlim(0.42, 1.02)
ax.legend(frameon=False, loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_per_child.pdf")
fig.savefig(f"{OUT}/fig_per_child.png", dpi=200)
plt.close(fig)

print("wrote", sorted(os.listdir(OUT)))

# ------------------------------------------------------------- figure 3
# confound illustration: per-joint tracking rasters for two segments
sys.path.insert(0, "stage1")
from dataset import SegmentDataset
ds = SegmentDataset(rows, train=False)
te = [r for r in rows if r["split"] == "test"]
smm = [r for r in te if r["binary_label"] == 1 and 250 <= r["n_frames"] <= 400]
non = [r for r in te if r["binary_label"] == 0 and 250 <= r["n_frames"] <= 400]
smm_ex = min(smm, key=lambda r: abs(r["q_lowconf"] - 0.25))
non_ex = min(non, key=lambda r: abs(r["q_lowconf"] - 0.55))

fig_, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
for ax, r, title in [(axes[0], smm_ex, "SMM segment"),
                     (axes[1], non_ex, "non-SMM segment")]:
    arr = ds.load_segment(r)
    tracked = (arr[..., 2] >= 0.3).T          # (17, T)
    fps = 25.0
    ax.imshow(tracked, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#e9e8e4", BLUE]),
              interpolation="nearest",
              extent=[0, tracked.shape[1] / fps, 16.5, -0.5])
    cov = tracked.mean()
    ax.set_title(f"{title} ({cov:.0%} joint-frames tracked)", fontsize=8.5)
    ax.set_xlabel("time (s)")
    for yline in (4.5, 10.5):
        ax.axhline(yline, color="#ffffff", linewidth=1.2)
axes[0].set_ylabel("joint")
axes[0].set_yticks([2, 7.5, 13.5], ["head", "arms", "legs"], fontsize=8)
fig_.tight_layout()
fig_.savefig(f"{OUT}/fig_confound.pdf")
fig_.savefig(f"{OUT}/fig_confound.png", dpi=200)
plt.close(fig_)
print("wrote fig_confound")
