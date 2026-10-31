"""Draw the CompactCTRGCN architecture diagram for the supplementary.

Two panels: (A) the network stack from input tensor to the detection
score, with the SAPL identity branch shown dashed since every detector
result outside the SAPL sweep uses lambda = 0 (main paper Section 3.5); (B) one
spatial-temporal block expanded into its CTR-GC branches and multi-scale
temporal branches. Numbers (widths, channel counts, kernel sizes) match
stage1/model.py and sections/math_details.tex; this figure adds no new
claims, it is a reading aid for the equations already in the appendix.

Usage:
    python results/make_architecture_figure.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
LIGHT_BLUE = "#e8f0fb"
LIGHT_GRID = "#f2f1ee"
OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "text.color": INK, "pdf.fonttype": 42,
})


def box(ax, x, y, w, h, text, fc="white", ec=INK, lw=1.1, fontsize=8.2,
        ls="-", textcolor=INK, weight="normal"):
    p = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.004,rounding_size=0.012",
                        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls,
                        zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=textcolor, weight=weight, zorder=4,
             linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, p1, p2, color=INK, lw=1.2, ls="-", connectionstyle=None):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=9,
                         linewidth=lw, color=color, linestyle=ls, zorder=2,
                         connectionstyle=connectionstyle, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def right(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def bottom(b):
    x, y, w, h = b
    return (x + w / 2, y)


fig = plt.figure(figsize=(7.0, 4.55))

# panel A, top
axA = fig.add_axes([0.0, 0.56, 1.0, 0.42])
axA.set_xlim(0, 11.3)
axA.set_ylim(-0.95, 2.2)
axA.axis("off")
axA.text(0.0, 2.08, "A. Network stack", fontsize=9.5, weight="bold", color=INK)

y0, h = 0.55, 1.0
xs = [0.0, 1.45, 2.90, 4.55, 6.20, 7.85, 9.05]
ws = [1.15, 1.15, 1.35, 1.35, 1.35, 0.95, 1.15]

b_in = box(axA, xs[0], y0, ws[0], h,
           "input\n$5\\times200\\times17$\n(C,T,V)", fc=LIGHT_GRID)
b_bn = box(axA, xs[1], y0, ws[1], h, "input BN\n(flattened\nC$\\cdot$V)", fc="white")
b_s1 = box(axA, xs[2], y0, ws[2], h,
           "stage 1\n2$\\times$ ST-block\nwidth 48", fc=LIGHT_BLUE)
b_s2 = box(axA, xs[3], y0, ws[3], h,
           "stage 2\n2$\\times$ ST-block\nwidth 96, /2", fc=LIGHT_BLUE)
b_s3 = box(axA, xs[4], y0, ws[4], h,
           "stage 3\n2$\\times$ ST-block\nwidth 136, /2", fc=LIGHT_BLUE)
b_gap = box(axA, xs[5], y0, ws[5], h, "GAP\nover T, V", fc="white")
b_fc = box(axA, xs[6], y0, ws[6], h,
           "linear\n136 $\\to$ 1", fc=LIGHT_GRID)

for a, b in [(b_in, b_bn), (b_bn, b_s1), (b_s1, b_s2), (b_s2, b_s3),
             (b_s3, b_gap), (b_gap, b_fc)]:
    arrow(axA, right(a), left(b))

arrow(axA, right(b_fc), (right(b_fc)[0] + 0.30, right(b_fc)[1]))
axA.text(right(b_fc)[0] + 0.36, right(b_fc)[1], "SMM\nscore", ha="left",
          va="center", fontsize=8.2, color=INK)

# SAPL identity branch, dashed, off the pooled features
b_grl = box(axA, xs[5] - 0.05, y0 - 0.98, 1.55, 0.62,
            "gradient reversal\n$\\times(-\\lambda)$", fc="white", ec=MUTED,
            ls=(0, (4, 2)), fontsize=7.6, textcolor=MUTED)
b_id = box(axA, xs[5] + 1.65, y0 - 0.98, 1.55, 0.62,
           "identity head\n$\\to$ child logits", fc="white", ec=MUTED,
           ls=(0, (4, 2)), fontsize=7.6, textcolor=MUTED)
arrow(axA, (bottom(b_gap)[0], bottom(b_gap)[1]), top(b_grl), color=MUTED,
      ls=(0, (4, 2)),
      connectionstyle="arc3,rad=0.0")
arrow(axA, right(b_grl), left(b_id), color=MUTED, ls=(0, (4, 2)))
axA.text(xs[5] - 0.05, y0 - 1.18,
         r"subject-adversarial branch only ($\lambda \ne 0$); every result outside the sweep uses $\lambda=0$ (Section 3.5)",
         fontsize=7.0, color=MUTED, ha="left", va="top")

# panel B, bottom
axB = fig.add_axes([0.0, 0.0, 1.0, 0.50])
axB.set_xlim(0, 10.4)
axB.set_ylim(0, 3.05)
axB.axis("off")
axB.text(0.0, 2.92, "B. One spatial-temporal (ST) block", fontsize=9.5,
          weight="bold", color=INK)

axB.text(0.0, 2.72, "(any of the six blocks; widths/strides per stage, panel A)",
          fontsize=7.2, color=MUTED)

by0, bh = 0.25, 0.55
b_bin = box(axB, 0.0, 1.15, 1.05, bh, "block\ninput", fc=LIGHT_GRID, fontsize=7.8)

# three CTR-GC branches summed; branch 1 carries the full form, branches
# 2-3 are the same equation with independent weights, so they stay terse
# rather than repeating it verbatim.
br_y = [2.05, 1.15, 0.25]
br_labels = [
    "CTR-GC branch 1\n$A^{(c)}=A^{base}+\\alpha\\,\\Delta_c(Q,K)$",
    "CTR-GC branch 2\n(same form, own weights)",
    "CTR-GC branch 3\n(same form, own weights)",
]
ctrgc_boxes = []
for lab, yy in zip(br_labels, br_y):
    bb = box(axB, 1.35, yy, 2.35, bh, lab, fc="white", fontsize=7.1)
    ctrgc_boxes.append(bb)
    arrow(axB, right(b_bin), left(bb),
          connectionstyle=f"arc3,rad={(yy - 1.425) * 0.18}")

b_sum = box(axB, 4.05, 1.15, 0.55, bh, "+", fc=LIGHT_BLUE, fontsize=13,
            weight="bold")
for bb in ctrgc_boxes:
    arrow(axB, right(bb), left(b_sum),
          connectionstyle=f"arc3,rad={(1.425 - (bb[1] + bb[3] / 2)) * 0.18}")

b_bnrelu1 = box(axB, 4.95, 1.15, 1.15, bh, "BN, residual,\nReLU", fc="white",
                fontsize=7.4)
arrow(axB, right(b_sum), left(b_bnrelu1))

# temporal unit: four parallel branches, one quarter of the width each,
# fanning directly out of the BN/residual/ReLU output (no junction box)
temp_labels = [
    "5$\\times$1 conv\ndilation 1",
    "5$\\times$1 conv\ndilation 2",
    "3$\\times$1\nmax pool",
    "1$\\times$1 conv\n(pointwise)",
]
temp_y = [2.35, 1.65, 0.95, 0.25]
temp_boxes = []
for lab, yy in zip(temp_labels, temp_y):
    bb = box(axB, 7.15, yy, 1.55, 0.5, lab, fc="white", fontsize=7.1)
    temp_boxes.append(bb)
    arrow(axB, right(b_bnrelu1), left(bb),
          connectionstyle=f"arc3,rad={(yy + 0.25 - 1.425) * 0.10}")

b_concat = box(axB, 9.05, 1.15, 0.55, bh, "concat\n+res.", fc=LIGHT_BLUE,
               fontsize=6.6)
for bb in temp_boxes:
    arrow(axB, right(bb), left(b_concat),
          connectionstyle=f"arc3,rad={(1.425 - (bb[1] + bb[3] / 2)) * 0.10}")

axB.annotate("BN, ReLU $\\to$\nnext block", right(b_concat),
             textcoords="offset points", xytext=(4, 0), ha="left",
             va="center", fontsize=7.0, color=INK)

fig.savefig(f"{OUT}/fig_architecture.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_architecture.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("wrote fig_architecture")
