#!/bin/bash
# End-to-end reproduction: stage0 sanity gates through the headline
# detector, the SAPL sweep, and every reported table and figure.
#
# Requires the public ASDPose release at data/StimmingADSMotion (see
# main paper Section 3.1). Stage 0 gates and the table/figure scripts
# are CPU-only; stage1/stage2 training need a CUDA GPU and take about
# 1.6 hours per run on an 8 GiB card (roughly 10 GPU-hours total for
# this script, not counting revision/run_w3.sh below).
#
# Usage: bash reproduce.sh   (from the repo root)
set -e
cd "$(dirname "$0")"
mkdir -p cache logs results/segment_scores

run () {
    echo "=== $(date '+%F %T') starting: $*"
    "$@"
}

echo "## Stage 0: sanity gates (Section 3.1, CPU-only)"
run python stage0/gate01_inventory.py --data-root data/StimmingADSMotion
run python stage0/gate02_pose_quality.py --data-root data/StimmingADSMotion \
    --sample 2000 --seed 0
run python stage0/map_segments_to_recordings.py --split test \
    --out cache/segment_map_test.csv
run python stage0/gate04_periodicity.py --split test
run python stage0/gate05_compute.py

echo "## Stage 0.3: released baseline reproduction (needs their checkpoint)"
run python stage0/gate03_segment_eval.py --out cache/gate03_segment_scores.csv

echo "## Stage 1: segment cache, detector (v3 recipe), baselines"
run python stage1/build_cache.py
run python stage1/eval_baselines.py
run python stage1/quantify_per_child.py --model baseline \
    --out cache/quant_baseline.csv

echo "## Stage 2: SAPL sweep, headline lambda=0 family (3 seeds) plus the tradeoff curve"
for seed in 0 1 2; do
    run python stage2/train_sapl.py --config configs/stage2_sapl.yaml \
        --lam 0 --seed $seed --out-prefix cache/stage2_sapl_lam0_seed$seed
    cp cache/stage2_sapl_lam0_seed${seed}_test_scores.csv results/segment_scores/
done
for lam in 0.1 0.3 0.5 0.7 1.0; do
    run python stage2/train_sapl.py --config configs/stage2_sapl.yaml \
        --lam $lam --out-prefix cache/stage2_sapl_lam$lam
    cp cache/stage2_sapl_lam${lam}_test_scores.csv results/segment_scores/
done
run python stage1/quantify_per_child.py --model ours \
    --weights cache/stage2_sapl_lam0_seed0.pt --out cache/quant_ours.csv
run python stage2/mi_probe.py --weights cache/stage2_sapl_lam0_seed0.pt --sapl
run python stage2/per_child_analysis.py \
    --scores results/segment_scores/stage2_sapl_lam0_seed0_test_scores.csv

echo "## Tables and figures, from the score CSVs above (no GPU needed)"
run python results/make_paper_tables.py
run python results/make_paper_figures.py
run python results/make_architecture_figure.py

echo "=== $(date '+%F %T') reproduce.sh complete"
echo
echo "Not run above (optional, ~6 more GPU-hours): the revision-round"
echo "controls (bootstrap CIs, length-decile check, quality-matched"
echo "control, and the rest of Section 4/5's stress tests):"
echo "    bash revision/run_w3.sh"
