#!/bin/bash
# W3 retrains, run serially on the single 8 GiB GPU, highest value first:
#   a) v3 without the confidence channel, seeds 0/1/2
#   b) SAPL lambda 0.5 and 0.7 (fills the 0.3 to 1.0 gap), seed 0
#   c) velocity-only and dropout-only arms at v3 normalization/sampling
# Each run's test scores are copied into results/segment_scores/.
# Usage: bash revision/run_w3.sh   (from the repo root)
set -e
cd "$(dirname "$0")/.."
mkdir -p logs cache

run () {
    echo "=== $(date '+%F %T') starting: $*"
    "$@"
}

for s in 0 1 2; do
    run python stage1/train.py --config configs/stage1_v3_noconf.yaml \
        --seed $s --out-prefix cache/stage1_v3_noconf_seed$s
    cp cache/stage1_v3_noconf_seed${s}_test_scores.csv results/segment_scores/
done

for lam in 0.5 0.7; do
    run python stage2/train_sapl.py --config configs/stage2_sapl.yaml \
        --lam $lam --out-prefix cache/stage2_sapl_lam$lam
    cp cache/stage2_sapl_lam${lam}_test_scores.csv results/segment_scores/
done

run python stage1/train.py --config configs/stage1_abl_velocity_v3norm.yaml \
    --out-prefix cache/stage1_abl_velocity_v3norm
cp cache/stage1_abl_velocity_v3norm_test_scores.csv results/segment_scores/

run python stage1/train.py --config configs/stage1_abl_dropout_v3norm.yaml \
    --out-prefix cache/stage1_abl_dropout_v3norm
cp cache/stage1_abl_dropout_v3norm_test_scores.csv results/segment_scores/

echo "=== $(date '+%F %T') W3 complete"
