# SAFE-SMM: code and configs

Code behind "SAFE-SMM: Shortcut-Aware Frozen Evaluation and Efficient
Detection of Stereotypical Motor Movements from Skeleton Video" (WACV
2027 submission). Anonymized for review: no git history, no run logs,
no author-identifying paths.

## Layout

    stage0/     data audit, pose-quality audit, baseline reproduction,
                periodicity check, compute check (main paper Section 3.1)
    stage1/     detector (CompactCTRGCN), training, evaluation protocols,
                per-child quantification
    stage2/     subject-adversarial training (SAPL), identity MI probe,
                per-child fairness analysis
    revision/   controls added during the review-response round
                (bootstrap CIs, length-decile check, quality-matched
                control, and related stress tests)
    configs/    one YAML per run; configs/protocols.yaml carries the
                frozen quality-filter thresholds referenced in main
                paper Section 3.2
    results/    per-segment scores for every reported number, plus the
                scripts that generate every table and figure

## Reproducing

Requires the public ASDPose release (Barami et al., 2024) at
`data/StimmingADSMotion`, and a CUDA GPU for training. Stage 0's gates
and the table/figure scripts are CPU-only.

    bash reproduce.sh          # stage0 gates through every table/figure
    bash revision/run_w3.sh    # optional, ~6 more GPU-hours of controls

Every reported number already has its score CSV under
`results/segment_scores/`, so `results/make_paper_tables.py` and
`results/make_paper_figures.py` reproduce every table and figure from
those directly, without retraining anything.
