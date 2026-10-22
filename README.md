# SAFE-SMM: code, configurations, scores and logs

Code behind "SAFE-SMM: Shortcut-Aware Frozen Evaluation and Efficient
Detection of Stereotypical Motor Movements from Skeleton Video", accepted
at WACV 2027 (Evaluations and Datasets track). Every number in the paper
and its supplementary material can be regenerated from this repository.

## Layout

    stage0/     data inventory, pose-quality audit, baseline reproduction,
                periodicity check, compute check (paper Section 3.1)
    stage1/     the detector (CompactCTRGCN), training, the two evaluation
                protocols, per-child quantification
    stage2/     subject-adversarial training, identity probe, per-child
                analysis
    revision/   controls added in response to review: bootstrap intervals,
                Delta ceiling scales, length-decile check, quality-matched
                control, and related stress tests
    configs/    one YAML per run; configs/protocols.yaml holds the frozen
                quality-filter thresholds of paper Section 3.2, committed
                before any model was trained
    results/    per-segment scores for every reported number, the scripts
                that produce every table and figure, and the figure sources
    logs/       per-epoch training logs for every run, the local machine
                spec, and the cloud instance spec

## Reproducing

The ASDPose release (Barami et al., 2024) is expected at
`data/StimmingADSMotion`. Stage 0 and every table and figure script run
on CPU. Training needs a CUDA GPU with 8 GiB or more and takes about 1.6
hours per run.

    bash reproduce.sh          # stage0 gates through every table and figure
    bash revision/run_w3.sh    # the review-round retrains, about 6 GPU-hours

Since every reported number already has its score CSV under
`results/segment_scores/`, the two scripts `results/make_paper_tables.py`
and `results/make_paper_figures.py` rebuild every table and figure
without retraining.

## Weights

Trained checkpoints for the headline v3 family, the no-confidence
control, the ablation arms, the breadth arms, and the subject-adversarial
sweep are published as release assets of this repository.

## Not released

Pooled feature vectors carry recoverable child-identity information
(paper Section 4.5) and are not released. The dataset itself is
distributed by its authors.
