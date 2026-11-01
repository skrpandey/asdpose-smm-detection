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

Trained checkpoints are attached to the v1.0 release of this repository:
26 files, one per training run, each named after the score CSV that run
produced under `results/segment_scores/`. `SHA256SUMS` in the release
lists a checksum for every file.

| checkpoints | paper |
|---|---|
| `stage1_detector` (v1), `stage1_detector_v2`, `stage1_v2_seed1`, `stage1_v2_seed2` | Tables 1 and 2 |
| `stage1_abl_velocity`, `stage1_abl_dropout`, `stage1_abl_velocity_v3norm`, `stage1_abl_dropout_v3norm`, `wacv_ours_base` | Table 2 |
| `wacv_stgcn_base`, `wacv_stgcn_remedies`, `wacv_agcn_base`, `wacv_agcn_remedies` | Table 3 |
| `stage1_detector_v3` (the single lambda = 0 run), `stage2_sapl_lam0.1`, `lam0.3` with `lam0.3_seed1` and `lam0.3_seed2`, `lam0.5`, `lam0.7`, `lam1.0` | Table 4 |
| `stage1_v3_noconf_seed0` to `seed2` | no-confidence control, Section 4.1 |
| `stage1_permuted`, `stage1_v2_permuted` | shuffled-label controls, Section 4.3 |

The checkpoints of the headline lambda = 0 three-seed family were not
retained: Table 1 "v3, 3 seeds", the "both (v3)" row of Table 2, and the
CompactCTRGCN remedies row of Table 3. Their per-segment scores are in
`results/segment_scores/stage2_sapl_lam0_seed{0,1,2}_test_scores.csv`,
so every number they support can still be recomputed. The closest
released weights are the no-confidence family (0.956 standard, 0.944
quality-controlled, three seeds) and `stage1_detector_v3`, a single run
of the same recipe (0.953 standard).

Each file is a dict holding `state_dict`, the training `config`, the
`seed`, and `val_auc`. CompactCTRGCN checkpoints load as in
`stage1/quantify_per_child.py`. Subject-adversarial checkpoints prefix
the detector's keys with `backbone.` and also hold the identity head used
during training; strip the prefix as `stage2/mi_probe.py --sapl` does.
The `wacv_stgcn_*` and `wacv_agcn_*` files load into `STGCN` and `AGCN`
from `stage1/baseline_archs.py` with `c_in` from the config; the widths
in their config are not used.

## Not released

Pooled feature vectors carry recoverable child-identity information
(paper Section 4.5) and are not released. The dataset itself is
distributed by its authors.

## License

Code and released checkpoints are under the MIT license (see `LICENSE`),
the same license as the ASDPose release. The dataset itself is not
included and stays under its authors' terms.
