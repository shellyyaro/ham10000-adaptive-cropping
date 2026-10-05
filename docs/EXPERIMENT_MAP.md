# Experiment map

This file maps each stage described in the final report to its reproducible artifacts.

| Report stage | Versioned configuration | Entry point | Main output |
|---|---|---|---|
| Preliminary ResNet training | `configs/resnet_ensemble/` | `scripts/01_train_resnet_ensemble.py` | 24 checkpoint directories |
| Predicted-label Grad-CAM fusion | Configuration constants in `generate_fused_masks.py` | `scripts/02_generate_masks.py` | Three split-specific fused-mask NPZ files |
| Full-image baseline | `FULL_IMAGE` in `report_experiment_configs.py` | `scripts/03_run_report_experiments.py` | Five fold checkpoints and OOF predictions |
| Fixed crop | `FIXED_CROP_BASELINE` in `report_experiment_configs.py` | `scripts/03_run_report_experiments.py` | Five fold checkpoints and OOF predictions |
| Adaptive baseline | `ADAPTIVE_INPUT` + `ADAPTIVE_BASELINE_OBJECTIVE` | `scripts/03_run_report_experiments.py` | Five fold checkpoints and OOF predictions |
| Adaptive alternative objective, common augmentation | `ADAPTIVE_INPUT` + `ALTERNATIVE_UNGATED_OBJECTIVE` | `scripts/03_run_report_experiments.py` | Five fold checkpoints and OOF predictions |
| Adaptive alternative objective, mild augmentation | ungated objective + `MILD_AUGMENTATION` | `scripts/03_run_report_experiments.py` | Five fold checkpoints and OOF predictions |
| Adaptive alternative objective, mild augmentation and gated KL | mild configuration + `GATED_KL` | `scripts/03_run_report_experiments.py` | Five fold checkpoints and OOF predictions |
| Figure 1: mask-area/category distribution | fused training masks at threshold 0.55 | `scripts/05_rebuild_report_artifacts.py` | `figure_1_mask_area_distribution.png` |
| Figure 2: representative fixed/adaptive crops | fused masks plus HAM10000 images `ISIC_0026236` and `ISIC_0032692` | `scripts/05_rebuild_report_artifacts.py` | `figure_2_crop_examples.png` |
| Figure 3: six pooled confusion matrices | six normalized confusion-matrix CSV files | `scripts/05_rebuild_report_artifacts.py` | `figure_3_confusion_matrices.png` |
| Table 1: five-fold mean ± sample SD | `five_fold_mean_sd.csv` | `scripts/05_rebuild_report_artifacts.py` | `table_1_five_fold_performance.csv` |
| Table 2: pooled class recall | diagonal of each pooled normalized confusion matrix | `scripts/05_rebuild_report_artifacts.py` | `table_2_pooled_class_recall.csv` |
| Table 3: published-results comparison | versioned literature extraction | `scripts/05_rebuild_report_artifacts.py` | `table_3_literature_comparison.csv` |

The exact mean and sample-standard-deviation values used for the six-row comparison are stored in `results/reported_metrics/five_fold_mean_sd.csv`. Compact pooled out-of-fold diagnostics for every row are stored in the corresponding subdirectory.

The exact PNG bytes embedded as Figures 1-3 are stored under `results/report_artifacts/canonical/`. The automated contract opens the DOCX as an archive and verifies that all three embedded images match those files byte-for-byte. It also parses all three Word tables and verifies their reported values against the generated CSV artifacts.

## Mask ensemble composition

The ensemble contains 24 separately trained ResNet-18 checkpoints:

| Series | Count | Main variation |
|---|---:|---|
| Regular baseline ablation | 9 | classifier width, dropout, learning rate, weight decay, blur, and max-pooling |
| Mid-level fusion | 8 | source layer, pooling method, and embedding dimension |
| Filter and multiscale series | 7 | stem kernel and multiscale stem/block structure |

The fixed preliminary split uses seed 0. The final classifier evaluation uses five lesion-disjoint train/validation/test folds generated with seed 42. The mask set is not regenerated inside each final fold; this matches the stated conditional evaluation scope in the report.

The versioned files `splits/mask_generation_fixed_split.csv` and `splits/report_five_fold_assignments.csv` record the exact path-independent assignments. Every training entry point verifies regenerated splits against these manifests before training.

For the full claim-by-claim verification, exact fold counts, and automated leakage checks, see `docs/REPORT_CODE_AUDIT.md`.
