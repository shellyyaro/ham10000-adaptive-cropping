# Report to code audit

This audit maps the methods and experiments stated in the submitted report to versioned code, configurations, split assignments, and compact results. It also records the automated checks that must pass before submission.

## Audit conclusion

The complete reported workflow is represented in the repository: the 24 ResNet-18 preliminary runs, fixed-mask generation, fixed and adaptive cropping, all six final EfficientNet-B1 configurations, five lesion-disjoint evaluation folds, checkpoint selection, and the reported summary artifacts. Two safeguards were strengthened during the audit:

1. every split path now fails immediately when a `lesion_id` crosses train, validation, or test; and
2. mask generation now verifies the exact reported ensemble composition of 9 + 8 + 7 = 24 checkpoints before a full report run.

A second source-to-submission comparison confirmed that all shared training
modules are present. Nine core modules are byte-for-byte identical to the
development project. `common_io.py` and `run_resnet_with_meta.py` differ only
where the legacy `config.txt` writer/call was removed in favor of JSON; no
model, data, loss, optimization, or metric behavior changed. Mask generation
also verifies the exact 24 versioned run tags, not only the three series
counts, and is forced to reuse the same fixed split manifest as the preliminary
training stage.

## Data partition contract

All partitioning is performed on a one-row-per-lesion table and is expanded to image rows only after the lesion assignments have been selected.

### Preliminary mask-generation split

The fixed assignment is stored in `splits/mask_generation_fixed_split.csv`.

| Partition | Images | Unique lesions |
|---|---:|---:|
| Train | 6,021 | 4,482 |
| Validation | 2,003 | 1,494 |
| Test | 1,991 | 1,494 |
| Total | 10,015 | 7,470 |

`make_or_load_splits()` in `scripts/splits.py` creates the assignment from lesion IDs, stratifies by diagnosis, and now raises an exception for any cross-partition lesion overlap whether the CSVs were freshly generated or loaded from disk. `train_resnet_ensemble.py` additionally compares the regenerated assignment with the versioned manifest before training.

### Final five-fold classification protocol

The exact assignments are stored in `splits/report_five_fold_assignments.csv`. The outer stratified five-fold split assigns every lesion to exactly one held-out test fold. Within each fold, 20% of the remaining lesions are assigned to validation, yielding approximately 64% train, 16% validation, and 20% test.

| Fold | Train images / lesions | Validation images / lesions | Test images / lesions |
|---:|---:|---:|---:|
| 0 | 6,402 / 4,780 | 1,616 / 1,196 | 1,997 / 1,494 |
| 1 | 6,402 / 4,780 | 1,585 / 1,196 | 2,028 / 1,494 |
| 2 | 6,431 / 4,780 | 1,609 / 1,196 | 1,975 / 1,494 |
| 3 | 6,394 / 4,780 | 1,600 / 1,196 | 2,021 / 1,494 |
| 4 | 6,424 / 4,780 | 1,597 / 1,196 | 1,994 / 1,494 |

The automated contract verifies all of the following:

- within every fold, train, validation, and test lesion sets are pairwise disjoint;
- all images of one lesion receive the same partition within a fold;
- every fold covers all 10,015 images and all 7,470 lesions;
- the five test sets are mutually exclusive at lesion and image level; and
- every image and every lesion appears in held-out test data exactly once across the five folds.

## Method and experiment traceability

| Report statement | Reproducible implementation | Status |
|---|---|---|
| 24 ImageNet-pretrained ResNet-18 mask models, trained with metadata | `configs/resnet_ensemble/`; `scripts/01_train_resnet_ensemble.py`; `train_resnet_ensemble.py` | Covered; exact 9/8/7 counts checked |
| Separate fixed lesion-level mask split | `make_or_load_splits()`; `mask_generation_fixed_split.csv` | Covered; fail-fast leakage gate and exact manifest check |
| Predicted-class Grad-CAM per model | `GradCAM` and `compute_majority_fused_mask_for_row()` | Covered |
| Strict majority of at least 13/24 | `get_majority_info()` computes `floor(N/2)+1` | Covered; exact 24-model gate added |
| Confidence weighting, agreement boost, and no-majority mean | `weighted_label_fusion()` and `compute_majority_fused_mask_for_row()` | Covered |
| Largest connected component, dilation, and low-variation full-image fallback | `keep_largest_region()` and the quality gate in mask generation | Covered |
| Masks generated once and fixed across final folds | split-specific mask NPZ files are combined once by `make_combined_mask_npz()` | Covered; masks are not regenerated per fold |
| Fixed threshold-margin crop baseline | `FIXED_TRAIN_POLICIES` and `FIXED_EVAL_POLICIES` in `report_experiment_configs.py` | Covered |
| Adaptive threshold 0.60 and occupancies 0.10–0.60 | `CANONICAL_ADAPTIVE_POLICIES`; `adaptive_crop_policy.py` | Covered |
| Square-first crop, balanced rectangle fallback, boundary shift, bbox/full-image fallback, achieved occupancy | `adaptive_crop_for_target_ratio()` | Covered |
| EfficientNet-B1, 1,280 visual features, shared model, metadata branch, RGB-only input | `models/EfficientNet.py`; common preprocessing sets `mask_mode=none` and uses the mask only for crop coordinates | Covered |
| Shared geometry and optional per-view color perturbations | `MultiViewMaskedBackgroundDataset` in `datasets_masked_background.py` | Covered |
| Training-only metadata imputation and standardization | `build_meta_features_leakage_free()` | Covered |
| AdamW, learning rate and weight decay 1e-4, up to 50 epochs, patience 7 | `BASE_CONFIG` in `report_experiment_configs.py`; final runner | Covered |
| Checkpoint chosen by validation balanced accuracy | final runner calls training with `early_stopping_metric="balanced_acc"` | Covered |
| Baseline averaged-probability CE + view CE + MSE consistency | `ADAPTIVE_BASELINE_OBJECTIVE` and multi-view loss implementation | Covered |
| Inverse-frequency CE weights computed from each training partition and label smoothing 0.05 | `common_imbalance.py`, `BASE_CONFIG`, and `test_core_training_contract.py` | Covered and numerically tested |
| Alternative margin + soft-wrong + KL objective, including detached true-margin gate | `ALTERNATIVE_UNGATED_OBJECTIVE`, `GATED_KL`, and multi-view loss implementation | Covered |
| Six final configurations over five folds | `CONFIGURATIONS` in `run_report_experiments.py` | Covered |
| Test metrics, per-class recall, macro-F1, and lesion consistency | result aggregation in `run_report_experiments.py` and `evaluate_report_checkpoints.py` | Covered |
| Reported five-fold mean and sample SD values | `results/reported_metrics/five_fold_mean_sd.csv` | Covered |
| Figure 1 mask-area/category distribution | `plot_mask_area_distribution()` and canonical report PNG | Covered; exact embedded-image hash checked |
| Figure 2 representative crop comparison | `plot_crop_examples()`, the fixed/adaptive crop implementations, and canonical report PNG | Covered; exact image IDs and embedded-image hash checked |
| Figure 3 six pooled confusion matrices | six normalized matrix CSVs and `plot_confusion_matrix_grid()` | Covered; rebuilt from versioned results and exact embedded-image hash checked |
| Tables 1-2 | `build_table_1()` and `build_table_2()` | Covered; Word table cells checked against generated CSV values |
| Table 3 literature comparison | versioned literature CSV and `build_table_3()` | Covered; present-study values checked against the Word table |

## Reproducibility boundary

The Git package intentionally does not include HAM10000 images, generated NPZ masks, or model checkpoints because they are large and/or subject to dataset distribution constraints. Their required locations and regeneration commands are documented in the repository README. The compact split manifests, all configurations, code, and the reported aggregate/diagnostic results are versioned.

## Verification command

Run from the repository root:

```powershell
python scripts/00_verify_submission.py
```

The tests are the submission gate for repository layout, exact experiment counts, fixed-split counts, lesion-level separation, one-time held-out coverage, and the 24-model ensemble contract.
