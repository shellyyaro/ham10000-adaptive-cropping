# Adaptive lesion occupancy cropping for HAM10000

This repository is the clean submission package for the final project. It contains the full reproducible path described in the report: training the 24-model ResNet-18 ensemble, generating fused Grad-CAM lesion masks, and evaluating the six EfficientNet-B1 configurations with five lesion-disjoint folds.

## Reproduction pipeline

The stages must be run in order because the classification experiments use the masks produced by the ResNet ensemble.

### 1. Train the 24 ResNet-18 configurations

```powershell
python scripts/01_train_resnet_ensemble.py --dry-run
python scripts/01_train_resnet_ensemble.py
```

The exact configurations are grouped into three reported preliminary ablation series:

- `regular_baseline`: 9 configurations
- `midlevel_fusion`: 8 configurations
- `filter_size`: 7 configurations

All models use the fixed lesion-level split with seed 0. Their checkpoints are written under `checkpoints/resnet_ensemble/`.

### 2. Generate fused lesion masks

```powershell
python scripts/02_generate_masks.py --save-all-splits
```

For every image, each of the 24 models predicts a class and produces a Grad-CAM map for that predicted class. A strict majority requires at least 13 votes. When a majority exists, maps are weighted by the model probability for the majority class and agreeing models receive an additional boost. Without a strict majority, the maps are averaged. The largest high-activation component is retained and dilated; low-variation maps fall back to the full image.

The resulting fixed preprocessing masks are stored under `data/generated_masks/`.

### 3. Run the six report experiments

```powershell
python scripts/03_run_report_experiments.py --dry-run
python scripts/03_run_report_experiments.py
```

The runner creates five mutually exclusive lesion-level test folds and evaluates:

1. full image;
2. fixed bounding-box crops;
3. adaptive occupancy crops with the baseline objective;
4. adaptive crops with the alternative objective, common augmentation, and ungated KL;
5. adaptive crops with the alternative objective, mild augmentation, and ungated KL;
6. adaptive crops with the alternative objective, mild augmentation, and true-margin-gated KL.

Adaptive views target lesion occupancies of 10%, 20%, 30%, 40%, 50%, and 60%. Outputs are written to `results/runs/report_experiments/`.

### 4. Recreate diagnostic figures from saved checkpoints

```powershell
python scripts/04_evaluate_report_checkpoints.py --dry-run
python scripts/04_evaluate_report_checkpoints.py
```

This optional step reproduces pooled out-of-fold confusion matrices, ROC curves, precision-recall curves, and summary metrics without retraining.

### 5. Rebuild the exact report figures and tables

The three figures embedded in `reports/final_report.docx` are versioned byte-for-byte under `results/report_artifacts/canonical/`. Tables 1-3 and Figure 3 can be rebuilt using only the compact files in Git:

```powershell
python scripts/05_rebuild_report_artifacts.py --available-only --verify-report-media
```

Figures 1-2 depend on the HAM10000 images and the fused masks produced in stage 2:

```powershell
python scripts/05_rebuild_report_artifacts.py `
  --data-root C:\path\to\HAM10000 `
  --mask-root data\generated_masks `
  --verify-report-media
```

The script rebuilds the mask-area distribution, the two representative crop examples, the six-panel confusion-matrix figure, the six-row performance table, the pooled class-recall table, and the curated literature-comparison table.

## Setup

Python 3.10 or 3.11 and a CUDA-capable PyTorch installation are recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For an editable install of the `src/` package, use `pip install -e .` instead.

Download HAM10000 separately and follow `data/README.md`, or set `HAM10000_ROOT`.

Optional environment variables:

- `HAM10000_ROOT`: dataset directory;
- `HAM10000_MASK_ROOT`: generated mask directory;
- `HAM10000_RESULTS_ROOT`: output directory for the six final experiments.

## Repository contents

- `scripts/00_verify_submission.py`: automated completeness and leakage-prevention gate;
- `configs/resnet_ensemble/`: the exact 24 ResNet configurations;
- `src/resnet_vit_pretrained_comparison/scripts/report_experiment_configs.py`: the exact six classifier configurations, defined directly in Python;
- `splits/`: compact, path-independent assignment manifests for the preliminary split and final five folds;
- `src/`: model, data, training, mask-generation, and evaluation code;
- `results/reported_metrics/`: compact pooled out-of-fold metrics and diagnostic figures for all six reported configurations;
- `results/reported_metrics/five_fold_mean_sd.csv`: the exact six-row mean ± sample-SD comparison used in the report;
- `results/report_artifacts/canonical/`: the exact three figure files embedded in the report;
- `results/report_artifacts/generated/`: rebuilt Tables 1-3, Figure 3, and a SHA-256 report-media manifest;
- `scripts/05_rebuild_report_artifacts.py`: the code that rebuilds every report figure and table;
- `docs/EXPERIMENT_MAP.md`: a direct mapping from report stages to code, configurations, and outputs;
- `docs/REPORT_CODE_AUDIT.md`: a claim-by-claim audit, exact split counts, and the leakage-proof verification contract;
- `docs/CODE_INVENTORY.md`: file-level comparison with the original development project and the reason excluded exploratory branches are not required;
- `reports/final_report.docx`: the submitted report;
- `data/` and `checkpoints/`: documented locations for non-versioned large artifacts.

## Reproducibility notes

- Splits are lesion-disjoint, so photographs of the same lesion cannot cross partitions.
- The mask-generation split uses seed 0 and is separate from the final five-fold evaluation, which uses seed 42.
- The masks are generated once and held fixed across all final cropping experiments, matching the scope stated in the report.
- Dataset images, generated NPZ mask packs, and model checkpoints are excluded from Git because of their size and licensing/distribution constraints.

Before submission, run the automated contract:

```powershell
python scripts/00_verify_submission.py
```
