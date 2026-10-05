# Code inventory and original-project comparison

This repository contains the complete executable code path for the work
described in the final report.  The number of files under `src/` is smaller
than in the development project because debug scripts and abandoned model
branches were not copied into the submission.  The shared training modules
were not removed.

## Complete reported workflow

| Stage | Entry point | Implementation and supporting code |
|---|---|---|
| Verify the package | `scripts/00_verify_submission.py` | Runs the full automated contract under `tests/` |
| Train the 24 mask models | `scripts/01_train_resnet_ensemble.py` | `train_resnet_ensemble.py`, `run_resnet_with_meta.py`, `models_cnn_resnet.py`, `training_cnn_resnet.py`, `common_training.py`, `common_imbalance.py`, `common_io.py`, and 24 JSON configurations |
| Generate fused masks | `scripts/02_generate_masks.py` | `generate_fused_masks.py`, Grad-CAM, strict-majority fusion, confidence weighting, largest-component filtering, dilation, and full-image fallback |
| Create leakage-free splits | called by stages 1 and 3 | `splits.py`, the fixed mask manifest, and the five-fold assignment manifest |
| Run the six classifier experiments | `scripts/03_run_report_experiments.py` | `report_experiment_configs.py`, `run_report_experiments.py`, `run_efficientnet_with_meta.py`, `EfficientNet.py`, adaptive/fixed crop datasets and policies |
| Recreate evaluation outputs | `scripts/04_evaluate_report_checkpoints.py` | `evaluate_report_checkpoints.py` and the same model/data/loss utilities used during training |

## Shared modules retained from the development project

The following important files are present in `src/` and were compared with the
development project. Nine shared modules are byte-for-byte identical to their
original counterparts:

- `training/common_imbalance.py`
- `training/common_training.py`
- `training/training_cnn_resnet.py`
- `data/adaptive_crop_policy.py`
- `data/datasets_masked_background.py`
- `data/mask_fusion_loader.py`
- `data/meta_features.py`
- `models/EfficientNet.py`
- `models/models_cnn_resnet.py`

The submission versions of the following files are intentionally adapted:

- `scripts/common_io.py` and `scripts/run_resnet_with_meta.py`: the only
  differences from their development counterparts remove the legacy
  `config.txt` writer/call and keep JSON as the executable configuration
  record. No model, data, loss, optimization, or metric behavior was changed.
- `scripts/run_efficientnet_with_meta.py`: retains the full training and loss
  implementation, including averaged-probability CE, view CE, margin loss,
  soft-wrong loss, ungated KL, and detached true-margin-gated KL.
- `scripts/splits.py`: adds fail-fast lesion-overlap and fold-coverage checks.
- `scripts/mask_generation/generate_fused_masks.py`: replaces machine-specific
  paths with repository paths, reads checkpoint configuration from JSON, uses
  the exact versioned preliminary split, and verifies the exact 9 + 8 + 7 run
  tags before generating masks.
- `scripts/run_report_experiments.py`: replaces old TXT/result-folder recovery
  with six authoritative Python configurations and adds exact five-fold
  manifest verification.
- `scripts/evaluate_report_checkpoints.py`: repository-local evaluation entry
  point for the six report configurations.
- `data/data_pipeline.py`: preserves the original reusable data-preparation API
  while explicitly directing report use to training-only, leakage-free metadata
  fitting.  The fold runner performs the same operations directly because each
  fold has different training indices.

## Class-imbalance and CE implementation

`training/common_imbalance.py` calculates inverse-frequency weights from the
training partition and passes them to `torch.nn.CrossEntropyLoss`.  The six
classifier configurations set `class_weighting="inverse_freq"` and
`label_smoothing=0.05`.  The fixed-crop and adaptive-baseline objectives apply
that loss both to the averaged probability distribution and, with weight 0.30,
to every view.  The 24 preliminary ResNet configurations also use
inverse-frequency cross-entropy.

These facts are checked numerically in `tests/test_core_training_contract.py`;
they are not inferred from previous TXT result records.

## Development files intentionally excluded

The original project also contains exploratory branches that are not part of
the submitted method or any result claimed in the final report.  They include
debug visualizers, alternative mask-gating networks, attention-model trials,
DenseNet/ConvNeXt comparison runners, teacher-mask experiments, raw-mask
prototypes, hair-removal trials, and one-off probability export scripts.
Including those files would make the submitted workflow ambiguous and could
suggest that unreported methods were required to reproduce the results.

Earlier orchestration scripts that depended on machine-specific result folders
or TXT configuration dumps were also excluded.  Their report-relevant behavior
is preserved in the repository-local numbered entry points and code-defined
configurations listed above.

The executable training paths write configuration and status artifacts as JSON.
They do not read configuration values from TXT files.  The only `.txt` file in
the repository is `requirements.txt`, which is the standard Python dependency
manifest and is not an experiment configuration or result source.

## Submission gate

Run this before committing:

```powershell
python scripts/00_verify_submission.py
```

The gate checks the required core modules, all 24 ResNet configurations, all six
classifier configurations, inverse-frequency CE and label smoothing, the fixed
mask split, five-fold lesion isolation and one-time held-out coverage, the exact
mask-ensemble composition, and the gated-KL implementation.
