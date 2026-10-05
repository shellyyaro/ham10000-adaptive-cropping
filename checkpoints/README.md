# Checkpoints

Stage 1 writes the 24 ResNet-18 checkpoints below:

```text
checkpoints/resnet_ensemble/
├── regular_baseline/   # 9 configurations
├── midlevel_fusion/    # 8 configurations
└── filter_size/        # 7 configurations
```

Model weights are ignored by Git. Each run directory still records its configuration and metrics, while the exact input configurations are versioned under `configs/resnet_ensemble/`.
