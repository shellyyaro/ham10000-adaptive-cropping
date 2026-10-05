from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resnet_vit_pretrained_comparison.scripts.report_experiment_configs import (
    REPORT_EXPERIMENT_CONFIGS,
)
from resnet_vit_pretrained_comparison.training.common_imbalance import (
    build_loss_and_logits_transform,
    weights_inverse_freq,
)


class CoreTrainingContractTests(unittest.TestCase):
    def test_core_modules_required_by_the_experiments_are_present(self) -> None:
        required = [
            "scripts/common_io.py",
            "training/common_imbalance.py",
            "training/common_training.py",
            "training/training_cnn_resnet.py",
            "data/data_loader.py",
            "data/data_pipeline.py",
            "data/meta_features.py",
            "data/adaptive_crop_policy.py",
            "data/datasets_masked_background.py",
            "data/mask_fusion_loader.py",
            "models/models_cnn_resnet.py",
            "models/EfficientNet.py",
            "scripts/splits.py",
            "scripts/run_resnet_with_meta.py",
            "scripts/run_efficientnet_with_meta.py",
        ]
        package = ROOT / "src" / "resnet_vit_pretrained_comparison"
        missing = [relative for relative in required if not (package / relative).is_file()]
        self.assertEqual([], missing)

    def test_report_ce_uses_inverse_frequency_weights_and_label_smoothing(self) -> None:
        counts = np.array([100.0, 25.0, 4.0], dtype=np.float64)
        expected = weights_inverse_freq(counts)
        train_transform, eval_transform, loss_fn = build_loss_and_logits_transform(
            loss_name="ce",
            class_weighting="inverse_freq",
            label_smoothing=0.05,
            focal_gamma=2.0,
            focal_alpha_mode="none",
            logit_adjustment="none",
            tau=1.0,
            class_counts_np=counts,
            device=torch.device("cpu"),
        )

        self.assertIsInstance(loss_fn, nn.CrossEntropyLoss)
        np.testing.assert_allclose(loss_fn.weight.numpy(), expected, rtol=1e-6)
        self.assertAlmostEqual(0.05, float(loss_fn.label_smoothing))

        logits = torch.tensor([[1.0, 2.0, 3.0]])
        torch.testing.assert_close(train_transform(logits), logits)
        torch.testing.assert_close(eval_transform(logits), logits)

    def test_all_six_configs_keep_the_reported_imbalance_settings(self) -> None:
        for name, config in REPORT_EXPERIMENT_CONFIGS.items():
            with self.subTest(config=name):
                self.assertEqual("ce", config["loss_name"])
                self.assertEqual("inverse_freq", config["class_weighting"])
                self.assertEqual(0.05, config["label_smoothing"])
                self.assertEqual("none", config["sampler"])

    def test_all_24_resnet_configs_use_inverse_frequency_ce(self) -> None:
        import json

        paths = sorted((ROOT / "configs" / "resnet_ensemble").rglob("*.json"))
        self.assertEqual(24, len(paths))
        for path in paths:
            config = json.loads(path.read_text(encoding="utf-8"))["config"]
            with self.subTest(config=config["run_tag"]):
                self.assertEqual("ce", config["loss_name"])
                self.assertEqual("inverse_freq", config["class_weighting"])


if __name__ == "__main__":
    unittest.main()
