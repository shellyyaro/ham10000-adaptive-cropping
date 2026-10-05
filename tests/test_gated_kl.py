from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resnet_vit_pretrained_comparison.scripts.run_efficientnet_with_meta import (
    _compute_true_margin_consistency_gate,
)


class GatedKLTests(unittest.TestCase):
    def test_true_margin_gate_uses_report_temperature(self) -> None:
        labels = torch.tensor([0, 0])
        averaged_log_probabilities = torch.log(
            torch.tensor(
                [
                    [0.80, 0.20],  # true class wins: gate should be near one
                    [0.20, 0.80],  # wrong class wins: gate should be near zero
                ]
            )
        )
        gate = _compute_true_margin_consistency_gate(
            averaged_log_probabilities,
            labels,
            temperature=0.05,
            margin_threshold=0.0,
        )
        self.assertGreater(float(gate[0]), 0.99)
        self.assertLess(float(gate[1]), 0.01)

    def test_training_path_detaches_and_applies_gate_to_per_sample_kl(self) -> None:
        source = (
            ROOT
            / "src"
            / "resnet_vit_pretrained_comparison"
            / "scripts"
            / "run_efficientnet_with_meta.py"
        ).read_text(encoding="utf-8")
        self.assertIn("logits_for_loss.detach()", source)
        self.assertIn("consistency_loss_per_sample * consistency_gate", source)
        self.assertIn("consistency_weight * consistency_loss", source)


if __name__ == "__main__":
    unittest.main()
