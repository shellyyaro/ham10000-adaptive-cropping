from __future__ import annotations

import json
import csv
import ast
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resnet_vit_pretrained_comparison.scripts.report_experiment_configs import (
    REPORT_EXPERIMENT_CONFIGS,
)


class SubmissionLayoutTests(unittest.TestCase):
    def test_all_internal_imports_resolve_to_submission_source(self) -> None:
        source_root = ROOT / "src"
        package_root = source_root / "resnet_vit_pretrained_comparison"
        missing: list[tuple[str, str]] = []
        for path in package_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = node.module or ""
                if not module.startswith("resnet_vit_pretrained_comparison"):
                    continue
                parts = module.split(".")[1:]
                target = package_root.joinpath(*parts)
                if not (
                    target.with_suffix(".py").is_file()
                    or (target / "__init__.py").is_file()
                ):
                    missing.append((str(path.relative_to(source_root)), module))
        self.assertEqual([], missing)

    def test_resnet_configuration_counts(self) -> None:
        expected = {
            "regular_baseline": 9,
            "midlevel_fusion": 8,
            "filter_size": 7,
        }
        base = ROOT / "configs" / "resnet_ensemble"
        for series, count in expected.items():
            paths = sorted((base / series).glob("*.json"))
            self.assertEqual(count, len(paths), series)
            for path in paths:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertIn("run_tag", payload["config"])

    def test_six_final_classifier_configs_are_defined_in_python(self) -> None:
        self.assertEqual(6, len(REPORT_EXPERIMENT_CONFIGS))
        self.assertFalse(list((ROOT / "configs" / "final_classifier").glob("*.txt")))

    def test_six_reported_result_directories_exist(self) -> None:
        paths = sorted(
            path
            for path in (ROOT / "results" / "reported_metrics").iterdir()
            if path.is_dir()
        )
        self.assertEqual(6, len(paths))
        for path in paths:
            self.assertTrue((path / "summary_metrics.json").is_file())

    def test_five_fold_summary_has_six_rows(self) -> None:
        path = ROOT / "results" / "reported_metrics" / "five_fold_mean_sd.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(6, len(rows))

    def test_five_fold_summary_matches_the_report_table(self) -> None:
        path = ROOT / "results" / "reported_metrics" / "five_fold_mean_sd.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = {
                row["configuration"]: row
                for row in csv.DictReader(handle)
            }

        expected = {
            "full_image_efficientnet_baseline": (
                "76.30% ± 3.08%", "72.29% ± 2.29%", "0.6031 ± 0.0294",
                "84.84% ± 0.57%", "0.7493 ± 0.0178",
            ),
            "fixed_crop_viewce030_msecons005_baseline": (
                "80.39% ± 2.86%", "76.16% ± 3.13%", "0.6782 ± 0.0392",
                "87.74% ± 1.27%", "0.7718 ± 0.0382",
            ),
            "adaptive_crop_viewce030_msecons005_baseline": (
                "78.47% ± 3.77%", "77.08% ± 2.63%", "0.6773 ± 0.0434",
                "89.70% ± 0.97%", "0.7824 ± 0.0257",
            ),
            "adaptive_margin_softwrong_ungated_kl020": (
                "84.48% ± 1.53%", "73.50% ± 2.44%", "0.7347 ± 0.0320",
                "91.80% ± 0.93%", "0.7441 ± 0.0216",
            ),
            "adaptive_ungated_kl_mild_aug": (
                "85.72% ± 1.78%", "75.54% ± 2.71%", "0.7459 ± 0.0282",
                "92.06% ± 1.21%", "0.7671 ± 0.0164",
            ),
            "adaptive_gated_kl_mild_aug": (
                "85.83% ± 1.30%", "76.69% ± 2.37%", "0.7587 ± 0.0240",
                "91.99% ± 1.09%", "0.7695 ± 0.0221",
            ),
        }
        self.assertEqual(set(expected), set(rows))
        for configuration, values in expected.items():
            with self.subTest(configuration=configuration):
                row = rows[configuration]
                self.assertEqual(values[0], row["accuracy"])
                self.assertEqual(values[1], row["balanced_accuracy"])
                self.assertEqual(values[2], row["test_macro_f1"])
                self.assertEqual(values[3], row["lesion_consistency"])
                self.assertEqual(values[4], row["validation_balanced_accuracy"])

    def test_no_machine_specific_paths_in_text_sources(self) -> None:
        suffixes = {".py", ".md", ".txt", ".json", ".toml", ".csv"}
        offenders = []
        temp_prefix = "C:" + "\\Temp"
        pycharm_root = "Pycharm" + "Projects"
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            relative = path.relative_to(ROOT)
            if relative.parts[:2] in {
                ("splits", "repeated_tvt"),
                ("splits", "mask_generation_fixed"),
                ("results", "runs"),
            }:
                continue
            text = path.read_text(encoding="utf-8")
            if temp_prefix in text or pycharm_root in text:
                offenders.append(str(relative))
        self.assertEqual([], offenders)


if __name__ == "__main__":
    unittest.main()
