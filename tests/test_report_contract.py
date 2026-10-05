from __future__ import annotations

import csv
import json
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resnet_vit_pretrained_comparison.scripts.report_experiment_configs import (
    ADAPTIVE_POLICIES,
    FIXED_EVAL_POLICIES,
    FIXED_TRAIN_POLICIES,
    REPORT_EXPERIMENT_CONFIGS,
)
from resnet_vit_pretrained_comparison.scripts.mask_generation.generate_fused_masks import (
    expected_run_tags_by_source,
    validate_loaded_model_sources,
)


def read_rows(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class SplitContractTests(unittest.TestCase):
    def test_mask_generation_split_matches_report_and_is_lesion_disjoint(self) -> None:
        rows = read_rows("splits/mask_generation_fixed_split.csv")
        expected_images = {"train": 6021, "val": 2003, "test": 1991}
        self.assertEqual(expected_images, Counter(row["split"] for row in rows))

        lesion_partitions: dict[str, set[str]] = defaultdict(set)
        image_partitions: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            lesion_partitions[row["lesion_id"]].add(row["split"])
            image_partitions[row["image_id"]].add(row["split"])

        self.assertTrue(all(len(parts) == 1 for parts in lesion_partitions.values()))
        self.assertTrue(all(len(parts) == 1 for parts in image_partitions.values()))
        self.assertEqual(7470, len(lesion_partitions))
        self.assertEqual(10015, len(image_partitions))

    def test_final_folds_have_no_lesion_or_image_leakage(self) -> None:
        rows = read_rows("splits/report_five_fold_assignments.csv")
        self.assertEqual({"0", "1", "2", "3", "4"}, {row["split_idx"] for row in rows})

        by_fold_lesion: dict[tuple[str, str], set[str]] = defaultdict(set)
        by_fold_image: dict[tuple[str, str], set[str]] = defaultdict(set)
        test_image_counts: Counter[str] = Counter()
        test_lesion_folds: dict[str, set[str]] = defaultdict(set)
        all_images: set[str] = set()
        all_lesions: set[str] = set()

        for row in rows:
            fold = row["split_idx"]
            partition = row["split"]
            image_id = row["image_id"]
            lesion_id = row["lesion_id"]
            by_fold_lesion[(fold, partition)].add(lesion_id)
            by_fold_image[(fold, partition)].add(image_id)
            all_images.add(image_id)
            all_lesions.add(lesion_id)
            if partition == "test":
                test_image_counts[image_id] += 1
                test_lesion_folds[lesion_id].add(fold)

        for fold in map(str, range(5)):
            lesion_sets = [by_fold_lesion[(fold, p)] for p in ("train", "val", "test")]
            image_sets = [by_fold_image[(fold, p)] for p in ("train", "val", "test")]
            self.assertFalse(lesion_sets[0] & lesion_sets[1], f"fold {fold}: train/val lesion leak")
            self.assertFalse(lesion_sets[0] & lesion_sets[2], f"fold {fold}: train/test lesion leak")
            self.assertFalse(lesion_sets[1] & lesion_sets[2], f"fold {fold}: val/test lesion leak")
            self.assertFalse(image_sets[0] & image_sets[1], f"fold {fold}: train/val image leak")
            self.assertFalse(image_sets[0] & image_sets[2], f"fold {fold}: train/test image leak")
            self.assertFalse(image_sets[1] & image_sets[2], f"fold {fold}: val/test image leak")
            self.assertEqual(all_images, set().union(*image_sets))
            self.assertEqual(all_lesions, set().union(*lesion_sets))

        self.assertEqual(10015, len(all_images))
        self.assertEqual(7470, len(all_lesions))
        self.assertTrue(all(count == 1 for count in test_image_counts.values()))
        self.assertEqual(all_images, set(test_image_counts))
        self.assertTrue(all(len(folds) == 1 for folds in test_lesion_folds.values()))
        self.assertEqual(all_lesions, set(test_lesion_folds))

    def test_every_lesion_has_one_partition_per_fold(self) -> None:
        rows = read_rows("splits/report_five_fold_assignments.csv")
        assignments: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in rows:
            assignments[(row["split_idx"], row["lesion_id"])].add(row["split"])
        self.assertEqual(5 * 7470, len(assignments))
        self.assertTrue(all(len(parts) == 1 for parts in assignments.values()))


class ReportConfigurationContractTests(unittest.TestCase):
    def test_all_six_report_experiments_are_complete_python_configs(self) -> None:
        configs = REPORT_EXPERIMENT_CONFIGS
        self.assertEqual(
            {
                "full_image_efficientnet_baseline",
                "fixed_crop_viewce030_msecons005_baseline",
                "adaptive_crop_viewce030_msecons005_baseline",
                "adaptive_margin_softwrong_ungated_kl020",
                "adaptive_ungated_kl_mild_aug",
                "adaptive_gated_kl_mild_aug",
            },
            set(configs),
        )

        full = configs["full_image_efficientnet_baseline"]
        self.assertEqual("original", full["input_source"])
        self.assertIs(full["use_crop_as_input"], False)
        self.assertIs(full["multi_view_crop"], False)

        fixed = configs["fixed_crop_viewce030_msecons005_baseline"]
        self.assertEqual(FIXED_TRAIN_POLICIES, fixed["multi_view_crop_policy_choices"])
        self.assertEqual(FIXED_EVAL_POLICIES, fixed["eval_crop_policy_choices"])
        self.assertEqual(0.30, fixed["view_ce_weight"])
        self.assertEqual(0.05, fixed["consistency_weight"])
        self.assertEqual("mse_probs", fixed["consistency_loss_type"])

        adaptive = configs["adaptive_crop_viewce030_msecons005_baseline"]
        self.assertEqual(ADAPTIVE_POLICIES, adaptive["multi_view_crop_policy_choices"])
        self.assertEqual(ADAPTIVE_POLICIES, adaptive["eval_crop_policy_choices"])
        self.assertEqual(0.60, adaptive["crop_threshold"])

    def test_alternative_and_gated_kl_objectives_are_code_defined(self) -> None:
        configs = REPORT_EXPERIMENT_CONFIGS
        alternatives = [
            configs["adaptive_margin_softwrong_ungated_kl020"],
            configs["adaptive_ungated_kl_mild_aug"],
            configs["adaptive_gated_kl_mild_aug"],
        ]
        for config in alternatives:
            self.assertEqual("avg_margin_hinge", config["main_loss_type"])
            self.assertEqual(0.10, config["main_margin_target"])
            self.assertEqual(0.30, config["wrong_crop_count_weight"])
            self.assertEqual(0.15, config["wrong_crop_count_temperature"])
            self.assertEqual("kl_probs", config["consistency_loss_type"])
            self.assertEqual(0.20, config["consistency_weight"])

        common, mild, gated = alternatives
        self.assertEqual(20.0, common["rotation_deg"])
        self.assertIs(common["hflip"], True)
        self.assertIs(common["vflip"], True)
        self.assertEqual("none", common["consistency_gate_mode"])

        for config in (mild, gated):
            self.assertEqual(0.0, config["rotation_deg"])
            self.assertIs(config["hflip"], False)
            self.assertIs(config["vflip"], False)
            self.assertEqual(0.10, config["color_jitter_brightness"])
            self.assertIs(config["per_view_color_jitter_brightness"], True)
            self.assertIs(config["per_view_color_jitter_contrast"], False)
            self.assertIs(config["per_view_color_jitter_saturation"], False)

        self.assertEqual("none", mild["consistency_gate_mode"])
        self.assertEqual("true_margin", gated["consistency_gate_mode"])
        self.assertEqual(0.05, gated["consistency_gate_temperature"])
        self.assertEqual(0.0, gated["consistency_gate_margin_threshold"])

    def test_final_runner_has_no_txt_configuration_dependency(self) -> None:
        runner = (
            ROOT
            / "src"
            / "resnet_vit_pretrained_comparison"
            / "scripts"
            / "run_report_experiments.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("parse_config_txt", runner)
        self.assertNotIn("source_config.txt", runner)
        self.assertIn("get_report_experiment_config", runner)

        execution_files = [
            ROOT
            / "src"
            / "resnet_vit_pretrained_comparison"
            / "scripts"
            / name
            for name in (
                "run_report_experiments.py",
                "run_efficientnet_with_meta.py",
                "run_resnet_with_meta.py",
                "run_vit_with_meta.py",
                "evaluate_report_checkpoints.py",
            )
        ]
        execution_source = "\n".join(
            path.read_text(encoding="utf-8") for path in execution_files
        )
        self.assertNotIn("write_config_txt", execution_source)
        self.assertNotIn('"config.txt"', execution_source)
        self.assertNotIn("selected_configuration.txt", execution_source)
        self.assertNotIn("ERROR.txt", execution_source)
        self.assertIn('"config.json"', execution_source)
        self.assertIn("effective_config.json", execution_source)
        self.assertIn(
            "Dry run completed from Python-defined configurations",
            runner,
        )
        self.assertLess(
            runner.index("if args.dry_run:"),
            runner.index("meta, _ = load_ham10000_metadata()"),
        )

    def test_checkpoint_evaluator_dry_run_precedes_checkpoint_validation(self) -> None:
        evaluator = (
            ROOT
            / "src"
            / "resnet_vit_pretrained_comparison"
            / "scripts"
            / "evaluate_report_checkpoints.py"
        ).read_text(encoding="utf-8")
        main_source = evaluator[evaluator.index("def main() -> None:") :]
        self.assertLess(
            main_source.index("if args.dry_run:"),
            main_source.index("validate_plan(configurations, args.folds)"),
        )

    def test_mask_ensemble_contains_the_reported_24_resnet18_variants(self) -> None:
        expected_counts = {"regular_baseline": 9, "midlevel_fusion": 8, "filter_size": 7}
        for series, expected_count in expected_counts.items():
            paths = sorted((ROOT / "configs" / "resnet_ensemble" / series).glob("*.json"))
            self.assertEqual(expected_count, len(paths), series)
            for path in paths:
                payload = json.loads(path.read_text(encoding="utf-8"))
                config = payload["config"]
                self.assertEqual("resnet18", config["arch"])
                self.assertIs(config["pretrained"], True)
                self.assertIs(config.get("no_meta", False), False)
                self.assertIs(config.get("use_fused_masks", False), False)
                self.assertEqual("none", config.get("mask_mode", "none"))

    def test_report_code_contains_exact_ensemble_gate(self) -> None:
        path = (
            ROOT
            / "src"
            / "resnet_vit_pretrained_comparison"
            / "scripts"
            / "mask_generation"
            / "generate_fused_masks.py"
        )
        source = path.read_text(encoding="utf-8")
        self.assertIn("EXPECTED_ENSEMBLE_SIZE", source)
        self.assertIn("exactly 24 checkpoints", source)
        self.assertIn('"resnet_regular_baseline_ablation_rerun_full_config_v1": 9', source)
        self.assertIn('"resnet_midlevel_fusion_rerun_v1": 8', source)
        self.assertIn('"resnet_filter_size_7_configs_rerun_v1": 7', source)
        self.assertIn("expected_run_tags_by_source", source)
        self.assertIn("exact 24 versioned ResNet", source)

    def test_mask_ensemble_gate_checks_all_exact_run_tags(self) -> None:
        expected = expected_run_tags_by_source()
        self.assertEqual(
            {
                "resnet_regular_baseline_ablation_rerun_full_config_v1": 9,
                "resnet_midlevel_fusion_rerun_v1": 8,
                "resnet_filter_size_7_configs_rerun_v1": 7,
            },
            {label: len(tags) for label, tags in expected.items()},
        )

        synthetic_models = [
            {
                "source_label": label,
                "model_cfg": {"run_tag": run_tag},
            }
            for label, run_tags in expected.items()
            for run_tag in sorted(run_tags)
        ]
        validate_loaded_model_sources(synthetic_models)

        synthetic_models[0]["model_cfg"]["run_tag"] = "wrong_run_tag"
        with self.assertRaisesRegex(RuntimeError, "exact 24 versioned ResNet"):
            validate_loaded_model_sources(synthetic_models)

    def test_mask_generation_reuses_the_versioned_preliminary_split(self) -> None:
        path = (
            ROOT
            / "src"
            / "resnet_vit_pretrained_comparison"
            / "scripts"
            / "mask_generation"
            / "generate_fused_masks.py"
        )
        source = path.read_text(encoding="utf-8")
        self.assertIn('REPO_ROOT / "splits" / "mask_generation_fixed"', source)
        self.assertIn('prefix="ham10000_mask_generation"', source)
        self.assertIn("validate_fixed_split_manifest(splits)", source)
        self.assertNotIn("parents[3] / \"splits\"", source)


if __name__ == "__main__":
    unittest.main()
