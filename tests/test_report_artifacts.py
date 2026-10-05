from __future__ import annotations

import csv
import hashlib
import unittest
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "results" / "report_artifacts"
CANONICAL = ARTIFACTS / "canonical"
GENERATED = ARTIFACTS / "generated"


class ReportArtifactTests(unittest.TestCase):
    @staticmethod
    def report_tables() -> list[list[list[str]]]:
        namespace = {
            "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        }
        with zipfile.ZipFile(ROOT / "reports" / "final_report.docx") as archive:
            document = ET.fromstring(archive.read("word/document.xml"))
        result: list[list[list[str]]] = []
        for table in document.findall(".//w:tbl", namespace):
            rows: list[list[str]] = []
            for row in table.findall("./w:tr", namespace):
                rows.append(
                    [
                        "".join(
                            node.text or ""
                            for node in cell.findall(".//w:t", namespace)
                        )
                        for cell in row.findall("./w:tc", namespace)
                    ]
                )
            result.append(rows)
        return result

    def test_every_embedded_report_figure_is_versioned_byte_for_byte(self) -> None:
        mapping = {
            "image1.png": "report_cover_logo.png",
            "image2.png": "figure_1_mask_area_distribution.png",
            "image3.png": "figure_2_crop_examples.png",
            "image4.png": "figure_3_confusion_matrices.png",
        }
        with zipfile.ZipFile(ROOT / "reports" / "final_report.docx") as archive:
            report_media = sorted(
                Path(name).name
                for name in archive.namelist()
                if name.startswith("word/media/")
            )
            self.assertEqual(sorted(mapping), report_media)
            for report_name, checked_in_name in mapping.items():
                with self.subTest(figure=report_name):
                    report_bytes = archive.read(f"word/media/{report_name}")
                    checked_in_bytes = (CANONICAL / checked_in_name).read_bytes()
                    self.assertEqual(
                        hashlib.sha256(report_bytes).hexdigest(),
                        hashlib.sha256(checked_in_bytes).hexdigest(),
                    )

    def test_table_1_artifact_matches_the_six_report_rows(self) -> None:
        with (GENERATED / "table_1_five_fold_performance.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(6, len(rows))
        self.assertEqual(
            [
                "Full image",
                "Fixed crop",
                "Adaptive baseline",
                "Adaptive + alternative objective (ungated KL, common aug.)",
                "Adaptive + alternative objective (ungated KL, mild aug.)",
                "Adaptive + alternative objective (gated KL, mild aug.)",
            ],
            [row["display_name"] for row in rows],
        )
        self.assertEqual("60.31% ± 2.94%", rows[0]["macro_f1"])
        self.assertEqual("75.87% ± 2.40%", rows[-1]["macro_f1"])
        self.assertEqual(
            "76.95% ± 2.21%", rows[-1]["validation_balanced_accuracy"]
        )

        report_rows = self.report_tables()[0][1:]
        for generated, report in zip(rows, report_rows):
            expected = [
                generated["accuracy"],
                generated["balanced_accuracy"],
                generated["macro_f1"],
                generated["lesion_consistency"],
                generated["validation_balanced_accuracy"],
            ]
            self.assertEqual(
                [value.replace("%", "") for value in expected],
                report[1:],
            )

    def test_table_2_is_derived_from_confusion_matrix_diagonals(self) -> None:
        with (GENERATED / "table_2_pooled_class_recall.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(6, len(rows))
        self.assertEqual("61.16", rows[0]["AKIEC"])
        self.assertEqual("92.25", rows[0]["VASC"])
        self.assertEqual("66.97", rows[-1]["AKIEC"])
        self.assertEqual("90.14", rows[-1]["VASC"])
        self.assertEqual(
            [[row[label] for label in ("AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC")] for row in rows],
            [report_row[1:] for report_row in self.report_tables()[1][1:]],
        )

    def test_literature_table_and_all_rebuild_entry_points_are_versioned(self) -> None:
        with (ARTIFACTS / "table_3_literature_comparison.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(7, len(rows))
        self.assertEqual("Present study", rows[-1]["study"])
        self.assertEqual("85.83 ± 1.30", rows[-1]["accuracy_percent"])
        report_table = self.report_tables()[2]
        self.assertEqual(8, len(report_table))
        self.assertEqual("Present study", report_table[-1][0])
        self.assertEqual("85.83 ± 1.30", report_table[-1][3])
        self.assertEqual("76.69 ± 2.37", report_table[-1][4])
        self.assertEqual("75.87 ± 2.40", report_table[-1][5])
        self.assertEqual("97.13 ± 0.40", report_table[-1][6])

        source = (ROOT / "scripts" / "05_rebuild_report_artifacts.py").read_text(
            encoding="utf-8"
        )
        for function_name in (
            "plot_mask_area_distribution",
            "plot_crop_examples",
            "plot_confusion_matrix_grid",
            "build_table_1",
            "build_table_2",
            "build_table_3",
            "verify_report_media",
        ):
            self.assertIn(f"def {function_name}", source)
        self.assertNotIn("C:\\Users\\", source)


if __name__ == "__main__":
    unittest.main()
