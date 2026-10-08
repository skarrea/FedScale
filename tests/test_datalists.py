import csv
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from shared_modules.datalists import create_datalists, load_case_info, write_datalists


class DatalistTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv_path = self.root / "cases.csv"
        self.splits_path = self.root / "splits.json"
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["patient_id", "study_id", "case_csPCa", "psa", "center"])
            writer.writerows([
                ["10000", "1000000", "NO", "7.7", "PCNN"],
                ["10000", "1000002", "YES", "", "PCNN"],
                ["10001", "1000001", "YES", "8.7", "RUMC"],
                ["00002", "0000003", "", "", "RUMC"],
            ])
        self.set_splits([{"train": [10000], "val": ["10001_1000001"]}])

    def set_splits(self, splits):
        self.splits_path.write_text(json.dumps(splits), encoding="utf-8")

    def create(self, **kwargs):
        return create_datalists(self.splits_path, self.csv_path, "imagesTR", "labelsTr", **kwargs)

    def test_patient_expansion_channel_order_and_fold_output(self):
        self.set_splits([
            {"train": [10000], "val": ["10001_1000001"]},
            {"train": ["10001_1000001"], "val": ["10000"]},
        ])
        datalists = self.create()
        self.assertEqual(datalists[0]["training"], [
            {"image": [f"imagesTR/10000_1000000_{s:04d}.nii.gz" for s in range(3)],
             "pca": "labelsTr/10000_1000000.nii.gz", "case_pca": 0},
            {"image": [f"imagesTR/10000_1000002_{s:04d}.nii.gz" for s in range(3)],
             "pca": "labelsTr/10000_1000002.nii.gz", "case_pca": 1},
        ])
        self.assertEqual(datalists[1]["validation"], datalists[0]["training"])
        self.assertEqual(datalists[0]["test"], [])
        paths = write_datalists(datalists, self.root / "output")
        self.assertEqual([p.name for p in paths], ["fold_0.json", "fold_1.json"])
        self.assertEqual(json.loads(paths[0].read_text()), datalists[0])

    def test_optional_masks_and_custom_sequence_order(self):
        sample = self.create(sequence_ids=[2, 0, 1], prostate_pred_dir="preds",
                             prostate_pred_suffix="", zones_dir="zones")[0]["training"][0]
        self.assertEqual(sample["image"][0], "imagesTR/10000_1000000_0002.nii.gz")
        self.assertEqual(sample["prostate_pred"], "preds/10000_1000000.nii.gz")
        self.assertEqual(sample["zones"], "zones/10000_1000000.nii.gz")

    def test_preserves_leading_zeros_and_missing_cancer_status(self):
        self.set_splits([{"train": ["00002"], "val": []}])
        sample = self.create()[0]["training"][0]
        self.assertEqual(sample["pca"], "labelsTr/00002_0000003.nii.gz")
        self.assertNotIn("case_pca", sample)

    def test_rejects_unknown_duplicate_and_overlapping_patients(self):
        scenarios = [
            ({"train": ["missing"], "val": []}, "Unknown ID"),
            ({"train": ["10000", "10000_1000000"], "val": []}, "Duplicate case"),
            ({"train": ["10000_1000000"], "val": ["10000_1000002"]}, "both train and val"),
            ({"train": ["10000"], "val": ["10000"]}, "both train and val"),
        ]
        for fold, message in scenarios:
            with self.subTest(fold=fold):
                self.set_splits([fold])
                with self.assertRaisesRegex(ValueError, message):
                    self.create()

    def test_file_checks_include_negative_case_and_optional_masks(self):
        self.set_splits([{"train": ["10000_1000000"], "val": []}])
        sample = self.create(prostate_pred_dir="preds", zones_dir="zones")[0]["training"][0]
        for name in sample["image"] + [sample["prostate_pred"], sample["zones"]]:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        with self.assertRaisesRegex(FileNotFoundError, "labelsTr/10000_1000000"):
            self.create(check_files=True, data_dir=self.root,
                        prostate_pred_dir="preds", zones_dir="zones")
        label = self.root / sample["pca"]
        label.parent.mkdir(parents=True)
        label.touch()
        self.create(check_files=True, data_dir=self.root,
                    prostate_pred_dir="preds", zones_dir="zones")
        (self.root / sample["zones"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "zones/10000_1000000"):
            self.create(check_files=True, data_dir=self.root,
                        prostate_pred_dir="preds", zones_dir="zones")

    def test_invalid_inputs_fail_with_context(self):
        for splits in ({"train": [], "val": []}, [], [{"train": []}],
                       [{"train": "10000", "val": []}]):
            with self.subTest(splits=splits):
                self.set_splits(splits)
                with self.assertRaises(ValueError):
                    self.create()
        self.set_splits([{"train": [], "val": []}])
        with self.assertRaisesRegex(ValueError, "three distinct"):
            self.create(sequence_ids=[0, 0, 2])
        self.csv_path.write_text("patient_id,study_id,case_csPCa\n10000,1000000,UNKNOWN\n")
        with self.assertRaisesRegex(ValueError, "Invalid case_csPCa for 10000_1000000"):
            load_case_info(self.csv_path)
        self.csv_path.write_text("patient_id,study_id\n10000,1000000\n10000,1000000\n")
        with self.assertRaisesRegex(ValueError, "Duplicate case"):
            load_case_info(self.csv_path)

    def test_cli_generates_fold_json(self):
        output_dir = self.root / "cli_output"
        result = subprocess.run([
            sys.executable, "-m", "shared_modules.datalists",
            "--splits", str(self.splits_path), "--cases-csv", str(self.csv_path),
            "--images-dir", "imagesTR", "--labels-dir", "labelsTr",
            "--output-dir", str(output_dir),
        ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("2 training, 1 validation", result.stdout)
        self.assertEqual(json.loads((output_dir / "fold_0.json").read_text()), self.create()[0])


if __name__ == "__main__":
    unittest.main()
