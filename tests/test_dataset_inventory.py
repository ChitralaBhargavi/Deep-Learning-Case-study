import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.inspect_ddr_dataset import inventory_dataset, main, render_markdown


class DatasetInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "fixture"
        self.root.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_grayscale_mask_requires_pixels_and_matching_dimensions(self):
        image_dir = self.root / "images" / "train"
        mask_dir = self.root / "masks" / "train"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        Image.new("RGB", (48, 32), (90, 80, 70)).save(image_dir / "sample_01.jpg")
        mask = Image.new("L", (48, 32), 0)
        mask.putpixel((10, 12), 255)
        mask.save(mask_dir / "sample_01_MA_mask.png")

        report = inventory_dataset(self.root)

        self.assertEqual(report["summary"]["image_file_count"], 2)
        self.assertEqual(report["summary"]["images_with_verified_annotations"], 1)
        annotation = report["annotations"][0]
        self.assertEqual(annotation["status"], "verified_pair")
        self.assertTrue(annotation["pixel_evidence"]["non_empty_pixels"])
        self.assertIn("microaneurysms", report["candidate_lesion_classes"])
        self.assertEqual(report["splits"]["train"]["image_files"], 2)

    def test_mask_name_alone_does_not_verify_color_image_as_mask(self):
        image_dir = self.root / "train"
        image_dir.mkdir()
        Image.new("RGB", (24, 24), (20, 30, 40)).save(image_dir / "tile.jpg")
        rng = np.random.default_rng(11)
        Image.fromarray(rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)).save(
            image_dir / "tile_MA_mask.png"
        )

        report = inventory_dataset(self.root)

        annotation = next(item for item in report["annotations"] if item["format"] == "candidate_mask_image")
        self.assertEqual(annotation["status"], "unverified_not_mask_like")
        self.assertEqual(report["summary"]["images_with_verified_annotations"], 0)

    def test_voc_boxes_are_checked_against_actual_image_dimensions(self):
        images = self.root / "images" / "train"
        labels = self.root / "annotations" / "train"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        Image.new("RGB", (64, 48), "black").save(images / "fundus.jpg")
        (labels / "fundus.xml").write_text(
            """<annotation>
            <filename>fundus.jpg</filename><size><width>64</width><height>48</height></size>
            <object><name>hard exudate</name><bndbox>
            <xmin>4</xmin><ymin>5</ymin><xmax>20</xmax><ymax>25</ymax>
            </bndbox></object></annotation>""",
            encoding="utf-8",
        )
        (labels / "bad.xml").write_text(
            """<annotation><filename>fundus.jpg</filename>
            <object><name>unknown</name><bndbox>
            <xmin>0</xmin><ymin>0</ymin><xmax>100</xmax><ymax>20</ymax>
            </bndbox></object></annotation>""",
            encoding="utf-8",
        )

        report = inventory_dataset(self.root)

        records = {Path(item["path"]).name: item for item in report["annotations"]}
        self.assertEqual(records["fundus.xml"]["status"], "verified_pair")
        self.assertEqual(records["bad.xml"]["status"], "invalid_box_coordinates")
        self.assertEqual(records["fundus.xml"]["boxes"][0]["class_name"], "hard exudate")

    def test_coco_boxes_require_image_reference_and_valid_coordinates(self):
        image_dir = self.root / "images" / "test"
        annotation_dir = self.root / "annotations" / "test"
        image_dir.mkdir(parents=True)
        annotation_dir.mkdir(parents=True)
        Image.new("RGB", (80, 60), "black").save(image_dir / "coco_01.jpg")
        (annotation_dir / "instances.json").write_text(
            json.dumps(
                {
                    "images": [{"id": 3, "file_name": "coco_01.jpg"}],
                    "categories": [{"id": 4, "name": "hemorrhage"}],
                    "annotations": [{"id": 9, "image_id": 3, "category_id": 4, "bbox": [5, 6, 20, 12]}],
                }
            ),
            encoding="utf-8",
        )

        report = inventory_dataset(self.root)

        annotation = next(item for item in report["annotations"] if item["format"] == "bounding_boxes")
        self.assertEqual(annotation["status"], "verified_pair")
        self.assertEqual(annotation["boxes"][0]["class_name"], "hemorrhage")
        self.assertEqual(annotation["matched_image_dimensions"], [80, 60])

    def test_yolo_boxes_and_class_names_are_parsed_from_contents(self):
        image_dir = self.root / "images" / "validation"
        label_dir = self.root / "labels" / "validation"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        Image.new("RGB", (100, 80), "black").save(image_dir / "img7.jpg")
        (label_dir / "classes.txt").write_text("microaneurysm\n", encoding="utf-8")
        (label_dir / "img7.txt").write_text("0 0.5 0.5 0.2 0.1\n", encoding="utf-8")

        report = inventory_dataset(self.root)

        annotation = next(item for item in report["annotations"] if item["format"] == "bounding_boxes")
        self.assertEqual(annotation["status"], "verified_pair")
        self.assertEqual(annotation["candidate_classes"], ["microaneurysm"])
        self.assertEqual(report["splits"]["validation"]["annotation_files"], 1)

    def test_image_level_csv_labels_are_not_reported_as_lesion_annotations(self):
        images = self.root / "images" / "train"
        images.mkdir(parents=True)
        Image.new("RGB", (32, 24), "black").save(images / "record_3.jpg")
        (self.root / "grading.csv").write_text("image_id,grade\nrecord_3,2\n", encoding="utf-8")

        report = inventory_dataset(self.root)

        self.assertEqual(report["summary"]["annotation_file_count"], 1)
        self.assertEqual(report["summary"]["image_level_label_file_count"], 1)
        self.assertEqual(report["summary"]["images_with_verified_annotations"], 0)
        self.assertEqual(
            report["annotations"][0]["format"],
            "image_level_labels_not_lesion_annotations",
        )

    def test_dimension_mismatch_and_corrupt_image_are_reported(self):
        image_dir = self.root / "images" / "test"
        mask_dir = self.root / "masks" / "test"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        Image.new("RGB", (40, 32), "black").save(image_dir / "case.jpg")
        Image.new("L", (20, 16), 0).save(mask_dir / "case_HE.png")
        (image_dir / "broken.png").write_bytes(b"not an image")

        report = inventory_dataset(self.root)

        mask = next(item for item in report["annotations"] if item["format"] == "candidate_mask_image")
        self.assertEqual(mask["status"], "dimension_mismatch")
        self.assertEqual(len(report["mismatched_files"]), 1)
        self.assertEqual(report["unreadable_files"][0]["path"], "images/test/broken.png")

    def test_license_evidence_and_markdown_are_reported_without_claiming_missing_terms(self):
        (self.root / "LICENSE.txt").write_text(
            "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)\n",
            encoding="utf-8",
        )
        Image.new("RGB", (16, 12), "white").save(self.root / "plain.png")

        report = inventory_dataset(self.root)
        rendered = render_markdown(report)
        serialized = json.dumps(report)

        self.assertIn("CC BY-NC-SA 4.0", serialized)
        self.assertIn("Images with structurally verified annotation pairs: **0**", rendered)
        self.assertIn("Reuse terms and attribution obligations remain unverified", render_markdown({
            **report, "license_and_attribution_notes": []
        }))

    def test_cli_writes_json_and_markdown_outside_dataset_root(self):
        output = self.root.parent / "inventory-output"

        result = main(["--root", str(self.root), "--output-dir", str(output)])

        self.assertEqual(result, 0)
        machine_report = json.loads((output / "ddr_inventory.json").read_text(encoding="utf-8"))
        self.assertEqual(machine_report["summary"]["image_file_count"], 0)
        self.assertTrue((output / "ddr_inventory.md").is_file())

    def test_cli_rejects_report_output_inside_dataset_root(self):
        output = self.root / "reports"

        result = main(["--root", str(self.root), "--output-dir", str(output)])

        self.assertEqual(result, 2)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
