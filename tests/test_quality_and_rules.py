import io
import unittest

import numpy as np
from PIL import Image

from src.image_quality import assess_image_quality
from src.screening_rules import assess_prediction, validate_model_output, validate_probabilities


def encode_image(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


class ImageQualityTests(unittest.TestCase):
    def test_valid_resolution_and_detail(self):
        rng = np.random.default_rng(7)
        image = rng.integers(30, 210, (400, 400, 3), dtype=np.uint8)
        result = assess_image_quality(encode_image(image))
        self.assertTrue(result["acceptable"])
        self.assertEqual(result["status"], "Good")
        self.assertEqual(result["checks"]["decoding"]["status"], "pass")
        self.assertEqual(result["checks"]["resolution"]["status"], "pass")

    def test_dark_image_is_flagged(self):
        image = np.full((400, 400, 3), 10, dtype=np.uint8)
        result = assess_image_quality(encode_image(image))
        self.assertFalse(result["acceptable"])
        self.assertEqual(result["checks"]["exposure"]["status"], "fail")

    def test_low_resolution_is_flagged(self):
        rng = np.random.default_rng(3)
        image = rng.integers(30, 210, (100, 100, 3), dtype=np.uint8)
        result = assess_image_quality(encode_image(image))
        self.assertFalse(result["acceptable"])
        self.assertEqual(result["checks"]["resolution"]["status"], "fail")

    def test_blur_warning_is_reported(self):
        image = np.full((400, 400, 3), 120, dtype=np.uint8)
        result = assess_image_quality(encode_image(image))
        self.assertEqual(result["checks"]["sharpness"]["status"], "warning")
        self.assertIn("blur", result["checks"]["sharpness"]["message"])

    def test_corrupt_file_has_structured_failure(self):
        result = assess_image_quality(b"not an image")
        self.assertFalse(result["acceptable"])
        self.assertEqual(result["checks"]["decoding"]["status"], "fail")
        self.assertTrue(result["suggested_action"])


class ScreeningRuleTests(unittest.TestCase):
    def test_small_top_two_margin_is_uncertain(self):
        result = assess_prediction([0.36, 0.35, 0.12, 0.10, 0.07], 0, {"acceptable": True})
        self.assertTrue(result["uncertainty"]["is_uncertain"])
        self.assertAlmostEqual(result["uncertainty"]["top_two_margin"], 0.01)
        self.assertEqual(result["screening_priority"]["level"], "professional_review")

    def test_low_confidence_is_uncertain(self):
        result = assess_prediction([0.48, 0.20, 0.15, 0.10, 0.07], 0, {"acceptable": True})
        self.assertIn("top_class_confidence_below_threshold", result["uncertainty"]["reasons"])

    def test_higher_predicted_grade_triggers_review_rule(self):
        result = assess_prediction([0.05, 0.05, 0.10, 0.70, 0.10], 3, {"acceptable": True})
        self.assertEqual(result["screening_priority"]["level"], "professional_review")
        self.assertTrue(any("higher-grade" in reason for reason in result["screening_priority"]["reasons"]))

    def test_failed_quality_means_output_is_not_reliable(self):
        result = assess_prediction([0.80, 0.08, 0.05, 0.04, 0.03], 0, {"acceptable": False})
        self.assertEqual(result["screening_priority"]["level"], "not_reliable")

    def test_invalid_probability_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            assess_prediction([float("nan"), 0.25, 0.25, 0.25, 0.25], 0, {"acceptable": True})

    def test_probabilities_that_do_not_sum_to_one_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_probabilities([0.1, 0.1, 0.1, 0.1, 0.1])

    def test_missing_model_output_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_model_output(np.empty((0, 5)))


if __name__ == "__main__":
    unittest.main()
