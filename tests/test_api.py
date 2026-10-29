import io
import unittest

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import STATE, app


def make_png():
    rng = np.random.default_rng(13)
    array = rng.integers(30, 210, (400, 400, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


class PredictionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def test_health_and_model_status(self):
        health = self.client.get("/api/health")
        model_info = self.client.get("/api/model-info")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["model_status"], "trained")
        self.assertTrue(model_info.json()["gradcam_available"])
        self.assertEqual(model_info.json()["class_names"], [
            "No DR", "Mild", "Moderate", "Severe", "Proliferative DR",
        ])

    def test_model_unavailable_does_not_expose_internal_error(self):
        original_predictor, original_error = STATE["predictor"], STATE["error"]
        try:
            STATE["predictor"] = None
            STATE["error"] = "C:\\private\\model-path.keras"
            health = self.client.get("/api/health")
            model_info = self.client.get("/api/model-info")
            self.assertEqual(health.status_code, 503)
            self.assertNotIn("private", health.text)
            self.assertNotIn("private", model_info.text)
        finally:
            STATE["predictor"], STATE["error"] = original_predictor, original_error

    def test_prediction_contains_five_probabilities_and_gradcam_assets(self):
        response = self.client.post(
            "/api/predict",
            files={"file": ("fundus.png", make_png(), "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        probabilities = result["probabilities"]
        self.assertEqual(len(probabilities), 5)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=3)
        self.assertIn("top_two_margin", result["uncertainty"])
        self.assertIn("screening_priority", result)
        self.assertEqual(result["image_quality"]["checks"]["decoding"]["status"], "pass")
        self.assertTrue(result["gradcam_available"])
        self.assertEqual(self.client.get(result["gradcam_url"]).status_code, 200)
        self.assertEqual(self.client.get(result["heatmap_url"]).status_code, 200)

    def test_corrupt_image_returns_structured_quality_error(self):
        response = self.client.post(
            "/api/predict",
            files={"file": ("corrupt.png", b"not an image", "image/png")},
        )
        self.assertEqual(response.status_code, 400)
        quality = response.json()["detail"]["image_quality"]
        self.assertEqual(quality["checks"]["decoding"]["status"], "fail")
        self.assertTrue(quality["suggested_action"])

    def test_mismatched_extension_and_image_contents_are_rejected(self):
        response = self.client.post(
            "/api/predict",
            files={"file": ("fundus.jpg", make_png(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "invalid_image")

    def test_unsupported_extension_is_rejected(self):
        response = self.client.post(
            "/api/predict",
            files={"file": ("fundus.gif", b"content", "image/gif")},
        )
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["detail"]["code"], "unsupported_format")

    def test_oversized_upload_is_rejected(self):
        response = self.client.post(
            "/api/predict",
            files={"file": ("fundus.png", b"x" * (10 * 1024 * 1024 + 1), "image/png")},
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"]["code"], "file_too_large")


if __name__ == "__main__":
    unittest.main()
