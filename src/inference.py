"""Real inference pipeline: quality check -> preprocessing -> EfficientNetB0 model -> Grad-CAM.

Preprocessing note: the trained model's EfficientNetB0 backbone contains its own Rescaling (1/255) and
Normalization layers.  The correct input is therefore RGB, 224x224, float32 in the RAW 0-255 range
(this is what keras.applications.efficientnet.preprocess_input does - it is a pass-through).
Do NOT divide by 255 here, or the image would be scaled twice.
"""
import io
import logging
import threading
import traceback
import uuid
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from src import config
from src.gradcam import GradCam, colorize_heatmap, overlay_heatmap, save_heatmap, save_overlay
from src.image_quality import INSUFFICIENT_MESSAGE, assess_image_quality
from src.screening_rules import assess_prediction, validate_model_output

log = logging.getLogger("retinaai.inference")
ROBUSTNESS_VARIANTS = (
    ("brightness_minus_8", -8.0, 1.0),
    ("brightness_plus_8", 8.0, 1.0),
    ("contrast_minus_5_percent", 0.0, 0.95),
    ("contrast_plus_5_percent", 0.0, 1.05),
)


def preprocess_image(rgb_uint8: np.ndarray) -> np.ndarray:
    """RGB uint8 (H,W,3) -> float32 batch (1,224,224,3) in 0..255."""
    img = tf.image.resize(rgb_uint8, (config.IMG_SIZE, config.IMG_SIZE), method="bilinear", antialias=True)
    img = tf.round(tf.clip_by_value(img, 0.0, 255.0))
    batch = tf.cast(img, tf.float32).numpy()[None, ...]
    return tf.keras.applications.efficientnet.preprocess_input(batch)


def load_rgb(contents: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(contents)) as im:
        im = ImageOps.exif_transpose(im)
        return np.array(im.convert("RGB"))


def make_appearance_variants(rgb: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Create mild RGB brightness/contrast variants before the shared preprocessing step."""
    variants = []
    for name, brightness, contrast in ROBUSTNESS_VARIANTS:
        adjusted = (rgb.astype(np.float32) - 127.5) * contrast + 127.5 + brightness
        variants.append((name, np.clip(adjusted, 0, 255).astype(np.uint8)))
    return variants


def summarize_robustness(original_probabilities, variant_probabilities) -> dict:
    """Compare validated variant outputs with the unchanged original prediction."""
    original = validate_probabilities(original_probabilities)
    if len(variant_probabilities) != len(ROBUSTNESS_VARIANTS):
        raise ValueError("Robustness check requires all configured appearance variants.")

    variants = []
    probability_changes = []
    original_class = int(np.argmax(original))
    for (name, _brightness, _contrast), output in zip(ROBUSTNESS_VARIANTS, variant_probabilities):
        probabilities = validate_probabilities(output)
        predicted_class = int(np.argmax(probabilities))
        changes = [abs(probability - baseline) for probability, baseline in zip(probabilities, original)]
        probability_changes.extend(changes)
        variants.append({
            "name": name,
            "predicted_class": predicted_class,
            "predicted_class_name": config.CLASS_NAMES[predicted_class],
            "probabilities": {label: probability for label, probability in zip(config.CLASS_NAMES, probabilities)},
            "mean_absolute_probability_change": round(float(np.mean(changes)), 6),
            "max_absolute_probability_change": round(float(np.max(changes)), 6),
        })

    consistent_count = sum(item["predicted_class"] == original_class for item in variants)
    consistent = consistent_count == len(variants)
    return {
        "available": True,
        "experimental": True,
        "status": "consistent" if consistent else "sensitive",
        "original_predicted_class": original_class,
        "original_predicted_class_name": config.CLASS_NAMES[original_class],
        "variant_count": len(variants),
        "consistent_variant_count": consistent_count,
        "consistency_percent": round(100 * consistent_count / len(variants), 1),
        "mean_absolute_probability_change": round(float(np.mean(probability_changes)), 6),
        "max_absolute_probability_change": round(float(np.max(probability_changes)), 6),
        "variants": variants,
        "message": (
            "The predicted class changed for one or more mild appearance variations."
            if not consistent else
            "The predicted class was unchanged for the tested mild appearance variations."
        ),
        "disclaimer": (
            "Experimental sensitivity check only; this is not a calibrated reliability score or "
            "a probability of medical correctness."
        ),
    }


class RetinaPredictor:
    def __init__(self, model_path=config.MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model file not found: {model_path}")
        self.model_path = model_path
        self.model = tf.keras.models.load_model(str(model_path), compile=False)
        in_shape, out_shape = self.model.input_shape, self.model.output_shape
        if tuple(in_shape[1:]) != config.IMG_SHAPE or out_shape[-1] != config.NUM_CLASSES:
            raise ValueError(f"Unexpected model shapes: input {in_shape}, output {out_shape}; "
                             f"expected {(None, *config.IMG_SHAPE)} -> (None, {config.NUM_CLASSES}).")
        self._lock = threading.Lock()
        self.gradcam, self.gradcam_error = None, None
        try:
            self.gradcam = GradCam(self.model)
        except Exception as exc:                     # prediction still works; Grad-CAM reports unavailable
            self.gradcam_error = str(exc)
            traceback.print_exc()

    def _run_robustness_check(self, rgb: np.ndarray, original_probabilities: list[float]) -> dict:
        try:
            variant_probabilities = []
            for _name, variant in make_appearance_variants(rgb):
                batch = preprocess_image(variant)
                model_output = np.asarray(self.model(batch, training=False))
                variant_probabilities.append(validate_model_output(model_output))
            return summarize_robustness(original_probabilities, variant_probabilities)
        except Exception:
            log.exception("Experimental robustness check could not be completed")
            return {
                "available": False,
                "experimental": True,
                "status": "unavailable",
                "message": "The experimental sensitivity check could not be completed.",
                "disclaimer": (
                    "This check is not a calibrated reliability score or a probability of medical correctness."
                ),
            }

    def predict_bytes(self, contents: bytes, include_robustness: bool = False) -> dict:
        quality = assess_image_quality(contents)
        if not quality["acceptable"]:
            return {
                "predicted_class": None,
                "image_quality": quality,
                "screening_priority": {
                    "level": "not_reliable",
                    "label": "Do not rely on this result",
                    "reasons": ["Image-quality checks failed; no prediction was produced."],
                    "validated": False,
                },
                "message": INSUFFICIENT_MESSAGE,
            }

        rgb = load_rgb(contents)
        batch = preprocess_image(rgb)
        with self._lock:
            model_output = np.asarray(self.model(batch, training=False))
            try:
                probs = validate_model_output(model_output)
            except ValueError as exc:
                raise RuntimeError("The model returned invalid class output.") from exc
            idx = int(np.argmax(probs))
            robustness = (
                self._run_robustness_check(rgb, probs) if include_robustness else None
            )
            cam = None
            if self.gradcam is not None:
                try:
                    cam = self.gradcam.compute(batch, idx)
                except Exception:
                    log.exception("Grad-CAM could not be computed")
        confidence = probs[idx]
        decision = assess_prediction(probs, idx, quality)

        gradcam_filename = None
        heatmap_filename = None
        if cam is not None:
            try:
                token = uuid.uuid4().hex[:12]
                path = save_overlay(overlay_heatmap(rgb, cam), f"gradcam_{token}.png")
                heatmap = colorize_heatmap(cam, rgb.shape[1], rgb.shape[0])
                heatmap_path = save_heatmap(heatmap, f"heatmap_{token}.png")
                gradcam_filename = path.name
                heatmap_filename = heatmap_path.name
            except Exception:
                log.exception("Grad-CAM visualization could not be saved")

        result = {
            "predicted_class": idx,
            "predicted_class_name": config.CLASS_NAMES[idx],
            "confidence": confidence,
            "confidence_percent": round(100 * confidence, 1),
            "low_confidence": decision["uncertainty"]["is_uncertain"],
            "probabilities": {n: p for n, p in zip(config.CLASS_NAMES, probs)},
            "image_quality": quality,
            **decision,
            "recommendation": "This is an automated model output, not a diagnosis or referral decision.",
            "model_status": "trained",
            "model_file": self.model_path.name,
            "gradcam_available": gradcam_filename is not None,
            "gradcam_filename": gradcam_filename,
            "heatmap_filename": heatmap_filename,
            "gradcam_message": None if gradcam_filename else "Grad-CAM is unavailable for this prediction.",
            "disclaimer": config.DISCLAIMER,
        }
        if robustness is not None:
            result["robustness"] = robustness
        return result
