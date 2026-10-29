"""Configurable decision-support flags; these rules are not clinically validated.

Uncertainty is flagged when either minimum-confidence or minimum-margin is missed.
Review priority is flagged at the configured class index or for uncertain outputs.
"""

import math
import os


def validate_probabilities(probabilities) -> list[float]:
    try:
        values = [float(probability) for probability in probabilities]
    except (TypeError, ValueError) as exc:
        raise ValueError("Model output must contain numeric class probabilities.") from exc
    if (len(values) != 5 or not all(math.isfinite(p) and 0 <= p <= 1 for p in values)
            or not math.isclose(sum(values), 1.0, abs_tol=1e-3)):
        raise ValueError("Model output must contain five finite probabilities summing to 1.")
    return values


def validate_model_output(model_output) -> list[float]:
    if getattr(model_output, "shape", None) != (1, 5):
        raise ValueError("Model output must have shape (1, 5).")
    return validate_probabilities(model_output[0])


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number between 0 and 1.") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a number between 0 and 1.")
    return value


def _env_class_index(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer from 1 to 4.") from exc
    if not 1 <= value <= 4:
        raise ValueError(f"{name} must be an integer from 1 to 4.")
    return value


UNCERTAINTY_MIN_CONFIDENCE = _env_float("RETINAAI_UNCERTAINTY_MIN_CONFIDENCE", 0.50)
UNCERTAINTY_MIN_MARGIN = _env_float("RETINAAI_UNCERTAINTY_MIN_MARGIN", 0.15)
PRIORITY_REVIEW_MIN_CLASS = _env_class_index("RETINAAI_PRIORITY_REVIEW_MIN_CLASS", 2)


def assess_prediction(probabilities: list[float], class_index: int, quality: dict) -> dict:
    """Apply transparent threshold rules to validated model probabilities."""
    probabilities = validate_probabilities(probabilities)
    if not 0 <= class_index < len(probabilities):
        raise ValueError("Predicted class index is outside the configured class range.")

    ordered = sorted(probabilities, reverse=True)
    margin = ordered[0] - ordered[1]
    confidence = probabilities[class_index]
    uncertainty_reasons = []
    if confidence < UNCERTAINTY_MIN_CONFIDENCE:
        uncertainty_reasons.append("top_class_confidence_below_threshold")
    if margin < UNCERTAINTY_MIN_MARGIN:
        uncertainty_reasons.append("top_two_probability_margin_below_threshold")

    uncertainty = {
        "is_uncertain": bool(uncertainty_reasons),
        "top_two_margin": round(margin, 6),
        "top_two_margin_percent": round(margin * 100, 1),
        "minimum_confidence_threshold": UNCERTAINTY_MIN_CONFIDENCE,
        "minimum_margin_threshold": UNCERTAINTY_MIN_MARGIN,
        "reasons": uncertainty_reasons,
    }

    if not quality.get("acceptable", False):
        priority = {
            "level": "not_reliable",
            "label": "Do not rely on this result",
            "reasons": ["Image-quality checks failed; no prediction should be used for screening decisions."],
        }
    else:
        reasons = []
        if uncertainty["is_uncertain"]:
            reasons.append("The prediction is uncertain under the configured confidence/margin rules.")
        if class_index >= PRIORITY_REVIEW_MIN_CLASS:
            reasons.append("The predicted grade meets the configured higher-grade review rule.")
        priority = {
            "level": "professional_review" if reasons else "no_rule_flag",
            "label": "Professional review recommended" if reasons else "No additional rule-based flag",
            "reasons": reasons or [
                "The configured uncertainty and higher-grade rules did not flag this output; this does not establish clinical safety."
            ],
        }

    return {
        "uncertainty": uncertainty,
        "screening_priority": {
            **priority,
            "rules": {
                "minimum_confidence_threshold": UNCERTAINTY_MIN_CONFIDENCE,
                "minimum_top_two_margin_threshold": UNCERTAINTY_MIN_MARGIN,
                "minimum_class_index_for_review": PRIORITY_REVIEW_MIN_CLASS,
            },
            "validated": False,
        },
    }
