"""Heuristic image-quality screening, not a clinical gradability assessment."""
import io

import cv2
import numpy as np
from PIL import Image, ImageOps

INSUFFICIENT_MESSAGE = "Image quality is insufficient for reliable screening."

MIN_SIDE_HARD = 128
MIN_SIDE_SOFT = 224
DARK_BRIGHTNESS = 30
BRIGHT_BRIGHTNESS = 220
LOW_CONTRAST = 12
BLUR_SHARPNESS = 8.0


def assess_image_quality(contents: bytes) -> dict:
    try:
        with Image.open(io.BytesIO(contents)) as im:
            im.load()
            rgb = np.array(ImageOps.exif_transpose(im).convert("RGB"))
    except Exception:
        message = "The image could not be decoded. Upload a readable JPG or PNG file."
        return {
            "status": "Poor",
            "acceptable": False,
            "checks": {
                "decoding": {"status": "fail", "value": None, "message": message},
            },
            "metrics": {},
            "warnings": [],
            "issues": [message],
            "messages": [message],
            "suggested_action": "Choose another valid JPG or PNG image.",
            "screening_note": "These simple checks cannot reliably identify every retinal image.",
        }

    h, w = rgb.shape[:2]
    scale = 512.0 / max(h, w)
    small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA) \
        if scale < 1 else rgb
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    fg = gray > 15                                   # ignore the black border around fundus images
    fg_fraction = float(fg.mean())
    region = gray[fg] if fg_fraction > 0.05 else gray.reshape(-1)
    brightness = float(region.mean())
    contrast = float(region.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    checks = {}
    problems, warnings = [], []
    checks["decoding"] = {
        "status": "pass",
        "value": "decoded",
        "message": "Image file decoded successfully.",
    }

    def add_check(name, status, value, message):
        checks[name] = {"status": status, "value": value, "message": message}
        if status == "fail":
            problems.append(message)
        elif status == "warning":
            warnings.append(message)

    if min(h, w) < MIN_SIDE_HARD:
        add_check("resolution", "fail", f"{w}x{h}px",
                  f"Resolution is too low ({w}x{h}px); image detail may be insufficient.")
    elif min(h, w) < MIN_SIDE_SOFT:
        add_check("resolution", "warning", f"{w}x{h}px",
                  f"Resolution is low ({w}x{h}px); small details may be lost.")
    else:
        add_check("resolution", "pass", f"{w}x{h}px", "Image dimensions meet the heuristic minimum.")

    if fg_fraction < 0.15:
        add_check("exposure", "fail", round(brightness, 1), "The image is almost entirely black.")
    elif brightness < DARK_BRIGHTNESS:
        add_check("exposure", "fail", round(brightness, 1), "The image appears very dark.")
    elif brightness > BRIGHT_BRIGHTNESS:
        add_check("exposure", "fail", round(brightness, 1), "The image appears over-exposed.")
    elif brightness < 50 or brightness > 190:
        add_check("exposure", "warning", round(brightness, 1),
                  "Image brightness is outside the usual range; exposure may affect details.")
    else:
        add_check("exposure", "pass", round(brightness, 1), "No extreme brightness was detected.")

    contrast_status = "fail" if contrast < LOW_CONTRAST else "pass"
    add_check("contrast", contrast_status, round(contrast, 1),
              "Image contrast is very low." if contrast_status != "pass"
              else "Image contrast is above the heuristic minimum.")

    blur_status = "warning" if sharpness < BLUR_SHARPNESS else "pass"
    add_check("sharpness", blur_status, round(sharpness, 1),
              "The image may be blurred or out of focus." if blur_status == "warning"
              else "No strong blur signal was detected.")

    status = "Poor" if problems else ("Fair" if warnings else "Good")
    messages = problems + warnings
    if problems:
        action = "Upload a sharper, well-lit image with the retina visible and centred."
    elif warnings:
        action = "If possible, upload a sharper, evenly exposed image; review the warnings before interpreting results."
    else:
        action = "No corrective action was suggested by these simple checks."

    return {
        "status": status,
        "acceptable": not problems,
        "checks": checks,
        "metrics": {
            "width": int(w), "height": int(h),
            "brightness": round(brightness, 1), "contrast": round(contrast, 1),
            "sharpness": round(sharpness, 1), "non_dark_area_fraction": round(fg_fraction, 2),
        },
        "warnings": warnings,
        "issues": problems,
        "messages": messages,
        "suggested_action": action,
        "screening_note": "These simple image statistics cannot reliably distinguish every retinal image from a non-retinal image.",
    }
