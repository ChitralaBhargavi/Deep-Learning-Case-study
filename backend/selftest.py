"""Optional check that the real model loads and predicts.   Run from backend/:  python selftest.py [image.jpg]

With no argument a SYNTHETIC test image is used. That only proves the pipeline runs end-to-end;
use a real fundus image to see a meaningful prediction.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

from src import config
from src.inference import RetinaPredictor

p = RetinaPredictor(config.MODEL_PATH)
print("model loaded:", config.MODEL_PATH, "| gradcam:", "ok" if p.gradcam else p.gradcam_error)
if len(sys.argv) > 1:
    data = Path(sys.argv[1]).read_bytes()
else:
    import cv2
    yy, xx = np.mgrid[:512, :512]
    r2 = (yy - 256) ** 2 + (xx - 256) ** 2
    disc = r2 < 240 ** 2
    rng = np.random.default_rng(1)
    base = np.stack([200 - r2 / 400, 100 - r2 / 900, 40 + 0 * r2], -1)
    img = np.zeros((512, 512, 3), "float32")
    img[disc] = base[disc] + rng.normal(0, 18, (int(disc.sum()), 3))
    img = cv2.GaussianBlur(img, (0, 0), 1.2) + rng.normal(0, 6, img.shape) * disc[..., None]
    img = np.clip(img, 0, 255).astype("uint8")
    buf = io.BytesIO(); Image.fromarray(img).save(buf, "PNG"); data = buf.getvalue()
r = p.predict_bytes(data)
print({k: v for k, v in r.items() if k not in ("disclaimer",)})
if r.get("predicted_class") is not None:
    assert len(r["probabilities"]) == 5 and abs(sum(r["probabilities"].values()) - 1) < 1e-3
    assert "uncertainty" in r and "screening_priority" in r
    assert r["gradcam_filename"] and (config.GRADCAM_DIR / r["gradcam_filename"]).is_file()
    assert r["heatmap_filename"] and (config.GRADCAM_DIR / r["heatmap_filename"]).is_file()
    print("OK: five valid probabilities, uncertainty rules, Grad-CAM overlay and heatmap generated.")
