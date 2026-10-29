"""Grad-CAM for the nested EfficientNetB0 model.

The trained model is  Input -> EfficientNetB0 (nested Functional model) -> GAP -> BN -> Dropout
-> Dense(256) -> Dropout -> Dense(5, softmax).  The layer "top_activation" lives INSIDE the nested
backbone, so it is not reachable from the outer model's graph.  We therefore:
  1. build a sub-model  backbone.input -> backbone.get_layer("top_activation").output,
  2. pass its 7x7x1280 feature map manually through the outer classification head layers,
  3. take the gradient of the chosen class score w.r.t. that feature map.

Grad-CAM is a model-attention visualisation, NOT a validated lesion detector.
"""
import cv2
import numpy as np
import tensorflow as tf

from src import config

OVERLAY_MAX_SIDE = 640


class GradCamUnavailable(RuntimeError):
    pass


def _find_backbone(model):
    try:
        layer = model.get_layer(config.BACKBONE_LAYER)
        layer.get_layer(config.GRADCAM_LAYER)
        return layer
    except Exception:
        pass
    for layer in model.layers:                      # fall back: any nested model containing the layer
        if hasattr(layer, "get_layer"):
            try:
                layer.get_layer(config.GRADCAM_LAYER)
                return layer
            except Exception:
                continue
    raise GradCamUnavailable(f"No nested backbone containing '{config.GRADCAM_LAYER}' was found.")


class GradCam:
    def __init__(self, model):
        self.backbone = _find_backbone(model)
        layers = list(model.layers)
        idx = layers.index(self.backbone)
        self.head = layers[idx + 1:]
        if not self.head:
            raise GradCamUnavailable("Classification head layers were not found after the backbone.")
        target = self.backbone.get_layer(config.GRADCAM_LAYER)
        self.conv_model = tf.keras.Model(self.backbone.inputs, target.output, name="gradcam_conv_model")
        self._verify(model)

    def _head(self, feats):
        y = feats
        for layer in self.head:
            y = layer(y, training=False)
        return y

    def _verify(self, model):
        """Check that conv_model + head reproduces the real model's output (no fake path)."""
        probe = np.random.default_rng(0).uniform(0, 255, (1, *config.IMG_SHAPE)).astype("float32")
        full = np.asarray(model(probe, training=False))
        split = np.asarray(self._head(self.conv_model(probe, training=False)))
        if not np.allclose(full, split, atol=1e-4):
            raise GradCamUnavailable("Split model (backbone + head) does not match the full model output.")

    def compute(self, batch: np.ndarray, class_index: int) -> np.ndarray:
        """batch: float32 (1,224,224,3), 0..255. Returns heat-map (7x7) scaled to 0..1."""
        x = tf.convert_to_tensor(batch, dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv = self.conv_model(x, training=False)
            tape.watch(conv)
            score = self._head(conv)[:, class_index]
        grads = tape.gradient(score, conv)
        if grads is None:
            raise GradCamUnavailable("Gradients could not be computed.")
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
        peak = float(cam.max())
        if not np.isfinite(peak) or peak <= 0:
            raise GradCamUnavailable("The model produced no positive activation for this class.")
        return (cam / peak).astype("float32")


def overlay_heatmap(original_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a JET heat-map over the original image (same aspect ratio). RGB uint8 in/out."""
    h, w = original_rgb.shape[:2]
    scale = min(1.0, OVERLAY_MAX_SIDE / max(h, w))
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    base = cv2.resize(original_rgb, size, interpolation=cv2.INTER_AREA)
    heat = np.clip(cv2.resize(cam, size, interpolation=cv2.INTER_CUBIC), 0, 1)
    colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    wgt = (alpha * heat)[..., None]
    out = base.astype("float32") * (1 - wgt) + colored.astype("float32") * wgt
    return np.clip(out, 0, 255).astype("uint8")


def colorize_heatmap(cam: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize and colorize a real Grad-CAM map to align with the source image."""
    scale = min(1.0, OVERLAY_MAX_SIDE / max(width, height))
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    heat = np.clip(cv2.resize(cam, size, interpolation=cv2.INTER_CUBIC), 0, 1)
    return cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )


def save_heatmap(heatmap_rgb: np.ndarray, filename: str):
    config.GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    path = config.GRADCAM_DIR / filename
    if not cv2.imwrite(str(path), cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)):
        raise OSError("Grad-CAM heatmap could not be saved.")
    _prune_gradcam_files()
    return path


def save_overlay(overlay_rgb: np.ndarray, filename: str):
    config.GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    path = config.GRADCAM_DIR / filename
    if not cv2.imwrite(str(path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)):
        raise OSError("Grad-CAM overlay could not be saved.")
    _prune_gradcam_files()
    return path


def _prune_gradcam_files():
    files = sorted(
        [*config.GRADCAM_DIR.glob("gradcam_*.png"), *config.GRADCAM_DIR.glob("heatmap_*.png")],
        key=lambda p: p.stat().st_mtime,
    )
    for old in files[:-config.MAX_GRADCAM_FILES]:
        old.unlink(missing_ok=True)
