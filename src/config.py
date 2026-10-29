"""Central configuration: paths, class names and disclaimer."""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The trained EfficientNetB0 model (shipped inside the project).
MODELS_DIR = PROJECT_ROOT / "backend" / "models"
MODEL_PATH = MODELS_DIR / "best_dr_model_v3.keras"

RESULTS_DIR = PROJECT_ROOT / "results"
GRADCAM_DIR = RESULTS_DIR / "gradcam"
MAX_GRADCAM_FILES = 100          # oldest overlays are deleted beyond this

# ----- Classes (index order is the model output order) -----
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
NUM_CLASSES = len(CLASS_NAMES)

UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ----- Model input -----
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# ----- Grad-CAM -----
# The EfficientNetB0 backbone is a nested model inside the outer model.
BACKBONE_LAYER = "efficientnetb0"
GRADCAM_LAYER = "top_activation"     # last convolutional activation inside the backbone

DISCLAIMER = (
    "This application is intended for AI-assisted screening and educational/research purposes. "
    "It does not provide a medical diagnosis. Users should consult a qualified ophthalmologist "
    "for clinical evaluation."
)


def ensure_dirs():
    for d in (MODELS_DIR, RESULTS_DIR, GRADCAM_DIR):
        d.mkdir(parents=True, exist_ok=True)
