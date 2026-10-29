"""RetinaAI FastAPI backend (EfficientNetB0 diabetic retinopathy screening).

Run (from the backend folder):   python -m uvicorn main:app --reload
API docs:                        http://127.0.0.1:8000/docs
"""
import io
import logging
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src import config
from src.image_quality import INSUFFICIENT_MESSAGE, assess_image_quality

log = logging.getLogger("retinaai")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
STATE = {"predictor": None, "error": None}


def load_model():
    """Load the trained model at startup. Failures are reported loudly, never hidden."""
    STATE["predictor"], STATE["error"] = None, None
    try:
        from src.inference import RetinaPredictor
        STATE["predictor"] = RetinaPredictor(config.MODEL_PATH)
        print(f"[RetinaAI] Trained model loaded: {config.MODEL_PATH}")
        if STATE["predictor"].gradcam_error:
            print(f"[RetinaAI] WARNING - Grad-CAM unavailable: {STATE['predictor'].gradcam_error}")
    except Exception as exc:
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        print("=" * 70)
        print(f"[RetinaAI] ERROR - the trained model could not be loaded from:\n  {config.MODEL_PATH}")
        print(f"  Reason: {STATE['error']}")
        print("  Predictions are DISABLED until this is fixed.")
        print("=" * 70)
        log.error(traceback.format_exc())


@asynccontextmanager
async def lifespan(app):
    config.ensure_dirs()
    load_model()
    yield


app = FastAPI(title="RetinaAI API", version="2.0.0",
              description="Diabetic retinopathy screening (educational, not a medical device)",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[],
                   allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
                   allow_methods=["*"], allow_headers=["*"])
config.ensure_dirs()
app.mount("/api/gradcam", StaticFiles(directory=str(config.GRADCAM_DIR)), name="gradcam")


def _err(status: int, code: str, message: str, **extra):
    return HTTPException(status_code=status, detail={"code": code, "message": message, **extra})


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": {
        "code": "invalid_request", "message": "No image received. Send a file in the 'file' form field."}})


@app.get("/api/health")
def health():
    if STATE["predictor"] is not None:
        return {"status": "ok", "model_status": "trained"}
    return JSONResponse(status_code=503, content={
        "status": "error", "model_status": "error",
        "message": "The trained model is unavailable. Check the backend logs for details."})


@app.get("/api/model-info")
def model_info():
    p = STATE["predictor"]
    trained = p is not None
    return {
        "trained": trained,
        "model_status": "trained" if trained else "error",
        "message": None if trained else "The trained model is unavailable. Check the backend logs for details.",
        "class_names": config.CLASS_NAMES,
        "model_file": config.MODEL_PATH.name,
        "architecture": "EfficientNetB0 (transfer learning), 224x224 RGB, 5-class softmax",
        "gradcam_available": bool(p and p.gradcam is not None),
    }


@app.post("/api/predict")
def predict(file: UploadFile = File(...), robustness_check: bool = Query(default=False)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in config.UPLOAD_EXTENSIONS:
        raise _err(415, "unsupported_format", "Unsupported file type. Please upload a JPG, JPEG or PNG image.")
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise _err(400, "empty_file", "The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise _err(413, "file_too_large", "The image is larger than 10 MB. Please upload a smaller file.")
    try:
        with Image.open(io.BytesIO(contents)) as im:
            im.load()
            expected_format = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}[suffix]
            if im.format != expected_format:
                raise ValueError("Image contents do not match the filename extension.")
    except Exception:
        quality = assess_image_quality(contents)
        raise _err(400, "invalid_image",
                   "The file could not be read as a supported image or does not match its file extension.",
                   image_quality=quality)

    predictor = STATE["predictor"]
    if predictor is None:
        raise _err(503, "model_unavailable",
                   "The trained model is unavailable; no result was produced. Do not rely on this application for screening decisions.")
    try:
        result = predictor.predict_bytes(contents, include_robustness=robustness_check)
    except Exception:
        log.exception("Prediction failed")
        raise _err(500, "server_error",
                   "The model could not produce a valid result. Do not rely on a prediction for screening decisions; please try again.")

    if result["predicted_class"] is None:
        raise _err(422, "poor_image_quality", INSUFFICIENT_MESSAGE,
                   image_quality=result["image_quality"],
                   screening_priority=result.get("screening_priority"))
    result["gradcam_url"] = f"/api/gradcam/{result['gradcam_filename']}" if result["gradcam_filename"] else None
    result["heatmap_url"] = (
        f"/api/gradcam/{result['heatmap_filename']}" if result["heatmap_filename"] else None
    )
    return result


if __name__ == "__main__":
    print("RetinaAI API: http://127.0.0.1:8000   (docs: http://127.0.0.1:8000/docs)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
