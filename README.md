# RetinaAI — Diabetic Retinopathy Screening

An educational/research application using the existing trained EfficientNetB0 model for five-class
diabetic-retinopathy screening support. The existing model is loaded at backend startup; this
project does not retrain or replace it.

| Index | Predicted class |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

> **Medical disclaimer.** This application is intended for AI-assisted screening and
> educational/research purposes. It does not provide a medical diagnosis. Users should consult a
> qualified ophthalmologist for clinical evaluation.

## Requirements

- Windows, Python 3.10–3.13, and Node.js LTS
- The included trained model: `backend/models/best_dr_model_v3.keras`

## Problem, objectives, and proposed solution

Diabetic retinopathy can damage retinal blood vessels, and screening commonly uses fundus
photographs to assess disease grade. This project demonstrates an existing deep-learning model
that classifies a fundus image into five DR grades. Its objectives are to provide a local screening
workflow, present model confidence and image-quality warnings, and make the model's output easier
to inspect with uncertainty rules and Grad-CAM. It is an educational/research demonstration, not a
clinical decision system.

## Architecture and end-to-end workflow

- **Frontend:** React and Vite. The in-app pages are Home, Screening, Results, Experiments, History,
  and About; page selection is handled in application state.
- **Backend:** FastAPI serves model information and health status, accepts image uploads, returns
  predictions, and serves generated Grad-CAM images.
- **Model:** the included Keras EfficientNetB0 five-class model is loaded once when the backend
  starts.
- **Request flow:** the browser previews a selected JPG/JPEG/PNG; on analysis it sends the image as
  multipart form data to `POST /api/predict`. The backend validates the upload, checks image
  quality, converts it to RGB, resizes it to 224×224, and runs inference. It validates the model
  probabilities, applies the configured uncertainty and screening-priority rules, and generates
  Grad-CAM assets when available. The frontend displays the returned result and keeps a summary in
  page memory only. PDF summaries are generated in the browser.
- Uploaded source images are processed in memory and not saved by the backend. Generated Grad-CAM
  assets are saved under `results/gradcam` and pruned to a bounded set.

## Dataset and model methodology

The project documentation identifies APTOS as the internal test evaluation and IDRiD as the
external evaluation; only the aggregate accuracy and macro-F1 values shown below are recorded.
The About page documents an ImageNet-pretrained EfficientNetB0 adapted for five classes, with a
global-average-pooling and batch-normalization head, dropout, a 256-unit ReLU dense layer, and a
five-class softmax output. Input preprocessing and the model's internal scaling are described in
the About page and inference code.

The repository contains the trained model file but no training script, dataset images, or
per-image evaluation labels/predictions. The available files do not establish dataset sizes,
split composition, optimizer, epoch count, or enough run configuration to reproduce the recorded
experiments. V4 and V5 are identified only by the documented changes in the experiment table.
These gaps prevent independent reproduction or detailed dataset inspection. DDR is not included,
downloaded, or inspected; the inventory script below only inspects a local archive extraction.

## Run locally

Open two PowerShell terminals at the project root. The commands below use the default backend and
frontend ports, `8000` and `5173`; check they are free before starting.

**Terminal 1 — backend**

```powershell
if (-not (Test-Path .\venv\Scripts\python.exe)) { python -m venv .\venv }
.\venv\Scripts\Activate.ps1
python -m pip install -r .\backend\requirements.txt
Set-Location .\backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

The included model must remain at `backend\models\best_dr_model_v3.keras`. On startup, the
backend logs whether it loaded. The API is at `http://127.0.0.1:8000`; health is at
`http://127.0.0.1:8000/api/health`, and interactive API documentation is at
`http://127.0.0.1:8000/docs`. A loaded model returns
`{"status":"ok","model_status":"trained"}`.

**Terminal 2 — frontend**

```powershell
Set-Location .\frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. If PowerShell blocks virtual-environment activation in Terminal 1,
run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` there, then activate the
environment again.

Check a port before starting either server:

```powershell
Get-NetTCPConnection -State Listen -LocalPort 8000
Get-NetTCPConnection -State Listen -LocalPort 5173
```

If a command returns a listener, identify it with
`Get-Process -Id <OwningProcess>` using the reported process ID; do not terminate an unknown
process. To use another free backend port, pass it to Uvicorn (for example, `--port 8001`) and set
`$env:VITE_API_URL = "http://127.0.0.1:8001"` in the frontend terminal before starting Vite. To
use another frontend port, pass it explicitly, for example `npm run dev -- --port 5174`. The Vite
server uses `strictPort`, so it will report an occupied port instead of silently switching.

## Implemented features

- Real inference using the included Keras model. RGB images are resized to 224×224 and passed as
  float32 values in the original 0–255 range; the model's EfficientNetB0 backbone performs its own
  scaling. The model is loaded once when the backend starts.
- Five-class probabilities, top-class confidence, and the difference between the highest and
  second-highest probabilities. Confidence is a model output, not clinical certainty.
- Configurable uncertainty rules. Defaults are a minimum top-class confidence of `0.50` and a
  minimum top-two margin of `0.15`. A result is marked uncertain if either threshold is not met.
  These are illustrative software thresholds, not medically validated; calibrate and validate them
  using a representative validation dataset before any clinical use.
- Structured heuristic quality checks for decoding, dimensions, brightness/exposure, contrast, and
  sharpness, with warnings and suggested corrective action. These checks are not a validated
  gradability model and cannot reliably distinguish every retinal image from a non-retinal image.
- Actual Grad-CAM for the model's predicted class, an opacity-adjustable overlay, show/hide control,
  side-by-side original/heatmap/overlay views, and overlay download. Grad-CAM is a visualization of
  regions influencing the prediction, not a validated lesion detector or proof of diagnosis.
- Configurable screening-support flags. The default rule marks uncertain results and class indices
  2–4 for professional review. It is not a diagnosis, treatment recommendation, or guaranteed
  referral decision; the rules are not clinically validated.
- A recorded experiment comparison page with APTOS and IDRiD accuracy and macro F1 charts.
- A model-reliability page explains that class-wise metrics and calibration results are unavailable
  without per-image labels and predictions; it does not fill gaps with estimates.
- Session-only screening history displays past results, class probabilities, quality warnings,
  uncertainty, and timestamps. History is held in React memory for the open page session only;
  refreshing or closing the page clears it. No account, backend history, or authentication is added.
- A downloadable PDF summary includes the input dimensions (not the uploaded image), prediction,
  five class probabilities, quality warnings, optional Grad-CAM visualization, limitations, and
  medical disclaimer. The temporary Grad-CAM image can expire before a later report is generated.
- Uploaded source images are processed in memory and are not saved as files. Generated Grad-CAM
  images are saved under `results/gradcam` and old generated images are pruned to a bounded set.

## Configuration

Set these environment variables before starting the backend. Values are validated at startup.

```powershell
$env:RETINAAI_UNCERTAINTY_MIN_CONFIDENCE = "0.50"
$env:RETINAAI_UNCERTAINTY_MIN_MARGIN = "0.15"
$env:RETINAAI_PRIORITY_REVIEW_MIN_CLASS = "2"
```

Confidence and margin values must be between 0 and 1. The priority class index must be between 1
and 4 in the configured class order. Review every rule with appropriate clinical expertise and
representative validation data before use beyond demonstration/research.

## Recorded experiments

These are recorded results, not live measurements:

| Experiment | APTOS test accuracy | IDRiD external accuracy | APTOS macro F1 | IDRiD macro F1 |
|---|---:|---:|---:|---:|
| V3 - selected model | 79.27% | 48.54% | 57.54% | 36.41% |
| V4 - retinal crop + CLAHE | 78.36% | 43.69% | 59.27% | 34.30% |
| V5 - class-weighted training | 75.27% | 31.07% | 58.80% | 28.56% |

V3 was selected because it had the strongest APTOS accuracy among these tested versions and the
best external IDRiD accuracy of the three. External IDRiD performance is substantially lower than
internal APTOS performance.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Backend/model readiness; does not expose internal paths or load errors |
| `GET /api/model-info` | Class order, architecture, and Grad-CAM availability |
| `POST /api/predict` | Multipart `file` field (JPG/JPEG/PNG, maximum 10 MB); returns prediction, class probabilities, quality checks, uncertainty, screening-support rules, and Grad-CAM URLs |
| `GET /api/gradcam/{filename}` | Generated Grad-CAM overlay or heatmap image |

The prediction endpoint rejects unsupported extensions, extension/content mismatches, corrupt
images, oversized uploads, unavailable models, and invalid model probability outputs. API errors
are returned without stack traces or server filesystem paths. `/docs` is available for interactive
API documentation.

## Validation

Run the focused Python tests from the project root:

```powershell
.\venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
```

This includes heuristic quality cases, controlled uncertainty/rule cases, API health, model
prediction, class-probability shape, Grad-CAM assets, and upload validation. The real-model test
requires the included model and installed backend dependencies.

Run the standalone real-model inference check:

```powershell
cd backend
python selftest.py
# Or provide an image: python selftest.py path\to\fundus_image.jpg
```

Run frontend report and evaluation-availability tests:

```powershell
cd frontend
npm test
```

Build the frontend:

```powershell
cd frontend
npm run build
```

### Verification and implementation status

Final local verification:

- `python -m unittest discover -s tests -v` — 29 backend tests passed, including model loading,
  API prediction and upload validation, image-quality rules, and uncertainty rules.
- `python -m unittest discover -s tests -p test_dataset_inventory.py -v` — 10 DDR inventory tests
  passed using synthetic temporary fixtures.
- From `backend\`, `python selftest.py` — the included model loaded and produced five valid,
  normalized class probabilities and Grad-CAM assets for its built-in synthetic input.
- From `frontend\`, `npm test` — 5 tests passed, covering PDF report generation and evaluation-data
  availability behavior.
- From `frontend\`, `npm run build` — Vite production build succeeded.
- Live local check — `/api/health` reported `ok`/`trained`, `/api/model-info` reported the model and
  Grad-CAM available, and `/docs` returned HTTP 200.
- Browser smoke check — the app showed the backend as ready; a generated synthetic noise image
  (not a retinal or patient image) passed through upload, prediction, and result display with five
  probabilities and Grad-CAM. The history entry appeared and cleared, and the experiment page
  displayed the recorded APTOS/IDRiD values and unavailable-metric notices.

These checks verify software execution, not prediction validity on real retinal images, clinical
performance, or generalization. No real patient image or local DDR dataset was used.

**Implemented and tested:** model startup/inference path, health/model-info and prediction APIs,
upload validation, image-quality and uncertainty rules, Grad-CAM generation, PDF report creation,
session-only history and clearing, experiment-page display, and production frontend build.

**Implemented but not clinically validated:** model predictions, heuristic quality checks,
uncertainty thresholds, screening-priority guidance, and Grad-CAM interpretations. No representative
real-image or clinical validation was performed in this verification.

**Not implemented/completed:** lesion detection or segmentation; DDR archive download or inspection;
per-class evaluation, confusion matrices, and probability calibration. The DDR inventory tool is
implemented and unit-tested, but no real DDR data was inspected.

### Session history and reliability notes

- The reliability page reuses only recorded aggregate APTOS/IDRiD accuracy and macro F1. The project
  contains no saved per-image labels, predicted classes, or probability vectors, so per-class
  precision/recall/F1, confusion matrices, reliability diagrams, ECE, and Brier score remain
  explicitly unavailable.
- Reliability metric definitions and the exact evaluation data needed to calculate them are
  documented in the interface. Model confidence is not described as calibrated probability or
  clinical certainty.
- Screening history is kept in application memory for the current page session and cleared on
  reload, close, or user action. There is no authentication or persistent database in this project,
  so no backend history endpoint or patient identity storage was introduced.
- PDF reports are generated in the browser; they include a prediction summary, class probabilities,
  quality warnings, an available Grad-CAM image, limitations, and the medical disclaimer. The
  uploaded source image is not stored in history or embedded in the report.
- Frontend unit tests verify PDF generation, malformed report inputs, unavailable temporary
  Grad-CAM handling, and absent evaluation data. The final browser smoke check verified history
  display and clearing; PDF report downloading was not exercised in that browser check.

### Phase I — DDR dataset archive inspection

The standalone `tools/inspect_ddr_dataset.py` script inventories an already downloaded and
extracted dataset. It recursively records decodable image files and dimensions, split-directory
names, candidate mask images, and structurally parseable VOC XML, COCO JSON, CSV, or YOLO box
annotations. A filename alone is never enough to mark a mask verified: the script checks pixel
content, a unique image match, and equal dimensions. Bounding boxes are checked against image
dimensions when possible. “Verified” means structurally paired/parsed only; it does not validate
the clinical correctness of annotations or class meanings.

No DDR archive is included in this project. The [DDR dataset repository](https://github.com/nkicsl/DDR-dataset)
provides the [Google Drive folder](https://drive.google.com/drive/folders/1z6tSFmxW_aNayUqVxx6h6bY4kwGzUTEC)
and [Baidu Drive folder](https://pan.baidu.com/s/1560JK2pzxTN9Ny1TcmNasQ) (the repository lists
access code `ue0t`), states the archive is distributed in ten chunks, and gives the archive-assembly
instructions. Use those repository-provided sources; this project does not download or redistribute
the dataset. The repository states that dataset images, annotations, labels, and related materials are under **CC BY-NC-SA 4.0**:
attribute the authors and cite the paper, restrict use to non-commercial purposes, and check the
ShareAlike condition for adaptations. Check the license files and terms accompanying your actual
download as well.

Keep the extracted hierarchy intact and place the dataset under `data\DDR` at the project root.
The source repository description does not specify a single required extracted directory layout;
the inspector discovers files recursively. If extraction creates an additional enclosing folder,
use the directory containing the extracted dataset task folders as `--root`. Do not copy reports
into the dataset directory.

```powershell
# From the project root, save the ten official chunks in data\DDR-download.
New-Item -ItemType Directory -Force data\DDR-download, data\DDR
$parts = Get-ChildItem -LiteralPath data\DDR-download -Filter 'DDR-dataset.zip.0*' -File | Sort-Object Name
if ($parts.Count -ne 10) { throw "Expected the ten archive chunks described by the DDR repository." }
$archivePath = Join-Path (Resolve-Path data\DDR-download) 'DDR-dataset.zip'
$archive = [System.IO.File]::Create($archivePath)
try {
    foreach ($part in $parts) {
        $inputStream = [System.IO.File]::OpenRead($part.FullName)
        try { $inputStream.CopyTo($archive) } finally { $inputStream.Dispose() }
    }
} finally { $archive.Dispose() }
Expand-Archive -LiteralPath $archivePath -DestinationPath data\DDR

# Inspect the extracted archive; report output is kept separately from the dataset.
python tools\inspect_ddr_dataset.py --root data\DDR --output-dir reports\ddr_inventory

# The script writes reports\ddr_inventory\ddr_inventory.json and
# reports\ddr_inventory\ddr_inventory.md; report output must be outside --root.
```

Run the inventory tests (all files are temporary synthetic fixtures, not clinical dataset records):

```powershell
python -m unittest discover -s tests -p test_dataset_inventory.py -v
```

Review the report, actual license terms, annotation-to-image matches, class evidence, missing or
mismatched files, and the official split before considering segmentation training. Do not train
until the annotation format, class mapping, reuse rights, and a leakage-safe split are confirmed.
The inventory does not modify dataset files, upload images, make predictions, or train a model.

## Limitations

- This is a college deep-learning case-study demonstration, not a certified medical device.
- Recorded test metrics and a model's output probabilities do not establish clinical reliability.
- Per-class APTOS/IDRiD metrics, confusion matrices, reliability diagrams, ECE, and Brier score were
  not available in the recorded artifacts: no paired per-image ground-truth labels and predictions
  or class probabilities were present. Aggregate accuracy and macro F1 cannot reconstruct them.
- Screening history is session-only, not authenticated or persisted; it is cleared on refresh/close
  and is not designed for shared or multi-user devices. Generated PDFs are downloaded to the user
  device and are outside the application's control.
- External IDRiD accuracy is substantially lower than internal APTOS test accuracy.
- Thresholds and priority rules are configurable examples requiring representative-data and
  clinical review; they are not validated.
- Quality statistics can flag some unsuitable images but cannot reliably identify all blur,
  exposure, retinal content, or gradability failures.
- Grad-CAM is a model-attention visualization and may not correspond to lesions or clinical
  reasoning.
- An uncertain result, failed image-quality check, or unavailable/invalid model output must not be
  relied upon for screening decisions.
