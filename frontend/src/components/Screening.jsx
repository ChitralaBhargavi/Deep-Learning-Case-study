import { useEffect, useRef, useState } from "react";
import { predictImage } from "../api.js";

const OK_TYPES = ["image/jpeg", "image/png"];
const MAX_MB = 10;

export default function Screening({ modelInfo, onResult }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);
  const transferredPreview = useRef(null);

  useEffect(() => () => {
    if (previewUrl && previewUrl !== transferredPreview.current) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  const choose = (f) => {
    setError(null);
    if (!f) return;
    const ext = f.name.toLowerCase().split(".").pop();
    if (!OK_TYPES.includes(f.type) && !["jpg", "jpeg", "png"].includes(ext)) {
      setError({ message: "Unsupported file type. Please choose a JPG, JPEG or PNG image." });
      return;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setError({ message: `The image is larger than ${MAX_MB} MB. Please choose a smaller file.` });
      return;
    }
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
  };

  const reset = () => {
    setFile(null);
    setPreviewUrl(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const analyze = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await predictImage(file);
      transferredPreview.current = previewUrl;
      onResult(result, previewUrl);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  };

  const noModel = modelInfo && !modelInfo.trained;

  return (
    <section className="section narrow">
      <h1 className="h1-sm">Upload a retinal image</h1>
      {noModel && (
        <p className="notice" role="alert">
          <strong>Model unavailable.</strong> {modelInfo.message} You can still preview an image, but
          no prediction can be produced until the backend model is loaded.
        </p>
      )}

      {!file ? (
        <div
          className={dragging ? "drop drop-on" : "drop"}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); choose(e.dataTransfer.files?.[0]); }}
        >
          <p className="drop-title">Drag and drop a fundus image here</p>
          <p className="muted">or</p>
          <button className="btn btn-primary" onClick={() => inputRef.current?.click()}>Browse files</button>
          <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png" hidden
                 onChange={(e) => choose(e.target.files?.[0])} />
          <p className="muted small">Nothing is uploaded until you click Analyze Image.</p>
        </div>
      ) : (
        <div className="preview-wrap">
          <div className={loading ? "preview scanning" : "preview"}>
            <img src={previewUrl} alt="Selected retinal image preview" />
            {loading && <span className="scan-line" />}
          </div>
          <div className="preview-side">
            <p className="file-name">{file.name}</p>
            <p className="muted small">{(file.size / 1024).toFixed(0)} KB</p>
            <div className="actions">
              <button className="btn btn-primary" onClick={analyze} disabled={loading}>
                {loading ? "Analyzing…" : "Analyze Image"}
              </button>
              <button className="btn btn-ghost" onClick={reset} disabled={loading}>Choose another</button>
            </div>
            {loading && <p className="muted" role="status">Checking quality, classifying and building the Grad-CAM view…</p>}
          </div>
        </div>
      )}

      {error && (
        <div className="error" role="alert">
          <p className="error-title">{error.message}</p>
          {error.quality?.messages?.length > 0 && (
            <ul>{error.quality.messages.map((m) => <li key={m}>{m}</li>)}</ul>
          )}
          {error.quality?.checks && (
            <ul className="check-list" aria-label="Image-quality checks">
              {Object.entries(error.quality.checks).map(([name, check]) => (
                <li key={name}><strong>{name} — {check.status}:</strong> {check.message}</li>
              ))}
            </ul>
          )}
          {error.quality?.suggested_action && <p>{error.quality.suggested_action}</p>}
          {error.screeningPriority && (
            <p className="warning-banner" role="alert">
              <strong>{error.screeningPriority.label}</strong>
              {error.screeningPriority.reasons?.join(" ")}
            </p>
          )}
          {error.code === "poor_image_quality" && (
            <button className="btn btn-ghost" onClick={reset}>Choose another image</button>
          )}
        </div>
      )}

      <div className="requirements">
        <h2>Image requirements</h2>
        <ul>
          <li>Supported formats: JPG, JPEG, PNG (up to {MAX_MB} MB)</li>
          <li>A colour fundus photograph of one eye, with the retina centred</li>
          <li>Prefer 224 × 224 pixels or higher; very small images may be rejected. Use an evenly lit image in focus.</li>
          <li>No screenshots, scans of documents or photos of faces</li>
        </ul>
      </div>
    </section>
  );
}
