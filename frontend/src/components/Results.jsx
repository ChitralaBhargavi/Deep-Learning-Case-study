import { useState } from "react";
import ProbabilityBars from "./ProbabilityBars.jsx";
import { getRobustnessPassport } from "../reliabilityPassport.js";

function loadImage(src, crossOrigin = false) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    if (crossOrigin) image.crossOrigin = "anonymous";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("An image needed for the overlay could not be loaded."));
    image.src = src;
  });
}

export default function Results({ analysis, go }) {
  const { result: r, previewUrl } = analysis;
  const q = r.image_quality;
  const [opacity, setOpacity] = useState(0.45);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [downloadError, setDownloadError] = useState("");
  const [downloadStatus, setDownloadStatus] = useState("");
  const qClass = q.status === "Good" ? "q-good" : q.status === "Fair" ? "q-fair" : "q-poor";
  const priority = r.screening_priority;
  const robustness = getRobustnessPassport(r.robustness);

  const downloadOverlay = async () => {
    setDownloadError("");
    setDownloadStatus("");
    try {
      const [original, heatmap] = await Promise.all([
        loadImage(previewUrl),
        loadImage(r.heatmap_full_url, true),
      ]);
      const canvas = document.createElement("canvas");
      canvas.width = original.naturalWidth;
      canvas.height = original.naturalHeight;
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Overlay image generation is unavailable in this browser.");
      context.drawImage(original, 0, 0, canvas.width, canvas.height);
      context.globalAlpha = opacity;
      context.drawImage(heatmap, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      if (!blob) throw new Error("The overlay image could not be generated.");
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "retinaai-gradcam-overlay.png";
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      setDownloadStatus("Overlay download started.");
    } catch (error) {
      setDownloadError(error.message || "The overlay could not be downloaded.");
    }
  };

  return (
    <section className="section">
      <h1 className="h1-sm">Screening result</h1>

      <section className="panel wide passport" aria-labelledby="passport-heading">
        <div className="passport-heading">
          <div>
            <h2 id="passport-heading">AI Reliability Passport</h2>
            <p className="small muted">Experimental appearance-sensitivity check; not a measure of medical correctness.</p>
          </div>
          <span className={`q-badge ${qClass}`}>Image quality: {q.status}</span>
        </div>
        <div className="passport-grid">
          <div>
            <span className="passport-label">Original-image prediction</span>
            <strong>{r.predicted_class_name}</strong>
          </div>
          <div>
            <span className="passport-label">Original confidence</span>
            <strong>{r.confidence_percent.toFixed(1)}%</strong>
          </div>
          <div>
            <span className="passport-label">Top-two probability margin</span>
            <strong>{Number.isFinite(r.uncertainty?.top_two_margin_percent)
              ? `${r.uncertainty.top_two_margin_percent.toFixed(1)} pp`
              : "Unavailable"}</strong>
          </div>
          <div>
            <span className="passport-label">Perturbation consistency</span>
            <strong>{r.robustness?.available
              ? `${r.robustness.consistent_variant_count}/${r.robustness.variant_count} variations`
              : "Unavailable"}</strong>
          </div>
          <figure className="passport-heatmap">
            {r.heatmap_full_url
              ? <img src={r.heatmap_full_url} alt="Grad-CAM heatmap for the original-image prediction" />
              : <span className="small muted">Grad-CAM unavailable</span>}
            <figcaption>Original prediction Grad-CAM</figcaption>
          </figure>
        </div>
        <p className={robustness.warning ? "warning-banner" : "small muted"} role={robustness.warning ? "alert" : "status"}>
          <strong>{robustness.label}.</strong>{" "}
          {robustness.warning
            ? "The result is sensitive to image appearance and should not be relied on alone. "
            : ""}
          {robustness.message}
        </p>
        {r.robustness?.available && (
          <>
            <p className="small muted">
              Mean absolute class-probability change: {(r.robustness.mean_absolute_probability_change * 100).toFixed(2)} pp;
              {" "}maximum: {(r.robustness.max_absolute_probability_change * 100).toFixed(2)} pp.
              These describe variation across the tested copies, not calibrated reliability.
            </p>
            <details className="passport-details">
              <summary>View predictions and probabilities for each appearance variation</summary>
              <ul>
                {r.robustness.variants.map((variant) => (
                  <li key={variant.name}>
                    <strong>{variant.name.replaceAll("_", " ")} — {variant.predicted_class_name}:</strong>{" "}
                    {Object.entries(variant.probabilities)
                      .map(([name, value]) => `${name} ${(value * 100).toFixed(1)}%`)
                      .join(", ")}
                  </li>
                ))}
              </ul>
            </details>
          </>
        )}
      </section>

      <div className="result-grid">
        <div className="panel">
          <h2>Model prediction</h2>
          <p className="muted small">Predicted DR grade</p>
          <p className="pred-class">{r.predicted_class_name}</p>
          <p className="muted small">Model confidence for this class</p>
          <p className="pred-conf">{r.confidence_percent.toFixed(1)}%</p>
          <p className="small muted">
            Confidence is the model's output probability, not clinical certainty and not a measure of diagnostic accuracy.
          </p>
          {r.uncertainty?.is_uncertain && (
            <p className="warning-banner" role="alert">
              <strong>Uncertain result — professional review recommended.</strong>
              <span>Top-two probability difference: {r.uncertainty.top_two_margin_percent.toFixed(1)} percentage points.</span>
            </p>
          )}
          <h3>Probability for each class</h3>
          <ProbabilityBars probabilities={r.probabilities} predicted={r.predicted_class_name} />
        </div>

        <div className="panel">
          <h2>Image quality</h2>
          <p><span className={`q-badge ${qClass}`}>{q.status}</span></p>
          <p className="small muted">Heuristic checks only; not a medically validated image-quality model or retinal-image detector.</p>
          <ul className="quality-checks">
            {Object.entries(q.checks || {}).map(([name, check]) => (
              <li key={name}>
                <span className={`check-status check-${check.status}`}>{check.status}</span>
                <span><strong>{name.replaceAll("_", " ")}</strong>: {check.message}</span>
              </li>
            ))}
          </ul>
          <dl className="qmetrics">
            <div><dt>Resolution</dt><dd>{q.metrics.width} × {q.metrics.height}px</dd></div>
            <div><dt>Brightness</dt><dd>{q.metrics.brightness}</dd></div>
            <div><dt>Contrast</dt><dd>{q.metrics.contrast}</dd></div>
            <div><dt>Sharpness</dt><dd>{q.metrics.sharpness}</dd></div>
          </dl>
          {q.warnings?.length > 0 && <ul className="small">{q.warnings.map((m) => <li key={m}>{m}</li>)}</ul>}
          <p className="small"><strong>Suggested action:</strong> {q.suggested_action}</p>
        </div>
      </div>

      <div className="panel wide">
        <h2>Screening-support priority</h2>
        <p className={`priority-badge priority-${priority?.level || "unknown"}`} role="status">
          {priority?.label || "No priority information"}
        </p>
        <ul>{(priority?.reasons || []).map((reason) => <li key={reason}>{reason}</li>)}</ul>
        <p className="small muted">
          This configurable rule-based indicator is not clinically validated and is not a diagnosis, treatment recommendation, or guaranteed referral decision.
          If the result is uncertain or image quality is inadequate, do not rely on it for screening decisions.
          The configured uncertainty thresholds require validation on a representative validation dataset.
        </p>
      </div>

      <div className="panel wide explainability">
        <h2>Model attention visualization</h2>
        <p className="small muted">
          Grad-CAM visualizes image regions that influenced the actual model prediction. It is not a validated lesion detector or proof of a diagnosis.
        </p>
        {r.heatmap_full_url ? (
          <>
            <div className="cam-controls">
              <label className="slider-label" htmlFor="heatmap-opacity">
                Overlay opacity <output htmlFor="heatmap-opacity">{Math.round(opacity * 100)}%</output>
              </label>
              <input id="heatmap-opacity" type="range" min="0" max="100" value={Math.round(opacity * 100)}
                onChange={(event) => setOpacity(Number(event.target.value) / 100)} />
              <label className="toggle-label">
                <input type="checkbox" checked={showHeatmap} onChange={(event) => setShowHeatmap(event.target.checked)} />
                Show heatmap on overlay
              </label>
            </div>
            <div className="cam-compare">
              <figure className="cam-card">
                <img src={previewUrl} alt="Uploaded retinal image used for model prediction" />
                <figcaption>Original image</figcaption>
              </figure>
              <figure className="cam-card">
                <img src={r.heatmap_full_url} alt="Grad-CAM heatmap generated for the predicted class" />
                <figcaption>Grad-CAM heatmap</figcaption>
              </figure>
              <figure className="cam-card">
                <div className="overlay-frame">
                  <img src={previewUrl} alt="" />
                  {showHeatmap && <img className="heat-layer" src={r.heatmap_full_url} alt="" style={{ opacity }} />}
                </div>
                <figcaption>Overlay on original image</figcaption>
              </figure>
            </div>
            <button className="btn btn-ghost" onClick={downloadOverlay}>Download overlay image</button>
            {downloadError && <p className="error-title" role="alert">{downloadError}</p>}
            {downloadStatus && <p className="small" role="status">{downloadStatus}</p>}
          </>
        ) : (
          <p className="notice small" role="status">{r.gradcam_message || "Grad-CAM is unavailable for this prediction."}</p>
        )}
      </div>

      <p className="disclaimer" role="note">{r.disclaimer}</p>
      <div className="actions">
        <button className="btn btn-primary" onClick={() => go("screen")}>Analyze Another Image</button>
      </div>
    </section>
  );
}
