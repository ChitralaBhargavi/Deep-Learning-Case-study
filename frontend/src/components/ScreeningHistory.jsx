import { useState } from "react";
import { downloadScreeningReport, generateScreeningReport } from "../reports/screeningReport.js";

const CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"];

function ReportButton({ entry }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");

  const download = async () => {
    setBusy(true);
    setError("");
    setStatus("");
    try {
      const report = await generateScreeningReport(entry);
      downloadScreeningReport(report, entry.timestamp);
      setStatus("PDF report generated. Check your browser downloads.");
    } catch (reason) {
      setError(reason.message || "The PDF report could not be generated.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <button className="btn btn-ghost" onClick={download} disabled={busy}>
        {busy ? "Preparing report…" : "Download PDF report"}
      </button>
      {error && <p className="error-title" role="alert">{error}</p>}
      {status && <p className="small" role="status">{status}</p>}
    </>
  );
}

export default function ScreeningHistory({ entries, clearHistory }) {
  return (
    <section className="section">
      <div className="history-heading">
        <div>
          <h1 className="h1-sm">Screening history</h1>
          <p className="muted">History is held only in this page session. It is not uploaded or saved to a database.</p>
        </div>
        {entries.length > 0 && (
          <button className="btn btn-ghost" onClick={clearHistory}>Clear session history</button>
        )}
      </div>
      <p className="notice">
        Uploaded retinal images are not retained in history. A generated Grad-CAM image may be included in a report while its temporary backend file remains available.
      </p>
      {entries.length === 0 ? (
        <div className="panel history-empty">
          <h2>No screenings in this session</h2>
          <p className="muted">Completed screening summaries will appear here until the page is reloaded or you clear the history.</p>
        </div>
      ) : (
        <div className="history-list" aria-label="Previous screening sessions">
          {entries.map((entry) => (
            <article className="panel history-entry" key={entry.id}>
              <div className="history-entry-heading">
                <div>
                  <p className="muted small">{new Date(entry.timestamp).toLocaleString()}</p>
                  <h2>{entry.predictedClassName}</h2>
                </div>
                <span className={`q-badge ${entry.imageQuality?.status === "Good" ? "q-good" : entry.imageQuality?.status === "Fair" ? "q-fair" : "q-poor"}`}>
                  Image quality: {entry.imageQuality?.status || "Unavailable"}
                </span>
              </div>
              <p>Model confidence: {Number(entry.confidencePercent).toFixed(1)}%</p>
              <p className={entry.isUncertain ? "history-uncertain" : "muted"}>
                {entry.isUncertain ? "Uncertain result — professional review recommended." : "Not flagged uncertain by configured thresholds."}
              </p>
              <h3>Class probabilities</h3>
              <ul className="history-probabilities">
                {CLASS_NAMES.map((name) => (
                  <li key={name}><span>{name}</span><strong>{(entry.probabilities[name] * 100).toFixed(1)}%</strong></li>
                ))}
              </ul>
              {(entry.imageQuality?.warnings?.length > 0 || entry.imageQuality?.issues?.length > 0) && (
                <>
                  <h3>Quality warnings</h3>
                  <ul className="small">
                    {[...(entry.imageQuality.issues || []), ...(entry.imageQuality.warnings || [])].map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                </>
              )}
              <ReportButton entry={entry} />
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
