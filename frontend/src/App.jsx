import { lazy, Suspense, useEffect, useState } from "react";
import { getModelInfo } from "./api.js";
import Header from "./components/Header.jsx";
import Home from "./components/Home.jsx";
import Screening from "./components/Screening.jsx";
import Results from "./components/Results.jsx";
import About from "./components/About.jsx";
import Experiments from "./components/Experiments.jsx";

const ScreeningHistory = lazy(() => import("./components/ScreeningHistory.jsx"));
let nextHistoryEntryId = 0;

export default function App() {
  const [page, setPage] = useState("home");
  const [modelInfo, setModelInfo] = useState(null);
  const [analysis, setAnalysis] = useState(null); // { result, previewUrl }
  const [history, setHistory] = useState([]);

  useEffect(() => {
    getModelInfo().then(setModelInfo);
  }, []);
  useEffect(() => () => {
    if (analysis?.previewUrl) URL.revokeObjectURL(analysis.previewUrl);
  }, [analysis?.previewUrl]);

  const go = (p) => {
    setPage(p);
    window.scrollTo({ top: 0 });
  };
  const onResult = (result, previewUrl) => {
    const historyEntry = {
      id: `${Date.now()}-${nextHistoryEntryId++}`,
      timestamp: new Date().toISOString(),
      predictedClassName: result.predicted_class_name,
      confidencePercent: result.confidence_percent,
      probabilities: result.probabilities,
      imageQuality: result.image_quality,
      isUncertain: result.uncertainty?.is_uncertain ?? Boolean(result.low_confidence),
      uncertaintyMarginPercent: result.uncertainty?.top_two_margin_percent ?? null,
      screeningPriority: result.screening_priority,
      heatmapUrl: result.heatmap_full_url,
      modelFile: result.model_file,
    };
    setHistory((entries) => [historyEntry, ...entries]);
    setAnalysis({ result, previewUrl });
    go("results");
  };

  return (
    <>
      <a className="skip" href="#main">Skip to content</a>
      <Header page={page} go={go} modelInfo={modelInfo} hasResult={!!analysis} />
      <main id="main">
        {page === "home" && <Home go={go} modelInfo={modelInfo} />}
        {page === "screen" && <Screening modelInfo={modelInfo} onResult={onResult} />}
        {page === "results" && analysis && (
          <Results analysis={analysis} go={go} />
        )}
        {page === "about" && <About modelInfo={modelInfo} />}
        {page === "experiments" && <Experiments />}
        {page === "history" && (
          <Suspense fallback={<p className="section" role="status">Loading screening history and report tools…</p>}>
            <ScreeningHistory entries={history} clearHistory={() => setHistory([])} />
          </Suspense>
        )}
      </main>
      <footer className="footer">
        <p>
          This application is intended for AI-assisted screening and educational/research purposes. It
          does not provide a medical diagnosis. Users should consult a qualified ophthalmologist for
          clinical evaluation.
        </p>
      </footer>
    </>
  );
}
