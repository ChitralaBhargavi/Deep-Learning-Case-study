import { experiments } from "../data/experiments.js";
import { calibrationDefinitions, evaluationAvailability } from "../data/evaluationAvailability.js";

function MetricChart({ title, aptosKey, idridKey }) {
  return (
    <section className="panel chart-panel" aria-label={title}>
      <h2>{title}</h2>
      <div className="chart-legend" aria-label="Dataset legend">
        <span><i className="legend-swatch aptos" />APTOS</span>
        <span><i className="legend-swatch idrid" />IDRiD external</span>
      </div>
      <div className="experiment-bars">
        {experiments.map((experiment) => (
          <div className="experiment-row" key={experiment.name}>
            <strong>{experiment.name}</strong>
            {[["APTOS", aptosKey, "aptos"], ["IDRiD", idridKey, "idrid"]].map(([label, key, className]) => (
              <div className="dataset-bar-row" key={label}>
                <span>{label}</span>
                <div className="metric-track" role="img" aria-label={`${label} ${title}: ${experiment[key].toFixed(2)} percent`}>
                  <div className={`metric-fill ${className}`} style={{ width: `${experiment[key]}%` }} />
                </div>
                <span className="metric-value">{experiment[key].toFixed(2)}%</span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </section>
  );
}

export default function Experiments() {
  return (
    <section className="section">
      <h1 className="h1-sm">Model reliability and experiments</h1>
      <p className="muted">
        Recorded results from the documented experiments. These are not live measurements and do not establish clinical performance.
      </p>
      <p className="small">
        APTOS is the internal test evaluation; IDRiD is the external evaluation. The selected V3 experiment records
        79.27% APTOS accuracy and 48.54% IDRiD accuracy. These aggregate results do not provide class-specific performance.
      </p>
      <div className="result-grid experiment-charts">
        <MetricChart title="Accuracy comparison" aptosKey="aptosAccuracy" idridKey="idridAccuracy" />
        <MetricChart title="Macro F1 comparison" aptosKey="aptosMacroF1" idridKey="idridMacroF1" />
      </div>
      <div className="panel wide">
        <h2>Recorded experiment results</h2>
        <div className="table-wrap">
          <table className="table experiment-table">
            <caption>APTOS test and IDRiD external results</caption>
            <thead><tr>
              <th scope="col">Experiment</th><th scope="col">APTOS test accuracy</th>
              <th scope="col">IDRiD external accuracy</th><th scope="col">APTOS macro F1</th>
              <th scope="col">IDRiD macro F1</th>
            </tr></thead>
            <tbody>{experiments.map((item) => (
              <tr key={item.name}>
                <th scope="row">{item.name}</th>
                <td>{item.aptosAccuracy.toFixed(2)}%</td><td>{item.idridAccuracy.toFixed(2)}%</td>
                <td>{item.aptosMacroF1.toFixed(2)}%</td><td>{item.idridMacroF1.toFixed(2)}%</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </div>
      <div className="panel wide">
        <h2>Interpretation and limitations</h2>
        <p>
          V3 was selected because it had the strongest APTOS test accuracy among these tested versions and the best external IDRiD accuracy of the three.
        </p>
        <p className="warning-banner" role="note">
          External IDRiD performance is substantially lower than internal APTOS performance. Results may not transfer across datasets, cameras, populations, or clinical settings.
        </p>
        <p className="small muted">These figures are recorded experiment results, not live measurements, new validation, or clinical claims.</p>
      </div>
      <section className="panel wide" aria-labelledby="class-metrics-heading">
        <h2 id="class-metrics-heading">Class-wise performance and confusion matrices</h2>
        <div className="evaluation-requirements">
          <div>
            <h3>Per-class precision, recall, and F1</h3>
            <p>Requires ground-truth labels and predicted class for each evaluated image for both APTOS and IDRiD.</p>
            <span className="availability-badge">
              {evaluationAvailability.perClassMetrics.available ? "Recorded metrics available" : "Not recorded"}
            </span>
          </div>
          <div>
            <h3>Confusion matrices</h3>
            <p>Requires per-image ground-truth and predicted classes using the same five-class mapping.</p>
            <span className="availability-badge">
              {evaluationAvailability.confusionMatrices.available ? "Recorded matrix available" : "Not recorded"}
            </span>
          </div>
        </div>
        {!evaluationAvailability.perClassMetrics.available && !evaluationAvailability.confusionMatrices.available && (
          <p className="notice">
            No class-level precision, recall, F1-score, or confusion-matrix values are displayed because the necessary per-image labels and predictions were not found.
          </p>
        )}
      </section>
      <section className="panel wide" aria-labelledby="calibration-heading">
        <h2 id="calibration-heading">Calibration and probability reliability</h2>
        {!evaluationAvailability.calibration.available && (
          <p className="notice">
            Reliability diagrams, Expected Calibration Error (ECE), and Brier scores are not reported: there are no recorded per-image ground-truth labels paired with the model&apos;s five-class probabilities.
          </p>
        )}
        <div className="calibration-definitions">
          {calibrationDefinitions.map((item) => (
            <div key={item.name}>
              <h3>{item.name}</h3>
              <p>{item.definition}</p>
              <span className="availability-badge">
                {evaluationAvailability.calibration.available ? "Recorded metric available" : "Unavailable — evaluation data missing"}
              </span>
            </div>
          ))}
        </div>
        <p className="small muted">
          Multiclass calibration should be evaluated on held-out, representative data with the exact ground-truth class and all five predicted probabilities for every image. ECE also depends on a documented binning scheme. None of these calibration results should be inferred from model confidence or aggregate accuracy/F1.
        </p>
      </section>
      <p className="disclaimer">
        Model confidence is the model&apos;s output probability for a class. Calibration measures whether predicted probabilities align with empirical outcomes; clinical certainty is a separate matter. No calibration or clinical validation is claimed here.
      </p>
    </section>
  );
}
