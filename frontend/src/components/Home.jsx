import FundusArt from "./FundusArt.jsx";

const FEATURES = [
  ["5-Class Classification", "No DR, Mild, Moderate, Severe and Proliferative diabetic retinopathy."],
  ["AI Confidence Score", "Every result comes with a probability for each of the five classes."],
  ["Grad-CAM Explainability", "A heatmap shows which image regions influenced the model most."],
  ["Image Quality Check", "Dark, blurry or tiny images are flagged before the model is used."],
];

export default function Home({ go, modelInfo }) {
  return (
    <>
      <section className="hero">
        <div className="hero-text">
          <h1>AI-Powered Diabetic Retinopathy Screening</h1>
          <p className="lead">
            Upload a retinal image and analyze it using an EfficientNetB0 deep learning model trained for five-class
            diabetic retinopathy classification.
          </p>
          <div className="actions">
            <button className="btn btn-primary" onClick={() => go("screen")}>Start Screening</button>
            <button className="btn btn-ghost" onClick={() => go("about")}>Learn More</button>
          </div>
          {modelInfo && !modelInfo.trained && (
            <p className="notice" role="alert">
              <strong>Model unavailable.</strong> {modelInfo.message}
            </p>
          )}
        </div>
        <FundusArt />
      </section>

      <section className="section">
        <h2>What the system does</h2>
        <dl className="features">
          {FEATURES.map(([t, d]) => (
            <div key={t} className="feature">
              <dt>{t}</dt>
              <dd>{d}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="section experiment-cta">
        <h2>Model reliability and experiments</h2>
        <p className="muted">Review the recorded APTOS and external IDRiD results for the tested model versions.</p>
        <button className="btn btn-ghost" onClick={() => go("experiments")}>View experiment comparison</button>
      </section>
    </>
  );
}
