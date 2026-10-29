const COLORS = ["#2E8B7A", "#9DBE5A", "#E3A33B", "#D2692F", "#A3341F"];

export default function ProbabilityBars({ probabilities, predicted }) {
  return (
    <ul className="bars" aria-label="Class probabilities">
      {Object.entries(probabilities).map(([name, p], i) => {
        const pct = p * 100;
        const top = name === predicted;
        return (
          <li key={name} className={top ? "bar-row top" : "bar-row"}>
            <span className="bar-name">{name}</span>
            <span className="bar-track" role="img" aria-label={`${name}: ${pct.toFixed(1)} percent`}>
              <span className="bar-fill" style={{ width: `${Math.max(pct, 0.6)}%`, background: COLORS[i] }} />
            </span>
            <span className="bar-pct">{pct.toFixed(1)}%</span>
          </li>
        );
      })}
    </ul>
  );
}
