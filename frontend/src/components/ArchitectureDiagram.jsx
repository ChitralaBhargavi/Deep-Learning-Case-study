const STEPS = [
  ["Fundus image", "JPG / PNG"],
  ["Quality check", "brightness, contrast, size"],
  ["Preprocess", "224×224 RGB, float32"],
  ["EfficientNetB0", "convolutional features"],
  ["Global avg pooling", "+ batch norm, dropout 0.3"],
  ["Dense 256", "ReLU + Dropout 0.2"],
  ["Softmax", "5 class probabilities"],
];

export default function ArchitectureDiagram() {
  const w = 150, gap = 18, h = 64;
  return (
    <div className="diagram" role="img" aria-label="Model architecture: image, quality check, preprocessing, EfficientNetB0, pooling, dense layer, softmax; Grad-CAM branches from the convolutional features">
      <svg viewBox={`0 0 ${STEPS.length * (w + gap)} 170`} width="100%">
        {STEPS.map(([t, s], i) => {
          const x = i * (w + gap);
          return (
            <g key={t}>
              <rect x={x} y="10" width={w} height={h} rx="6" fill={i === 3 ? "var(--teal)" : "var(--mist)"} stroke="var(--line)" />
              <text x={x + w / 2} y="40" textAnchor="middle" fontSize="14" fontWeight="600" fill={i === 3 ? "#fff" : "var(--ink)"}>{t}</text>
              <text x={x + w / 2} y="60" textAnchor="middle" fontSize="11" fill={i === 3 ? "#d6ecee" : "var(--muted)"}>{s}</text>
              {i < STEPS.length - 1 && <path d={`M${x + w + 3} 42 h${gap - 6}`} stroke="var(--muted)" strokeWidth="1.5" markerEnd="url(#ah)" />}
            </g>
          );
        })}
        <defs>
          <marker id="ah" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
            <path d="M0 0 L6 3.5 L0 7 z" fill="var(--muted)" />
          </marker>
        </defs>
        <path d={`M${3 * (w + gap) + w / 2} ${10 + h} V112`} fill="none" stroke="var(--amber)" strokeWidth="1.5" strokeDasharray="4 3" />
        <rect x={3 * (w + gap) + w / 2 - 80} y="112" width="160" height="42" rx="6" fill="#fff" stroke="var(--amber)" />
        <text x={3 * (w + gap) + w / 2} y="131" textAnchor="middle" fontSize="13" fontWeight="600" fill="var(--ink)">Grad-CAM</text>
        <text x={3 * (w + gap) + w / 2} y="146" textAnchor="middle" fontSize="11" fill="var(--muted)">gradients of top_activation</text>
      </svg>
    </div>
  );
}
