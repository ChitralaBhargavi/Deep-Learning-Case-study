export default function Header({ page, go, modelInfo, hasResult }) {
  const links = [
    ["home", "Home"],
    ["screen", "Screening"],
    ...(hasResult ? [["results", "Results"]] : []),
    ["experiments", "Experiments"],
    ["history", "History"],
    ["about", "About"],
  ];
  let status = { cls: "pill", text: "Checking model…" };
  if (modelInfo) {
    if (!modelInfo.online) status = { cls: "pill pill-warn", text: "Backend offline" };
    else if (modelInfo.trained) status = { cls: "pill pill-ok", text: "Model ready" };
    else status = { cls: "pill pill-warn", text: "Model unavailable" };
  }
  return (
    <header className="header">
      <button className="brand" onClick={() => go("home")} aria-label="RetinaAI home">
        <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true">
          <circle cx="16" cy="16" r="14" fill="var(--teal)" />
          <circle cx="16" cy="16" r="8" fill="none" stroke="var(--amber)" strokeWidth="1.5" />
          <circle cx="16" cy="16" r="3.2" fill="var(--amber)" />
        </svg>
        <span>RetinaAI</span>
      </button>
      <nav aria-label="Main">
        {links.map(([key, label]) => (
          <button
            key={key}
            className={page === key ? "navlink active" : "navlink"}
            aria-current={page === key ? "page" : undefined}
            onClick={() => go(key)}
          >
            {label}
          </button>
        ))}
      </nav>
      <span className={status.cls} role="status">{status.text}</span>
    </header>
  );
}
