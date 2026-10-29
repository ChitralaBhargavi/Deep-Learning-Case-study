/* Stylised illustration only - not a real retinal image and not model output. */
export default function FundusArt() {
  return (
    <figure className="fundus" aria-hidden="true">
      <svg viewBox="0 0 400 400" role="img">
        <defs>
          <radialGradient id="ret" cx="45%" cy="48%" r="60%">
            <stop offset="0%" stopColor="#E9A94A" />
            <stop offset="55%" stopColor="#C4562E" />
            <stop offset="100%" stopColor="#4E160F" />
          </radialGradient>
          <radialGradient id="disc" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#FBE9A8" />
            <stop offset="100%" stopColor="#EDB750" />
          </radialGradient>
          <radialGradient id="heat" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#ff2d1a" stopOpacity="0.85" />
            <stop offset="45%" stopColor="#ffd400" stopOpacity="0.5" />
            <stop offset="100%" stopColor="#23c4a5" stopOpacity="0" />
          </radialGradient>
          <clipPath id="clip"><circle cx="200" cy="200" r="188" /></clipPath>
        </defs>
        <circle cx="200" cy="200" r="190" fill="#0b1417" />
        <g clipPath="url(#clip)">
          <circle cx="200" cy="200" r="188" fill="url(#ret)" />
          <g fill="none" stroke="#7E2417" strokeLinecap="round" opacity="0.85">
            <path d="M278 196 C240 170 205 150 150 120" strokeWidth="5" />
            <path d="M278 196 C236 210 196 236 160 290" strokeWidth="5" />
            <path d="M278 196 C225 190 170 196 100 190" strokeWidth="4" />
            <path d="M205 150 C190 120 188 100 170 70" strokeWidth="3" />
            <path d="M196 236 C190 266 196 296 214 330" strokeWidth="3" />
            <path d="M160 205 C130 215 112 238 92 262" strokeWidth="2.5" />
          </g>
          <ellipse cx="282" cy="196" rx="26" ry="32" fill="url(#disc)" />
          <circle cx="170" cy="204" r="16" fill="#6B1E12" opacity="0.55" />
          <circle className="heat" cx="150" cy="150" r="82" fill="url(#heat)" />
        </g>
        <circle cx="200" cy="200" r="190" fill="none" stroke="#1d3a44" strokeWidth="4" />
      </svg>
      <figcaption>Stylised illustration of a fundus photograph with an attention overlay</figcaption>
    </figure>
  );
}
