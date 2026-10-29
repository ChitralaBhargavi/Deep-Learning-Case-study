export const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const OFFLINE =
  "Cannot reach the RetinaAI backend. Start it with: cd backend, then python -m uvicorn main:app --reload (then reload this page).";

export async function getModelInfo() {
  try {
    const r = await fetch(`${API_URL}/api/model-info`);
    if (!r.ok) throw new Error("bad status");
    return { online: true, ...(await r.json()) };
  } catch {
    return { online: false, trained: false, message: OFFLINE };
  }
}

/** Returns the prediction JSON or throws {message, quality?}. */
export async function predictImage(file) {
  const body = new FormData();
  body.append("file", file);
  let res;
  try {
    res = await fetch(`${API_URL}/api/predict?robustness_check=true`, { method: "POST", body });
  } catch {
    throw { message: OFFLINE };
  }
  let data = null;
  try {
    data = await res.json();
  } catch {
    /* non-JSON error */
  }
  if (!res.ok) {
    const d = data?.detail;
    throw {
      message: (typeof d === "object" && d?.message) || "The server could not analyse this image.",
      code: d?.code,
      quality: d?.image_quality,
      screeningPriority: d?.screening_priority,
    };
  }
  return {
    ...data,
    gradcam_full_url: data.gradcam_url ? `${API_URL}${data.gradcam_url}` : null,
    heatmap_full_url: data.heatmap_url ? `${API_URL}${data.heatmap_url}` : null,
  };
}
