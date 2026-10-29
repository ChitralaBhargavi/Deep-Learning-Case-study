import test from "node:test";
import assert from "node:assert/strict";
import { evaluationAvailability } from "../src/data/evaluationAvailability.js";
import { generateScreeningReport } from "../src/reports/screeningReport.js";

function validSession() {
  return {
    id: "session-1",
    timestamp: "2026-10-24T12:00:00.000Z",
    predictedClassName: "Moderate",
    confidencePercent: 65,
    probabilities: {
      "No DR": 0.1,
      Mild: 0.1,
      Moderate: 0.65,
      Severe: 0.1,
      "Proliferative DR": 0.05,
    },
    imageQuality: {
      status: "Fair",
      metrics: { width: 640, height: 480 },
      warnings: ["The image may be blurry."],
      issues: [],
    },
    isUncertain: false,
    uncertaintyMarginPercent: 50,
    heatmapUrl: null,
  };
}

test("generates a non-empty PDF report for a valid screening entry", async () => {
  const pdf = await generateScreeningReport(validSession());
  assert.ok(pdf instanceof Blob);
  assert.ok(pdf.size > 500);
  const signature = new Uint8Array(await pdf.slice(0, 5).arrayBuffer());
  assert.equal(new TextDecoder().decode(signature), "%PDF-");
});

test("rejects an absent screening entry", async () => {
  await assert.rejects(() => generateScreeningReport(null), {
    name: "TypeError",
    message: "A screening history entry is required to generate a report.",
  });
});

test("rejects malformed probabilities and invalid confidence", async () => {
  const badProbabilities = validSession();
  badProbabilities.probabilities.Mild = 0.7;
  await assert.rejects(() => generateScreeningReport(badProbabilities), /five valid class probabilities/);

  const badConfidence = validSession();
  badConfidence.confidencePercent = Number.NaN;
  await assert.rejects(() => generateScreeningReport(badConfidence), /invalid confidence/);
});

test("still generates a PDF when a temporary Grad-CAM image is unavailable", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response("missing", { status: 404 });
  try {
    const entry = { ...validSession(), heatmapUrl: "https://example.invalid/temporary-heatmap.png" };
    const pdf = await generateScreeningReport(entry);
    assert.ok(pdf.size > 500);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("evaluation data stays explicitly unavailable without labels and predictions", () => {
  assert.equal(evaluationAvailability.perClassMetrics.available, false);
  assert.equal(evaluationAvailability.confusionMatrices.available, false);
  assert.equal(evaluationAvailability.calibration.available, false);
  assert.ok(evaluationAvailability.calibration.required.includes("ground-truth labels"));
  assert.ok(evaluationAvailability.calibration.required.includes("five-class predicted probabilities for every evaluated image"));
});
