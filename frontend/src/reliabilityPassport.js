export function getRobustnessPassport(robustness) {
  if (!robustness?.available) {
    return {
      status: "unavailable",
      label: "Sensitivity check unavailable",
      warning: false,
      message: robustness?.message || "No experimental sensitivity result was provided.",
    };
  }

  if (robustness.status === "sensitive") {
    return {
      status: "sensitive",
      label: "Appearance-sensitive result",
      warning: true,
      message: robustness.message,
    };
  }

  if (robustness.status === "consistent") {
    return {
      status: "consistent",
      label: "No class change in tested variations",
      warning: false,
      message: robustness.message,
    };
  }

  return {
    status: "unavailable",
    label: "Sensitivity check unavailable",
    warning: false,
    message: "The experimental sensitivity result was invalid.",
  };
}
