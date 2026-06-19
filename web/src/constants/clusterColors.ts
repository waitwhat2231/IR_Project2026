const GOLDEN_ANGLE = 137.508;
const SATURATION = 72;
const LIGHTNESS = 62;

/**
 * Deterministic, evenly-distinguishable color per cluster id. Uses the
 * golden-angle technique instead of a fixed palette array so it scales to
 * any n_clusters the backend reports, not just the current k=20 default.
 */
export function clusterColor(clusterId: number): string {
  const hue = (clusterId * GOLDEN_ANGLE) % 360;
  return `hsl(${String(hue)}deg ${String(SATURATION)}% ${String(LIGHTNESS)}%)`;
}
