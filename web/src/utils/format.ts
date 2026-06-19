/** Full-precision score display — never round, the report needs exact values. */
export function formatScore(score: number): string {
  return score.toFixed(6);
}

const compactNumberFormatter = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

const exactNumberFormatter = new Intl.NumberFormat("en-US");

export function formatDocCount(count: number | null): string {
  if (count === null) return "unknown";
  return `${exactNumberFormatter.format(count)} docs`;
}

export function formatCompactCount(count: number): string {
  return compactNumberFormatter.format(count);
}
