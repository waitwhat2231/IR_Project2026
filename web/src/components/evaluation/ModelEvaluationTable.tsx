import type { ModelEvaluation } from "@/api/types";

interface ModelEvaluationTableProps {
  title: string;
  models: ModelEvaluation[];
}

export function ModelEvaluationTable({ title, models }: ModelEvaluationTableProps) {
  const metricKeys = Array.from(
    new Set(models.flatMap((m) => Object.keys(m.aggregate))),
  );

  if (models.length === 0) {
    return (
      <div className="rounded-card border border-dashed border-border-subtle px-4 py-6 text-center text-xs text-text-muted">
        {title} — not run yet
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-card border border-border-subtle">
      <table className="w-full min-w-[640px] border-collapse text-left text-sm">
        <thead>
          <tr className="border-b border-border-subtle bg-surface-elevated">
            <th className="px-3 py-2.5 font-medium text-text-secondary">{title}</th>
            {metricKeys.map((key) => (
              <th
                key={key}
                className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-xs font-medium text-text-secondary"
              >
                {key}
              </th>
            ))}
            <th className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-xs font-medium text-text-secondary">
              elapsed_sec
            </th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <tr key={model.name} className="border-b border-border-subtle last:border-0">
              <td className="px-3 py-2.5 font-medium text-text">{model.name}</td>
              {metricKeys.map((key) => (
                <td
                  key={key}
                  className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-xs tabular-nums text-text-secondary"
                >
                  {key in model.aggregate ? model.aggregate[key].toFixed(4) : "—"}
                </td>
              ))}
              <td className="whitespace-nowrap px-3 py-2.5 text-right font-mono text-xs tabular-nums text-text-muted">
                {model.elapsed_sec !== null ? model.elapsed_sec.toFixed(2) : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
