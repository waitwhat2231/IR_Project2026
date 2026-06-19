import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ModelEvaluation } from "@/api/types";

interface MetricsBarChartProps {
  metric: string;
  baselineModels: ModelEvaluation[];
  enhancedModels: ModelEvaluation[];
}

interface ChartRow {
  model: string;
  baseline: number | null;
  enhanced: number | null;
}

export function MetricsBarChart({
  metric,
  baselineModels,
  enhancedModels,
}: MetricsBarChartProps) {
  const rows = useMemo<ChartRow[]>(() => {
    const modelNames = new Set<string>();
    for (const m of baselineModels) modelNames.add(m.name);
    for (const m of enhancedModels) modelNames.add(m.name);

    const baselineByName = new Map(baselineModels.map((m) => [m.name, m]));
    const enhancedByName = new Map(enhancedModels.map((m) => [m.name, m]));

    return Array.from(modelNames).map((name) => ({
      model: name,
      baseline: baselineByName.get(name)?.aggregate[metric] ?? null,
      enhanced: enhancedByName.get(name)?.aggregate[metric] ?? null,
    }));
  }, [metric, baselineModels, enhancedModels]);

  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={rows} margin={{ top: 8, right: 12, left: -8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" />
        <XAxis
          dataKey="model"
          tick={{ fill: "var(--color-text-secondary)", fontSize: 12 }}
          tickLine={false}
          axisLine={{ stroke: "var(--color-border)" }}
        />
        <YAxis
          tick={{ fill: "var(--color-text-secondary)", fontSize: 12 }}
          tickLine={false}
          axisLine={{ stroke: "var(--color-border)" }}
          width={48}
        />
        <Tooltip
          cursor={{ fill: "var(--color-surface-hover)" }}
          contentStyle={{
            background: "var(--color-surface-elevated)",
            border: "1px solid var(--color-border-strong)",
            borderRadius: 8,
            fontSize: 12,
          }}
          labelStyle={{ color: "var(--color-text)" }}
        />
        <Legend wrapperStyle={{ fontSize: 12, color: "var(--color-text-secondary)" }} />
        <Bar
          dataKey="baseline"
          name="Baseline"
          fill="var(--color-info)"
          radius={[4, 4, 0, 0]}
        />
        <Bar
          dataKey="enhanced"
          name="Enhanced"
          fill="var(--color-accent)"
          radius={[4, 4, 0, 0]}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}
