import { useMemo } from 'react';
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { ClusterInfo, ScatterPoint } from '@/api/types';
import { clusterColor } from '@/constants/clusterColors';

interface ClusterScatterChartProps {
  points: ScatterPoint[];
  clusters: ClusterInfo[];
  /** Owned by the parent so Browse-tab cards and Map-tab chips share one selection. */
  selectedClusterId: number | null;
}

interface ScatterTooltipProps {
  active?: boolean;
  payload?: { payload: ScatterPoint }[];
  clusterLabel: Map<number, string>;
}

export function ClusterScatterChart({
  points,
  clusters,
  selectedClusterId,
}: ClusterScatterChartProps) {
  const pointsByCluster = useMemo(() => {
    const grouped = new Map<number, ScatterPoint[]>();
    for (const point of points) {
      const existing = grouped.get(point.cluster_id);
      if (existing) {
        existing.push(point);
      } else {
        grouped.set(point.cluster_id, [point]);
      }
    }
    return grouped;
  }, [points]);

  const clusterLabel = useMemo(() => {
    const labels = new Map<number, string>();
    for (const cluster of clusters) {
      labels.set(
        cluster.id,
        cluster.top_terms.slice(0, 2).join(', ') ||
          `cluster ${String(cluster.id)}`,
      );
    }
    return labels;
  }, [clusters]);

  return (
    <ResponsiveContainer width="100%" height={480}>
      <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="var(--color-border-subtle)"
        />
        <XAxis
          type="number"
          dataKey="x"
          tick={false}
          axisLine={{ stroke: 'var(--color-border)' }}
          tickLine={false}
          label={{
            value: 'UMAP-1',
            position: 'insideBottom',
            offset: -4,
            fill: 'var(--color-text-muted)',
            fontSize: 11,
          }}
        />
        <YAxis
          type="number"
          dataKey="y"
          tick={false}
          axisLine={{ stroke: 'var(--color-border)' }}
          tickLine={false}
          width={28}
          label={{
            value: 'UMAP-2',
            angle: -90,
            position: 'insideLeft',
            fill: 'var(--color-text-muted)',
            fontSize: 11,
          }}
        />
        <Tooltip
          cursor={{ stroke: 'var(--color-border-strong)' }}
          content={(props) => {
            const payload = isScatterPayload(props.payload)
              ? props.payload
              : undefined;
            return (
              <ScatterTooltip
                active={props.active}
                payload={payload}
                clusterLabel={clusterLabel}
              />
            );
          }}
        />
        {clusters.map((cluster) => {
          const isDimmed =
            selectedClusterId !== null && selectedClusterId !== cluster.id;
          return (
            <Scatter
              key={cluster.id}
              name={`cluster ${String(cluster.id)}`}
              data={pointsByCluster.get(cluster.id) ?? []}
              fill={clusterColor(cluster.id)}
              fillOpacity={isDimmed ? 0.05 : 0.75}
              isAnimationActive={false}
            />
          );
        })}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function isScatterPayload(
  payload: unknown,
): payload is { payload: ScatterPoint }[] {
  return (
    Array.isArray(payload) &&
    payload.every(
      (item) => typeof item === 'object' && item !== null && 'payload' in item,
    )
  );
}

function ScatterTooltip({
  active,
  payload,
  clusterLabel,
}: ScatterTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;

  const point = payload[0].payload;

  return (
    <div className="rounded-md border border-border-strong bg-surface-elevated px-3 py-2 text-xs shadow-xl">
      <p className="font-mono text-text">{point.doc_id}</p>
      <p className="mt-0.5 text-text-muted">
        cluster {point.cluster_id} · {clusterLabel.get(point.cluster_id) ?? '—'}
      </p>
    </div>
  );
}
