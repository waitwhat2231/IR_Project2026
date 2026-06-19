import { useState } from "react";
import { AlertTriangle, Boxes, LayoutGrid, Map as MapIcon } from "lucide-react";
import { useDatasets } from "@/hooks/useDatasets";
import { useClusters } from "@/hooks/useClusters";
import { useClusterScatter } from "@/hooks/useClusterScatter";
import { Segmented } from "@/components/ui/Segmented";
import { Badge } from "@/components/ui/Badge";
import { ClusterCard } from "@/components/clusters/ClusterCard";
import { ClusterScatterChart } from "@/components/clusters/ClusterScatterChart";
import { clusterColor } from "@/constants/clusterColors";
import { formatCompactCount } from "@/utils/format";
import clsx from "clsx";

type SubView = "browse" | "map";

export function ClustersView() {
  const [subView, setSubView] = useState<SubView>("browse");
  const [selectedClusterId, setSelectedClusterId] = useState<number | null>(null);

  const datasetsQuery = useDatasets();
  const dataset = datasetsQuery.data?.datasets[0];

  const clustersQuery = useClusters(dataset?.name ?? "", !!dataset);
  const scatterQuery = useClusterScatter(dataset?.name ?? "", !!dataset && subView === "map");

  if (!dataset) return null;

  if (clustersQuery.isPending) {
    return (
      <div className="mx-auto max-w-5xl px-5 py-10">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }, (_, i) => (
            <div
              key={i}
              className="h-40 animate-pulse rounded-card border border-border-subtle bg-surface/60"
            />
          ))}
        </div>
      </div>
    );
  }

  if (clustersQuery.isError) {
    return (
      <div className="mx-auto max-w-2xl px-5 py-20">
        <div className="flex flex-col items-center gap-4 rounded-card border border-dashed border-border bg-surface/40 px-8 py-14 text-center">
          <AlertTriangle size={26} className="text-text-muted" />
          <div className="space-y-1.5">
            <p className="text-sm font-medium text-text">
              Clustering hasn't been run for this dataset yet
            </p>
            <p className="text-xs leading-relaxed text-text-secondary">
              {clustersQuery.error instanceof Error
                ? clustersQuery.error.message
                : "Run python offline/step10_cluster.py, then reopen this tab."}
            </p>
          </div>
        </div>
      </div>
    );
  }

  const { clusters, n_clusters, n_docs, embedding_source, generated_at } = clustersQuery.data;

  return (
    <div className="mx-auto max-w-5xl px-5 py-8">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2.5">
          <Boxes size={18} className="text-accent" />
          <h1 className="text-lg font-semibold text-text">Document clusters</h1>
        </div>
        <div className="flex flex-wrap items-center gap-1.5 font-mono text-xs text-text-muted">
          <Badge tone="neutral" mono>{n_clusters} clusters</Badge>
          <Badge tone="neutral" mono>{formatCompactCount(n_docs)} docs</Badge>
          <Badge tone="neutral" mono>{embedding_source}</Badge>
          <span className="hidden sm:inline">· generated {generated_at}</span>
        </div>
      </div>

      <div className="mb-6 w-full max-w-xs">
        <Segmented
          aria-label="Cluster view"
          value={subView}
          onChange={setSubView}
          options={[
            { value: "browse", label: "Browse" },
            { value: "map", label: "Map" },
          ]}
        />
      </div>

      {subView === "browse" ? (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {clusters.map((cluster) => (
            <ClusterCard
              key={cluster.id}
              cluster={cluster}
              selected={selectedClusterId === cluster.id}
              onClick={() => {
                setSelectedClusterId((current) =>
                  current === cluster.id ? null : cluster.id,
                );
              }}
            />
          ))}
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-1.5">
            {clusters.map((cluster) => {
              const selected = selectedClusterId === cluster.id;
              return (
                <button
                  key={cluster.id}
                  type="button"
                  onClick={() => {
                    setSelectedClusterId((current) =>
                      current === cluster.id ? null : cluster.id,
                    );
                  }}
                  className={clsx(
                    "inline-flex cursor-pointer items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors duration-150",
                    selected
                      ? "border-accent/60 bg-accent-subtle text-text"
                      : "border-border bg-surface-elevated text-text-secondary hover:text-text",
                  )}
                >
                  <span
                    className="h-2 w-2 shrink-0 rounded-full"
                    style={{ backgroundColor: clusterColor(cluster.id) }}
                    aria-hidden="true"
                  />
                  {cluster.top_terms.slice(0, 2).join(", ") || `cluster ${String(cluster.id)}`}
                </button>
              );
            })}
          </div>

          <div className="rounded-card border border-border-subtle bg-surface/40 p-2">
            {scatterQuery.isPending ? (
              <div className="flex h-[480px] items-center justify-center gap-2 text-sm text-text-muted">
                <MapIcon size={16} className="animate-pulse" />
                Loading {formatCompactCount(10_000)} sample points…
              </div>
            ) : scatterQuery.isError ? (
              <div className="flex h-[480px] items-center justify-center text-sm text-text-muted">
                Couldn't load the scatter sample.
              </div>
            ) : (
              <ClusterScatterChart
                points={scatterQuery.data.points}
                clusters={scatterQuery.data.clusters}
                selectedClusterId={selectedClusterId}
              />
            )}
          </div>

          {scatterQuery.isSuccess && (
            <p className="flex items-center gap-1.5 text-xs text-text-muted">
              <LayoutGrid size={12} />
              {scatterQuery.data.n_points} of {formatCompactCount(n_docs)} documents
              shown (stratified sample) · hover a point for details, click a chip
              above to isolate a cluster.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
