import type { KeyboardEvent } from "react";
import type { ClusterInfo } from "@/api/types";
import { clusterColor } from "@/constants/clusterColors";
import { CopyButton } from "@/components/ui/CopyButton";
import { formatCompactCount } from "@/utils/format";
import clsx from "clsx";

interface ClusterCardProps {
  cluster: ClusterInfo;
  selected?: boolean;
  onClick?: () => void;
}

export function ClusterCard({ cluster, selected = false, onClick }: ClusterCardProps) {
  const color = clusterColor(cluster.id);

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (!onClick) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onClick();
    }
  };

  return (
    <div
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      onClick={onClick}
      onKeyDown={handleKeyDown}
      className={clsx(
        "w-full rounded-card border p-4 text-left transition-colors duration-150",
        onClick && "cursor-pointer",
        selected
          ? "border-accent/60 bg-accent-subtle"
          : "border-border-subtle bg-surface/60 hover:border-border",
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <span
            className="h-2.5 w-2.5 shrink-0 rounded-full"
            style={{ backgroundColor: color }}
            aria-hidden="true"
          />
          <span className="font-mono text-xs text-text-muted">
            cluster {cluster.id}
          </span>
        </div>
        <span className="shrink-0 font-mono text-xs text-text-secondary">
          {formatCompactCount(cluster.size)} docs · {cluster.pct.toFixed(1)}%
        </span>
      </div>

      <div className="mt-2.5 flex flex-wrap gap-1.5">
        {cluster.top_terms.slice(0, 6).map((term) => (
          <span
            key={term}
            className="rounded-full border border-border bg-surface-elevated px-2 py-0.5 text-xs text-text-secondary"
          >
            {term}
          </span>
        ))}
      </div>

      {cluster.representative_doc_ids.length > 0 && (
        <div className="mt-3 space-y-1 border-t border-border-subtle pt-2.5">
          <p className="text-[11px] font-medium uppercase tracking-wide text-text-muted">
            Representative doc_ids
          </p>
          <div className="space-y-1">
            {cluster.representative_doc_ids.map((docId) => (
              <div key={docId} className="flex items-center gap-1.5">
                <code className="truncate font-mono text-xs text-text-secondary">
                  {docId}
                </code>
                <span onClick={(e) => { e.stopPropagation(); }}>
                  <CopyButton value={docId} label="doc_id" />
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
