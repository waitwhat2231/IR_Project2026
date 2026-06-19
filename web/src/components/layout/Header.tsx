import { SearchCode } from "lucide-react";
import { useHealth } from "@/hooks/useHealth";
import { useDatasets } from "@/hooks/useDatasets";
import { useUiStore } from "@/stores/uiStore";
import type { AppView } from "@/stores/uiStore";
import { StatusDot } from "@/components/ui/StatusDot";
import { Badge } from "@/components/ui/Badge";
import { Segmented } from "@/components/ui/Segmented";
import { formatDocCount } from "@/utils/format";

const VIEW_OPTIONS: { value: AppView; label: string }[] = [
  { value: "search", label: "Search" },
  { value: "clusters", label: "Clusters" },
  { value: "evaluation", label: "Evaluation" },
];

export function Header() {
  const healthQuery = useHealth();
  const datasetsQuery = useDatasets();
  const activeView = useUiStore((s) => s.activeView);
  const setActiveView = useUiStore((s) => s.setActiveView);

  const healthStatus =
    healthQuery.isPending ? "pending" : healthQuery.data?.status === "ok"
      ? "online"
      : "offline";

  const dataset = datasetsQuery.data?.datasets[0];

  return (
    <header className="sticky top-0 z-40 border-b border-border-subtle bg-bg/85 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between gap-6 px-5">
        <div className="flex items-center gap-2.5 text-text">
          <div className="flex h-8 w-8 items-center justify-center rounded-control bg-accent-subtle text-accent">
            <SearchCode size={17} strokeWidth={2.25} />
          </div>
          <span className="font-semibold tracking-tight">
            IR Search Console
          </span>
        </div>

        <nav className="hidden w-72 sm:block">
          <Segmented
            aria-label="Switch view"
            options={VIEW_OPTIONS}
            value={activeView}
            onChange={setActiveView}
          />
        </nav>

        <div className="flex items-center gap-3">
          {dataset && (
            <Badge tone="accent" mono className="hidden md:inline-flex">
              {dataset.name}
              <span className="text-text-muted">·</span>
              {formatDocCount(dataset.document_count)}
            </Badge>
          )}

          <div
            className="flex items-center gap-1.5 rounded-full border border-border bg-surface px-2.5 py-1.5"
            title={
              healthStatus === "online"
                ? "Gateway is reachable"
                : healthStatus === "offline"
                  ? "Gateway is unreachable"
                  : "Checking gateway status"
            }
          >
            <StatusDot status={healthStatus} />
            <span className="hidden text-xs font-medium text-text-secondary lg:inline">
              {healthStatus === "online"
                ? "Online"
                : healthStatus === "offline"
                  ? "Offline"
                  : "Checking"}
            </span>
          </div>
        </div>
      </div>

      {/* Mobile view switch — shown below the brand row on narrow screens */}
      <div className="px-5 pb-3 sm:hidden">
        <Segmented
          aria-label="Switch view"
          options={VIEW_OPTIONS}
          value={activeView}
          onChange={setActiveView}
        />
      </div>
    </header>
  );
}
