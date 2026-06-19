import type { SearchResponse } from "@/api/types";
import { RETRIEVAL_MODE_META } from "@/constants/retrievalModels";

interface MetaStripProps {
  response: SearchResponse;
}

export function MetaStrip({ response }: MetaStripProps) {
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-xs text-text-muted">
      <span>
        <span className="text-text-secondary">{response.total_results}</span>{" "}
        result{response.total_results === 1 ? "" : "s"}
      </span>
      <span aria-hidden="true">·</span>
      <span>{response.dataset}</span>
      <span aria-hidden="true">·</span>
      <span className="capitalize">{response.execution_mode}</span>
      <span aria-hidden="true">·</span>
      <span>{RETRIEVAL_MODE_META[response.retrieval_mode].label}</span>
    </div>
  );
}
