import { SearchX } from "lucide-react";

interface EmptyStateProps {
  query: string;
}

export function EmptyState({ query }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-card border border-dashed border-border py-16 text-center">
      <SearchX size={28} className="text-text-muted" />
      <div className="space-y-1">
        <p className="text-sm font-medium text-text">
          No results for “{query}”
        </p>
        <p className="text-xs text-text-muted">
          Try a different model, broaden the query, or check the Enhanced
          execution mode for spell correction and synonym expansion.
        </p>
      </div>
    </div>
  );
}
