export function ResultsSkeleton() {
  return (
    <div className="space-y-3" aria-hidden="true">
      {Array.from({ length: 5 }, (_, index) => (
        <div
          key={index}
          className="animate-pulse rounded-card border border-border-subtle bg-surface/60 p-4"
        >
          <div className="flex items-center justify-between">
            <div className="h-4 w-1/3 rounded bg-surface-elevated" />
            <div className="h-4 w-16 rounded bg-surface-elevated" />
          </div>
          <div className="mt-3 h-3 w-1/4 rounded bg-surface-elevated" />
          <div className="mt-3 space-y-2">
            <div className="h-3 w-full rounded bg-surface-elevated" />
            <div className="h-3 w-full rounded bg-surface-elevated" />
            <div className="h-3 w-2/3 rounded bg-surface-elevated" />
          </div>
        </div>
      ))}
    </div>
  );
}
