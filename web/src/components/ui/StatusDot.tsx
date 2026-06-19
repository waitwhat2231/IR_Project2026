import clsx from "clsx";

type StatusTone = "online" | "offline" | "pending";

interface StatusDotProps {
  status: StatusTone;
  className?: string;
}

const TONE_CLASSES: Record<StatusTone, string> = {
  online: "bg-success shadow-[0_0_8px_var(--color-success)]",
  offline: "bg-danger shadow-[0_0_8px_var(--color-danger)]",
  pending: "bg-text-muted",
};

export function StatusDot({ status, className }: StatusDotProps) {
  return (
    <span className={clsx("relative inline-flex h-2 w-2", className)}>
      {status === "online" && (
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-success opacity-60" />
      )}
      <span
        className={clsx(
          "relative inline-flex h-2 w-2 rounded-full",
          TONE_CLASSES[status],
        )}
      />
    </span>
  );
}
