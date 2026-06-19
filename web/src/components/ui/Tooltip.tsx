import type { ReactNode } from "react";

interface TooltipProps {
  content: string;
  children: ReactNode;
}

export function Tooltip({ content, children }: TooltipProps) {
  return (
    <span className="group/tooltip relative inline-flex">
      {children}
      <span
        role="tooltip"
        className="pointer-events-none absolute -top-2 left-1/2 z-50 w-max max-w-64 -translate-x-1/2 -translate-y-full rounded-md border border-border-strong bg-surface-elevated px-2.5 py-1.5 text-xs leading-snug text-text-secondary opacity-0 shadow-lg transition-opacity duration-150 group-hover/tooltip:opacity-100"
      >
        {content}
      </span>
    </span>
  );
}
