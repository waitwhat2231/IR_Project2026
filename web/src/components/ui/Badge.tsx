import type { ReactNode } from "react";
import clsx from "clsx";

type BadgeTone = "neutral" | "accent" | "success" | "info" | "danger";

interface BadgeProps {
  children: ReactNode;
  tone?: BadgeTone;
  mono?: boolean;
  className?: string;
}

const TONE_CLASSES: Record<BadgeTone, string> = {
  neutral: "bg-surface-elevated text-text-secondary border-border",
  accent: "bg-accent-subtle text-accent-warm border-accent/30",
  success: "bg-success-subtle text-success border-success/30",
  info: "bg-info-subtle text-info border-info/30",
  danger: "bg-danger-subtle text-danger border-danger/30",
};

export function Badge({
  children,
  tone = "neutral",
  mono = false,
  className,
}: BadgeProps) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium leading-none",
        mono && "font-mono",
        TONE_CLASSES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
