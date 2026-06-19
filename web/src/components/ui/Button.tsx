import { forwardRef } from "react";
import type { ButtonHTMLAttributes } from "react";
import clsx from "clsx";

type ButtonVariant = "primary" | "secondary" | "ghost" | "outline";
type ButtonSize = "sm" | "md" | "lg";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  primary:
    "bg-accent text-[#140a04] hover:bg-accent-hover shadow-[0_0_0_1px_rgba(240,94,35,0.4),0_8px_24px_-8px_rgba(240,94,35,0.55)] hover:shadow-[0_0_0_1px_rgba(240,94,35,0.55),0_10px_28px_-6px_rgba(240,94,35,0.65)]",
  secondary:
    "bg-surface-elevated text-text border border-border-strong hover:border-text-muted hover:bg-surface-hover",
  ghost: "bg-transparent text-text-secondary hover:text-text hover:bg-surface-hover",
  outline:
    "bg-transparent text-text border border-border hover:border-border-strong hover:bg-surface-hover",
};

const SIZE_CLASSES: Record<ButtonSize, string> = {
  sm: "h-8 px-3 text-sm gap-1.5",
  md: "h-10 px-4 text-sm gap-2",
  lg: "h-12 px-6 text-base gap-2.5",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  function Button(
    { variant = "primary", size = "md", className, disabled, ...props },
    ref,
  ) {
    return (
      <button
        ref={ref}
        disabled={disabled}
        className={clsx(
          "inline-flex items-center justify-center whitespace-nowrap rounded-control font-semibold",
          "transition-all duration-150 ease-out cursor-pointer select-none",
          "disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none",
          "active:scale-[0.98]",
          VARIANT_CLASSES[variant],
          SIZE_CLASSES[size],
          className,
        )}
        {...props}
      />
    );
  },
);
